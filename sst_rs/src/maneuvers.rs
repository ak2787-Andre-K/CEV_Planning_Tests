//! Precomputed maneuvers (`sst/maneuvers.py`): turnarounds, and the maneuver library the Driver uses to get onto a
//! known route nearby.
use std::collections::{BTreeSet, HashSet};
use std::f64::consts::TAU;
use std::sync::OnceLock;

use crate::config::{
    FOOTPRINT_OFFSETS, FOOTPRINT_RADIUS, FOOTPRINT_REACH, FREE, JOIN_SPEED_OVER, JOIN_TOLERANCE_SCALE, LIB_BINS,
    LIB_CANDIDATES, LIB_DEPTH, LIB_KEY_CELL, LIB_MIN_SPEED, LIB_SPEEDS, LIB_STEP_FRAMES, LIB_TARGET_OFFSETS,
    MAX_ACCELERATION, MAX_LATERAL_ACCELERATION, MAX_SPEED, MAX_STEER, MAX_STEER_RATE, NUDGE_NODE_FRAMES, PLAN_DT,
    RECONNECT_JOIN_WINDOW, SAFETY_MARGIN, SCREEN_HEIGHT, SCREEN_WIDTH, WHEELBASE,
};
use crate::kinematics::{Action, can_stop_clear, feasible_accel, max_steer_at_speed, propagate, wrap};
use crate::map::{Map, cell};
use crate::route::{Route, join_tolerance, route_join};
use crate::tree::State;

/// Drive a turnaround from `start`, one PLAN_DT step at a time through propagate: for each phase (steering fraction,
/// speed, total heading change), stop, set the wheel in place, then drive until the heading change reaches the
/// target; finally stop and straighten the wheel in place. Returns the states, or an empty list if it hits something
/// or runs out of frames.
pub fn run_turnaround(map: &Map, start: &State, phases: &[(f64, f64, f64)]) -> Vec<State> {
    const MAX_FRAMES: usize = 3000;
    struct Drive<'a> {
        map: &'a Map,
        x: State,
        states: Vec<State>,
        turned: f64,
        frames: usize,
    }
    impl Drive<'_> {
        fn step(&mut self, target_speed: f64, target_steer: f64) -> bool {
            self.frames += 1;
            if self.frames > MAX_FRAMES {
                return false;
            }
            let (accel, lo, hi) = feasible_accel(&self.x, PLAN_DT, (target_speed - self.x.s) / PLAN_DT);
            let rate = ((target_steer - self.x.steer) / PLAN_DT).clamp(lo, lo.max(hi));
            let mut seg = propagate(self.map, &self.x, Action { rate, accel }, 1);
            let Some(last) = seg.last_mut() else { return false };
            self.turned += wrap(last.a - self.x.a);
            if self.frames.is_multiple_of(NUDGE_NODE_FRAMES) {
                last.node = true; // a node, so the safe-speed look-ahead has points along it
            }
            self.x = *last;
            self.states.extend(seg);
            true
        }

        fn stop_and_set_wheel(&mut self, target_steer: f64) -> bool {
            while self.x.s.abs() > 1e-3 || (self.x.steer - target_steer).abs() > 1e-3 {
                if !self.step(0.0, target_steer) {
                    return false;
                }
            }
            true
        }
    }
    let mut d = Drive { map, x: *start, states: Vec::new(), turned: 0.0, frames: 0 };
    for &(steer_fraction, speed, until) in phases {
        let wheel = steer_fraction * max_steer_at_speed(speed.abs());
        if !d.stop_and_set_wheel(wheel) {
            return Vec::new();
        }
        // Stop driving early by the turn the car will still make while braking to a stop at this lock
        while d.turned.abs() + d.x.s * d.x.s / (2.0 * MAX_ACCELERATION) * d.x.steer.tan().abs() / WHEELBASE < until {
            if !d.step(speed, wheel) {
                return Vec::new();
            }
        }
    }
    if !d.stop_and_set_wheel(0.0) {
        return Vec::new();
    }
    if let Some(last) = d.states.last_mut() {
        last.node = true;
    }
    d.states
}

/// Radius of free circle the car needs, entering at `speed`, to come out at any heading.
pub fn turn_circle_radius(speed: f64) -> f64 {
    23.5f64.max(speed * speed / (4.0 * MAX_ACCELERATION) + 9.0)
}

/// How far (px) from its start the library explores for a start speed: across the circle it needs.
pub fn lib_range(speed: f64) -> f64 {
    60f64.max(2.0 * turn_circle_radius(speed) + 10.0)
}

/// Radius of the known-free circle around (cx, cy): the distance to the nearest cell that isn't known FREE (unknown
/// counts as blocked, and so does the map's edge), at most max_radius.
fn free_radius(map: &Map, cx: f64, cy: f64, max_radius: f64) -> f64 {
    let edge = cx.min(cy).min(SCREEN_WIDTH as f64 - 1.0 - cx).min(SCREEN_HEIGHT as f64 - 1.0 - cy);
    let x0 = (cx - max_radius).floor().max(0.0) as usize;
    let x1 = ((cx + max_radius).ceil() + 1.0).max(0.0).min(SCREEN_WIDTH as f64) as usize;
    let y0 = (cy - max_radius).floor().max(0.0) as usize;
    let y1 = ((cy + max_radius).ceil() + 1.0).max(0.0).min(SCREEN_HEIGHT as f64) as usize;
    let mut nearest = f64::INFINITY;
    for x in x0..x1 {
        for y in y0..y1 {
            if map.occupancy[cell(x, y)] != FREE {
                nearest = nearest.min((x as f64 - cx).hypot(y as f64 - cy));
            }
        }
    }
    max_radius.min(edge).min(nearest).max(0.0)
}

const LIB_ACCELS: [f64; 3] = [-MAX_ACCELERATION, 0.0, MAX_ACCELERATION];
const LIB_RATES: [f64; 3] = [-MAX_STEER_RATE, 0.0, MAX_STEER_RATE];

/// The (acceleration, steering rate) of each step of a library sequence, first step first (codes are base 9, first
/// step most significant).
fn lib_controls(mut code: u64, depth: usize) -> Vec<(f64, f64)> {
    let mut steps = Vec::with_capacity(depth);
    for _ in 0..depth {
        let c = (code % 9) as usize;
        code /= 9;
        steps.push((LIB_ACCELS[c / 3], LIB_RATES[c % 3]));
    }
    steps.reverse();
    steps
}

/// LIB_STEP_FRAMES PLAN_DT frames of (acc, rate) from (x, y, a, v, s), the steering kept within the lateral limit and
/// the speed within [0, MAX_SPEED], integrated like propagate.
fn lib_step(mut p: [f64; 5], acc: f64, rate: f64) -> [f64; 5] {
    let [x, y, a, v, s] = &mut p;
    for _ in 0..LIB_STEP_FRAMES {
        let v_end = (*v + acc * PLAN_DT).clamp(0.0, MAX_SPEED);
        let top = v.max(v_end).max(1e-9);
        let limit = MAX_STEER.min((MAX_LATERAL_ACCELERATION * WHEELBASE / (top * top)).atan());
        *s = (*s + rate * PLAN_DT).clamp(-limit, limit);
        let k = s.tan() / WHEELBASE;
        let d = (*v + v_end) / 2.0 * PLAN_DT;
        let a_end = *a + k * d;
        if k.abs() < 1e-9 {
            (*x, *y) = (*x + d * a.cos(), *y + d * a.sin());
        } else {
            (*x, *y) = (*x + (a_end.sin() - a.sin()) / k, *y - (a_end.cos() - a.cos()) / k);
        }
        (*a, *v) = (a_end, v_end);
    }
    p
}

/// Lookup key of an end position and heading.
fn lib_key(x: f64, y: f64, a: f64) -> i64 {
    let (cell, acell) = LIB_KEY_CELL;
    let xb = (x / cell).round_ties_even() as i64 + 10000;
    let yb = (y / cell).round_ties_even() as i64 + 10000;
    let bins = (TAU / acell).round() as i64;
    let ab = (a.rem_euclid(TAU) / acell).round_ties_even() as i64 % bins;
    (xb * 20001 + yb) * 64 + ab
}

/// The library for one start speed: end states relative to a start at the origin facing +x with the wheel straight,
/// sequence codes and depths, sorted by lookup key.
struct Entries {
    end: Vec<[f32; 5]>, // x, y, heading, speed, steering (f32, as in Python: millions of entries)
    code: Vec<u64>,
    depth: Vec<u8>,
    key: Vec<i64>,
}

/// Explore every sequence for a start speed, keeping the first one into each bin of end position, heading, speed and
/// steering (LIB_BINS), within lib_range of the start and never slower than LIB_MIN_SPEED.
fn build_entries(v0: f64) -> Entries {
    let (bx, ba, bv, bs) = LIB_BINS;
    let dedup = |p: &[f64; 5]| -> i64 {
        let r = |q: f64| q.round_ties_even() as i64;
        ((((r(p[0] / bx) + 1000) * 2001 + r(p[1] / bx) + 1000) * 64 + r(p[2].rem_euclid(TAU) / ba) % 36) * 32
            + r(p[3] / bv))
            * 16
            + r(p[4] / bs)
            + 8
    };
    let reach = lib_range(v0);
    let start = [0.0, 0.0, 0.0, v0, 0.0];
    let mut frontier = vec![(start, 0u64)];
    let mut visited: HashSet<i64> = HashSet::from([dedup(&start)]);
    let mut found: Vec<([f64; 5], u64, usize)> = Vec::new();
    for d in 1..=LIB_DEPTH {
        if frontier.is_empty() {
            break;
        }
        // Children in parent order, then the first of each bin, in bin order (like numpy's unique)
        let mut children: Vec<(i64, usize, [f64; 5], u64)> = Vec::new();
        for (p, code) in &frontier {
            for c in 0..9 {
                let q = lib_step(*p, LIB_ACCELS[c / 3], LIB_RATES[c % 3]);
                if q[3] >= LIB_MIN_SPEED && q[0].hypot(q[1]) <= reach {
                    children.push((dedup(&q), children.len(), q, code * 9 + c as u64));
                }
            }
        }
        children.sort_by_key(|c| (c.0, c.1));
        children.dedup_by_key(|c| c.0);
        frontier.clear();
        for (key, _, q, code) in children {
            if visited.insert(key) {
                frontier.push((q, code));
                found.push(([q[0], q[1], q[2].rem_euclid(TAU), q[3], q[4]], code, d));
            }
        }
    }
    let mut keyed: Vec<(i64, usize)> = found.iter().enumerate().map(|(i, f)| (lib_key(f.0[0], f.0[1], f.0[2]), i)).collect();
    keyed.sort();
    Entries {
        end: keyed.iter().map(|&(_, i)| found[i].0.map(|v| v as f32)).collect(),
        code: keyed.iter().map(|&(_, i)| found[i].1).collect(),
        depth: keyed.iter().map(|&(_, i)| found[i].2 as u8).collect(),
        key: keyed.iter().map(|&(k, _)| k).collect(),
    }
}

/// The maneuver library, one set of entries per LIB_SPEEDS start speed (about 10 million in all), built the first
/// time it's needed (about 20 s) unless warm_up started it earlier.
fn library() -> &'static Vec<Entries> {
    static LIBRARY: OnceLock<Vec<Entries>> = OnceLock::new();
    LIBRARY.get_or_init(|| LIB_SPEEDS.iter().map(|&v0| build_entries(v0)).collect())
}

/// A library maneuver from `start` (moving forward at most LIB_SPEEDS' top + 5 px/s) onto `route` near state i, or
/// None. The known-free circle around the midpoint of the two must be at least turn_circle_radius for the speed and
/// hold both ends, and the maneuver must stay inside it. Candidates are the entries whose end (rotated to the start)
/// lies within a lookup bin of route state i + LIB_TARGET_OFFSETS; they're ranked by time plus the time later lost
/// speeding up to the route's speed, and the best LIB_CANDIDATES are re-driven from the real start through propagate
/// until one joins the route within the live join tolerance at a state the car can stop from. Returns (cost of the
/// maneuver to the join plus the gap, states, join index).
pub fn library_join(map: &Map, start: &State, route: &Route, i: usize) -> Option<(f64, Vec<State>, usize)> {
    let top_speed = LIB_SPEEDS[LIB_SPEEDS.len() - 1];
    if start.s < 0.0 || start.s > top_speed + 5.0 {
        return None;
    }
    let q = &route.states[i];
    let dist = (q.x - start.x).hypot(q.y - start.y);
    let reach = lib_range(start.s);
    if q.s <= 0.0 || dist > reach {
        return None;
    }
    let pad = FOOTPRINT_RADIUS + SAFETY_MARGIN;
    let (cx, cy) = ((start.x + q.x) / 2.0, (start.y + q.y) / 2.0);
    let radius = free_radius(map, cx, cy, reach);
    if radius < turn_circle_radius(start.s).max(dist / 2.0 + FOOTPRINT_REACH + pad) {
        return None;
    }
    let nearest = (0..LIB_SPEEDS.len())
        .min_by(|&a, &b| (LIB_SPEEDS[a] - start.s).abs().total_cmp(&(LIB_SPEEDS[b] - start.s).abs()))
        .unwrap();
    let lib = &library()[nearest];
    let (sn, c) = start.a.sin_cos();
    let (cell_size, acell) = LIB_KEY_CELL;
    let first = route.valid_from.get().max(1);
    let targets: BTreeSet<usize> = LIB_TARGET_OFFSETS
        .iter()
        .map(|&o| i as i64 + o)
        .filter(|&t| t >= first as i64 && (t as usize) < route.states.len())
        .map(|t| t as usize)
        .collect();
    let (mut picks, mut wants) = (Vec::new(), Vec::new());
    for &t in &targets {
        let p = &route.states[t];
        if p.s <= 0.0 {
            continue;
        }
        let (dx, dy) = (p.x - start.x, p.y - start.y);
        let (tx, ty, ta) = (dx * c + dy * sn, -dx * sn + dy * c, (p.a - start.a).rem_euclid(TAU));
        let mut keys = BTreeSet::new();
        for a_ in [-1.0, 0.0, 1.0] {
            for b_ in [-1.0, 0.0, 1.0] {
                for k_ in [-1.0, 0.0, 1.0] {
                    keys.insert(lib_key(tx + a_ * cell_size, ty + b_ * cell_size, ta + k_ * acell));
                }
            }
        }
        for key in keys {
            let lo = lib.key.partition_point(|&k| k < key);
            let hi = lib.key.partition_point(|&k| k <= key);
            for e in lo..hi {
                picks.push(e);
                wants.push([tx, ty, ta, p.s, p.steer]);
            }
        }
    }
    // Keep the entries that end slow enough and with about the right steering, scored by time + speed lost + miss
    let mut scored: Vec<(f64, usize)> = Vec::new();
    let mut seen = HashSet::new();
    for (&e, w) in picks.iter().zip(&wants) {
        let end = lib.end[e].map(|v| v as f64);
        let steer_tol = 2.0 * join_tolerance(w[3]).2 + LIB_BINS.3;
        if end[3] > w[3] + JOIN_SPEED_OVER || (end[4] - w[4]).abs() > steer_tol || !seen.insert(e) {
            continue;
        }
        let duration = lib.depth[e] as f64 * LIB_STEP_FRAMES as f64 * PLAN_DT;
        let lost = (w[3] - end[3]).max(0.0).powi(2) / (2.0 * MAX_ACCELERATION * w[3].max(1.0));
        let miss = (end[0] - w[0]).hypot(end[1] - w[1]) / MAX_SPEED + wrap(end[2] - w[2]).abs() * 0.1;
        scored.push((duration + lost + miss, e));
    }
    scored.sort_by(|a, b| a.0.total_cmp(&b.0));
    scored.truncate(LIB_CANDIDATES);
    let (t_first, t_last) = (*targets.first()?, *targets.last()?);
    let lo_i = (t_first.saturating_sub(RECONNECT_JOIN_WINDOW.0)).max(first) - 1;
    let hi_i = route.states.len().min(t_last + RECONNECT_JOIN_WINDOW.1);
    let mut best: Option<(f64, Vec<State>, usize, f64)> = None;
    for &(_, e) in &scored {
        let steps = lib_controls(lib.code[e], lib.depth[e] as usize);
        let total_frames = steps.len() * LIB_STEP_FRAMES;
        let mut x = *start;
        let (mut states, mut found, mut frames) = (Vec::new(), None, 0usize);
        'drive: for &(acc, rate) in &steps {
            for _ in 0..LIB_STEP_FRAMES {
                frames += 1;
                let acc_f = acc.clamp(-x.s / PLAN_DT, (MAX_SPEED - x.s) / PLAN_DT);
                let limit = max_steer_at_speed(x.s.max(x.s + acc_f * PLAN_DT));
                let rate_f = rate.clamp((-limit - x.steer) / PLAN_DT, (limit - x.steer) / PLAN_DT);
                let mut seg = propagate(map, &x, Action { rate: rate_f, accel: acc_f }, 1);
                let Some(p) = seg.last_mut() else { break 'drive };
                let (ps, pc) = p.a.sin_cos();
                if FOOTPRINT_OFFSETS.iter().any(|o| (p.x + o * pc - cx).hypot(p.y + o * ps - cy) > radius - pad) {
                    break 'drive;
                }
                if frames.is_multiple_of(NUDGE_NODE_FRAMES) {
                    p.node = true;
                }
                x = *p;
                states.extend(seg);
                if frames >= total_frames / 2
                    && let Some(join) = route_join(&x, route, lo_i, hi_i, JOIN_TOLERANCE_SCALE)
                    && can_stop_clear(map, &x)
                {
                    found = Some(join);
                    break 'drive;
                }
            }
        }
        let Some((j, gap)) = found else { continue };
        let cost = x.cost - start.cost + gap;
        if best.as_ref().is_none_or(|b| cost + route.remaining[j] < b.0 + route.remaining[b.2]) {
            best = Some((cost, states, j, gap));
        }
    }
    let (cost, mut states, j, gap) = best?;
    let last = states.last_mut().unwrap();
    last.node = true;
    last.cost += gap;
    Some((cost, states, j))
}

/// Start building the library on a background thread, so the first library_join doesn't stall the drive. A
/// library_join before it's done waits for it.
pub fn warm_up() {
    std::thread::spawn(library);
}
