//! Shortcutting a planned path into a smoother one (`smoothPath` in `sst/smoothing.py`).
use std::collections::BTreeSet;
use std::f64::consts::PI;

use crate::config::{
    LOOP_DETOUR_RATIO, LOOP_SPAN, LOOP_TURN, NUDGE_NODE_FRAMES, PLAN_DT, SMOOTH_BUMPS, SMOOTH_JOIN_SLACK,
    SMOOTH_LOOKAHEADS, SMOOTH_LOOP_SPAN, SMOOTH_MIN_GAIN, SMOOTH_MIN_SPEED, SMOOTH_NODES_AHEAD, SMOOTH_RETRIES,
    SMOOTH_STEER_BLEND, SMOOTH_SWEEPS, SMOOTH_TANGENTS, SMOOTH_TARGET_SPACING, SMOOTH_TOLERANCE_SCALE, WHEELBASE,
};
use crate::kinematics::{Action, feasible_accel, max_steer_at_speed, propagate, wrap};
use crate::map::Map;
use crate::route::{Route, route_join, splice_route};
use crate::tree::State;

/// Whether the stretch of `route` from node index a to a later node at most LOOP_SPAN px further along is a loop or
/// big wiggle: driven at least LOOP_DETOUR_RATIO times the straight line across, or turning through LOOP_TURN.
pub fn route_loopy(route: &Route, a: usize) -> bool {
    let k = route.nodes.iter().position(|&n| n == a).unwrap();
    let p = &route.states[a];
    let (mut turning, mut last) = (0.0, a);
    for &c in &route.nodes[k + 1..] {
        if route.arc[c] - route.arc[a] > LOOP_SPAN {
            break;
        }
        turning += (last..c).map(|i| wrap(route.states[i + 1].a - route.states[i].a).abs()).sum::<f64>();
        last = c;
        let q = &route.states[c];
        if turning >= LOOP_TURN || route.arc[c] - route.arc[a] >= LOOP_DETOUR_RATIO * (q.x - p.x).hypot(q.y - p.y).max(1.0) {
            return true;
        }
    }
    false
}

/// Dense points and their arc lengths along the cubic Hermite curve from p's position and heading to q's, with both
/// end tangents `tangent` times the straight distance.
fn hermite_curve(p: &State, q: &State, tangent: f64) -> (Vec<(f64, f64)>, Vec<f64>) {
    let d = (q.x - p.x).hypot(q.y - p.y);
    let (t0x, t0y) = (p.a.cos() * tangent * d, p.a.sin() * tangent * d);
    let (t1x, t1y) = (q.a.cos() * tangent * d, q.a.sin() * tangent * d);
    let n = 20.max((d / 2.0) as usize);
    let points: Vec<(f64, f64)> = (0..n)
        .map(|k| {
            let u = k as f64 / (n - 1) as f64;
            let (u2, u3) = (u * u, u * u * u);
            let (h00, h10, h01, h11) = (2.0 * u3 - 3.0 * u2 + 1.0, u3 - 2.0 * u2 + u, -2.0 * u3 + 3.0 * u2, u3 - u2);
            (h00 * p.x + h10 * t0x + h01 * q.x + h11 * t1x, h00 * p.y + h10 * t0y + h01 * q.y + h11 * t1y)
        })
        .collect();
    let mut arcs = vec![0.0];
    for w in points.windows(2) {
        arcs.push(arcs.last().unwrap() + (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1));
    }
    (points, arcs)
}

/// A shortcut that rejoins the route: total cost to the goal, its states, the route index it joins at, the gap cost.
pub struct Shortcut {
    pub total: f64,
    pub states: Vec<State>,
    pub join: usize,
    pub gap: f64,
}

/// Drive from route state p (index a) along a Hermite curve to route state q with pure pursuit (`lookahead` px ahead
/// on the curve), the speed going linearly from p's to q's plus `bump` times the faster in the middle, the steering
/// blended toward q's over the last SMOOTH_STEER_BLEND. Every PLAN_DT step goes through propagate, so the planner's
/// limits and collision checks apply. Past half way, each step is checked for a join within `scale` of the join
/// tolerance (indices a+1 to hi). Returns the cheapest join.
#[allow(clippy::too_many_arguments)]
pub fn shortcut_drive(map: &Map, p: &State, q: &State, route: &Route, a: usize, hi: usize, tangent: f64,
                      lookahead: f64, bump: f64, scale: f64) -> Option<Shortcut> {
    let (points, arcs) = hermite_curve(p, q, tangent);
    let length = *arcs.last().unwrap();
    if length < 1.0 {
        return None;
    }
    let mut x = State { steer: p.steer, ..State::new(p.x, p.y, p.a, p.s) };
    let top = p.s.abs().max(q.s.abs());
    let (mut states, mut best, mut k): (Vec<State>, Option<Shortcut>, usize) = (Vec::new(), None, 0);
    let max_steps = (1.5 * length / p.s.abs().min(q.s.abs()).max(20.0) / PLAN_DT) as usize + 40;
    for step in 0..max_steps {
        // Progress: the nearest curve point, searching forward from the last one
        let window = &points[k..(k + 40).min(points.len())];
        let nearest = (0..window.len())
            .min_by(|&i, &j| {
                let d = |(px, py): (f64, f64)| (px - x.x).hypot(py - x.y);
                d(window[i]).total_cmp(&d(window[j]))
            })
            .unwrap();
        k += nearest;
        let f = arcs[k] / length;
        if f >= 0.999 && step > 0 && best.is_some() {
            break;
        }
        let j = arcs.partition_point(|&s| s < arcs[k] + lookahead).min(points.len() - 1);
        let (mut tx, mut ty) = points[j];
        if j == points.len() - 1 {
            // past the curve: aim along q's heading
            let extra = arcs[k] + lookahead - length;
            tx += extra * q.a.cos();
            ty += extra * q.a.sin();
        }
        let alpha = (ty - x.y).atan2(tx - x.x) - x.a;
        let dist = (tx - x.x).hypot(ty - x.y).max(1e-6);
        let mut desired = (2.0 * alpha.sin() * WHEELBASE / dist).atan();
        let w = ((f - (1.0 - SMOOTH_STEER_BLEND)) / SMOOTH_STEER_BLEND).clamp(0.0, 1.0);
        desired = (1.0 - w) * desired + w * q.steer;
        let v_ref = p.s.abs() + (q.s.abs() - p.s.abs()) * f + bump * top * (PI * f).sin();
        let v_ref = v_ref.max(if f < 0.9 { SMOOTH_MIN_SPEED } else { q.s.abs() });
        let (accel, lo, hi_rate) = feasible_accel(&x, PLAN_DT, (v_ref - x.s) / PLAN_DT);
        let limit = max_steer_at_speed(x.s.abs().max((x.s + accel * PLAN_DT).abs()));
        let desired = desired.clamp(-limit, limit);
        let action = Action { rate: ((desired - x.steer) / PLAN_DT).clamp(lo, lo.max(hi_rate)), accel };
        let mut seg = propagate(map, &x, action, 1);
        if seg.is_empty() {
            return best;
        }
        if step % NUDGE_NODE_FRAMES == NUDGE_NODE_FRAMES - 1 {
            seg.last_mut().unwrap().node = true;
        }
        states.extend(&seg);
        x = *seg.last().unwrap();
        if f >= 0.5
            && let Some((i, gap)) = route_join(&x, route, a + 1, hi, scale) {
                let total = x.cost + gap + route.remaining[i];
                if best.as_ref().is_none_or(|b| total < b.total) {
                    best = Some(Shortcut { total, states: states.clone(), join: i, gap });
                }
            }
    }
    best
}

/// One sweep over the route's nodes: from each, try shortcut_drive to route states up to SMOOTH_NODES_AHEAD nodes
/// ahead (every SMOOTH_TARGET_SPACING px, plus the nodes; out to SMOOTH_LOOP_SPAN px if the stretch is loopy) with
/// every combination of tangent, lookahead and bump, and splice in the cheapest if it's at least SMOOTH_MIN_GAIN
/// cheaper to the goal (then try the same node again). Returns the route and the number of splices made.
fn smooth_route(map: &Map, mut route: Route) -> (Route, usize) {
    let (mut made, mut k, mut tries_here) = (0, 0, 0);
    while k + 1 < route.nodes.len() {
        let a = route.nodes[k];
        let p = route.states[a];
        if p.s < 0.0 {
            // reversing: leave it
            (k, tries_here) = (k + 1, 0);
            continue;
        }
        let ahead = &route.nodes[k + 1..(k + 1 + SMOOTH_NODES_AHEAD).min(route.nodes.len())];
        let mut end = *ahead.last().unwrap();
        if route_loopy(&route, a) {
            while end + 1 < route.states.len() && route.arc[end + 1] - route.arc[a] <= SMOOTH_LOOP_SPAN {
                end += 1;
            }
        }
        let mut targets: BTreeSet<usize> = ahead.iter().copied().collect();
        let mut last_arc = route.arc[a];
        for i in a + 1..=end {
            if route.arc[i] - last_arc >= SMOOTH_TARGET_SPACING {
                targets.insert(i);
                last_arc = route.arc[i];
            }
        }
        let hi = route.states.len().min(end + SMOOTH_JOIN_SLACK);
        let mut best: Option<Shortcut> = None;
        for &t in &targets {
            let q = route.states[t];
            if q.s < 0.0 {
                continue;
            }
            for tangent in SMOOTH_TANGENTS {
                for lookahead in SMOOTH_LOOKAHEADS {
                    for bump in SMOOTH_BUMPS {
                        if let Some(found) = shortcut_drive(map, &p, &q, &route, a, hi, tangent, lookahead, bump,
                                                                    SMOOTH_TOLERANCE_SCALE)
                            && best.as_ref().is_none_or(|b| found.total < b.total) {
                                best = Some(found);
                            }
                    }
                }
            }
        }
        if let Some(Shortcut { total, mut states, join, gap }) = best
            && total < route.remaining[a] * (1.0 - SMOOTH_MIN_GAIN) {
                let last = states.last_mut().unwrap();
                last.node = true;
                last.cost += gap; // the bit from the shortcut's end on to the join, so the route's costs stay right
                route = splice_route(&route, a, states, Some((&route, join)));
                made += 1;
                tries_here += 1;
                if tries_here < SMOOTH_RETRIES {
                    continue; // same node again, on the new route
                }
            }
        (k, tries_here) = (k + 1, 0);
    }
    (route, made)
}

/// Smooth a planned path (states from the root) with smooth_route sweeps until one makes no change or SMOOTH_SWEEPS
/// have run. The returned states' costs run consistently along the path (cost so far, from the remaining costs), and
/// their times follow from the speed along the arc.
pub fn smooth_path(map: &Map, path: Vec<State>) -> Vec<State> {
    let mut route = Route::new(path, None);
    for _ in 0..SMOOTH_SWEEPS {
        let made;
        (route, made) = smooth_route(map, route);
        if made == 0 {
            break;
        }
    }
    let mut t = 0.0;
    (0..route.states.len())
        .map(|i| {
            let mut s = route.states[i];
            s.cost = route.remaining[0] - route.remaining[i];
            if i > 0 {
                let ds = route.arc[i] - route.arc[i - 1];
                t += ds / ((s.s.abs() + route.states[i - 1].s.abs()) / 2.0).max(1.0);
            }
            s.time = t;
            s
        })
        .collect()
}
