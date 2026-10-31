//! Car motion (`sst/car.py` and `sst/kinematics.py`): the steering limits, random actions, and propagating an action
//! through the bicycle model with a collision check at every step.
use crate::config::{
    CHECK_SPACING, COST_ACCEL, COST_DECEL, COST_STEER_ANGLE, COST_STEER_RATE, EARLY_ACCEL_SIGMA, FOOTPRINT_OFFSETS,
    FOOTPRINT_RADIUS, FOOTPRINT_REACH, MAX_ACCELERATION, MAX_LATERAL_ACCELERATION, MAX_REVERSE_SPEED, MAX_SPEED,
    MAX_STEER, MAX_STEER_RATE, PLAN_DT, SAFE_SPEED_STEP, SAFETY_MARGIN, TIME_PROP, WHEELBASE,
};
use crate::map::Map;
use crate::rng::Random;
use crate::tree::State;

/// A control held over an edge.
#[derive(Clone, Copy, Debug)]
pub struct Action {
    pub rate: f64,  // steering rate, rad/s
    pub accel: f64, // acceleration, px/s^2
}

/// Wrap an angle difference into [-pi, pi).
#[inline]
pub fn wrap(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

/// The most steering the lateral acceleration limit allows at `speed`.
pub fn max_steer_at_speed(speed: f64) -> f64 {
    if speed == 0.0 {
        return MAX_STEER;
    }
    MAX_STEER.min((MAX_LATERAL_ACCELERATION * WHEELBASE / (speed * speed)).atan())
}

/// Steering rates that keep every frame of an edge within the lateral limit for this acceleration: (lo, hi), with
/// lo > hi when no rate works. Frame k drives with steer0 + rate * k * PLAN_DT at up to that frame's top speed.
fn steer_rate_bounds(speed0: f64, steer0: f64, accel: f64, frames: usize) -> (f64, f64) {
    let (mut lo, mut hi) = (-MAX_STEER_RATE, MAX_STEER_RATE);
    let mut v_prev = speed0;
    for k in 1..=frames {
        let v = speed0 + accel * k as f64 * PLAN_DT;
        let limit = max_steer_at_speed(v_prev.abs().max(v.abs()));
        let t = k as f64 * PLAN_DT;
        lo = lo.max((-limit - steer0) / t);
        hi = hi.min((limit - steer0) / t);
        v_prev = v;
    }
    (lo, hi)
}

/// Accelerations that keep the speed within its limits over an edge of `time`.
pub fn accel_range(x: &State, time: f64) -> (f64, f64) {
    (
        (-MAX_ACCELERATION).max((-MAX_REVERSE_SPEED - x.s) / time),
        MAX_ACCELERATION.min((MAX_SPEED - x.s) / time),
    )
}

/// Clamp `accel` to accel_range, then, if no steering rate keeps the edge within the lateral limit, move it halfway
/// toward braking until one does. Returns (accel, rate_lo, rate_hi).
pub fn feasible_accel(x: &State, time: f64, accel: f64) -> (f64, f64, f64) {
    let frames = ((time / PLAN_DT).round() as usize).max(1);
    let (a_lo, a_hi) = accel_range(x, time);
    // max then min, not clamp: like Python's min(max(...)) the upper bound wins if the range is ever reversed
    #[allow(clippy::manual_clamp)]
    let mut accel = accel.max(a_lo).min(a_hi);
    #[allow(clippy::manual_clamp)]
    let braking = (-x.s / time).max(-MAX_ACCELERATION).min(MAX_ACCELERATION);
    for _ in 0..20 {
        let (lo, hi) = steer_rate_bounds(x.s, x.steer, accel, frames);
        if lo <= hi {
            return (accel, lo, hi);
        }
        accel = (accel + braking) / 2.0;
    }
    let (lo, hi) = steer_rate_bounds(x.s, x.steer, braking, frames);
    (braking, lo, hi)
}

/// Random (steering rate, acceleration) to hold for `time`: acceleration uniform over what keeps the speed in range
/// (or a normal around `accel_mean`), and the rate that reaches a normal target steering angle (spread `steer_sigma`
/// times the most allowed) at the end of the edge, capped so every frame stays within the lateral limit.
pub fn random_action(rng: &mut Random, x: &State, time: f64, steer_sigma: f64, accel_mean: Option<f64>) -> Action {
    let accel = match accel_mean {
        None => {
            let (lo, hi) = accel_range(x, time);
            rng.uniform(lo, hi)
        }
        Some(mean) => rng.gauss(mean, EARLY_ACCEL_SIGMA),
    };
    let (accel, lo, hi) = feasible_accel(x, time, accel);
    let v_end = x.s + accel * time;
    let target = rng.gauss(0.0, steer_sigma * max_steer_at_speed(x.s.abs().max(v_end.abs())));
    Action { rate: ((target - x.steer) / time).clamp(lo, lo.max(hi)), accel }
}

/// Rear-axle pose after driving signed arc length d from (x, y) at heading a with constant curvature.
#[inline]
pub fn arc_pose(x: f64, y: f64, a: f64, curvature: f64, d: f64) -> (f64, f64, f64) {
    if curvature.abs() < 1e-9 {
        return (x + d * a.cos(), y + d * a.sin(), a);
    }
    let a_end = a + curvature * d;
    (x + (a_end.sin() - a.sin()) / curvature, y - (a_end.cos() - a.cos()) / curvature, a_end)
}

/// Gap between the car's footprint circles and the nearest obstacle, rear axle at (x, y) with heading a.
/// 0 or less means a collision (or a circle center off the map).
pub fn body_clearance(map: &Map, x: f64, y: f64, a: f64) -> f64 {
    let (s, c) = a.sin_cos();
    let mut clearance = f64::INFINITY;
    for offset in FOOTPRINT_OFFSETS {
        match map.sdf((x + offset * c).round() as i64, (y + offset * s).round() as i64) {
            Some(d) => clearance = clearance.min(d as f64 - FOOTPRINT_RADIUS),
            None => return 0.0,
        }
    }
    clearance
}

/// Hold `action` from x for `frames` PLAN_DT steps. Each frame first applies the steering rate (clamped to
/// MAX_STEER), then drives that frame in equal time steps in which every footprint circle moves at most
/// CHECK_SPACING. Returns those check points (x itself not included, the last one the new state), or an empty list if
/// one comes within SAFETY_MARGIN of an obstacle. Cost grows per second by the action's rate plus
/// COST_STEER_ANGLE * |steering| of the frame.
pub fn propagate(map: &Map, x: &State, action: Action, frames: usize) -> Vec<State> {
    let control_rate = 1.0 + COST_ACCEL * action.accel.abs() + COST_STEER_RATE * action.rate.abs()
        + COST_DECEL * (-action.accel).max(0.0);
    let (mut px, mut py, mut pa, mut v) = (x.x, x.y, x.a, x.s);
    let (mut steer, mut cost) = (x.steer, x.cost);
    let mut out: Vec<State> = Vec::new();
    for frame in 0..frames {
        steer = (steer + action.rate * PLAN_DT).clamp(-MAX_STEER, MAX_STEER);
        let cost_rate = control_rate + COST_STEER_ANGLE * steer.abs(); // steering is held for the frame
        let v_end = v + action.accel * PLAN_DT;
        let curvature = steer.tan() / WHEELBASE;
        let travel = v.abs().max(v_end.abs()) * PLAN_DT * 1f64.hypot(FOOTPRINT_REACH * curvature);
        let steps = ((travel / CHECK_SPACING).ceil() as usize).max(1);
        for i in 1..=steps {
            let t = PLAN_DT * i as f64 / steps as f64;
            let (sx, sy, sa) = arc_pose(px, py, pa, curvature, v * t + action.accel * t * t / 2.0);
            if body_clearance(map, sx, sy, sa) < SAFETY_MARGIN {
                return Vec::new();
            }
            out.push(State {
                x: sx,
                y: sy,
                a: sa.rem_euclid(std::f64::consts::TAU),
                s: v + action.accel * t,
                steer,
                cost: cost + t * cost_rate,
                time: x.time + frame as f64 * PLAN_DT + t,
                node: false,
            });
        }
        let last = out.last().unwrap();
        (px, py, pa, v) = (last.x, last.y, last.a, v_end);
        cost += PLAN_DT * cost_rate;
    }
    out
}

/// SST's Monte Carlo propagation: a random action held from x for a random whole number of PLAN_DT frames.
pub fn monte_carlo_prop(map: &Map, rng: &mut Random, x: &State, steer_sigma: f64, accel_mean: Option<f64>) -> Vec<State> {
    let frames = rng.randint(1, ((TIME_PROP / PLAN_DT).round() as usize).max(1));
    let action = random_action(rng, x, frames as f64 * PLAN_DT, steer_sigma, accel_mean);
    propagate(map, x, action, frames)
}

/// Like body_clearance, but unknown space counts as an obstacle too.
pub fn known_clearance(map: &Map, x: f64, y: f64, a: f64) -> f64 {
    let (s, c) = a.sin_cos();
    let mut clearance = f64::INFINITY;
    for offset in FOOTPRINT_OFFSETS {
        match map.known_sdf((x + offset * c).round() as i64, (y + offset * s).round() as i64) {
            Some(d) => clearance = clearance.min(d as f64 - FOOTPRINT_RADIUS),
            None => return 0.0,
        }
    }
    clearance
}

/// Whether every pose (x, y, heading) keeps SAFETY_MARGIN, from obstacles or (with unknown_too) also unknown space.
pub fn poses_clear(map: &Map, poses: &[(f64, f64, f64)], unknown_too: bool) -> bool {
    poses.iter().all(|&(x, y, a)| {
        let c = if unknown_too { known_clearance(map, x, y, a) } else { body_clearance(map, x, y, a) };
        c >= SAFETY_MARGIN
    })
}

/// Whether every state keeps SAFETY_MARGIN from obstacles.
pub fn path_is_clear(map: &Map, states: &[State]) -> bool {
    states.iter().all(|s| body_clearance(map, s.x, s.y, s.a) >= SAFETY_MARGIN)
}

/// The emergency stops tried, gentlest first: full braking while turning the wheel at these steering rates.
const STOP_RATES: [f64; 3] = [0.0, MAX_STEER_RATE, -MAX_STEER_RATE]; // straight, hard right, hard left

/// Brake as hard as possible from pose (0, 0, heading 0) at forward `speed` and steering `steer`, turning the wheel at
/// `rate` but never past the lateral limit. Returns the (x, y, heading) points after the start, spaced so every
/// footprint circle moves at most CHECK_SPACING between them, the last one where the car stops.
fn simulate_stop(speed: f64, mut steer: f64, rate: f64) -> Vec<(f64, f64, f64)> {
    let (mut x, mut y, mut a, mut v) = (0.0, 0.0, 0.0, speed);
    let mut out = Vec::new();
    while v > 0.0 {
        let t_move = PLAN_DT.min(v / MAX_ACCELERATION);
        let limit = max_steer_at_speed(v);
        steer = (steer + rate * PLAN_DT).clamp(-limit, limit);
        let curvature = steer.tan() / WHEELBASE;
        let steps = ((v * t_move * 1f64.hypot(FOOTPRINT_REACH * curvature) / CHECK_SPACING).ceil() as usize).max(1);
        for i in 1..=steps {
            let t = t_move * i as f64 / steps as f64;
            out.push(arc_pose(x, y, a, curvature, v * t - MAX_ACCELERATION * t * t / 2.0));
        }
        (x, y, a) = *out.last().unwrap();
        v -= MAX_ACCELERATION * t_move;
        if v < 1e-9 {
            v = 0.0;
        }
    }
    out
}

/// The world poses of the stop at steering `rate` from state s (empty if s is already stopped). Braking in reverse
/// traces the forward stop mirrored front to back.
fn stop_poses(s: &State, rate: f64) -> Vec<(f64, f64, f64)> {
    let (sn, c) = s.a.sin_cos();
    simulate_stop(s.s.abs(), s.steer, rate)
        .into_iter()
        .map(|(x, y, h)| {
            let (x, h) = if s.s < 0.0 { (-x, -h) } else { (x, h) };
            (s.x + x * c - y * sn, s.y + x * sn + y * c, (s.a + h).rem_euclid(std::f64::consts::TAU))
        })
        .collect()
}

/// Whether the car, at this pose, speed (signed) and steering, can brake to a stop keeping SAFETY_MARGIN from
/// obstacles and unknown space with one of the emergency stops. The pose itself must be clear too.
pub fn can_stop_from(map: &Map, x: f64, y: f64, a: f64, speed: f64, steer: f64) -> bool {
    if known_clearance(map, x, y, a) < SAFETY_MARGIN {
        return false;
    }
    let s = State { steer, ..State::new(x, y, a, speed) };
    STOP_RATES.iter().any(|&rate| poses_clear(map, &stop_poses(&s, rate), true))
}

/// The highest of p's planned speed and SAFE_SPEED_STEP steps below it (down to 0) from which the car can still stop
/// before obstacles and unknown space, found by bisection (slower never stops later). 0 if even stopping at p isn't
/// safe.
pub fn safe_speed(map: &Map, p: &State) -> f64 {
    let top = p.s.abs();
    let mut speeds = vec![top];
    let mut v = (top / SAFE_SPEED_STEP).floor() * SAFE_SPEED_STEP;
    while v >= 0.0 {
        if v < top {
            speeds.push(v);
        }
        v -= SAFE_SPEED_STEP;
    }
    if *speeds.last().unwrap() != 0.0 {
        speeds.push(0.0);
    }
    let sign = if p.s < 0.0 { -1.0 } else { 1.0 };
    let ok = |i: usize| can_stop_from(map, p.x, p.y, p.a, sign * speeds[i], p.steer);
    if !ok(speeds.len() - 1) {
        return 0.0;
    }
    let (mut lo, mut hi) = (0, speeds.len() - 1); // speeds[hi] is safe; find the first (fastest) safe one
    while lo < hi {
        let mid = (lo + hi) / 2;
        if ok(mid) { hi = mid } else { lo = mid + 1 }
    }
    speeds[lo]
}

/// Whether the car can brake to a stop from `s` with one of the emergency stops while keeping SAFETY_MARGIN from
/// known obstacles (unknown space counts as free, as in planning).
pub fn can_stop_clear(map: &Map, s: &State) -> bool {
    STOP_RATES.iter().any(|&rate| poses_clear(map, &stop_poses(s, rate), false))
}
