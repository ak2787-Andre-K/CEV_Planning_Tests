//! Car motion (`sst/car.py` and `sst/kinematics.py`): the steering limits, random actions, and propagating an action
//! through the bicycle model with a collision check at every step.
use crate::config::{
    CHECK_SPACING, COST_ACCEL, COST_DECEL, COST_STEER_ANGLE, COST_STEER_RATE, EARLY_ACCEL_SIGMA, FOOTPRINT_OFFSETS,
    FOOTPRINT_RADIUS, FOOTPRINT_REACH, MAX_ACCELERATION, MAX_LATERAL_ACCELERATION, MAX_REVERSE_SPEED, MAX_SPEED,
    MAX_STEER, MAX_STEER_RATE, PLAN_DT, SAFETY_MARGIN, TIME_PROP, WHEELBASE,
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
