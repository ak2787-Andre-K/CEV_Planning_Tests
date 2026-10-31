//! Routes, and joining or splicing new states onto them (`sst/route.py`).
use crate::config::{
    CHECK_SPACING, COST_STEER_ANGLE, JOIN_SPEED_OVER, MAX_LATERAL_ACCELERATION, MAX_STEER, MAX_STEER_RATE,
    SAFETY_MARGIN, WHEELBASE,
};
use crate::kinematics::wrap;
use crate::tree::State;

/// A drivable path: states (the start, then check points) into the goal region, which of them are tree nodes, the
/// arc length along it, and the remaining cost from each state to its end.
pub struct Route {
    pub states: Vec<State>,
    pub nodes: Vec<usize>,
    pub arc: Vec<f64>,
    pub remaining: Vec<f64>,
}

impl Route {
    pub fn new(states: Vec<State>, remaining: Option<Vec<f64>>) -> Route {
        let n = states.len();
        let nodes = (0..n).filter(|&i| i == 0 || i == n - 1 || states[i].node).collect();
        let mut arc = vec![0.0];
        for w in states.windows(2) {
            arc.push(arc.last().unwrap() + (w[1].x - w[0].x).hypot(w[1].y - w[0].y));
        }
        let end = states.last().map_or(0.0, |s| s.cost);
        let remaining = remaining.unwrap_or_else(|| states.iter().map(|p| end - p.cost).collect());
        Route { states, nodes, arc, remaining }
    }
}

/// How far a state may be from a route and still join it, so a tracking controller can take over without leaving the
/// SAFETY_MARGIN: (sideways px, heading rad, steering rad).
pub fn join_tolerance(speed: f64) -> (f64, f64, f64) {
    let v = speed.abs().max(1e-6);
    let curvature = (MAX_STEER.tan() / WHEELBASE).min(MAX_LATERAL_ACCELERATION / (v * v));
    let heading = std::f64::consts::PI.min((2.0 * curvature * SAFETY_MARGIN).sqrt());
    let steering = (heading * WHEELBASE * MAX_STEER_RATE / v).sqrt();
    (SAFETY_MARGIN, heading, steering)
}

/// Where state n joins the route between indices lo and hi: the state i such that n lies between states i-1 and i
/// along the route, within `scale` times join_tolerance of state i (sideways, heading, steering; same direction, at
/// most JOIN_SPEED_OVER faster). Returns (i, cost of the gap from n on to state i).
pub fn route_join(n: &State, route: &Route, lo: usize, hi: usize, scale: f64) -> Option<(usize, f64)> {
    let (side_tol, heading_tol, steer_tol) = join_tolerance(n.s);
    let (side_tol, heading_tol, steer_tol) = (scale * side_tol, scale * heading_tol, scale * steer_tol);
    for i in lo.max(1)..hi.min(route.states.len()) {
        let (p, q) = (&route.states[i], &route.states[i - 1]);
        if (p.x - n.x).hypot(p.y - n.y) > side_tol + CHECK_SPACING {
            continue;
        }
        if n.s * p.s < 0.0 || n.s.abs() > p.s.abs() + JOIN_SPEED_OVER {
            continue;
        }
        if wrap(n.a - p.a).abs() > heading_tol || (n.steer - p.steer).abs() > steer_tol {
            continue;
        }
        let (dx, dy) = (p.x - q.x, p.y - q.y);
        let seg = dx.hypot(dy);
        if seg == 0.0 {
            continue;
        }
        let along = ((n.x - q.x) * dx + (n.y - q.y) * dy) / seg;
        let side = ((n.x - q.x) * dy - (n.y - q.y) * dx).abs() / seg;
        if (0.0..=seg).contains(&along) && side <= side_tol {
            let gap = (seg - along) / p.s.abs().max(1.0) * (1.0 + COST_STEER_ANGLE * p.steer.abs());
            return Some((i, gap));
        }
    }
    None
}

/// A Route that follows prefix.states[..=upto], then new_states (planned from prefix.states[upto], costs counted
/// from 0), then join's states after join_index.
pub fn splice_route(prefix: &Route, upto: usize, new_states: Vec<State>, join: &Route, join_index: usize) -> Route {
    let join_rem = join.remaining[join_index];
    let new_end = new_states.last().map_or(0.0, |s| s.cost);
    let mut remaining: Vec<f64> =
        (0..=upto).map(|i| prefix.remaining[i] - prefix.remaining[upto] + new_end + join_rem).collect();
    remaining.extend(new_states.iter().map(|p| new_end - p.cost + join_rem));
    remaining.extend(&join.remaining[join_index + 1..]);
    let mut states = prefix.states[..=upto].to_vec();
    states.extend(new_states);
    states.extend(&join.states[join_index + 1..]);
    Route::new(states, Some(remaining))
}
