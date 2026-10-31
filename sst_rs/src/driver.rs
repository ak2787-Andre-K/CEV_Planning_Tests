//! The Driver (`sst/driver.py`): drives the car along the best route while lidar scans reveal the world, and gets
//! past whatever blocks the road.
//!
//! The logic is the Python Driver's, step for step: the same speed limits, the same reactions to a blocked route
//! (a nudge, a reconnect, a replan from a bit before the blockage, or a stop), the same fallbacks once stopped
//! (forward tests, a library maneuver, a reconnect, committing to a branch of the replan tree, backing up), and the
//! same route improvements while nothing is blocked. Its random draws differ from Python's. Left out: the geometric
//! RRT (off in Python too), the obstacle log, and drawing.
//!
//! Speed: the next SAFE_SPEED_NODES tree nodes each get a safe speed (cached per map version); the car drives at
//! min(planned speed, min over those nodes of sqrt(safe^2 + 2 a distance)), so it can always brake to their safe speeds
//! in time. It can also always stop before the first route state ahead in an obstacle or unknown space.
//!
//! When a scan adds obstacles, every route's states are re-checked. If the current route is blocked ahead, replanning
//! starts right away: from R (REPLAN_BACK_NODES nodes before L, the last node before the blocked part, or the node just
//! before L if the car is already past that) if the car hasn't reached it, with the car able to stop at L meanwhile;
//! otherwise from where the car will stop, at rest. If the car reaches R without a new route, replanning restarts from
//! L at rest while the car brakes there.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::f64::consts::{PI, TAU};
use std::rc::Rc;

use crate::config::{
    CHECK_SPACING, CREEP_SPEED, DEAD_REPLAN_ITERATIONS, DEAD_REPLAN_NODES, FPS, GROUP_BACK_NODES, GROUP_G_WEIGHT,
    GROUP_HEADING_SCALE, GROUP_K, GROUP_MIN_DISTANCE, GROUP_MIN_SIZE, GROUP_PAST_FIRST, GROUP_RETRY_INTERVAL,
    GROUP_SAMPLES, GROUP_SPEED_SCALE, GROUP_STUCK_TIME, H_CELL, IMPROVE_MIN_GAIN, IMPROVE_SKIP_FRAMES,
    JOIN_SPEED_OVER, JOIN_TOLERANCE_SCALE, LIB_SPEEDS, LIB_TRIES_EVERY, MAX_ACCELERATION, MAX_LATERAL_ACCELERATION,
    MAX_REVERSE_SPEED, MAX_SPEED, MAX_STEER, MOTION_DT, NEAR_FOCUS_SHARE, NEAR_JOIN_SCALE, NEAR_LOOKAHEAD,
    NEAR_SAMPLE_SIGMA, NONSTOP, NONSTOP_LEAD, NUDGE_BACK_LENGTH, NUDGE_NODE_FRAMES, NUDGE_SHIFT_LENGTH, NUDGE_VARIANTS,
    PAST_BEYOND, PAST_FOCUS_SHARE, PAST_LOOKAHEAD, PAST_RADIUS, PAST_SAMPLE_SIGMA, PLAN_DT, RECONNECT_JOIN_WINDOW,
    RECONNECT_LOOKAHEADS, RECONNECT_MAX_TARGETS, RECONNECT_MAX_TURN, RECONNECT_RADIUS, RECONNECT_TANGENTS,
    RECONNECT_TARGET_SPACING, REPLAN_BACK_NODES, REPLAN_ITERATIONS_PER_FRAME, REPLAN_SAMPLE_GROW_EVERY,
    REPLAN_SAMPLE_GROWTH, REPLAN_SAMPLE_RADIUS, SAFE_SPEED_NODES, SCREEN_HEIGHT, SCREEN_WIDTH, SQUIGGLY_STEER_FRACTION,
    STALL_REPLAN_TIME, STOP_FAN_FRACTION, STOP_FAN_TESTS, TARGET, TARGET_RAD, TURN_HEADING_TOL, TURN_MAX_ENTRY_SPEED,
    TURN_NEAR_RADIUS, TURN_PRIMITIVE_STUCK_TIME, TURNAROUND_MANEUVERS, W_A, W_ST, W_V, WHEELBASE,
};
use crate::kinematics::{
    Action, can_stop_clear, feasible_accel, max_steer_at_speed, path_is_clear, propagate, safe_speed, wrap,
};
use crate::maneuvers::{lib_range, library_join, run_turnaround};
use crate::map::Map;
use crate::planner::{Guide, Key, Planner, Sample, goal_distance_grid, grid_distance, grid_value};
use crate::rng::Random;
use crate::route::{Route, join_tolerance, splice_route};
use crate::smoothing::{route_loopy, shortcut_drive};
use crate::tree::{State, Tree};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Follow,
    Replan, // planning from replan_index while driving on
    Stop,   // stopping at stop_index and planning from rest there
    Done,
}

/// A way for a planner node (or a plain state) to join a route or the goal: the total cost to the goal, the node,
/// the route and index it joins at (None for the goal region itself), and any maneuver states to drive after the node.
#[derive(Clone)]
struct Join {
    cost: f64,
    node: usize,
    route: Option<Rc<Route>>,
    index: usize,
    extra: Option<Vec<State>>,
}

/// A dodge that joins: its states, and the route and index it joins at (route None for the goal).
type Dodge = (Vec<State>, Option<Rc<Route>>, usize);

/// A route state a planner node may join, filed by position.
struct JoinPoint {
    x: f64,
    y: f64,
    route: Rc<Route>,
    index: usize,
}

/// Join points in 10 px cells.
#[derive(Default)]
struct JoinMap {
    cells: HashMap<(i64, i64), Vec<usize>>,
    points: Vec<JoinPoint>,
}

impl JoinMap {
    const CELL: f64 = 10.0;

    fn add(&mut self, p: JoinPoint) {
        let key = ((p.x / Self::CELL).floor() as i64, (p.y / Self::CELL).floor() as i64);
        self.cells.entry(key).or_default().push(self.points.len());
        self.points.push(p);
    }

    /// The join points within `delta` (straight line) of (x, y).
    fn nbhd(&self, x: f64, y: f64, delta: f64) -> Vec<&JoinPoint> {
        let mut out = Vec::new();
        for i in ((x - delta) / Self::CELL).floor() as i64..=((x + delta) / Self::CELL).floor() as i64 {
            for j in ((y - delta) / Self::CELL).floor() as i64..=((y + delta) / Self::CELL).floor() as i64 {
                for &k in self.cells.get(&(i, j)).map(Vec::as_slice).unwrap_or(&[]) {
                    let p = &self.points[k];
                    if (p.x - x).hypot(p.y - y) <= delta {
                        out.push(p);
                    }
                }
            }
        }
        out
    }
}

fn same(a: &Rc<Route>, b: &Rc<Route>) -> bool {
    Rc::ptr_eq(a, b)
}

fn state_with(p: &State, speed: f64) -> State {
    State { steer: p.steer, ..State::new(p.x, p.y, p.a, speed) }
}

/// Labels (0..k-1) of k-means on the rows of `features`, started with k-means++.
fn k_means(rng: &mut Random, features: &[[f64; 4]], k: usize) -> Vec<usize> {
    let d2 = |a: &[f64; 4], b: &[f64; 4]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();
    let n = features.len();
    let mut centers = vec![features[rng.randint(0, n - 1)]];
    for _ in 1..k {
        let dist: Vec<f64> = features.iter().map(|f| centers.iter().map(|c| d2(f, c)).fold(f64::INFINITY, f64::min)).collect();
        let total: f64 = dist.iter().sum();
        let pick = if total > 0.0 {
            let mut r = rng.random() * total;
            dist.iter().position(|&d| { r -= d; r < 0.0 }).unwrap_or(n - 1)
        } else {
            rng.randint(0, n - 1)
        };
        centers.push(features[pick]);
    }
    let mut labels = vec![0; n];
    for _ in 0..25 {
        for (l, f) in labels.iter_mut().zip(features) {
            *l = (0..k).min_by(|&a, &b| d2(f, &centers[a]).total_cmp(&d2(f, &centers[b]))).unwrap();
        }
        let moved: Vec<[f64; 4]> = (0..k)
            .map(|g| {
                let members: Vec<&[f64; 4]> = features.iter().zip(&labels).filter(|(_, l)| **l == g).map(|(f, _)| f).collect();
                if members.is_empty() {
                    return centers[g];
                }
                let mut mean = [0.0; 4];
                for m in &members {
                    for d in 0..4 {
                        mean[d] += m[d] / members.len() as f64;
                    }
                }
                mean
            })
            .collect();
        let close = moved.iter().zip(&centers).all(|(m, c)| m.iter().zip(c).all(|(a, b)| (a - b).abs() <= 1e-8 + 1e-5 * b.abs()));
        if close {
            break;
        }
        centers = moved;
    }
    labels
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 { values[n / 2] } else { (values[n / 2 - 1] + values[n / 2]) / 2.0 }
}

/// Everything the Driver knows besides the map and its planners, so a planner's hooks can update it while it runs.
struct Ctx {
    routes: Vec<Rc<Route>>,
    route: Rc<Route>,
    s: f64, // arc length along the route
    v: f64, // speed
    mode: Mode,
    joins: Vec<Join>,
    replan_index: Option<usize>,
    stop_index: Option<usize>,
    fan_tried: bool,                      // whether the forward tests have been tried at the current stop
    hold_index: Option<usize>,            // after joining a plan made from rest: come to rest here before going on
    hold_route: Option<Rc<Route>>,        // the route hold_index is an index into
    safe_cache: HashMap<[u64; 5], (u64, f64)>, // state -> (map version, safe speed)
    seen_version: Option<u64>,
    bad_index: Option<usize>,             // first blocked state of the current blockage
    bad_end: Option<usize>,               // its last; route states after it are past the obstacle
    past_nodes: BinaryHeap<Reverse<(Key, usize, usize, usize)>>, // (h, order, replanner node, route index)
    near_nodes: BinaryHeap<Reverse<(Key, usize, usize, usize)>>, // (h, order, planner node, index into near_targets)
    near_targets: Vec<(Rc<Route>, usize)>,
    unsafe_states: Vec<usize>,            // indices of route states in an obstacle or unknown space
    join_map: JoinMap,
    join_rows: Vec<[f64; 5]>,             // (x, y, heading, speed, steering) of every join point, for replan_h
    join_remaining: Vec<f64>,
    join_grid: Option<Vec<f64>>,
    h_version: u64,
    heap_order: usize,
    improve_joins: Vec<Join>,
    improve_index: Option<usize>,
    improve_route: Option<Rc<Route>>,
    no_squiggles: Option<Rc<Route>>,      // the route last found to have no squiggly edge ahead
    stopped_time: f64,                    // s at rest while waiting on a replan
    stalled_time: f64,                    // s at rest while just following the route
    group_tried: bool,
    group_end: Option<(Rc<Route>, usize)>, // end of the branch commit_branch last drove: no backing up from there
    lib_tried_at: i64,
    group_retry: f64,
    turn_tried: HashSet<(i64, i64, i64)>,
    rng: Random,
    verbose: bool,
    events: Vec<String>,
}

impl Ctx {
    fn say(&mut self, message: String) {
        if self.verbose {
            println!("Driver: {message}");
        }
        self.events.push(message);
    }

    fn pose(&self) -> (f64, f64, f64, f64, f64) {
        self.route.pose_at(self.s)
    }

    fn set_unsafe(&mut self, map: &Map) {
        self.unsafe_states = self.route.blocked_indices(map, true);
    }

    fn next_order(&mut self) -> usize {
        self.heap_order += 1;
        self.heap_order
    }

    fn safe_at(&mut self, map: &Map, p: &State) -> f64 {
        let key = [p.x, p.y, p.a, p.s, p.steer].map(f64::to_bits);
        match self.safe_cache.get(&key) {
            Some(&(version, safe)) if version == map.version => safe,
            _ => {
                let safe = safe_speed(map, p);
                self.safe_cache.insert(key, (map.version, safe));
                safe
            }
        }
    }

    /// Where the car must stop: the stop point while replanning or stopping, else the hold point. A hold on a route
    /// that has since been replaced went with it. (Python only drops it at the start of the next frame, so a route
    /// replaced mid-frame, e.g. a reconnect from rest then an improvement, left an index into the wrong route and
    /// crashed hardStopArc.)
    fn stop_point(&self) -> Option<usize> {
        if matches!(self.mode, Mode::Replan | Mode::Stop) {
            return self.stop_index;
        }
        self.hold_index.filter(|_| self.hold_route.as_ref().is_some_and(|r| same(r, &self.route)))
    }

    fn unsafe_ahead(&self, here: usize) -> Option<usize> {
        self.unsafe_states.iter().copied().find(|&i| i > here)
    }

    /// The fastest the car may go now.
    fn speed_limit(&mut self, map: &Map) -> f64 {
        let stop_at = self.stop_point();
        self.speed_limit_at(map, self.s, stop_at)
    }

    /// Arc length the car must not drive past: the last clear state before the first one ahead in an obstacle or
    /// unknown space, or the stop point while replanning, stopping or holding (if it's ahead).
    fn hard_stop_arc(&self) -> f64 {
        let r = &self.route;
        let mut limit = *r.arc.last().unwrap();
        if let Some(u) = self.unsafe_ahead(r.index_at(self.s)) {
            limit = limit.min(self.s.max(r.arc[u - 1]));
        }
        if let Some(stop) = self.stop_point()
            && r.arc[stop] >= self.s
        {
            limit = limit.min(r.arc[stop]);
        }
        limit
    }

    /// The fastest the car may go at arc length s: it must be able to brake to the safe speed of each of the next
    /// SAFE_SPEED_NODES nodes in time, to a stop before the first route state ahead in an obstacle or unknown space,
    /// and to a stop at stop_at.
    fn speed_limit_at(&mut self, map: &Map, s: f64, stop_at: Option<usize>) -> f64 {
        let route = self.route.clone();
        let mut limit = MAX_SPEED;
        for &k in route.nodes.iter().filter(|&&k| route.arc[k] > s).take(SAFE_SPEED_NODES) {
            let safe = self.safe_at(map, &route.states[k]);
            limit = limit.min((safe * safe + 2.0 * MAX_ACCELERATION * (route.arc[k] - s)).sqrt());
        }
        if let Some(u) = self.unsafe_ahead(route.index_at(s)) {
            // Stop at the state before it: that one was still clear
            limit = limit.min((2.0 * MAX_ACCELERATION * (route.arc[u - 1] - s).max(0.0)).sqrt());
        }
        if let Some(stop) = stop_at {
            limit = limit.min((2.0 * MAX_ACCELERATION * (route.arc[stop] - s).max(0.0)).sqrt());
        }
        limit
    }

    /// The speed the car can actually have at route index i on the way to a stop at `last`: the route's planned speed,
    /// held below the speed limit.
    fn speed_at(&mut self, map: &Map, i: usize, last: usize) -> f64 {
        let p = self.route.states[i];
        let arc = self.route.arc[i];
        p.s.abs().min(self.speed_limit_at(map, arc, Some(last))).copysign(p.s)
    }

    /// Join candidates: the still-valid tails of every route.
    fn build_join_map(&mut self) {
        self.join_map = JoinMap::default();
        self.join_rows.clear();
        self.join_remaining.clear();
        for r in &self.routes {
            for i in r.valid_from.get().max(1)..r.states.len() {
                let p = &r.states[i];
                self.join_map.add(JoinPoint { x: p.x, y: p.y, route: r.clone(), index: i });
                self.join_rows.push([p.x, p.y, p.a, p.s, p.steer]);
                self.join_remaining.push(r.remaining[i]);
            }
        }
        self.join_grid = None;
        self.h_version += 1;
    }

    /// Estimated cost to the goal from every grid cell, in px: around known obstacles to a join point plus that
    /// route's remaining cost (times MAX_SPEED), or to the goal region, whichever is least.
    fn join_grid(&mut self, map: &Map) -> &Vec<f64> {
        if self.join_grid.is_none() {
            let (nx, ny) = (SCREEN_WIDTH / H_CELL, SCREEN_HEIGHT / H_CELL);
            let center = |i: usize| (i * H_CELL + H_CELL / 2) as f64;
            let mut sources: Vec<f64> = (0..nx * ny)
                .map(|k| {
                    let inside = (center(k / ny) - TARGET.0).hypot(center(k % ny) - TARGET.1) <= TARGET_RAD;
                    if inside { 0.0 } else { f64::INFINITY }
                })
                .collect();
            for (p, rem) in self.join_rows.iter().zip(&self.join_remaining) {
                let i = ((p[0] / H_CELL as f64).floor().max(0.0) as usize).min(nx - 1);
                let j = ((p[1] / H_CELL as f64).floor().max(0.0) as usize).min(ny - 1);
                sources[i * ny + j] = sources[i * ny + j].min(rem * MAX_SPEED);
            }
            self.join_grid = Some(grid_distance(map, &sources));
        }
        self.join_grid.as_ref().unwrap()
    }

    /// Replanner h, an estimate of the cost to the goal (px over MAX_SPEED): the larger of the join grid (around known
    /// obstacles to a join point plus its route's remaining cost, or to the goal) and the smaller of the straight-line
    /// distance to the goal region and, over the join points, the state distance plus the route's remaining cost.
    fn replan_h(&mut self, map: &Map, n: &State) -> f64 {
        let mut d = ((n.x - TARGET.0).hypot(n.y - TARGET.1) - TARGET_RAD).max(0.0);
        for (j, rem) in self.join_rows.iter().zip(&self.join_remaining) {
            let dist = (j[0] - n.x).hypot(j[1] - n.y) + W_A * wrap(j[2] - n.a).abs() + W_V * (j[3] - n.s).abs()
                + W_ST * (j[4] - n.steer).abs() + rem * MAX_SPEED;
            d = d.min(dist);
        }
        d.max(grid_value(self.join_grid(map), n.x, n.y)) / MAX_SPEED
    }

    /// A random state in a circle around (cx, cy) (avoiding known obstacles), REPLAN_SAMPLE_RADIUS to start and
    /// REPLAN_SAMPLE_GROWTH times bigger every REPLAN_SAMPLE_GROW_EVERY iterations of the planner using it.
    fn sample_around(map: &Map, rng: &mut Random, cx: f64, cy: f64, it: usize) -> State {
        let growths = (it / REPLAN_SAMPLE_GROW_EVERY).min(30) as i32;
        let radius = (REPLAN_SAMPLE_RADIUS * REPLAN_SAMPLE_GROWTH.powi(growths)).min((SCREEN_WIDTH + SCREEN_HEIGHT) as f64);
        let (mut x, mut y) = (cx, cy);
        for _ in 0..50 {
            let (r, t) = (radius * rng.random().sqrt(), rng.uniform(0.0, TAU));
            x = (cx + r * t.cos()).clamp(0.0, SCREEN_WIDTH as f64 - 1.0);
            y = (cy + r * t.sin()).clamp(0.0, SCREEN_HEIGHT as f64 - 1.0);
            if !map.occupied(x as usize, y as usize) {
                break;
            }
        }
        let speed = rng.uniform(-MAX_REVERSE_SPEED, MAX_SPEED);
        let limit = max_steer_at_speed(speed.abs());
        let a = rng.uniform(0.0, TAU);
        State { steer: rng.uniform(-limit, limit), ..State::new(x, y, a, speed) }
    }

    /// The lowest-h entry of a near/past heap whose node is still active, dropping inactive ones off the top.
    fn best_active(heap: &mut BinaryHeap<Reverse<(Key, usize, usize, usize)>>, tree: &Tree) -> Option<(usize, usize)> {
        while let Some(Reverse((_, _, n, _))) = heap.peek() {
            if tree.nodes[*n].active {
                break;
            }
            heap.pop();
        }
        heap.peek().map(|Reverse((_, _, n, extra))| (*n, *extra))
    }

    /// If a near miss is still active, NEAR_FOCUS_SHARE of the time: a sample on its route NEAR_LOOKAHEAD states ahead
    /// of the best one, lined up with it.
    fn sample_near(&mut self, tree: &Tree, rng: &mut Random) -> Option<State> {
        let (_, target) = Self::best_active(&mut self.near_nodes, tree)?;
        if rng.random() >= NEAR_FOCUS_SHARE {
            return None;
        }
        let (r, i) = &self.near_targets[target];
        let t = r.states[(i + NEAR_LOOKAHEAD).min(r.states.len() - 1)];
        let limit = max_steer_at_speed(t.s.abs());
        let x = rng.gauss(t.x, NEAR_SAMPLE_SIGMA).clamp(0.0, SCREEN_WIDTH as f64 - 1.0);
        let y = rng.gauss(t.y, NEAR_SAMPLE_SIGMA).clamp(0.0, SCREEN_HEIGHT as f64 - 1.0);
        let a = rng.gauss(t.a, 0.05).rem_euclid(TAU);
        Some(State { steer: t.steer.clamp(-limit, limit), ..State::new(x, y, a, t.s) })
    }

    /// Replan sampling: a near miss's route (sample_near); else, if a node got past the obstacle, PAST_FOCUS_SHARE of
    /// the time near the route PAST_LOOKAHEAD states ahead of the best one; otherwise the growing circle around the car.
    fn sample_replan(&mut self, map: &Map, tree: &Tree, it: usize, rng: &mut Random) -> State {
        if let Some(near) = self.sample_near(tree, rng) {
            return near;
        }
        if let Some((_, i)) = Self::best_active(&mut self.past_nodes, tree)
            && rng.random() < PAST_FOCUS_SHARE
        {
            let t = self.route.states[(i + PAST_LOOKAHEAD).min(self.route.states.len() - 1)];
            let speed = rng.gauss(t.s, 15.0).clamp(-MAX_REVERSE_SPEED, MAX_SPEED);
            let limit = max_steer_at_speed(speed.abs());
            let x = rng.gauss(t.x, PAST_SAMPLE_SIGMA).clamp(0.0, SCREEN_WIDTH as f64 - 1.0);
            let y = rng.gauss(t.y, PAST_SAMPLE_SIGMA).clamp(0.0, SCREEN_HEIGHT as f64 - 1.0);
            let a = rng.gauss(t.a, 0.1).rem_euclid(TAU);
            return State { steer: t.steer.clamp(-limit, limit), ..State::new(x, y, a, speed) };
        }
        let (x, y, ..) = self.pose();
        Self::sample_around(map, rng, x, y, it)
    }

    /// Every way state n can join: (total cost to the goal, route, index), with route None for the goal region. States
    /// of the current route at or before index `after` are skipped. With `near`, every (route, index) within
    /// NEAR_JOIN_SCALE times the tolerances (a near miss) is collected too.
    fn joins_for(&self, n: &State, after: Option<usize>, mut near: Option<&mut Vec<(Rc<Route>, usize)>>)
                 -> Vec<(f64, Option<Rc<Route>>, usize)> {
        if (n.x - TARGET.0).hypot(n.y - TARGET.1) <= TARGET_RAD {
            return vec![(n.cost, None, 0)];
        }
        let (side_tol, heading_tol, steer_tol) = join_tolerance(n.s);
        let (side_tol, heading_tol, steer_tol) =
            (JOIN_TOLERANCE_SCALE * side_tol, JOIN_TOLERANCE_SCALE * heading_tol, JOIN_TOLERANCE_SCALE * steer_tol);
        let scale = if near.is_some() { NEAR_JOIN_SCALE } else { 1.0 };
        let mut found = Vec::new();
        for jp in self.join_map.nbhd(n.x, n.y, scale * side_tol + CHECK_SPACING) {
            let (r, i) = (&jp.route, jp.index);
            if after.is_some_and(|after| same(r, &self.route) && i <= after) {
                continue;
            }
            let p = &r.states[i];
            if n.s * p.s < 0.0 || n.s.abs() > p.s.abs() + JOIN_SPEED_OVER {
                continue;
            }
            let (heading, steer) = (wrap(n.a - p.a).abs(), (n.steer - p.steer).abs());
            if heading > scale * heading_tol || steer > scale * steer_tol {
                continue;
            }
            // Sideways distance: to the route's line through this state's neighbors
            let q = &r.states[i - 1];
            let (dx, dy) = (p.x - q.x, p.y - q.y);
            let seg = dx.hypot(dy);
            let side = if seg == 0.0 { (n.x - p.x).hypot(n.y - p.y) } else { ((n.x - q.x) * dy - (n.y - q.y) * dx).abs() / seg };
            if side > scale * side_tol {
                continue;
            }
            if side <= side_tol && heading <= heading_tol && steer <= steer_tol {
                found.push((n.cost + r.remaining[i], Some(r.clone()), i));
            } else if let Some(near) = near.as_deref_mut() {
                near.push((r.clone(), i));
            }
        }
        found
    }

    fn record_near(&mut self, map: &Map, node: usize, state: &State, near: Vec<(Rc<Route>, usize)>) {
        for target in near {
            let h = self.replan_h(map, state);
            let order = self.next_order();
            self.near_targets.push(target);
            self.near_nodes.push(Reverse((Key(h), order, node, self.near_targets.len() - 1)));
        }
    }

    /// If node n is within TURN_NEAR_RADIUS of a joinable route state, facing almost the opposite way and no faster
    /// than TURN_MAX_ENTRY_SPEED, try every turnaround from it; those whose end joins a route give joins with the
    /// maneuver's states attached. Each spot (10 px cell, 30 degree heading bin) is tried once per planner.
    fn turnaround_joins(&mut self, map: &Map, node: usize, n: &State, after: Option<usize>) -> Vec<Join> {
        if n.s.abs() > TURN_MAX_ENTRY_SPEED {
            return Vec::new();
        }
        let facing_back = self.join_map.nbhd(n.x, n.y, TURN_NEAR_RADIUS).into_iter().any(|jp| {
            let skip = after.is_some_and(|after| same(&jp.route, &self.route) && jp.index <= after);
            !skip && wrap(n.a - jp.route.states[jp.index].a).abs() >= PI - TURN_HEADING_TOL
        });
        if !facing_back {
            return Vec::new();
        }
        let spot = ((n.x / 10.0).floor() as i64, (n.y / 10.0).floor() as i64,
                    (n.a.rem_euclid(TAU) / (30.0 * PI / 180.0)).floor() as i64);
        if !self.turn_tried.insert(spot) {
            return Vec::new();
        }
        let mut found = Vec::new();
        for phases in TURNAROUND_MANEUVERS {
            let states = run_turnaround(map, n, phases);
            if let Some(end) = states.last() {
                for (cost, route, index) in self.joins_for(end, after, None) {
                    found.push(Join { cost, node, route, index, extra: Some(states.clone()) });
                }
            }
        }
        found
    }

    /// Library maneuvers from state n onto the most promising join points within lib_range of it (at most one per
    /// route). Returns (joins as (total cost, route, index, maneuver states), whether any maneuver was tried).
    #[allow(clippy::type_complexity)]
    fn library_joins_for(&self, map: &Map, n: &State) -> (Vec<(f64, Rc<Route>, usize, Vec<State>)>, bool) {
        if !(0.0 <= n.s && n.s <= LIB_SPEEDS[LIB_SPEEDS.len() - 1] + 5.0) {
            return (Vec::new(), false);
        }
        let mut best: Vec<(f64, Rc<Route>, usize)> = Vec::new(); // per route, in first-seen order
        for jp in self.join_map.nbhd(n.x, n.y, lib_range(n.s)) {
            let score = (jp.x - n.x).hypot(jp.y - n.y) / n.s.max(20.0) + jp.route.remaining[jp.index];
            match best.iter_mut().find(|b| same(&b.1, &jp.route)) {
                Some(b) if score < b.0 => *b = (score, jp.route.clone(), jp.index),
                Some(_) => {}
                None => best.push((score, jp.route.clone(), jp.index)),
            }
        }
        if best.is_empty() {
            return (Vec::new(), false);
        }
        best.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut joins = Vec::new();
        for (_, r, i) in best.into_iter().take(3) {
            if let Some((cost, states, j)) = library_join(map, n, &r, i) {
                joins.push((n.cost + cost + r.remaining[j], r, j, states));
            }
        }
        (joins, true)
    }

    /// The replanner's hook for every node it keeps: joins (direct, turnaround, library), near misses, and whether
    /// the node got past the obstacle.
    fn try_join(&mut self, map: &Map, tree: &Tree, it: usize, node: usize) {
        let n = tree.nodes[node].state;
        let mut near = Vec::new();
        for (cost, route, index) in self.joins_for(&n, None, Some(&mut near)) {
            self.joins.push(Join { cost, node, route, index, extra: None });
        }
        self.record_near(map, node, &n, near);
        let turns = self.turnaround_joins(map, node, &n, None);
        self.joins.extend(turns);
        // Close to a known route in a free circle: a library maneuver onto it (at most once per LIB_TRIES_EVERY)
        if it as i64 - self.lib_tried_at >= LIB_TRIES_EVERY as i64 {
            let (joins, tried) = self.library_joins_for(map, &n);
            if tried {
                self.lib_tried_at = it as i64;
            }
            for (cost, r, index, states) in joins {
                self.joins.push(Join { cost, node, route: Some(r), index, extra: Some(states) });
            }
        }
        // Did it get past the obstacle? (near the current route, beyond the blocked stretch)
        if let Some(bad_end) = self.bad_end {
            let past = self.join_map.nbhd(n.x, n.y, PAST_RADIUS).into_iter().find(|jp| {
                same(&jp.route, &self.route) && self.route.arc[jp.index] >= self.route.arc[bad_end] + PAST_BEYOND
            });
            if let Some(index) = past.map(|jp| jp.index) {
                let h = self.replan_h(map, &n);
                let order = self.next_order();
                self.past_nodes.push(Reverse((Key(h), order, node, index)));
            }
        }
    }

    /// The improver's hook: joins at least IMPROVE_MIN_GAIN cheaper than the route from where it plans.
    fn try_improve_join(&mut self, map: &Map, tree: &Tree, node: usize) {
        let Some(improve_index) = self.improve_index else { return };
        let threshold = self.route.remaining[improve_index] * (1.0 - IMPROVE_MIN_GAIN);
        let n = tree.nodes[node].state;
        let mut near = Vec::new();
        let mut joins: Vec<Join> = self
            .joins_for(&n, Some(improve_index), Some(&mut near))
            .into_iter()
            .map(|(cost, route, index)| Join { cost, node, route, index, extra: None })
            .collect();
        self.record_near(map, node, &n, near);
        joins.extend(self.turnaround_joins(map, node, &n, Some(improve_index)));
        self.improve_joins.extend(joins.into_iter().filter(|j| j.cost < threshold));
    }

    /// Re-drive the current route from start_index past the blocked stretch, dodging it: follow the route, shift
    /// sideways over the NUDGE_SHIFT_LENGTH px ending where the obstacle starts, stay shifted alongside it, and shift
    /// back over NUDGE_BACK_LENGTH px after it. Each shift steers like the route plus a lane-change-shaped offset, at
    /// most `fraction` of the steering allowed (its sign picks the side). Speed follows the route's (from 0 with
    /// from_rest). Returns the states, or an empty list if it hits something or the obstacle isn't ahead.
    fn nudge(&self, map: &Map, start_index: usize, fraction: f64, from_rest: bool) -> Vec<State> {
        let (Some(bad_index), Some(bad_end)) = (self.bad_index, self.bad_end) else { return Vec::new() };
        let r = &self.route;
        let base = r.arc[start_index];
        let (obstacle_start, obstacle_end) = (r.arc[bad_index] - base, r.arc[bad_end] - base);
        if obstacle_end <= 0.0 {
            return Vec::new();
        }
        let shift_start = (obstacle_start - NUDGE_SHIFT_LENGTH).max(0.0);
        let shift_end = obstacle_start.max(shift_start + 20.0);
        let back_start = obstacle_end.max(shift_end);
        let length = back_start + NUDGE_BACK_LENGTH;
        let lane_change = |w: f64| if w < 0.25 { 4.0 * w } else if w < 0.75 { 2.0 - 4.0 * w } else { 4.0 * w - 4.0 };
        let p = &r.states[start_index];
        let mut x = state_with(p, if from_rest { 0.0 } else { p.s });
        let (mut states, mut travelled) = (Vec::new(), 0.0);
        for step in 0..4000 {
            if travelled >= length {
                return states;
            }
            let shape = if shift_start <= travelled && travelled < shift_end {
                lane_change((travelled - shift_start) / (shift_end - shift_start))
            } else if travelled >= back_start {
                -lane_change((travelled - back_start) / NUDGE_BACK_LENGTH)
            } else {
                0.0
            };
            let (_, _, _, ref_speed, ref_steer) = r.pose_at(base + travelled);
            let limit = max_steer_at_speed(x.s.abs().max(ref_speed.abs()));
            let desired = (ref_steer + fraction * limit * shape).clamp(-limit, limit);
            let (accel, lo, hi) = feasible_accel(&x, PLAN_DT, (ref_speed - x.s) / PLAN_DT);
            let mut seg = propagate(map, &x, Action { rate: ((desired - x.steer) / PLAN_DT).clamp(lo, lo.max(hi)), accel }, 1);
            let Some(last) = seg.last_mut() else { return Vec::new() };
            travelled += (last.x - x.x).hypot(last.y - x.y);
            if step % NUDGE_NODE_FRAMES == NUDGE_NODE_FRAMES - 1 {
                last.node = true; // a node, so the safe-speed look-ahead has points along it
            }
            x = *last;
            states.extend(seg);
        }
        Vec::new()
    }

    /// The cheapest of a set of dodges (nudges from start_index) that ends within join tolerance of a route beyond
    /// the obstacle (or another route, or the goal): (states, route, index).
    fn best_dodge(&mut self, map: &Map, start_index: usize, fractions: &[f64], from_rest: bool) -> Option<Dodge> {
        self.build_join_map();
        let mut best: Option<(f64, Dodge)> = None;
        for &fraction in fractions {
            let mut states = self.nudge(map, start_index, fraction, from_rest);
            let Some(end) = states.last_mut() else { continue };
            end.node = true;
            let end = *end;
            let joins = self.joins_for(&end, self.bad_end, None);
            if let Some((cost, route, index)) = joins.into_iter().min_by(|a, b| a.0.total_cmp(&b.0))
                && best.as_ref().is_none_or(|b| cost < b.0)
            {
                best = Some((cost, (states, route, index)));
            }
        }
        best.map(|(_, dodge)| dodge)
    }

    /// Before a full replan: NUDGE_VARIANTS nudges each way from start_index; splice in the cheapest that joins.
    fn try_nudge(&mut self, map: &Map, start_index: usize) -> bool {
        let fractions: Vec<f64> = (1..=NUDGE_VARIANTS)
            .flat_map(|k| [1.0, -1.0].map(|side| side * k as f64 / NUDGE_VARIANTS as f64))
            .collect();
        let Some((states, route, index)) = self.best_dodge(map, start_index, &fractions, false) else { return false };
        self.route = Rc::new(splice_route(&self.route, start_index, states, route.as_deref().map(|r| (r, index))));
        self.routes.push(self.route.clone());
        (self.bad_index, self.bad_end) = (None, None);
        self.set_unsafe(map);
        true
    }

    /// Whether the route's edge from node index a to node index b is very squiggly: mean |steering| / most allowed
    /// at least SQUIGGLY_STEER_FRACTION.
    fn squiggly(&self, a: usize, b: usize) -> bool {
        let states = &self.route.states[a + 1..b + 1];
        !states.is_empty()
            && states.iter().map(|p| p.steer.abs() / max_steer_at_speed(p.s.abs())).sum::<f64>() / states.len() as f64
                >= SQUIGGLY_STEER_FRACTION
    }
}

/// The replanner's guide: h aims at joinable routes, samples focus on near misses, nodes past the obstacle and the
/// car's surroundings, edges must end where the car can stop, and every kept node is checked for joins.
struct ReplanGuide<'a> {
    c: &'a mut Ctx,
    map: &'a Map,
}

impl Guide for ReplanGuide<'_> {
    fn h(&mut self, s: &State) -> f64 {
        self.c.replan_h(self.map, s)
    }
    fn version(&self) -> u64 {
        self.c.h_version
    }
    fn sample(&mut self, tree: &Tree, it: usize, rng: &mut Random) -> Option<Sample> {
        Some(Sample { state: self.c.sample_replan(self.map, tree, it, rng), nearest_only: false })
    }
    fn keep(&mut self, path: &[State]) -> bool {
        // Doomed nodes (too fast to make the next corner) would otherwise soak up the search
        can_stop_clear(self.map, path.last().unwrap())
    }
    fn on_child(&mut self, tree: &Tree, it: usize, node: usize) {
        self.c.try_join(self.map, tree, it, node);
    }
}

/// The improver's guide: like the replanner's, but sampling around where it plans from and looking for cheaper joins.
struct ImproveGuide<'a> {
    c: &'a mut Ctx,
    map: &'a Map,
    center: (f64, f64),
}

impl Guide for ImproveGuide<'_> {
    fn h(&mut self, s: &State) -> f64 {
        self.c.replan_h(self.map, s)
    }
    fn version(&self) -> u64 {
        self.c.h_version
    }
    fn sample(&mut self, tree: &Tree, it: usize, rng: &mut Random) -> Option<Sample> {
        let state = match self.c.sample_near(tree, rng) {
            Some(s) => s,
            None => Ctx::sample_around(self.map, rng, self.center.0, self.center.1, it),
        };
        Some(Sample { state, nearest_only: false })
    }
    fn on_child(&mut self, tree: &Tree, _it: usize, node: usize) {
        self.c.try_improve_join(self.map, tree, node);
    }
}

/// Drives the car along the best route while lidar scans (merged into `map` from outside) reveal the world.
pub struct Driver {
    pub map: Map,
    replanner: Option<Planner>,
    improver: Option<Planner>,
    c: Ctx,
}

impl Driver {
    /// A Driver on `map` with the planned `routes` (each from the start into the goal region), on the cheapest.
    pub fn new(map: Map, routes: Vec<Vec<State>>, seed: u64) -> Driver {
        let routes: Vec<Rc<Route>> = routes.into_iter().map(|p| Rc::new(Route::new(p, None))).collect();
        let route = routes.iter().min_by(|a, b| a.remaining[0].total_cmp(&b.remaining[0])).unwrap().clone();
        let c = Ctx {
            routes, route, s: 0.0, v: 0.0, mode: Mode::Follow, joins: Vec::new(), replan_index: None, stop_index: None,
            fan_tried: false, hold_index: None, hold_route: None, safe_cache: HashMap::new(), seen_version: None,
            bad_index: None, bad_end: None, past_nodes: BinaryHeap::new(), near_nodes: BinaryHeap::new(),
            near_targets: Vec::new(), unsafe_states: Vec::new(), join_map: JoinMap::default(), join_rows: Vec::new(),
            join_remaining: Vec::new(), join_grid: None, h_version: 0, heap_order: 0, improve_joins: Vec::new(),
            improve_index: None, improve_route: None, no_squiggles: None, stopped_time: 0.0, stalled_time: 0.0,
            group_tried: false, group_end: None, lib_tried_at: -(LIB_TRIES_EVERY as i64), group_retry: 0.0,
            turn_tried: HashSet::new(), rng: Random::seeded(seed), verbose: false, events: Vec::new(),
        };
        Driver { map, replanner: None, improver: None, c }
    }

    pub fn set_verbose(&mut self, verbose: bool) {
        self.c.verbose = verbose;
    }

    pub fn mode(&self) -> Mode {
        self.c.mode
    }

    /// (x, y, heading, planned speed, steering) of the car.
    pub fn pose(&self) -> (f64, f64, f64, f64, f64) {
        self.c.pose()
    }

    /// The car's speed.
    pub fn speed(&self) -> f64 {
        self.c.v
    }

    /// The route the car is on.
    pub fn route(&self) -> &[State] {
        &self.c.route.states
    }

    /// Everything the Driver has reported so far ("route blocked ahead, replan", "joined a route", ...).
    pub fn events(&self) -> &[String] {
        &self.c.events
    }

    fn new_planner(&mut self, start: State, improve_center: Option<(f64, f64)>) -> Planner {
        let seed = self.c.rng.next_u64();
        let mut planner = match improve_center {
            None => Planner::new(start, (TARGET.0, TARGET.1, TARGET_RAD), seed, &mut ReplanGuide { c: &mut self.c, map: &self.map }),
            Some(center) => Planner::new(start, (TARGET.0, TARGET.1, TARGET_RAD), seed,
                                         &mut ImproveGuide { c: &mut self.c, map: &self.map, center }),
        };
        planner.verbose = self.c.verbose;
        planner
    }

    fn start_replan(&mut self, start: State) {
        self.improver = None; // a real replan takes over
        self.replanner = Some(self.new_planner(start, None));
        self.c.joins.clear();
        self.c.past_nodes.clear();
        self.c.near_nodes.clear();
        self.c.turn_tried.clear();
        self.c.lib_tried_at = -(LIB_TRIES_EVERY as i64);
        self.c.build_join_map();
    }

    /// Stop at route state at_index (or, if None, as soon as the car can) and start planning from there, at rest.
    fn begin_stop(&mut self, at_index: Option<usize>) {
        let c = &mut self.c;
        let here = c.route.index_at(c.s);
        let at_index = at_index.unwrap_or_else(|| {
            let stop_arc = c.s + c.v * c.v / (2.0 * MAX_ACCELERATION);
            let mut at = c.route.arc.partition_point(|&a| a < stop_arc).min(c.route.states.len() - 1);
            if let Some(u) = c.unsafe_ahead(here) {
                at = at.min(u - 1);
            }
            at.max(here)
        });
        (c.mode, c.stop_index, c.fan_tried) = (Mode::Stop, Some(at_index), false);
        let p = c.route.states[at_index];
        self.start_replan(state_with(&p, 0.0));
    }

    /// Follow the current route up to start_index, then `states`, then route r after index j.
    fn adopt_splice(&mut self, start_index: usize, states: Vec<State>, r: &Route, j: usize, from_rest: bool) {
        let c = &mut self.c;
        c.route = Rc::new(splice_route(&c.route, start_index, states, Some((r, j))));
        c.routes.push(c.route.clone());
        if from_rest {
            (c.hold_index, c.hold_route) = (Some(start_index), Some(c.route.clone())); // rest at the splice first
        }
        c.mode = Mode::Follow;
        self.replanner = None;
        c.joins.clear();
        (c.bad_index, c.bad_end, c.replan_index, c.stop_index) = (None, None, None, None);
        c.set_unsafe(&self.map);
    }

    /// After obstacles were added: re-check every route, and react if the current one is blocked ahead.
    fn check_routes(&mut self) {
        let map = &self.map;
        let c = &mut self.c;
        for r in &c.routes {
            r.valid_from.set(r.blocked_indices(map, false).last().map_or(0, |&b| b + 1));
        }
        c.routes.retain(|r| r.valid_from.get() < r.states.len());
        let blocked = c.route.blocked_indices(map, false);
        let here = c.route.index_at(c.s);
        let Some(&bad) = blocked.iter().find(|&&i| i > here) else { return };
        if c.mode == Mode::Done {
            return;
        }
        let blocked_set: HashSet<usize> = blocked.iter().copied().collect();
        let stretch_end = |bad: usize| {
            let mut end = bad;
            while blocked_set.contains(&(end + 1)) {
                end += 1;
            }
            end
        };
        if matches!(c.mode, Mode::Replan | Mode::Stop) && let Some(bad_index) = c.bad_index {
            if bad >= bad_index {
                // More of the same obstacle (or one further on): keep the current replan, just refresh where it can join
                if self.replanner.is_some() {
                    c.build_join_map();
                }
                return;
            }
            // More of the obstacle showed up closer: keep planning from where the replan started, if that's still
            // ahead of (or, when stopping, at) the car and before the obstacle; only the fallback stop point moves
            let keep_from = if c.mode == Mode::Replan { c.replan_index } else { c.stop_index };
            let usable = keep_from.is_some_and(|k| k < bad && if c.mode == Mode::Replan { k > here } else { k >= here });
            if usable {
                (c.bad_index, c.bad_end) = (Some(bad), Some(stretch_end(bad)));
                if c.mode == Mode::Replan {
                    c.stop_index = c.route.nodes.iter().copied().rfind(|&k| k < bad);
                }
                if self.replanner.is_some() {
                    c.build_join_map();
                }
                return;
            }
        }
        (c.bad_index, c.bad_end) = (Some(bad), Some(stretch_end(bad)));
        let before: Vec<usize> = c.route.nodes.iter().copied().filter(|&k| k < bad).collect();
        let last = *before.last().unwrap(); // L
        // R: REPLAN_BACK_NODES nodes before L, or the node just before L if the car is already past that one
        let replan_from = [REPLAN_BACK_NODES, 1]
            .iter()
            .filter(|&&back| back < before.len())
            .map(|&back| before[before.len() - 1 - back])
            .find(|&k| k > here);
        if let Some(r) = replan_from {
            if c.try_nudge(map, r) {
                c.say("route blocked ahead; a small nudge gets past it".into());
                return;
            }
            let speed = c.speed_at(map, r, last);
            if self.try_reconnect(r, speed) {
                self.c.say("route blocked ahead; reconnected to a route nearby".into());
                return;
            }
            let c = &mut self.c;
            (c.mode, c.replan_index, c.stop_index) = (Mode::Replan, Some(r), Some(last));
            // Start at the speed the car can actually have at R, not the route's planned one: a start too fast to
            // stop before the obstacle makes every edge fail can_stop_clear, so the replan couldn't grow at all
            let speed = c.speed_at(&self.map, r, last);
            let p = c.route.states[r];
            self.start_replan(state_with(&p, speed));
        } else {
            self.begin_stop(if last > here { Some(last) } else { None });
        }
        let what = if self.c.mode == Mode::Replan { "replan" } else { "stop" };
        let n = self.c.routes.len();
        self.c.say(format!("route blocked ahead, {what} ({n} routes still reach the goal)"));
    }

    /// From the current route's state at start_index, at `speed` (0 from a stop), try Hermite shortcuts to states of
    /// every still-valid route within RECONNECT_RADIUS (beyond the blockage on the current route). Targets the car
    /// can't reach without reversing (inside a tightest-turn circle at the start) or facing more than
    /// RECONNECT_MAX_TURN away are skipped; the RECONNECT_MAX_TARGETS most promising are tried with every tangent and
    /// lookahead. The cheapest that rejoins within the live join tolerance and ends where the car can stop is spliced
    /// in. Returns whether one was.
    fn try_reconnect(&mut self, start_index: usize, speed: f64) -> bool {
        let (map, c) = (&self.map, &self.c);
        let p = c.route.states[start_index];
        if p.s < 0.0 || speed < 0.0 {
            return false;
        }
        let limit = max_steer_at_speed(speed);
        let start = State { steer: p.steer.clamp(-limit, limit), ..State::new(p.x, p.y, p.a, speed) };
        let (sn, cs) = p.a.sin_cos();
        let mut targets: Vec<(f64, Rc<Route>, usize)> = Vec::new();
        for r in &c.routes {
            let mut first = r.valid_from.get().max(1);
            if same(r, &c.route) {
                first = first.max(c.bad_end.unwrap_or(start_index) + 1);
            }
            let mut last_arc = f64::NEG_INFINITY;
            for i in first..r.states.len() {
                let q = &r.states[i];
                if r.arc[i] - last_arc < RECONNECT_TARGET_SPACING || q.s <= 0.0 {
                    continue;
                }
                let (dx, dy) = (q.x - p.x, q.y - p.y);
                let dist = dx.hypot(dy);
                if !(20.0..=RECONNECT_RADIUS).contains(&dist) {
                    continue;
                }
                last_arc = r.arc[i];
                let (fwd, left) = (dx * cs + dy * sn, -dx * sn + dy * cs);
                if wrap(q.a - p.a).abs() > RECONNECT_MAX_TURN {
                    continue;
                }
                // Dubins-style reachability: not inside either tightest-turn circle at the slower end's speed
                let v = speed.min(q.s);
                let radius = (WHEELBASE / MAX_STEER.tan()).max(v * v / MAX_LATERAL_ACCELERATION);
                if fwd.hypot(left.abs() - radius) < radius {
                    continue;
                }
                targets.push((dist / speed.max(q.s).max(1.0) + r.remaining[i], r.clone(), i));
            }
        }
        targets.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut best: Option<(crate::smoothing::Shortcut, Rc<Route>)> = None;
        for (_, r, i) in targets.into_iter().take(RECONNECT_MAX_TARGETS) {
            let lo = i.saturating_sub(RECONNECT_JOIN_WINDOW.0).max(r.valid_from.get().max(1)) - 1;
            let hi = r.states.len().min(i + RECONNECT_JOIN_WINDOW.1);
            for tangent in RECONNECT_TANGENTS {
                for lookahead in RECONNECT_LOOKAHEADS {
                    let Some(found) = shortcut_drive(map, &start, &r.states[i], &r, lo, hi, tangent, lookahead, 0.0,
                                                     JOIN_TOLERANCE_SCALE) else { continue };
                    if best.as_ref().is_some_and(|b| found.total >= b.0.total) {
                        continue;
                    }
                    if can_stop_clear(map, found.states.last().unwrap()) {
                        best = Some((found, r.clone()));
                    }
                }
            }
        }
        let Some((mut found, r)) = best else { return false };
        let last = found.states.last_mut().unwrap();
        last.node = true;
        last.cost += found.gap;
        self.adopt_splice(start_index, found.states, &r, found.join, speed == 0.0);
        true
    }

    /// Stopped at stop_index with no new route: STOP_FAN_TESTS forward nudges from rest with offsets spread over
    /// +-STOP_FAN_FRACTION of the allowed steering; splice in the cheapest that joins.
    fn try_forward_fan(&mut self) -> bool {
        let stop = self.c.stop_index.unwrap();
        if self.c.bad_end.is_none_or(|e| e <= stop) {
            return false;
        }
        let fractions: Vec<f64> = (0..STOP_FAN_TESTS)
            .map(|k| -STOP_FAN_FRACTION + 2.0 * STOP_FAN_FRACTION * k as f64 / (STOP_FAN_TESTS - 1) as f64)
            .collect();
        let Some((states, route, index)) = self.c.best_dodge(&self.map, stop, &fractions, true) else { return false };
        let c = &mut self.c;
        c.route = Rc::new(splice_route(&c.route, stop, states, route.as_deref().map(|r| (r, index))));
        c.routes.push(c.route.clone());
        c.mode = Mode::Follow;
        self.replanner = None;
        c.joins.clear();
        (c.bad_index, c.bad_end, c.replan_index, c.stop_index) = (None, None, None, None);
        c.set_unsafe(&self.map);
        true
    }

    /// At rest at route state start_index: a library maneuver onto a known route nearby, spliced in if one works.
    fn try_library_at(&mut self, start_index: usize) -> bool {
        self.c.build_join_map();
        let start = state_with(&self.c.route.states[start_index], 0.0);
        let (joins, _) = self.c.library_joins_for(&self.map, &start);
        let Some((_, r, j, states)) = joins.into_iter().min_by(|a, b| a.0.total_cmp(&b.0)) else { return false };
        self.adopt_splice(start_index, states, &r, j, true);
        true
    }

    /// The forward tests failed: back up along the route to the previous node (already driven, so clear) and restart
    /// the replan from there at rest.
    fn back_up(&mut self) {
        let Some(stop) = self.c.stop_index else { return };
        let Some(previous) = self.c.route.nodes.iter().copied().rfind(|&k| k < stop) else { return };
        self.c.stop_index = Some(previous);
        let p = self.c.route.states[previous];
        self.start_replan(state_with(&p, 0.0));
    }

    /// Stuck at the stop point, or driving on with no join yet: drive the most promising partial branch of the replan
    /// tree (rooted at route index at_index) instead of waiting. Only branches ending no further from the target than
    /// the car count. Takes the GROUP_SAMPLES best scoring (cost to go plus GROUP_G_WEIGHT times own cost) active nodes
    /// at least GROUP_BACK_NODES + 1 deep and GROUP_MIN_DISTANCE from the car whose branches don't contain one another,
    /// k-means their end states into GROUP_K groups, picks the group of at least GROUP_MIN_SIZE with the lowest median
    /// score and takes its best branch. Returns whether it committed to one.
    fn group_commit(&mut self, at_index: usize) -> bool {
        let Some(planner) = self.replanner.as_mut() else { return false };
        let (map, c) = (&self.map, &mut self.c);
        let root = planner.start();
        let mut guide = ReplanGuide { c, map };
        let target_grid = goal_distance_grid(map, (TARGET.0, TARGET.1, TARGET_RAD));
        let here = grid_value(&target_grid, root.x, root.y);
        let past: HashSet<usize> = guide.c.past_nodes.iter().map(|Reverse((_, _, n, _))| *n).collect();
        let mut candidates: Vec<(f64, usize)> = Vec::new();
        let mut past_candidates: Vec<(f64, usize)> = Vec::new();
        for n in 0..planner.tree.nodes.len() {
            let node = &planner.tree.nodes[n];
            if !node.active || !node.alive {
                continue;
            }
            let s = node.state;
            let is_past = past.contains(&n);
            if !is_past && (s.x - root.x).hypot(s.y - root.y) < GROUP_MIN_DISTANCE {
                continue;
            }
            // (not <=, as in Python: an unreachable end, inf or NaN, doesn't count either)
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(grid_value(&target_grid, s.x, s.y) <= here) {
                continue;
            }
            let (mut depth, mut v) = (0, n);
            while let Some(parent) = planner.tree.nodes[v].parent {
                (depth, v) = (depth + 1, parent);
            }
            let score = planner.h(n, &mut guide) + GROUP_G_WEIGHT * s.cost;
            if v == 0 && depth > GROUP_BACK_NODES && score.is_finite() {
                if is_past { past_candidates.push((score, n)) } else { candidates.push((score, n)) }
            }
        }
        if GROUP_PAST_FIRST && let Some(&(_, chosen)) = past_candidates.iter().min_by(|a, b| a.0.total_cmp(&b.0)) {
            return self.commit_branch(chosen, at_index, "the best branch past the obstacle".into());
        }
        candidates.extend(past_candidates);
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut picked, mut picked_set, mut ancestors) = (Vec::new(), HashSet::new(), HashSet::new());
        for &(score, n) in &candidates {
            if ancestors.contains(&n) {
                continue; // a picked branch already runs through it
            }
            let mut chain = Vec::new();
            let mut v = planner.tree.nodes[n].parent;
            while let Some(p) = v {
                chain.push(p);
                v = planner.tree.nodes[p].parent;
            }
            if chain.iter().any(|p| picked_set.contains(p)) {
                continue; // it extends a picked branch
            }
            picked.push((score, n));
            picked_set.insert(n);
            ancestors.extend(chain);
            if picked.len() == GROUP_SAMPLES {
                break;
            }
        }
        if picked.len() < GROUP_K * GROUP_MIN_SIZE / 2 {
            return false;
        }
        let features: Vec<[f64; 4]> = picked
            .iter()
            .map(|&(_, n)| {
                let s = planner.tree.nodes[n].state;
                [s.x, s.y, GROUP_HEADING_SCALE * s.a.cos(), GROUP_HEADING_SCALE * s.a.sin()]
            })
            .collect();
        let labels = k_means(&mut guide.c.rng, &features, GROUP_K.min(picked.len()));
        let mut groups: Vec<Vec<usize>> = (0..=*labels.iter().max().unwrap())
            .map(|g| (0..picked.len()).filter(|&i| labels[i] == g).collect())
            .collect();
        let big: Vec<Vec<usize>> = groups.iter().filter(|g| g.len() >= GROUP_MIN_SIZE).cloned().collect();
        groups = if big.is_empty() { vec![groups.into_iter().max_by_key(|g| g.len()).unwrap()] } else { big };
        let medians: Vec<f64> = groups.iter().map(|g| median(&mut g.iter().map(|&i| picked[i].0).collect::<Vec<_>>())).collect();
        let best_group = (0..groups.len()).min_by(|&a, &b| medians[a].total_cmp(&medians[b])).unwrap();
        let chosen = groups[best_group].iter().map(|&i| picked[i]).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().1;
        let mut sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
        sizes.sort_by(|a, b| b.cmp(a));
        let what = format!("grouped {} branches into {sizes:?}, the best of the best group (median score {:.2})",
                           picked.len(), medians[best_group]);
        self.commit_branch(chosen, at_index, what)
    }

    /// Drive the replan tree's branch to `chosen`, spliced in at route index at_index (where the tree's root is), and
    /// start the next replan GROUP_BACK_NODES nodes before its end. Returns whether it did.
    fn commit_branch(&mut self, chosen: usize, at_index: usize, what: String) -> bool {
        let Some(planner) = self.replanner.as_mut() else { return false };
        let (map, c) = (&self.map, &mut self.c);
        let mut new_states: Vec<State> = planner.tree.path_states(chosen)[1..].to_vec();
        if !path_is_clear(map, &new_states) {
            return false;
        }
        for p in &mut new_states {
            p.s *= GROUP_SPEED_SCALE; // the same geometry slower only needs less grip
        }
        // The car's route up to the replan start, then the branch; remaining costs guess the rest with replan_h
        let spliced = splice_route(&c.route, at_index, new_states, None);
        let h = planner.h(chosen, &mut ReplanGuide { c, map });
        let route = Rc::new(Route::new(spliced.states, Some(spliced.remaining.iter().map(|r| r + h).collect())));
        let nodes_after: Vec<usize> = route.nodes.iter().copied().filter(|&k| k > at_index).collect();
        if nodes_after.len() <= GROUP_BACK_NODES {
            return false;
        }
        let how = if c.mode == Mode::Stop {
            format!("stuck {:.1} s", c.stopped_time)
        } else {
            "nearing the replan start without a join".into()
        };
        c.say(format!("{how}; driving {what} and replanning on the way"));
        let use_turns = planner.use_turns;
        c.route = route.clone();
        c.mode = Mode::Replan;
        let replan_index = nodes_after[nodes_after.len() - 1 - GROUP_BACK_NODES];
        (c.replan_index, c.stop_index) = (Some(replan_index), Some(route.states.len() - 1));
        c.group_end = Some((route.clone(), route.states.len() - 1)); // stopping there later replans without backing up
        (c.bad_index, c.bad_end) = (None, None);
        c.set_unsafe(map);
        let speed = c.speed_at(map, replan_index, route.states.len() - 1);
        let p = route.states[replan_index];
        self.start_replan(state_with(&p, speed));
        self.replanner.as_mut().unwrap().use_turns = use_turns; // still stuck-ish: keep the turn primitives on
        true
    }

    /// Plan from the start of the next very squiggly edge or loopy stretch ahead (skipping one the car reaches within
    /// IMPROVE_SKIP_FRAMES frames), looking for a cheaper way on.
    fn start_improve(&mut self) {
        self.improver = None;
        self.c.improve_index = None;
        if self.c.no_squiggles.as_ref().is_some_and(|r| same(r, &self.c.route)) {
            return;
        }
        let c = &self.c;
        let here = c.route.index_at(c.s);
        let skip_time = IMPROVE_SKIP_FRAMES / FPS;
        let found = c.route.nodes.windows(2).map(|w| (w[0], w[1])).find(|&(a, b)| {
            a > here && (c.route.arc[a] - c.s) / c.v.max(1.0) > skip_time && (c.squiggly(a, b) || route_loopy(&c.route, a))
        });
        let Some((a, _)) = found else {
            self.c.no_squiggles = Some(self.c.route.clone());
            return;
        };
        let p = self.c.route.states[a];
        (self.c.improve_index, self.c.improve_route) = (Some(a), Some(self.c.route.clone()));
        self.c.improve_joins.clear();
        self.c.near_nodes.clear();
        self.c.turn_tried.clear();
        self.c.build_join_map();
        self.improver = Some(self.new_planner(state_with(&p, p.s), Some((p.x, p.y))));
    }

    /// Splice in the cheapest improvement found, if it's still clear.
    fn finish_improve(&mut self) {
        let improver = self.improver.take().unwrap();
        let c = &mut self.c;
        let join = c.improve_joins.iter().min_by(|a, b| a.cost.total_cmp(&b.cost)).unwrap().clone();
        let mut new_states = improver.tree.path_states(join.node)[1..].to_vec();
        new_states.extend(join.extra.unwrap_or_default());
        if path_is_clear(&self.map, &new_states) {
            let improve_index = c.improve_index.unwrap();
            c.route = Rc::new(splice_route(&c.route, improve_index, new_states, join.route.as_deref().map(|r| (r, join.index))));
            c.routes.push(c.route.clone());
            c.set_unsafe(&self.map);
            let to_go = c.route.remaining[c.route.index_at(c.s)];
            c.say(format!("improved the route, {to_go:.2} to go"));
        }
        c.improve_joins.clear();
        c.improve_index = None;
    }

    /// Splice in the cheapest join found and follow it.
    fn finish_replan(&mut self) {
        let replanner = self.replanner.as_ref().unwrap();
        let c = &mut self.c;
        let join = c.joins.iter().min_by(|a, b| a.cost.total_cmp(&b.cost)).unwrap().clone();
        let with_maneuver = join.extra.is_some();
        let mut new_states = replanner.tree.path_states(join.node)[1..].to_vec();
        new_states.extend(join.extra.unwrap_or_default());
        if !path_is_clear(&self.map, &new_states) {
            // The map changed under it; plan again from the same start
            let start = replanner.start();
            self.start_replan(start);
            return;
        }
        let upto = if c.mode == Mode::Replan { c.replan_index } else { c.stop_index }.unwrap();
        c.route = Rc::new(splice_route(&c.route, upto, new_states, join.route.as_deref().map(|r| (r, join.index))));
        // A plan made from rest starts from rest: the car comes to a stop at the splice point first
        c.hold_index = if c.mode == Mode::Stop { Some(upto) } else { None };
        c.hold_route = Some(c.route.clone());
        c.routes.push(c.route.clone());
        c.mode = Mode::Follow;
        self.replanner = None;
        c.joins.clear();
        (c.bad_index, c.replan_index, c.stop_index, c.bad_end) = (None, None, None, None);
        c.past_nodes.clear();
        c.set_unsafe(&self.map);
        let to_go = c.route.remaining[c.route.index_at(c.s)];
        let onto = if join.route.is_none() { "the goal" } else { "a route" };
        let how = if with_maneuver { " with a maneuver" } else { "" };
        c.say(format!("joined {onto}{how}, {to_go:.2} to go"));
    }

    /// Advance the Driver by dt seconds: react to map changes, plan for this frame, move the car, and handle stops.
    pub fn update(&mut self, dt: f64) {
        if self.c.mode == Mode::Done {
            return;
        }
        if self.c.hold_index.is_some() && !self.c.hold_route.as_ref().is_some_and(|r| same(r, &self.c.route)) {
            self.c.hold_index = None; // the route was replaced since: the hold went with it
        }
        if self.c.seen_version != Some(self.map.version) {
            self.c.seen_version = Some(self.map.version);
            self.c.set_unsafe(&self.map);
            if self.map.new_obstacles {
                self.map.new_obstacles = false;
                self.check_routes();
            }
        }
        let c = &mut self.c;
        c.stopped_time = if c.mode == Mode::Stop && c.v.abs() < 1.0 { c.stopped_time + dt } else { 0.0 };
        if c.stopped_time == 0.0 {
            c.group_tried = false;
        }
        c.stalled_time = if c.mode == Mode::Follow && c.v.abs() < 1.0 { c.stalled_time + dt } else { 0.0 };
        if let Some(r) = self.replanner.as_mut()
            && c.stopped_time >= TURN_PRIMITIVE_STUCK_TIME
        {
            r.use_turns = true;
        }

        // Planning budget for this frame: a replan if one is running, otherwise improving a squiggly stretch ahead
        if let Some(planner) = self.replanner.as_mut() {
            let mut guide = ReplanGuide { c: &mut self.c, map: &self.map };
            let mut joined = false;
            for _ in 0..REPLAN_ITERATIONS_PER_FRAME {
                planner.step(&self.map, &mut guide);
                if !guide.c.joins.is_empty() {
                    joined = true;
                    break;
                }
            }
            if joined {
                self.finish_replan();
            }
        } else if self.c.mode == Mode::Follow {
            let c = &self.c;
            let restart = self.improver.is_none()
                || !c.improve_route.as_ref().is_some_and(|r| same(r, &c.route))
                || c.improve_index.is_none_or(|i| i <= c.route.index_at(c.s)
                    || (c.route.arc[i] - c.s) / c.v.max(1.0) <= IMPROVE_SKIP_FRAMES / FPS);
            if restart {
                self.start_improve();
            }
            if let Some(improver) = self.improver.as_mut() {
                let center = (improver.start().x, improver.start().y);
                let mut guide = ImproveGuide { c: &mut self.c, map: &self.map, center };
                let mut found = false;
                for _ in 0..REPLAN_ITERATIONS_PER_FRAME {
                    improver.step(&self.map, &mut guide);
                    if !guide.c.improve_joins.is_empty() {
                        found = true;
                        break;
                    }
                }
                if found {
                    self.finish_improve();
                }
            }
        }

        // Speed and motion along the route (backwards while backing up to a stop point behind the car), in substeps
        // of at most MOTION_DT with the speed limit re-checked each one
        let map = &self.map;
        let c = &mut self.c;
        let hard_stop = c.hard_stop_arc();
        let steps = ((dt / MOTION_DT).ceil() as usize).max(1);
        let h = dt / steps as f64;
        let mut limit = 0.0;
        for _ in 0..steps {
            limit = c.speed_limit(map);
            let stop_arc = c.stop_index.map(|i| c.route.arc[i]);
            if c.mode == Mode::Stop && let Some(stop_arc) = stop_arc && c.s > stop_arc + 0.5 {
                let target = -MAX_REVERSE_SPEED.min((2.0 * MAX_ACCELERATION * (c.s - stop_arc)).sqrt());
                c.v += (target - c.v).clamp(-MAX_ACCELERATION * h, MAX_ACCELERATION * h);
                c.s = (c.s + c.v * h).max(stop_arc);
                if c.s <= stop_arc + 0.5 {
                    c.v = 0.0;
                }
            } else {
                let planned = c.pose().3.abs();
                let mut target = planned.min(limit);
                if limit > CREEP_SPEED {
                    target = target.max(CREEP_SPEED);
                }
                c.v += (target - c.v).clamp(-MAX_ACCELERATION * h, MAX_ACCELERATION * h);
                c.s = (c.s + c.v * h).min(*c.route.arc.last().unwrap());
                if c.s > hard_stop {
                    // backstop for rounding: never past the last clear state or the stop point
                    (c.s, c.v) = (hard_stop, 0.0);
                }
            }
        }

        let at_stop = |c: &Ctx| c.stop_index.is_some_and(|i| (c.s - c.route.arc[i]).abs() <= 1.0);
        // Stuck at the stop point for GROUP_STUCK_TIME: drive the most promising partial branch
        if self.c.mode == Mode::Stop && self.replanner.is_some() && !self.c.group_tried
            && self.c.stopped_time >= GROUP_STUCK_TIME && at_stop(&self.c)
        {
            self.c.group_tried = true;
            self.group_commit(self.c.stop_index.unwrap());
        }
        // Driving a committed branch with no join yet, or (NONSTOP) within NONSTOP_LEAD s of the replan start: every
        // GROUP_RETRY_INTERVAL, try to commit to a branch of the replan from there, so the car goes straight on
        if self.c.mode == Mode::Replan && self.replanner.is_some()
            && let Some(r) = self.c.replan_index
            && self.c.route.index_at(self.c.s) < r
        {
            let c = &self.c;
            let committed = c.group_end.as_ref().is_some_and(|(route, _)| same(route, &c.route));
            let lead = (c.route.arc[r] - c.s) / c.v.max(1.0);
            if committed || (NONSTOP && lead <= NONSTOP_LEAD) {
                self.c.group_retry -= dt;
                if self.c.group_retry <= 0.0 {
                    self.c.group_retry = GROUP_RETRY_INTERVAL;
                    self.group_commit(r);
                }
            }
        }
        // Stopped at the stop point with no new route yet: forward tests at slightly different angles, else the
        // library, a reconnect, a committed branch, or backing up
        if self.c.mode == Mode::Stop && !self.c.fan_tried && self.c.v.abs() < 1.0 && at_stop(&self.c) {
            self.c.fan_tried = true;
            let stop = self.c.stop_index.unwrap();
            if self.try_forward_fan() {
                self.c.say("stopped; a forward test at a slightly different angle gets past".into());
            } else if self.try_library_at(stop) {
                self.c.say("stopped; a library maneuver gets onto a route nearby".into());
            } else if self.try_reconnect(stop, 0.0) {
                self.c.say("stopped; reconnected to a route nearby".into());
            } else if self.group_commit(stop) {
                // committed to a branch of the replan from here: on we go, no backing up
            } else if self.c.group_end.as_ref().is_some_and(|(r, i)| same(r, &self.c.route) && *i == stop) {
                // at the end of a branch group_commit drove: no backing up, the replan goes on from here
            } else {
                self.back_up();
                self.c.say("stopped; no forward test got past, backing up a node and replanning from there".into());
            }
        }
        // At the replan start without a join: one last try to commit to a branch from here before braking
        if NONSTOP && self.c.mode == Mode::Replan && self.replanner.is_some()
            && let Some(r) = self.c.replan_index
            && self.c.route.index_at(self.c.s) >= r
            && self.c.s - self.c.route.arc[r] <= (self.c.v * dt).max(2.0)
        {
            self.group_commit(r);
        }
        // Out of time to replan from R: replan from L at rest right away, while braking to stop there
        if self.c.mode == Mode::Replan && self.c.replan_index.is_some_and(|r| self.c.route.index_at(self.c.s) >= r) {
            self.c.say("reached the replanning point without a new route; replanning from the last safe node".into());
            self.begin_stop(self.c.stop_index);
        }
        // Stuck short of the stop point (nothing safe ahead): replan from right here instead
        if self.c.mode == Mode::Stop && self.c.v < 1.0 && limit < CREEP_SPEED
            && self.c.stop_index.is_some_and(|i| self.c.s < self.c.route.arc[i] - 1.0)
        {
            self.c.v = 0.0;
            self.begin_stop(Some(self.c.route.index_at(self.c.s)));
            self.c.say("stopped short of the planned stop; replanning from here".into());
        }
        // Stuck while just following the route (nothing ahead it can safely drive into): replan from here
        if self.c.mode == Mode::Follow && self.c.stalled_time >= STALL_REPLAN_TIME && limit < CREEP_SPEED {
            self.c.stalled_time = 0.0;
            self.c.v = 0.0;
            (self.c.bad_index, self.c.bad_end) = (None, None);
            self.begin_stop(Some(self.c.route.index_at(self.c.s)));
            self.c.say("stalled while following the route; replanning from here".into());
        }
        // Came to rest at the start of a plan made from rest: go on (a hold on a replaced route is just dropped)
        if let Some(hold) = self.c.hold_index {
            let current = self.c.hold_route.as_ref().is_some_and(|r| same(r, &self.c.route));
            if !current || (self.c.v < 1.0 && self.c.s >= self.c.route.arc[hold] - 1.0) {
                self.c.hold_index = None;
            }
        }
        if self.c.s >= self.c.route.arc.last().unwrap() - 1e-6 && self.c.mode == Mode::Follow {
            self.c.mode = Mode::Done;
            self.c.say("reached the goal".into());
        }
        // Dead replan (its start can't grow): don't wait on it forever
        if let Some(r) = self.replanner.as_ref()
            && matches!(self.c.mode, Mode::Replan | Mode::Stop)
            && r.it >= DEAD_REPLAN_ITERATIONS
            && r.tree.active_count < DEAD_REPLAN_NODES
        {
            if self.c.mode == Mode::Replan {
                self.c.say("the replan can't grow from its start; stopping and replanning from the stop point".into());
                self.begin_stop(self.c.stop_index);
            } else if self.c.v.abs() < 1.0 {
                self.c.say("the replan can't grow from here; backing up a node and replanning from there".into());
                self.c.group_end = None;
                self.back_up();
            }
        }
    }
}
