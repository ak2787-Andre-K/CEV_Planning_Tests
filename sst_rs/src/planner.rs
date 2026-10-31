//! The global SST* planner with informed expansion (`GlobalPlanner` and `planGlobal` in `sst/global_planner.py`).
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::config::{
    ANNEAL_FACTOR, ANNEAL_MIN_GAIN, ANNEAL_WINDOW, BLOSSOM_RANDOM, DELTA_B_N_MIN, DELTA_B_N_START, EARLY_ACCEL_MEAN,
    EARLY_STEER_ITERATIONS, EARLY_STEER_VARIANCE_SCALE, EXPANSION_PENALTY, GREEDY_CHAIN_MAX, GREEDY_SHARE,
    GRID_INFLATION, H_CELL, MAX_ACCELERATION, MAX_SPEED, MAX_STEER_RATE, PLAN_DT, PRIMITIVES, SCREEN_HEIGHT,
    SCREEN_WIDTH, STEER_TARGET_SIGMA, TIE_BREAK, TURN_PRIMITIVE_ANGLE, TURN_PRIMITIVE_MAX_BRAKE,
    TURN_PRIMITIVE_MAX_FRAMES, TURN_PRIMITIVE_SPEEDS, WHEELBASE,
};
use crate::kinematics::{Action, feasible_accel, max_steer_at_speed, monte_carlo_prop, propagate, wrap};
use crate::map::{Map, cell};
use crate::rng::Random;
use crate::tree::{State, Tree, Witness, WitnessMap};

/// An f64 that orders totally, for the heaps.
#[derive(Clone, Copy, PartialEq)]
pub struct Key(pub f64);

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Grid Dijkstra distance (px) from every H_CELL cell to the nearest source through free space, 8-connected.
/// `sources` holds each cell's starting cost in px (inf for cells that aren't sources); a cell gets the least of
/// distance plus starting cost over the sources. A cell is free if its center is at least GRID_INFLATION from known
/// obstacles (unknown space counts as free); only free sources count. Unreachable cells are inf. Indexed i * ny + j.
pub fn grid_distance(map: &Map, sources: &[f64]) -> Vec<f64> {
    let (nx, ny) = (SCREEN_WIDTH / H_CELL, SCREEN_HEIGHT / H_CELL);
    let free: Vec<bool> = (0..nx * ny)
        .map(|k| map.obstacle_sdf[cell(grid_center(k / ny), grid_center(k % ny))] as f64 >= GRID_INFLATION)
        .collect();
    let mut dist = vec![f64::INFINITY; nx * ny];
    let mut heap = BinaryHeap::new();
    for k in 0..nx * ny {
        if free[k] && sources[k].is_finite() {
            dist[k] = sources[k];
            heap.push(Reverse((Key(sources[k]), k)));
        }
    }
    let (step, diag) = (H_CELL as f64, H_CELL as f64 * 2f64.sqrt());
    while let Some(Reverse((Key(d), k))) = heap.pop() {
        if d > dist[k] {
            continue;
        }
        let (i, j) = ((k / ny) as i64, (k % ny) as i64);
        for di in -1..=1i64 {
            for dj in -1..=1i64 {
                let (a, b) = (i + di, j + dj);
                if (di == 0 && dj == 0) || a < 0 || b < 0 || a >= nx as i64 || b >= ny as i64 {
                    continue;
                }
                let n = a as usize * ny + b as usize;
                let nd = d + if di != 0 && dj != 0 { diag } else { step };
                if free[n] && nd < dist[n] {
                    dist[n] = nd;
                    heap.push(Reverse((Key(nd), n)));
                }
            }
        }
    }
    dist
}

/// Center (px) of grid cell i along either axis.
pub fn grid_center(i: usize) -> usize {
    i * H_CELL + H_CELL / 2
}

/// Grid distance from every cell to the goal circle (grid_distance from the cells whose center is inside it).
pub fn goal_distance_grid(map: &Map, goal: (f64, f64, f64)) -> Vec<f64> {
    let ny = SCREEN_HEIGHT / H_CELL;
    let sources: Vec<f64> = (0..(SCREEN_WIDTH / H_CELL) * ny)
        .map(|k| {
            let inside = (grid_center(k / ny) as f64 - goal.0).hypot(grid_center(k % ny) as f64 - goal.1) <= goal.2;
            if inside { 0.0 } else { f64::INFINITY }
        })
        .collect();
    grid_distance(map, &sources)
}

/// A grid_distance grid's value at (x, y); if that cell is unreachable (too close to a wall), the best way in through
/// one of its neighbors.
pub fn grid_value(grid: &[f64], x: f64, y: f64) -> f64 {
    let (nx, ny) = (SCREEN_WIDTH / H_CELL, SCREEN_HEIGHT / H_CELL);
    let i = ((x / H_CELL as f64).floor().max(0.0) as usize).min(nx - 1);
    let k = ((y / H_CELL as f64).floor().max(0.0) as usize).min(ny - 1);
    if grid[i * ny + k].is_finite() {
        return grid[i * ny + k];
    }
    let mut best = f64::INFINITY;
    for a in i.saturating_sub(1)..(i + 2).min(nx) {
        for b in k.saturating_sub(1)..(k + 2).min(ny) {
            let step = H_CELL as f64 * (a as f64 - i as f64).hypot(b as f64 - k as f64);
            best = best.min(grid[a * ny + b] + step);
        }
    }
    best
}

/// Propagate one of the PRIMITIVES from x: its acceleration (made feasible) and the steering rate that reaches its
/// target steering at the end, capped to the allowed rates.
fn primitive_path(map: &Map, x: &State, (accel_frac, steer_frac, duration): (f64, f64, f64)) -> Vec<State> {
    let frames = ((duration / PLAN_DT).round() as usize).max(1);
    let time = frames as f64 * PLAN_DT;
    let (accel, lo, hi) = feasible_accel(x, time, accel_frac * MAX_ACCELERATION);
    let v_end = x.s + accel * time;
    let target = steer_frac * max_steer_at_speed(x.s.abs().max(v_end.abs()));
    propagate(map, x, Action { rate: ((target - x.steer) / time).clamp(lo, lo.max(hi)), accel }, frames)
}

/// Brake (or speed up) to corner_speed with the wheel going straight, then turn by `turn` rad (signed) at the most
/// steering that speed allows, unwinding the wheel in time to come out lined up. Every PLAN_DT step goes through
/// propagate. Returns the states, or an empty list if it hits something, x is reversing, or the speed change takes
/// over TURN_PRIMITIVE_MAX_BRAKE.
fn turn_path(map: &Map, x: &State, turn: f64, corner_speed: f64) -> Vec<State> {
    if x.s < 0.0 || (x.s - corner_speed).abs() / MAX_ACCELERATION > TURN_PRIMITIVE_MAX_BRAKE {
        return Vec::new();
    }
    let side = if turn > 0.0 { 1.0 } else { -1.0 };
    let goal_heading = x.a + turn;
    let (mut x, mut states, mut turning) = (*x, Vec::new(), false);
    for _ in 0..TURN_PRIMITIVE_MAX_FRAMES {
        let (accel, lo, hi) = feasible_accel(&x, PLAN_DT, (corner_speed - x.s) / PLAN_DT);
        let left = side * wrap(goal_heading - x.a); // heading still to turn, > 0
        if !turning && (x.s - corner_speed).abs() < 2.0 {
            turning = true;
        }
        let target = if turning {
            // Unwind once the heading the wheel turns while straightening would finish it
            let unwind = x.steer * x.steer * x.s.max(1.0) / (2.0 * WHEELBASE * MAX_STEER_RATE);
            let target = if left <= unwind { 0.0 } else { side * max_steer_at_speed(x.s.max(corner_speed)) };
            if left <= 0.0 || (target == 0.0 && x.steer.abs() < 1e-3) {
                return states;
            }
            target
        } else {
            0.0
        };
        let seg = propagate(map, &x, Action { rate: ((target - x.steer) / PLAN_DT).clamp(lo, lo.max(hi)), accel }, 1);
        let Some(&last) = seg.last() else { return Vec::new() };
        states.extend(seg);
        x = last;
    }
    Vec::new()
}

/// A sample for a normal-SST step; with nearest_only the planner extends the nearest node to it, not the cheapest
/// one nearby.
pub struct Sample {
    pub state: State,
    pub nearest_only: bool,
}

/// What steers a planner beyond plain SST: its estimate of the cost to the goal, and optional hooks for sampling,
/// filtering edges and hearing about every node it keeps. The plain global plan uses `GoalGrid`; the Driver's
/// replanners aim at the routes they can join instead.
pub trait Guide {
    /// Estimated cost to the goal from s.
    fn h(&mut self, s: &State) -> f64;
    /// Bumped whenever h changes, so values cached on nodes are dropped.
    fn version(&self) -> u64 {
        0
    }
    /// A sample for the normal-SST steps, or None for a uniform random one.
    fn sample(&mut self, _tree: &Tree, _it: usize, _rng: &mut Random) -> Option<Sample> {
        None
    }
    /// Whether an edge (a nonempty propagated path) may be kept.
    fn keep(&mut self, _path: &[State]) -> bool {
        true
    }
    /// Called with every node the tree keeps.
    fn on_child(&mut self, _tree: &Tree, _it: usize, _node: usize) {}
}

/// h = grid distance to the goal circle (around known obstacles) / MAX_SPEED.
pub struct GoalGrid(pub Vec<f64>);

impl GoalGrid {
    pub fn new(map: &Map, goal: (f64, f64, f64)) -> GoalGrid {
        GoalGrid(goal_distance_grid(map, goal))
    }
}

impl Guide for GoalGrid {
    fn h(&mut self, s: &State) -> f64 {
        let ny = SCREEN_HEIGHT / H_CELL;
        let i = ((s.x / H_CELL as f64).floor().max(0.0) as usize).min(SCREEN_WIDTH / H_CELL - 1);
        let j = ((s.y / H_CELL as f64).floor().max(0.0) as usize).min(ny - 1);
        self.0[i * ny + j] / MAX_SPEED
    }
}

/// SST* with informed expansion, run in steps.
///
/// Each step expands one node. With probability GREEDY_SHARE it's the active node with the lowest g + h (g = cost,
/// h from the guide, plus EXPANSION_PENALTY per earlier expansion and a small random tie-break). A node's first
/// expansion "blossoms": the PRIMITIVES and BLOSSOM_RANDOM random actions are tried and the child with the best g + h
/// the guide keeps is kept; later expansions try one random action. With use_turns, every turn primitive that fits is
/// also added as an extra child, once per node. A kept child closer to the goal than its parent is expanded next, up
/// to GREEDY_CHAIN_MAX in a row. The other steps are normal SST (a sample, best-first selection, one random action).
///
/// Once a path exists, every ANNEAL_WINDOW steps the best cost is compared with the window's start; if it improved by
/// less than ANNEAL_MIN_GAIN, delta_b_n and delta_s shrink by ANNEAL_FACTOR (not below DELTA_B_N_MIN).
pub struct Planner {
    pub rng: Random,
    pub tree: Tree,
    witnesses: WitnessMap,
    pub delta_b_n: f64,
    pub delta_s: f64,
    h_cache: Vec<Option<(u64, f64)>>, // per node: (guide version, h)
    open: BinaryHeap<Reverse<(Key, usize, usize)>>, // (g + h, counter, node); stale entries are skipped when popped
    counter: usize,
    chain: Option<usize>,
    chain_depth: usize,
    pub it: usize,
    pub found_at: Option<usize>, // iteration the first path was found at
    window: Option<(usize, f64)>, // (iteration, best cost) at the start of the improvement window, once a path exists
    pub use_turns: bool, // add turn primitives as extra children (set by the Driver once stuck)
    turns_tried: Vec<bool>, // per node
    pub verbose: bool,
}

impl Planner {
    pub fn new(start: State, goal: (f64, f64, f64), seed: u64, guide: &mut dyn Guide) -> Planner {
        let tree = Tree::new(start, goal);
        let mut witnesses = WitnessMap::new(DELTA_B_N_START / 4.0);
        witnesses.add(Witness { center: tree.nodes[0].state, rep: Some(0) });
        let mut planner = Planner {
            rng: Random::seeded(seed),
            tree,
            witnesses,
            delta_b_n: DELTA_B_N_START,
            delta_s: DELTA_B_N_START / 4.0,
            h_cache: Vec::new(),
            open: BinaryHeap::new(),
            counter: 0,
            chain: None,
            chain_depth: 0,
            it: 0,
            found_at: None,
            window: None,
            use_turns: false,
            turns_tried: Vec::new(),
            verbose: false,
        };
        planner.push(0, guide);
        planner
    }

    /// The start state (the tree's root).
    pub fn start(&self) -> State {
        self.tree.nodes[0].state
    }

    /// h of node n, cached until the guide's version changes.
    pub fn h(&mut self, n: usize, guide: &mut dyn Guide) -> f64 {
        if self.h_cache.len() <= n {
            self.h_cache.resize(n + 1, None);
        }
        let version = guide.version();
        if let Some((v, h)) = self.h_cache[n] && v == version {
            return h;
        }
        let h = guide.h(&self.tree.nodes[n].state);
        self.h_cache[n] = Some((version, h));
        h
    }

    fn push(&mut self, n: usize, guide: &mut dyn Guide) {
        let h = self.h(n, guide);
        let node = &self.tree.nodes[n];
        let priority = node.state.cost + h + EXPANSION_PENALTY * node.expansions as f64 + self.rng.uniform(0.0, TIE_BREAK);
        if priority.is_finite() {
            self.counter += 1;
            self.open.push(Reverse((Key(priority), self.counter, n)));
        }
    }

    /// Add an edge to the tree; a kept child is queued and reported to the guide.
    fn add(&mut self, path: Vec<State>, parent: usize, guide: &mut dyn Guide) -> Option<usize> {
        let child = self.tree.add(path, parent, &mut self.witnesses, self.delta_s)?;
        self.push(child, guide);
        guide.on_child(&self.tree, self.it, child);
        Some(child)
    }

    /// One planning iteration.
    pub fn step(&mut self, map: &Map, guide: &mut dyn Guide) {
        self.it += 1;
        let early = self.it <= EARLY_STEER_ITERATIONS;
        let steer_sigma = STEER_TARGET_SIGMA * if early { EARLY_STEER_VARIANCE_SCALE.sqrt() } else { 1.0 };
        let accel_mean = if early { Some(EARLY_ACCEL_MEAN) } else { None };

        // Pick the node to expand: continue a greedy chain, the lowest g + h, or (otherwise) normal SST selection
        let mut node = None;
        if let Some(c) = self.chain.filter(|&c| self.tree.nodes[c].active) {
            node = Some(c);
        } else if self.rng.random() < GREEDY_SHARE {
            while let Some(Reverse((_, _, candidate))) = self.open.pop() {
                if self.tree.nodes[candidate].active {
                    node = Some(candidate);
                    break;
                }
            }
            self.chain_depth = 0;
        }
        self.chain = None;

        if let Some(n) = node {
            let x = self.tree.nodes[n].state;
            let path = if self.tree.nodes[n].expansions == 0 {
                // Blossom: the primitives plus a few random actions; the best g + h the guide keeps
                let mut paths: Vec<Vec<State>> = PRIMITIVES.iter().map(|&p| primitive_path(map, &x, p)).collect();
                for _ in 0..BLOSSOM_RANDOM {
                    paths.push(monte_carlo_prop(map, &mut self.rng, &x, steer_sigma, accel_mean));
                }
                let mut scored: Vec<(f64, Vec<State>)> = paths
                    .into_iter()
                    .filter(|p| !p.is_empty())
                    .map(|p| {
                        let end = p.last().unwrap();
                        (end.cost + guide.h(end), p)
                    })
                    .collect();
                scored.sort_by(|a, b| a.0.total_cmp(&b.0));
                scored.into_iter().map(|(_, p)| p).find(|p| guide.keep(p)).unwrap_or_default()
            } else {
                let path = monte_carlo_prop(map, &mut self.rng, &x, steer_sigma, accel_mean);
                if !path.is_empty() && guide.keep(&path) { path } else { Vec::new() }
            };
            self.tree.nodes[n].expansions += 1;
            if self.tree.nodes[n].active {
                self.push(n, guide); // back in line, a little further back for each expansion
            }
            if !path.is_empty()
                && let Some(child) = self.add(path, n, guide)
            {
                let closer = self.h(child, guide) < self.h(n, guide);
                if closer && self.chain_depth < GREEDY_CHAIN_MAX {
                    self.chain = Some(child);
                    self.chain_depth += 1;
                }
            }
            if self.turns_tried.len() <= n {
                self.turns_tried.resize(n + 1, false);
            }
            if self.use_turns && !self.turns_tried[n] {
                // Stuck: every turn that fits becomes an extra child
                self.turns_tried[n] = true;
                for side in [-1.0, 1.0] {
                    for v in TURN_PRIMITIVE_SPEEDS {
                        let turn = turn_path(map, &x, side * TURN_PRIMITIVE_ANGLE, v);
                        if !turn.is_empty() && guide.keep(&turn) {
                            self.add(turn, n, guide);
                        }
                    }
                }
            }
        } else {
            let sample = guide.sample(&self.tree, self.it, &mut self.rng);
            let parent = match sample {
                Some(Sample { state, nearest_only: true }) => self.tree.nearest_active(&self.witnesses, &state),
                Some(Sample { state, .. }) => {
                    self.tree.select_near(&self.witnesses, &state, self.delta_b_n, self.delta_s)
                }
                None => self.tree.select(map, &mut self.rng, &self.witnesses, self.delta_b_n, self.delta_s),
            };
            if let Some(parent) = parent {
                let x = self.tree.nodes[parent].state;
                let path = monte_carlo_prop(map, &mut self.rng, &x, steer_sigma, accel_mean);
                if !path.is_empty() && guide.keep(&path) {
                    self.add(path, parent, guide);
                }
            }
        }

        let Some(best) = self.tree.best_goal().map(|g| self.tree.nodes[g].state.cost) else {
            if self.verbose && self.it.is_multiple_of(10_000) {
                println!("planner: iteration {}, no path yet, {} active nodes", self.it, self.tree.active_count);
            }
            return;
        };
        match self.window {
            None => {
                if self.verbose {
                    println!("planner: path found at iteration {}, cost {best:.2}", self.it);
                }
                self.window = Some((self.it, best));
                self.found_at = Some(self.it);
            }
            Some((start, cost)) if self.it - start >= ANNEAL_WINDOW => {
                let gain = (cost - best) / cost;
                let anneal = gain < ANNEAL_MIN_GAIN && self.delta_b_n * ANNEAL_FACTOR >= DELTA_B_N_MIN;
                if anneal {
                    self.delta_b_n *= ANNEAL_FACTOR;
                    self.delta_s *= ANNEAL_FACTOR;
                    let old = std::mem::replace(&mut self.witnesses, WitnessMap::new(self.delta_s));
                    self.witnesses = old.rebuilt(self.delta_s);
                }
                if self.verbose {
                    let radii = if anneal {
                        format!(", annealed to delta_B_N {:.2}, delta_S {:.2}", self.delta_b_n, self.delta_s)
                    } else {
                        String::new()
                    };
                    println!(
                        "planner: iteration {}, best cost {best:.2} ({:.1}% better in the last {ANNEAL_WINDOW}){radii}, \
                         {} active nodes",
                        self.it, 100.0 * gain, self.tree.active_count
                    );
                }
                self.window = Some((self.it, best));
            }
            Some(_) => {}
        }
    }

    pub fn run(&mut self, map: &Map, guide: &mut dyn Guide, iterations: usize) {
        for _ in 0..iterations {
            self.step(map, guide);
        }
    }

    /// The states from the start to the cheapest goal node, if a path exists.
    pub fn best_path(&self) -> Option<Vec<State>> {
        self.tree.best_goal().map(|g| self.tree.path_states(g))
    }
}
