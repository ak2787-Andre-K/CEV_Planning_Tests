//! The global SST* planner with informed expansion (`GlobalPlanner` and `planGlobal` in `sst/global_planner.py`).
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::config::{
    ANNEAL_FACTOR, ANNEAL_MIN_GAIN, ANNEAL_WINDOW, BLOSSOM_RANDOM, DELTA_B_N_MIN, DELTA_B_N_START, EARLY_ACCEL_MEAN,
    EARLY_STEER_ITERATIONS, EARLY_STEER_VARIANCE_SCALE, EXPANSION_PENALTY, GREEDY_CHAIN_MAX, GREEDY_SHARE,
    GRID_INFLATION, H_CELL, MAX_ACCELERATION, MAX_SPEED, PLAN_DT, PRIMITIVES, SCREEN_HEIGHT, SCREEN_WIDTH,
    STEER_TARGET_SIGMA, TIE_BREAK,
};
use crate::kinematics::{Action, feasible_accel, max_steer_at_speed, monte_carlo_prop, propagate};
use crate::map::{Map, cell};
use crate::rng::Random;
use crate::tree::{State, Tree, Witness, WitnessMap};

/// An f64 that orders totally, for the heaps.
#[derive(Clone, Copy, PartialEq)]
struct Key(f64);

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

/// Grid Dijkstra distance (px) from every H_CELL cell to the goal circle through free space, 8-connected. A cell is
/// free if its center is at least GRID_INFLATION from known obstacles. Unreachable cells are inf. Indexed i * ny + j.
pub fn goal_distance_grid(map: &Map, goal: (f64, f64, f64)) -> Vec<f64> {
    let (nx, ny) = (SCREEN_WIDTH / H_CELL, SCREEN_HEIGHT / H_CELL);
    let center = |i: usize| (i * H_CELL + H_CELL / 2) as f64;
    let free: Vec<bool> = (0..nx * ny)
        .map(|k| map.obstacle_sdf[cell(center(k / ny) as usize, center(k % ny) as usize)] as f64 >= GRID_INFLATION)
        .collect();
    let mut dist = vec![f64::INFINITY; nx * ny];
    let mut heap = BinaryHeap::new();
    for k in 0..nx * ny {
        if free[k] && (center(k / ny) - goal.0).hypot(center(k % ny) - goal.1) <= goal.2 {
            dist[k] = 0.0;
            heap.push(Reverse((Key(0.0), k)));
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

/// SST* with informed expansion, run in steps.
///
/// Each step expands one node. With probability GREEDY_SHARE it's the active node with the lowest g + h (g = cost,
/// h = grid distance to the goal / MAX_SPEED, plus EXPANSION_PENALTY per earlier expansion and a small random
/// tie-break). A node's first expansion "blossoms": the PRIMITIVES and BLOSSOM_RANDOM random actions are tried and
/// the child with the best g + h is kept; later expansions try one random action. A kept child closer to the goal
/// than its parent is expanded next, up to GREEDY_CHAIN_MAX in a row. The other steps are normal SST (random sample,
/// best-first selection, one random action).
///
/// Once a path exists, every ANNEAL_WINDOW steps the best cost is compared with the window's start; if it improved by
/// less than ANNEAL_MIN_GAIN, delta_b_n and delta_s shrink by ANNEAL_FACTOR (not below DELTA_B_N_MIN).
pub struct Planner<'a> {
    map: &'a Map,
    pub rng: Random,
    pub tree: Tree,
    witnesses: WitnessMap,
    pub delta_b_n: f64,
    pub delta_s: f64,
    h_grid: Vec<f64>,
    open: BinaryHeap<Reverse<(Key, usize, usize)>>, // (g + h, counter, node); stale entries are skipped when popped
    counter: usize,
    chain: Option<usize>,
    chain_depth: usize,
    pub it: usize,
    pub found_at: Option<usize>, // iteration the first path was found at
    window: Option<(usize, f64)>, // (iteration, best cost) at the start of the improvement window, once a path exists
    pub verbose: bool,
}

impl<'a> Planner<'a> {
    pub fn new(map: &'a Map, start: State, goal: (f64, f64, f64), seed: u64) -> Planner<'a> {
        let tree = Tree::new(start, goal);
        let mut witnesses = WitnessMap::new(DELTA_B_N_START / 4.0);
        witnesses.add(Witness { center: tree.nodes[0].state, rep: Some(0) });
        let mut planner = Planner {
            map,
            rng: Random::seeded(seed),
            tree,
            witnesses,
            delta_b_n: DELTA_B_N_START,
            delta_s: DELTA_B_N_START / 4.0,
            h_grid: goal_distance_grid(map, goal),
            open: BinaryHeap::new(),
            counter: 0,
            chain: None,
            chain_depth: 0,
            it: 0,
            found_at: None,
            window: None,
            verbose: false,
        };
        planner.push(0);
        planner
    }

    fn h(&self, s: &State) -> f64 {
        let ny = SCREEN_HEIGHT / H_CELL;
        let i = ((s.x / H_CELL as f64).floor().max(0.0) as usize).min(SCREEN_WIDTH / H_CELL - 1);
        let j = ((s.y / H_CELL as f64).floor().max(0.0) as usize).min(ny - 1);
        self.h_grid[i * ny + j] / MAX_SPEED
    }

    fn push(&mut self, n: usize) {
        let node = &self.tree.nodes[n];
        let priority = node.state.cost + self.h(&node.state) + EXPANSION_PENALTY * node.expansions as f64
            + self.rng.uniform(0.0, TIE_BREAK);
        if priority.is_finite() {
            self.counter += 1;
            self.open.push(Reverse((Key(priority), self.counter, n)));
        }
    }

    fn add(&mut self, path: Vec<State>, parent: usize) -> Option<usize> {
        let child = self.tree.add(path, parent, &mut self.witnesses, self.delta_s)?;
        self.push(child);
        Some(child)
    }

    /// One planning iteration.
    pub fn step(&mut self) {
        self.it += 1;
        let early = self.it <= EARLY_STEER_ITERATIONS;
        let steer_sigma = STEER_TARGET_SIGMA * if early { EARLY_STEER_VARIANCE_SCALE.sqrt() } else { 1.0 };
        let accel_mean = if early { Some(EARLY_ACCEL_MEAN) } else { None };
        let map = self.map;

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
                // Blossom: the primitives plus a few random actions, keeping the best g + h
                let mut paths: Vec<Vec<State>> = PRIMITIVES.iter().map(|&p| primitive_path(map, &x, p)).collect();
                for _ in 0..BLOSSOM_RANDOM {
                    paths.push(monte_carlo_prop(map, &mut self.rng, &x, steer_sigma, accel_mean));
                }
                paths
                    .into_iter()
                    .filter(|p| !p.is_empty())
                    .min_by(|p, q| {
                        let f = |path: &Vec<State>| path.last().map_or(f64::INFINITY, |s| s.cost + self.h(s));
                        f(p).total_cmp(&f(q))
                    })
                    .unwrap_or_default()
            } else {
                monte_carlo_prop(map, &mut self.rng, &x, steer_sigma, accel_mean)
            };
            self.tree.nodes[n].expansions += 1;
            if self.tree.nodes[n].active {
                self.push(n); // back in line, a little further back for each expansion
            }
            if !path.is_empty()
                && let Some(child) = self.add(path, n) {
                    let closer = self.h(&self.tree.nodes[child].state) < self.h(&x);
                    if closer && self.chain_depth < GREEDY_CHAIN_MAX {
                        self.chain = Some(child);
                        self.chain_depth += 1;
                    }
                }
        } else if let Some(parent) = self.tree.select(map, &mut self.rng, &self.witnesses, self.delta_b_n, self.delta_s) {
            let x = self.tree.nodes[parent].state;
            let path = monte_carlo_prop(map, &mut self.rng, &x, steer_sigma, accel_mean);
            if !path.is_empty() {
                self.add(path, parent);
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

    pub fn run(&mut self, iterations: usize) {
        for _ in 0..iterations {
            self.step();
        }
    }

    /// The states from the start to the cheapest goal node, if a path exists.
    pub fn best_path(&self) -> Option<Vec<State>> {
        self.tree.best_goal().map(|g| self.tree.path_states(g))
    }
}
