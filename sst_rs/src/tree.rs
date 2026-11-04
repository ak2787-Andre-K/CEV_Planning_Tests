//! SST tree pieces (`sst/tree.py`): states, the tree with SST's witness step and pruning, the witness map, node
//! selection and the state distance.
//!
//! Nodes live in an arena (`Tree::nodes`) and refer to each other by index. Pruned nodes stay in the arena, marked
//! not `alive`, so indices never move.
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::config::{MAX_REVERSE_SPEED, MAX_SPEED, SCREEN_HEIGHT, SCREEN_WIDTH, W_A, W_ST, W_V};
use crate::kinematics::{max_steer_at_speed, wrap};
use crate::map::Map;
use crate::rng::Random;

/// A car state: rear-axle pose, speed, steering, and the cost and driving time from the root.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct State {
    pub x: f64,
    pub y: f64,
    #[serde(rename = "heading")]
    pub a: f64, // rad
    #[serde(rename = "speed")]
    pub s: f64, // px/s (negative = reversing)
    pub steer: f64, // front wheel angle, rad
    pub cost: f64,
    pub time: f64, // s of driving from the root
    /// Whether this state is a tree node (the end of an edge) and not just a check point along one.
    pub node: bool,
}

impl State {
    pub fn new(x: f64, y: f64, a: f64, s: f64) -> State {
        State { x, y, a, s, ..Default::default() }
    }
}

/// Distance between states: euclidean px + W_A * heading difference + W_V * speed difference + W_ST * steering
/// difference. (The Python version stands this in for a Reeds-Shepp distance; it's symmetric.)
pub fn state_distance(s0: &State, s1: &State) -> f64 {
    (s1.x - s0.x).hypot(s1.y - s0.y) + W_A * wrap(s1.a - s0.a).abs() + W_V * (s1.s - s0.s).abs()
        + W_ST * (s1.steer - s0.steer).abs()
}

/// (position px, speed px/s, heading rad) two states within distance delta can differ by at most.
fn distance_bounds(delta: f64) -> (f64, f64, f64) {
    (delta, delta / W_V, delta / W_A)
}

pub struct Node {
    pub state: State,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// The check points of the edge from the parent to this node; the last one is this node's state.
    pub traj: Vec<State>,
    pub active: bool,
    pub alive: bool, // false once pruned
    pub expansions: usize, // times the g + h selection has expanded this node
}

/// An SST witness: a fixed point in state space (a copy of the node it was made for) and the node representing it.
pub struct Witness {
    pub center: State,
    pub rep: Option<usize>,
}

/// Witnesses in a position-only hash with cells as wide as the position bound for the radius it's built for.
/// Searching a smaller radius is still correct; rebuild once the radius has shrunk a lot.
pub struct WitnessMap {
    cell: f64,
    cells: HashMap<(i64, i64), Vec<usize>>,
    pub witnesses: Vec<Witness>,
}

impl WitnessMap {
    pub fn new(delta: f64) -> WitnessMap {
        WitnessMap { cell: distance_bounds(delta).0, cells: HashMap::new(), witnesses: Vec::new() }
    }

    fn key(&self, x: f64, y: f64) -> (i64, i64) {
        ((x / self.cell).floor() as i64, (y / self.cell).floor() as i64)
    }

    pub fn add(&mut self, w: Witness) -> usize {
        let index = self.witnesses.len();
        let key = self.key(w.center.x, w.center.y);
        self.cells.entry(key).or_default().push(index);
        self.witnesses.push(w);
        index
    }

    /// The same witnesses (and reps) in a map built for a new radius.
    pub fn rebuilt(self, delta: f64) -> WitnessMap {
        let mut map = WitnessMap::new(delta);
        for w in self.witnesses {
            map.add(w);
        }
        map
    }

    /// Witnesses that could be within distance delta of s (position, speed and heading checked; the caller still
    /// checks the exact distance).
    pub fn nbhd(&self, s: &State, delta: f64) -> Vec<usize> {
        let (r_pos, r_speed, r_heading) = distance_bounds(delta);
        let (i_lo, j_lo) = self.key(s.x - r_pos, s.y - r_pos);
        let (i_hi, j_hi) = self.key(s.x + r_pos, s.y + r_pos);
        let mut result = Vec::new();
        for i in i_lo..=i_hi {
            for j in j_lo..=j_hi {
                for &w in self.cells.get(&(i, j)).map(Vec::as_slice).unwrap_or(&[]) {
                    let o = &self.witnesses[w].center;
                    if (o.x - s.x).hypot(o.y - s.y) <= r_pos && (o.s - s.s).abs() <= r_speed
                        && wrap(o.a - s.a).abs() <= r_heading
                    {
                        result.push(w);
                    }
                }
            }
        }
        result
    }
}

/// The SST tree, rooted at node 0.
pub struct Tree {
    pub nodes: Vec<Node>,
    pub goal_nodes: Vec<usize>, // kept nodes inside the goal region
    pub goal: (f64, f64, f64), // goal region: x, y, radius
    pub active_count: usize,
}

impl Tree {
    pub fn new(root: State, goal: (f64, f64, f64)) -> Tree {
        let mut state = root;
        state.node = true;
        let node = Node {
            state, parent: None, children: Vec::new(), traj: Vec::new(), active: true, alive: true, expansions: 0,
        };
        Tree { nodes: vec![node], goal_nodes: Vec::new(), goal, active_count: 1 }
    }

    fn set_active(&mut self, n: usize, active: bool) {
        if self.nodes[n].active != active {
            self.nodes[n].active = active;
            if active { self.active_count += 1 } else { self.active_count -= 1 }
        }
    }

    /// This node and everything below it.
    pub fn subtree(&self, n: usize) -> Vec<usize> {
        let (mut stack, mut branch) = (vec![n], Vec::new());
        while let Some(v) = stack.pop() {
            branch.push(v);
            stack.extend(&self.nodes[v].children);
        }
        branch
    }

    /// Prune from n upward: if nothing in its branch is active, cut the whole branch off its parent, then check the
    /// parent the same way, stopping at the first node still needed. Returns the removed nodes.
    fn remove_until_active(&mut self, n: usize) -> Vec<usize> {
        let mut removed = Vec::new();
        let mut node = Some(n);
        while let Some(v) = node {
            let branch = self.subtree(v);
            if branch.iter().any(|&u| self.nodes[u].active) {
                break;
            }
            for &u in &branch {
                self.nodes[u].alive = false;
            }
            removed.extend(branch);
            let parent = self.nodes[v].parent.take();
            if let Some(p) = parent {
                self.nodes[p].children.retain(|&c| c != v);
            }
            node = parent;
        }
        removed
    }

    /// SST's witness step for a freshly propagated edge `path` (its last state is the new node) from `parent`.
    ///
    /// With no witness within delta_s of the new node, the node becomes a new witness and its rep. Otherwise the
    /// closest witness decides: if the new node is cheaper than its rep it takes over, and the old rep is made
    /// inactive and pruned; if not, it's dropped. Returns the new node's index if kept.
    pub fn add(&mut self, mut path: Vec<State>, parent: usize, witnesses: &mut WitnessMap, delta_s: f64) -> Option<usize> {
        let last = path.len() - 1;
        path[last].node = true;
        let state = path[last];
        let mut closest: Option<(usize, f64)> = None;
        for w in witnesses.nbhd(&state, delta_s) {
            let d = state_distance(&witnesses.witnesses[w].center, &state);
            if d <= delta_s && closest.is_none_or(|(_, best)| d < best) {
                closest = Some((w, d));
            }
        }
        let n = self.nodes.len();
        let (witness, old_rep) = match closest {
            None => (witnesses.add(Witness { center: state, rep: None }), None),
            Some((w, _)) => match witnesses.witnesses[w].rep {
                Some(rep) if state.cost >= self.nodes[rep].state.cost => return None,
                rep => (w, rep),
            },
        };

        // Attach before pruning, so the old rep's branch still counts the new node if it hangs below it
        self.nodes.push(Node {
            state, parent: Some(parent), children: Vec::new(), traj: path, active: false, alive: true, expansions: 0,
        });
        self.nodes[parent].children.push(n);
        self.set_active(n, true);
        witnesses.witnesses[witness].rep = Some(n);

        let (gx, gy, gr) = self.goal;
        if (state.x - gx).hypot(state.y - gy) <= gr {
            self.goal_nodes.push(n);
        }
        if let Some(old) = old_rep {
            self.set_active(old, false);
            let removed = self.remove_until_active(old);
            self.goal_nodes.retain(|g| !removed.contains(g));
        }
        Some(n)
    }

    /// The cheapest goal node, if there is one.
    pub fn best_goal(&self) -> Option<usize> {
        self.goal_nodes.iter().copied().min_by(|&a, &b| self.nodes[a].state.cost.total_cmp(&self.nodes[b].state.cost))
    }

    /// The states from the root (included) to `goal` along the tree: the root, then every edge's check points.
    pub fn path_states(&self, goal: usize) -> Vec<State> {
        let mut edges = Vec::new();
        let mut node = goal;
        while let Some(parent) = self.nodes[node].parent {
            edges.push(node);
            node = parent;
        }
        let mut states = vec![self.nodes[node].state];
        for &n in edges.iter().rev() {
            states.extend(&self.nodes[n].traj);
        }
        states
    }

    /// The cheapest rep within delta_b_n of s, or None. Every such rep belongs to a witness within
    /// delta_b_n + delta_s of s (a rep sits within delta_s of its witness).
    pub fn best_near(&self, witnesses: &WitnessMap, s: &State, delta_b_n: f64, delta_s: f64) -> Option<usize> {
        witnesses
            .nbhd(s, delta_b_n + delta_s)
            .into_iter()
            .filter_map(|w| witnesses.witnesses[w].rep)
            .filter(|&n| state_distance(&self.nodes[n].state, s) <= delta_b_n)
            .min_by(|&a, &b| self.nodes[a].state.cost.total_cmp(&self.nodes[b].state.cost))
    }

    /// The rep closest to s, over every witness.
    pub fn nearest_active(&self, witnesses: &WitnessMap, s: &State) -> Option<usize> {
        witnesses
            .witnesses
            .iter()
            .filter_map(|w| w.rep)
            .min_by(|&a, &b| state_distance(&self.nodes[a].state, s).total_cmp(&state_distance(&self.nodes[b].state, s)))
    }

    /// SST's best-first selection: sample a random state and pick the cheapest active node within delta_b_n of it, or
    /// the nearest active node if none is that close.
    pub fn select(&self, map: &Map, rng: &mut Random, witnesses: &WitnessMap, delta_b_n: f64, delta_s: f64) -> Option<usize> {
        self.select_near(witnesses, &random_state(map, rng), delta_b_n, delta_s)
    }

    /// Best-first selection for a given sample.
    pub fn select_near(&self, witnesses: &WitnessMap, sample: &State, delta_b_n: f64, delta_s: f64) -> Option<usize> {
        self.best_near(witnesses, sample, delta_b_n, delta_s).or_else(|| self.nearest_active(witnesses, sample))
    }
}

/// Uniform over the map, heading and the full speed range (reverse included), with a steering angle the lateral
/// limit allows at that speed. Positions on known obstacles are redrawn (up to 1000 tries).
pub fn random_state(map: &Map, rng: &mut Random) -> State {
    let (mut x, mut y) = (0.0, 0.0);
    for _ in 0..1000 {
        x = rng.uniform(0.0, SCREEN_WIDTH as f64);
        y = rng.uniform(0.0, SCREEN_HEIGHT as f64);
        let (px, py) = ((x as usize).min(SCREEN_WIDTH - 1), (y as usize).min(SCREEN_HEIGHT - 1));
        if !map.occupied(px, py) {
            break;
        }
    }
    let a = rng.uniform(0.0, std::f64::consts::TAU);
    let s = rng.uniform(-MAX_REVERSE_SPEED, MAX_SPEED);
    let limit = max_steer_at_speed(s.abs());
    let steer = rng.uniform(-limit, limit);
    State { steer, ..State::new(x, y, a, s) }
}
