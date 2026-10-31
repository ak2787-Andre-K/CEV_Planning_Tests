//! The SST global planner from `sst/` in Rust, plus the JSON messages its zenoh node exchanges.
//!
//! Same map, car model, planner and smoothing as the Python package, with the same settings. Its random draws differ
//! from Python's, so a seed doesn't give the same path as in Python, but the planner behaves the same way.
pub mod config;
pub mod driver;
pub mod kinematics;
pub mod library;
pub mod maneuvers;
pub mod map;
pub mod planner;
pub mod rng;
pub mod route;
pub mod smoothing;
pub mod tree;
pub mod world;

use std::time::Instant;

use serde::{Deserialize, Serialize};

use config::{MAX_PLAN_ITERATIONS, START, START_HEADING, TARGET, TARGET_RAD};
use map::{Map, Shape};
use planner::{GoalGrid, Planner};
use tree::State;

/// Key expressions the planner node uses.
pub const REQUEST_KEY: &str = "cev/planner/sst/request";
pub const PATH_KEY: &str = "cev/planner/sst/path";

/// Ask the planner for a path. Every field is optional; missing ones use the `sst` defaults.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct PlanRequest {
    #[serde(default)]
    pub id: u64,
    pub start: Option<(f64, f64)>,
    pub start_heading: Option<f64>, // rad; 0 = +x, positive = clockwise on screen
    pub goal: Option<(f64, f64)>,
    pub goal_radius: Option<f64>,
    pub iterations: Option<usize>,
    pub seed: Option<u64>,
    /// Whether to smooth the path with shortcuts after planning (default true).
    pub smooth: Option<bool>,
    /// Extra obstacles to add to the street map for this request.
    #[serde(default)]
    pub obstacles: Vec<Shape>,
}

/// The planner's answer, published on `PATH_KEY`. `path` is empty when no path was found; otherwise it runs from the
/// start into the goal region, one state per collision check point, with `node` marking the tree nodes.
#[derive(Debug, Serialize, Deserialize)]
pub struct PlanResponse {
    pub id: u64,
    pub found: bool,
    pub cost: f64,
    pub raw_cost: f64, // cost before smoothing
    pub time: f64, // s of driving
    pub path: Vec<State>,
    pub iterations: usize,
    pub found_at: Option<usize>, // iteration the first path was found at
    pub active_nodes: usize,
    pub plan_ms: f64,
    pub smooth_ms: f64,
}

/// Plan (and, unless the request says not to, smooth) a path for `request`. `streets` is the plain street map, reused
/// when the request adds no obstacles.
pub fn plan(streets: &Map, request: &PlanRequest, verbose: bool) -> PlanResponse {
    let with_obstacles;
    let map = if request.obstacles.is_empty() {
        streets
    } else {
        with_obstacles = Map::streets(&request.obstacles);
        &with_obstacles
    };
    let (x, y) = request.start.unwrap_or(START);
    let start = State::new(x, y, request.start_heading.unwrap_or(START_HEADING), 0.0);
    let (gx, gy) = request.goal.unwrap_or(TARGET);
    let goal = (gx, gy, request.goal_radius.unwrap_or(TARGET_RAD));
    let seed = request.seed.unwrap_or_else(rand::random);

    let timer = Instant::now();
    let mut guide = GoalGrid::new(map, goal);
    let mut planner = Planner::new(start, goal, seed, &mut guide);
    planner.verbose = verbose;
    planner.run(map, &mut guide, request.iterations.unwrap_or(MAX_PLAN_ITERATIONS));
    let plan_ms = timer.elapsed().as_secs_f64() * 1000.0;

    let timer = Instant::now();
    let mut path = planner.best_path().unwrap_or_default();
    let raw_cost = path.last().map_or(0.0, |s| s.cost);
    if !path.is_empty() && request.smooth.unwrap_or(true) {
        path = smoothing::smooth_path(map, path);
    }
    let smooth_ms = timer.elapsed().as_secs_f64() * 1000.0;
    let last = path.last().copied().unwrap_or_default();
    PlanResponse {
        id: request.id,
        found: !path.is_empty(),
        cost: last.cost,
        raw_cost,
        time: last.time,
        path,
        iterations: planner.it,
        found_at: planner.found_at,
        active_nodes: planner.tree.active_count,
        plan_ms,
        smooth_ms,
    }
}

/// Zenoh config from the command line: `--connect <endpoint>` and `--listen <endpoint>`,
/// e.g. `--listen tcp/0.0.0.0:7447` on one node and `--connect tcp/<ip>:7447` on the other.
/// With neither, nodes find each other by multicast scouting on the local network.
pub fn zenoh_config() -> zenoh::Result<zenoh::Config> {
    let mut config = zenoh::Config::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let key = match arg.as_str() {
            "--connect" => "connect/endpoints",
            "--listen" => "listen/endpoints",
            _ => continue,
        };
        let endpoint = args.next().ok_or(format!("{arg} needs an endpoint"))?;
        config.insert_json5(key, &serde_json::to_string(&[endpoint])?)?;
    }
    Ok(config)
}
