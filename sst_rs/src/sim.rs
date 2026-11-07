//! The messages between `fake_world` (standing in for the sim) and `sim_planner`, and converting them to and from the
//! planner's units.
//!
//! These are PLACEHOLDERS until Cam's centralized types exist; swap them for those then. Everything on the wire is in
//! meters and radians, x and y on the ground with heading 0 along +x and positive toward +y (the sim's x and z).
use serde::{Deserialize, Serialize};

use crate::config::{FREE, METERS_PER_UNIT, OCCUPIED, SCREEN_HEIGHT, SCREEN_WIDTH, UNKNOWN};
use crate::map::{Map, cell};
use crate::tree::State;

pub const MAP_KEY: &str = "cev/sim/map";
pub const POSE_KEY: &str = "cev/sim/pose";
pub const GOAL_KEY: &str = "cev/sim/goal";
pub const SIM_PATH_KEY: &str = "cev/planner/path";

/// An occupancy grid, laid out like ROS's nav_msgs/OccupancyGrid: `data` row by row (row 0 at `origin`, along +x),
/// -1 unknown, 0 free up to 100 occupied (50 or more counts as occupied).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OccupancyGrid {
    pub resolution: f64, // m per cell
    pub width: usize,    // cells along x
    pub height: usize,   // cells along y
    pub origin: (f64, f64), // m, the world position of cell (0, 0)'s corner
    pub data: Vec<i8>,
}

impl OccupancyGrid {
    /// The value of the cell containing (x, y) meters, or None off the grid.
    pub fn at(&self, x: f64, y: f64) -> Option<i8> {
        let (col, row) = ((x - self.origin.0) / self.resolution, (y - self.origin.1) / self.resolution);
        if col < 0.0 || row < 0.0 || col as usize >= self.width || row as usize >= self.height {
            return None;
        }
        Some(self.data[row as usize * self.width + col as usize])
    }

    /// The planner's map: each of its cells takes the value of the grid cell under its center (unknown off the grid).
    pub fn to_map(&self) -> Map {
        let mut occupancy = vec![UNKNOWN; SCREEN_WIDTH * SCREEN_HEIGHT];
        for x in 0..SCREEN_WIDTH {
            for y in 0..SCREEN_HEIGHT {
                occupancy[cell(x, y)] = match self.at(x as f64 * METERS_PER_UNIT, y as f64 * METERS_PER_UNIT) {
                    Some(v) if v >= 50 => OCCUPIED,
                    Some(v) if v >= 0 => FREE,
                    _ => UNKNOWN,
                };
            }
        }
        Map::from_occupancy(occupancy)
    }
}

/// Where the car is.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    #[serde(default)]
    pub speed: f64,
}

/// Where the car should go: within `radius` of (x, y).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Goal {
    pub x: f64,
    pub y: f64,
    pub radius: f64,
}

/// One point of a planned path.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PathPoint {
    pub x: f64,
    pub y: f64,
    pub heading: f64,
    pub speed: f64, // m/s
    pub steer: f64, // rad, front wheel angle
    pub time: f64,  // s from the start
}

/// The planner's answer: the path from the car's pose into the goal region (empty if none was found).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlannedPath {
    pub found: bool,
    pub cost: f64,
    pub time: f64, // s of driving
    pub plan_ms: f64,
    pub path: Vec<PathPoint>,
}

/// Meters to planner units, and back.
pub fn to_units(m: f64) -> f64 {
    m / METERS_PER_UNIT
}

pub fn to_meters(u: f64) -> f64 {
    u * METERS_PER_UNIT
}

impl From<&State> for PathPoint {
    fn from(s: &State) -> PathPoint {
        PathPoint { x: to_meters(s.x), y: to_meters(s.y), heading: s.a, speed: to_meters(s.s), steer: s.steer, time: s.time }
    }
}

/// The merceves car's body and wheelbase in meters, for drawing it (from Jaiden, 2026-11-06).
pub const CAR_LENGTH_M: f64 = 43.0 * 0.0254;
pub const CAR_WIDTH_M: f64 = 22.5 * 0.0254;
pub const WHEELBASE_M: f64 = 26.0 * 0.0254;
