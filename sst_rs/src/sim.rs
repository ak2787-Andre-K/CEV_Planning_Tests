//! The messages between the sim (or `fake_world`, standing in for it) and `sim_planner`: the team's protobuf types
//! from cev-interfaces, and converting them to and from the planner's units.
//!
//! Everything on the wire follows cev-interfaces' conventions: SI units, radians, world x east / y north, heading
//! counterclockwise from +x. Poses are the middle of the rear axle.
pub use cev_interfaces::{Goal, Header, Path, PathPoint, Point2, Pose2, VehicleState};

use crate::config::{FREE, METERS_PER_UNIT, OCCUPIED, SCREEN_HEIGHT, SCREEN_WIDTH, UNKNOWN};
use crate::map::{Map, cell};
use crate::tree::State;

/// The lidar's global map (OccupancyGrid in cev-interfaces' lidar.proto).
pub const MAP_KEY: &str = "cev/sensors/lidar/occupancy_grid/global";
pub const STATE_KEY: &str = "cev/vehicle/state";
pub const GOAL_KEY: &str = "cev/planning/goal";
pub const GLOBAL_PATH_KEY: &str = "cev/planning/global_path";

/// The lidar's occupancy grid, field for field as in cev-interfaces' lidar.proto (cornellev/cev-interfaces#3, not
/// merged yet, so not in the generated crate). Protobuf only goes by field numbers, so this encodes and decodes the
/// same bytes; use `cev_interfaces::OccupancyGrid` instead once #3 merges.
#[derive(Clone, PartialEq, prost::Message)]
pub struct OccupancyGrid {
    #[prost(message, optional, tag = "1")]
    pub header: Option<Header>,
    #[prost(uint32, tag = "2")]
    pub size_x: u32,
    #[prost(uint32, tag = "3")]
    pub size_y: u32,
    #[prost(float, tag = "4")]
    pub resolution_m: f32,
    #[prost(float, tag = "5")]
    pub origin_x: f32,
    #[prost(float, tag = "6")]
    pub origin_y: f32,
    /// Row-major: cell (x, y) is at index y * size_x + x. Values follow nav2: 0 free, 1-252 cost, 253 inscribed,
    /// 254 lethal, 255 unknown.
    #[prost(bytes = "vec", tag = "7")]
    pub data: Vec<u8>,
}

/// nav2 cell values.
pub const NAV2_FREE: u8 = 0;
pub const NAV2_INSCRIBED: u8 = 253; // 253 and up: the car's center there would hit something
pub const NAV2_LETHAL: u8 = 254;
pub const NAV2_UNKNOWN: u8 = 255;

impl OccupancyGrid {
    /// The value of the cell containing (x, y) meters, or None off the grid.
    pub fn at(&self, x: f64, y: f64) -> Option<u8> {
        let res = self.resolution_m as f64;
        let (col, row) = ((x - self.origin_x as f64) / res, (y - self.origin_y as f64) / res);
        if col < 0.0 || row < 0.0 || col as usize >= self.size_x as usize || row as usize >= self.size_y as usize {
            return None;
        }
        Some(self.data[row as usize * self.size_x as usize + col as usize])
    }

    /// The planner's map: each of its cells takes the value of the grid cell under its center. 253-254 (inscribed,
    /// lethal) are obstacles, 255 and off the grid are unknown, and 0-252 are free: the planner keeps its own
    /// clearance from obstacles, so it doesn't need the grid's costs.
    pub fn to_map(&self) -> Map {
        let mut occupancy = vec![UNKNOWN; SCREEN_WIDTH * SCREEN_HEIGHT];
        for x in 0..SCREEN_WIDTH {
            for y in 0..SCREEN_HEIGHT {
                occupancy[cell(x, y)] = match self.at(x as f64 * METERS_PER_UNIT, y as f64 * METERS_PER_UNIT) {
                    Some(NAV2_UNKNOWN) | None => UNKNOWN,
                    Some(v) if v >= NAV2_INSCRIBED => OCCUPIED,
                    Some(_) => FREE,
                };
            }
        }
        Map::from_occupancy(occupancy)
    }
}

/// Meters to planner units, and back.
pub fn to_units(m: f64) -> f64 {
    m / METERS_PER_UNIT
}

pub fn to_meters(u: f64) -> f64 {
    u * METERS_PER_UNIT
}

/// A pose in meters.
pub fn pose2(x: f64, y: f64, heading: f64) -> Pose2 {
    Pose2 { position: Some(Point2 { x, y }), heading }
}

/// (x, y, heading) of a pose, in meters (an unset position is the origin).
pub fn xy_heading(pose: Option<&Pose2>) -> (f64, f64, f64) {
    pose.map_or((0.0, 0.0, 0.0), |p| {
        let position = p.position.unwrap_or_default();
        (position.x, position.y, p.heading)
    })
}

/// A header stamped now.
pub fn header_now(frame_id: &str) -> Header {
    let stamp_ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    Header { stamp_ns, frame_id: frame_id.into(), seq: 0 }
}

/// The heading wrapped into (-pi, pi], as cev-interfaces' Pose2 wants it.
fn wrap_heading(a: f64) -> f64 {
    let w = (a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
    if w == -std::f64::consts::PI { std::f64::consts::PI } else { w }
}

impl From<&State> for PathPoint {
    fn from(s: &State) -> PathPoint {
        PathPoint {
            pose: Some(pose2(to_meters(s.x), to_meters(s.y), wrap_heading(s.a))),
            speed_mps: to_meters(s.s),
            steering_rad: s.steer,
            time_s: s.time,
        }
    }
}

/// The merceves car's body and wheelbase in meters, for drawing it (from Jaiden, 2026-11-06).
pub const CAR_LENGTH_M: f64 = 43.0 * 0.0254;
pub const CAR_WIDTH_M: f64 = 22.5 * 0.0254;
pub const WHEELBASE_M: f64 = 26.0 * 0.0254;
