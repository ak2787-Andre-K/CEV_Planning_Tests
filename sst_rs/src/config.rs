//! The settings the global planner uses, with the same values as `sst/config.py`.
use std::f64::consts::PI;

pub const SCREEN_WIDTH: usize = 2000;
pub const SCREEN_HEIGHT: usize = 1000;
pub const START: (f64, f64) = (250.0, 100.0);
pub const START_HEADING: f64 = 0.0; // rad; 0 = facing right (+x), positive = clockwise on screen
pub const TARGET: (f64, f64) = (1950.0, 700.0);
pub const TARGET_RAD: f64 = 70.0;
pub const MAX_PLAN_ITERATIONS: usize = 100_000;

// SST* annealing: once a path exists, every ANNEAL_WINDOW iterations the best cost is compared with the window's
// start; if it improved by less than ANNEAL_MIN_GAIN, delta_B_N and delta_S shrink by ANNEAL_FACTOR
pub const ANNEAL_WINDOW: usize = 10_000;
pub const ANNEAL_MIN_GAIN: f64 = 0.10;
pub const ANNEAL_FACTOR: f64 = 0.8;

// Car geometry (px)
pub const WHEELBASE: f64 = 6.0; // distance between front and rear axles
// Collision footprint: 3 circles in a row along the 10 x 5 px body (+1 px for grid rounding)
pub const FOOTPRINT_RADIUS: f64 = 4.004626062886658; // hypot(10 / 3 / 2, 5 / 2) + 1
pub const FOOTPRINT_OFFSETS: [f64; 3] = [-1.0 / 3.0, 3.0, 19.0 / 3.0]; // circle centers, px ahead of the rear axle
pub const FOOTPRINT_REACH: f64 = 19.0 / 3.0; // farthest circle center from the rear axle
pub const SAFETY_MARGIN: f64 = 2.0; // px of clearance every checked state must have
pub const CHECK_SPACING: f64 = 2.0 * SAFETY_MARGIN; // most any footprint circle moves between check points
pub const SDF_CAP: f64 = 20.0;

// Selection and witness radii, in state distance units (roughly px)
pub const DELTA_B_N_START: f64 = 60.0;
pub const DELTA_B_N_MIN: f64 = DELTA_B_N_START / 20.0; // never annealed below this
// State distance: euclidean px + W_A * heading diff (rad) + W_V * speed diff (px/s) + W_ST * steering diff (rad)
pub const W_A: f64 = 14.32394487827058; // 1/4 per degree
pub const W_V: f64 = 1.0 / 5.0;
pub const W_ST: f64 = 1.0 / 10.0;
pub const TIME_PROP: f64 = 0.5; // s, longest edge

// Edge cost per second: 1 + COST_ACCEL |accel| + COST_DECEL braking + COST_STEER_RATE |rate| + COST_STEER_ANGLE |steer|
pub const COST_ACCEL: f64 = 0.0;
pub const COST_STEER_RATE: f64 = 0.0;
pub const COST_STEER_ANGLE: f64 = 10.8;
pub const COST_DECEL: f64 = 0.01;
pub const PLAN_DT: f64 = 1.0 / 80.0; // s, planner step; edges last a whole number of these

// Car dynamics
pub const MAX_ACCELERATION: f64 = 120.0; // px/s^2
pub const MAX_SPEED: f64 = 300.0; // px/s
pub const MAX_REVERSE_SPEED: f64 = 100.0;
pub const MAX_STEER_RATE: f64 = 20.0 * PI / 180.0; // rad/s
pub const MAX_STEER: f64 = 35.0 * PI / 180.0; // rad, front wheel steering angle
pub const MAX_LATERAL_ACCELERATION: f64 = 200.0;
pub const STEER_TARGET_SIGMA: f64 = 0.5; // random target steering spread, times the most allowed at the edge's top speed
pub const EARLY_STEER_ITERATIONS: usize = 100; // the first iterations sample steering with a narrower spread...
pub const EARLY_STEER_VARIANCE_SCALE: f64 = 0.25; // ...this fraction of the normal variance
pub const EARLY_ACCEL_MEAN: f64 = MAX_ACCELERATION / 2.0; // ...and acceleration from a normal around this
pub const EARLY_ACCEL_SIGMA: f64 = MAX_ACCELERATION / 2.0 * std::f64::consts::FRAC_1_SQRT_2; // with this spread

// Informed expansion: most iterations expand the active node with the lowest g + h instead of a random sample's
pub const GREEDY_SHARE: f64 = 0.75;
pub const GREEDY_CHAIN_MAX: usize = 15; // a child closer to the goal is expanded next, at most this many in a row
pub const EXPANSION_PENALTY: f64 = 0.5; // added to a node's g + h each time it's expanded
pub const TIE_BREAK: f64 = 0.05; // random 0..this added to g + h
pub const BLOSSOM_RANDOM: usize = 5; // a node's first expansion tries the PRIMITIVES plus this many random actions
// (acceleration as a fraction of MAX_ACCELERATION, target steering as a fraction of the most allowed, duration s)
pub const PRIMITIVES: [(f64, f64, f64); 5] = [
    (1.0, 0.0, 0.3),  // speed up, wheel straight
    (0.0, 0.0, 0.3),  // hold speed, wheel straight
    (0.5, 0.3, 0.3),  // speed up gently, gentle right
    (0.5, -0.3, 0.3), // speed up gently, gentle left
    (-0.5, 0.0, 0.3), // brake gently, wheel straight
];
pub const H_CELL: usize = 10; // px, cell size of the grid h is computed on
pub const GRID_INFLATION: f64 = FOOTPRINT_RADIUS + SAFETY_MARGIN + 2.0; // clearance a free grid cell's center needs

// Smoothing (smooth_path): Hermite-curve shortcuts from every node to route states up to SMOOTH_NODES_AHEAD nodes ahead
pub const JOIN_SPEED_OVER: f64 = 10.0; // px/s, how much faster than a route a state may be when joining it
pub const SMOOTH_TOLERANCE_SCALE: f64 = 2.0 / 3.0;
pub const SMOOTH_NODES_AHEAD: usize = 2;
pub const SMOOTH_LOOP_SPAN: f64 = 250.0; // px; from the start of a loopy stretch, targets reach this far instead
pub const SMOOTH_TARGET_SPACING: f64 = 15.0; // px between target states tried along the path
pub const SMOOTH_JOIN_SLACK: usize = 30; // route states past the last target a shortcut may still join at
pub const SMOOTH_TANGENTS: [f64; 5] = [0.3, 0.5, 0.75, 1.0, 1.3]; // Hermite end tangents, times the straight distance
pub const SMOOTH_LOOKAHEADS: [f64; 3] = [10.0, 20.0, 35.0]; // px, pure pursuit lookahead
pub const SMOOTH_BUMPS: [f64; 2] = [0.0, 0.15]; // extra speed mid-shortcut, times the faster end's speed
pub const SMOOTH_STEER_BLEND: f64 = 0.3; // over this last fraction of the curve steering blends toward the target's
pub const SMOOTH_MIN_SPEED: f64 = 20.0; // px/s, least reference speed mid-shortcut
pub const SMOOTH_MIN_GAIN: f64 = 0.002; // a shortcut must make the cost to the goal at least this fraction cheaper
pub const SMOOTH_RETRIES: usize = 4; // times the same node is tried again after a splice
pub const SMOOTH_SWEEPS: usize = 3; // sweeps over the whole path, stopping early once one changes nothing
pub const NUDGE_NODE_FRAMES: usize = 8; // a shortcut gets a node every this many PLAN_DT steps
pub const LOOP_SPAN: f64 = 150.0; // px; a stretch up to this long counts as loopy if...
pub const LOOP_DETOUR_RATIO: f64 = 1.4; // ...it's at least this many times the straight line across it,
pub const LOOP_TURN: f64 = 270.0 * PI / 180.0; // ...or its heading turns through at least this much

// Street grid: road centerlines (px); everything between roads is a city block
pub const STREET_X: [f64; 6] = [250.0, 620.0, 980.0, 1330.0, 1680.0, 1950.0];
pub const STREET_Y: [f64; 3] = [100.0, 390.0, 700.0];
pub const STREET_WIDTH: f64 = 80.0;
pub const BLOCK_CORNER: f64 = 15.0; // rounded block corners, px
pub const DIAGONALS: [((f64, f64), (f64, f64)); 2] =
    [((0.0, 1000.0), (980.0, 100.0)), ((1330.0, 100.0), (2000.0, 760.0))];
