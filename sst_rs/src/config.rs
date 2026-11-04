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

// Occupancy grid cell states
pub const FREE: u8 = 0;
pub const OCCUPIED: u8 = 1;
pub const UNKNOWN: u8 = 2;
pub const DIRTY_TILE: usize = 50; // px, map updates are grouped on a grid this coarse

// Hidden obstacles on the roads, found by the lidar while driving
pub const LOCAL_OBSTACLE_COUNT: usize = 60;
pub const LOCAL_OBSTACLE_RADIUS: (f64, f64) = (7.0, 20.0); // px, min/max for circles
pub const LOCAL_BOX_LENGTH: (f64, f64) = (15.0, 45.0); // px, min/max for boxes
pub const LOCAL_BOX_WIDTH: (f64, f64) = (8.0, 20.0);
pub const BLOCKED_ROADS: usize = 2; // road segments fully blocked by one big obstacle
pub const LIDAR_BEAMS: usize = 3600; // long range, so fine angular steps: ~1.7 px apart at 1000 px
pub const LIDAR_DT: f64 = 1.0 / 20.0; // s between lidar scans
pub const FPS: f64 = 60.0; // display frames per second the Driver's timings assume

// Driving the planned routes (Driver)
pub const SAFE_SPEED_NODES: usize = 3; // tree nodes ahead whose safe speed limits the car
pub const SAFE_SPEED_STEP: f64 = 10.0; // px/s, steps a node's speed is lowered in until the car can stop from it
pub const REPLAN_BACK_NODES: usize = 3; // a replan starts this many nodes before the last safe node
pub const REPLAN_ITERATIONS_PER_FRAME: usize = 500; // planning budget per display frame while replanning or improving
pub const MOTION_DT: f64 = 1.0 / 120.0; // s, longest step the car moves in one go
pub const CREEP_SPEED: f64 = 5.0; // px/s, least target speed while the safe-speed limit allows it
pub const JOIN_TOLERANCE_SCALE: f64 = 1.0; // live joins use join_tolerance as is
// A node within NEAR_JOIN_SCALE times the join tolerances of a route is a near miss: NEAR_FOCUS_SHARE of the samples
// then go onto that route NEAR_LOOKAHEAD states ahead of it, so it can line up
pub const NEAR_JOIN_SCALE: f64 = 3.0;
pub const NEAR_FOCUS_SHARE: f64 = 0.5;
pub const NEAR_LOOKAHEAD: usize = 3; // route states
pub const NEAR_SAMPLE_SIGMA: f64 = 5.0; // px
// The normal-SST share of a replan samples in a circle around the car, growing while no join is found
pub const REPLAN_SAMPLE_RADIUS: f64 = 150.0; // px to start
pub const REPLAN_SAMPLE_GROWTH: f64 = 1.5; // radius factor...
pub const REPLAN_SAMPLE_GROW_EVERY: usize = 500; // ...every this many replanner iterations
// Replans focus on nodes that got past the obstacle (within PAST_RADIUS of the route, PAST_BEYOND past the blockage)
pub const PAST_RADIUS: f64 = 30.0; // px
pub const PAST_BEYOND: f64 = 40.0; // px
pub const PAST_FOCUS_SHARE: f64 = 0.8;
pub const PAST_LOOKAHEAD: usize = 5; // route states
pub const PAST_SAMPLE_SIGMA: f64 = 10.0; // px
// Improving the route while driving: plan again from the start of the next very squiggly edge ahead
pub const SQUIGGLY_STEER_FRACTION: f64 = 0.6; // mean |steering| / (most allowed) of a very squiggly edge
pub const IMPROVE_SKIP_FRAMES: f64 = 5.0; // skip a squiggly edge the car reaches within this many frames
pub const IMPROVE_MIN_GAIN: f64 = 0.02; // an improvement must be at least this fraction cheaper
// Before a full replan, nudge the route: re-drive it with a lane-change-shaped steering offset past the obstacle
pub const NUDGE_VARIANTS: usize = 5;
pub const NUDGE_SHIFT_LENGTH: f64 = 120.0; // px of route a nudge uses to shift sideways, ending where the obstacle starts
pub const NUDGE_BACK_LENGTH: f64 = 60.0; // px after the obstacle to shift back onto the route
pub const STOP_FAN_TESTS: usize = 50; // once stopped with no new route: forward tests from rest at different angles...
pub const STOP_FAN_FRACTION: f64 = 0.5; // ...with offsets over +-this fraction of the allowed steering
pub const STALL_REPLAN_TIME: f64 = 2.0; // s stopped while just following a route before replanning from there
// Turnarounds: tried from a replan node near a route but facing almost the opposite way. Phases: (steering as a
// fraction of the most allowed at TURN_SPEED, target speed, total heading change to stop at)
pub const TURN_SPEED: f64 = 25.0; // px/s
pub const TURN_NEAR_RADIUS: f64 = 25.0; // px
pub const TURN_HEADING_TOL: f64 = 35.0 * PI / 180.0;
pub const TURN_MAX_ENTRY_SPEED: f64 = 120.0; // px/s
pub const TURNAROUND_MANEUVERS: [&[(f64, f64, f64)]; 3] = [
    &[(-1.0, TURN_SPEED, PI)], // u-turn left
    &[(1.0, TURN_SPEED, PI)],  // u-turn right
    &[(-1.0, TURN_SPEED, 70.0 * PI / 180.0), (1.0, -TURN_SPEED, 130.0 * PI / 180.0), (-1.0, TURN_SPEED, PI)], // 3-point
];
// Turn primitives, once the car has been stuck at rest for TURN_PRIMITIVE_STUCK_TIME: brake to each speed and turn
// left and right by TURN_PRIMITIVE_ANGLE at the most steering that speed allows
pub const TURN_PRIMITIVE_STUCK_TIME: f64 = 5.0; // s
pub const TURN_PRIMITIVE_SPEEDS: [f64; 3] = [30.0, 60.0, 90.0]; // px/s
pub const TURN_PRIMITIVE_ANGLE: f64 = PI / 2.0;
pub const TURN_PRIMITIVE_MAX_BRAKE: f64 = 0.6; // s; skip a corner speed that takes longer than this to reach
pub const TURN_PRIMITIVE_MAX_FRAMES: usize = 240; // PLAN_DT steps, cap on one turn primitive
// Maneuver library: every sequence of LIB_DEPTH steps of LIB_STEP_FRAMES frames with full brake / coast / throttle
// and full-left / held / full-right steering rate, from each LIB_SPEEDS start, the fastest into each end bin
pub const LIB_SPEEDS: [f64; 16] =
    [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0];
pub const LIB_STEP_FRAMES: usize = 20; // PLAN_DT frames per control step (0.25 s)
pub const LIB_DEPTH: usize = 12; // steps (3 s)
pub const LIB_MIN_SPEED: f64 = 8.0; // px/s; maneuvers never slow below this
pub const LIB_BINS: (f64, f64, f64, f64) = (3.0, 10.0 * PI / 180.0, 20.0, 7.0 * PI / 180.0); // dedup bins
pub const LIB_KEY_CELL: (f64, f64) = (3.0, 10.0 * PI / 180.0); // lookup bins of end position and heading
pub const LIB_CANDIDATES: usize = 20; // best predicted entries re-driven from the real state per try
pub const LIB_TARGET_OFFSETS: [i64; 7] = [-6, -3, 0, 3, 6, 9, 12]; // route states around the chosen one looked up
pub const LIB_TRIES_EVERY: usize = 500; // a replan tries the library at most once per this many iterations
// Stuck at a stop (or about to stop while replanning, NONSTOP): drive the most promising branch of the replan tree,
// picked by k-means grouping of the best GROUP_SAMPLES branches' end states
pub const GROUP_STUCK_TIME: f64 = 5.0; // s
pub const GROUP_SAMPLES: usize = 100;
pub const GROUP_K: usize = 4;
pub const GROUP_MIN_SIZE: usize = 5;
pub const GROUP_HEADING_SCALE: f64 = 40.0; // px per unit of (cos, sin) heading in the end-state features
pub const GROUP_BACK_NODES: usize = 2;
pub const GROUP_MIN_DISTANCE: f64 = 120.0; // px; only branches ending at least this far from the car
pub const GROUP_SPEED_SCALE: f64 = 1.0; // the committed branch is driven at this fraction of its planned speed
pub const GROUP_RETRY_INTERVAL: f64 = 0.25; // s between tries to chain the next branch on while driving one
pub const GROUP_PAST_FIRST: bool = false; // false: always the grouping's pick, branches past the obstacle included
pub const GROUP_G_WEIGHT: f64 = 0.2; // branches score their cost to go plus this times their own cost
pub const NONSTOP: bool = true;
pub const NONSTOP_LEAD: f64 = 0.5; // s
pub const DEAD_REPLAN_ITERATIONS: usize = 20_000; // a replan with fewer than DEAD_REPLAN_NODES active nodes after
pub const DEAD_REPLAN_NODES: usize = 5; // this many iterations is dead
// Reconnecting: Hermite shortcuts from the replan start onto states of any still-valid route
pub const RECONNECT_RADIUS: f64 = 250.0; // px from the start a target may be
pub const RECONNECT_TARGET_SPACING: f64 = 15.0; // px along a route between targets
pub const RECONNECT_MAX_TARGETS: usize = 20; // most promising targets tried
pub const RECONNECT_TANGENTS: [f64; 3] = [0.5, 0.8, 1.1];
pub const RECONNECT_LOOKAHEADS: [f64; 2] = [15.0, 30.0];
pub const RECONNECT_MAX_TURN: f64 = 120.0 * PI / 180.0; // skip targets whose heading differs more than this
pub const RECONNECT_JOIN_WINDOW: (usize, usize) = (10, 30); // route states before and after a target to join at
