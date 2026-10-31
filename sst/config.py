"""Every SST setting: map, car, planner, replanning and driver constants."""
import math
import os
import random
from types import SimpleNamespace

import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # saved files live next to the sst package
SCREEN_WIDTH = 2000
SCREEN_HEIGHT = 1000
SCREEN_SIZE = (2000, 1000)
SCALE = 1
FPS = 60
START = (250, 100)
START_HEADING = 0.0  # rad; 0 = facing right (+x), positive = clockwise on screen
TARGET = (1950, 700)
TARGET_RAD = (70)
TARGET_SPEED = 5
MAX_PLAN_ITERATIONS = 100000  # planGlobal runs this many iterations
RESTART_DELAY = 1.5  # s the window waits after the ghost reaches the goal before starting the drive over
REPLAY_MAX_STOP = 0.5  # s; after each live drive, the drive is replayed at full speed with every stop cut to at most this
# Replanning demo in main(): plan until a path exists, improve it for REPLAN_IMPROVE_ITERATIONS, then block every path
# found so far with a fresh random circle of BLOCK_RADIUS (all the old circles are removed) and replan, on the same tree
REPLAN_DEMO = True
# Path library: the preprocessed (smoothed) paths saved to PATH_LIBRARY_FILE, next to this file.
#   'add': preprocess as usual, add the new paths to the library (making it if there is none) and drive with all of them
#   'use': skip preprocessing and drive with the library's paths (preprocesses and starts one if there is no library yet)
PATH_LIBRARY_MODE = 'use'
# Obstacle log: every time a new blockage is found on the route, the Driver records the episode (the map around the
# blockage, the car, the routes, the decision, and whether the car had to stop) and saves it as an .npz file in a folder
# per run under OBSTACLE_LOG_DIR once it's resolved (see Driver.openEvent / closeEvent)
OBSTACLE_LOG = True
OBSTACLE_LOG_DIR = os.path.join(ROOT_DIR, 'obstacle_log')
OBSTACLE_LOG_PATCH = 150  # px, half the side of the square of map saved around the blocked stretch
PATH_LIBRARY_FILE = os.path.join(ROOT_DIR, 'path_library.npz')
REPLAN_PATHS = 10  # a preprocessing batch stops at this many paths...
PREPROCESS_MAX_ITERATIONS = 750000  # ...or after this many planner iterations in total, whichever comes first
PREPROCESS_BATCHES = 1  # main() runs this many independent batches from scratch and keeps every batch's paths
REPLAN_IMPROVE_ITERATIONS = 100000  # iterations each path gets to improve once found, before the circles are redrawn
REPLAN_MAX_ITERATIONS = 100000  # a round without a path after this many iterations redraws the circles and goes on
BLOCK_RADIUS = 60  # px; on an 80 px road this leaves no gap the car fits through, wherever the node sits
# Driving the planned routes (Driver)
SAFE_SPEED_NODES = 3  # tree nodes ahead whose safe speed limits the car
REPLAN_BACK_NODES = 3  # a replan starts this many nodes before the last safe node, or 1 before if the car is past that
SAFE_SPEED_STEP = 10  # px/s, steps a node's speed is lowered in until the car can stop from it
REPLAN_ITERATIONS_PER_FRAME = 500  # planning budget per display frame while replanning or improving
MOTION_DT = 1 / 120  # s, longest step the car moves in one go; a longer frame is split into steps of at most this
CREEP_SPEED = 5  # px/s, least target speed while the safe-speed limit allows it, so zero-speed points don't stall the car
JOIN_SPEED_OVER = 10  # px/s, how much faster than a route a replanned node may be when joining it
JOIN_TOLERANCE_SCALE = 1  # joins use joinTolerance as is (relaxing it let the car jump ~45 degrees at a splice)
# A node within NEAR_JOIN_SCALE times the join tolerances of a route is a near miss: NEAR_FOCUS_SHARE of its planner's
# normal-SST samples then go onto that route NEAR_LOOKAHEAD states ahead of it (noise NEAR_SAMPLE_SIGMA px, the route's
# heading, speed and steering), so it can line up properly; it still only joins within the plain tolerances
NEAR_JOIN_SCALE = 3
NEAR_FOCUS_SHARE = 0.5
NEAR_LOOKAHEAD = 3  # route states
NEAR_SAMPLE_SIGMA = 5  # px
# The normal-SST share of a replan samples in a circle around the car, growing while no join is found
REPLAN_SAMPLE_RADIUS = 150  # px to start
REPLAN_SAMPLE_GROWTH = 1.5  # radius factor...
REPLAN_SAMPLE_GROW_EVERY = 500  # ...every this many replanner iterations
# Replans focus on nodes that got past the obstacle: a node within PAST_RADIUS of the current route at least PAST_BEYOND
# along it past the blocked stretch has made it through; PAST_FOCUS_SHARE of the normal-SST samples then go near the route PAST_LOOKAHEAD states
# ahead of the best such node (position noise PAST_SAMPLE_SIGMA, roughly the route's heading, speed and steering there)
PAST_RADIUS = 30  # px
PAST_BEYOND = 40  # px of route past the end of the blocked stretch before a nearby node counts as past the obstacle
PAST_FOCUS_SHARE = 0.8
PAST_LOOKAHEAD = 5  # route states
PAST_SAMPLE_SIGMA = 10  # px
# Improving the route while driving: plan again from the start of the next very squiggly edge ahead
SQUIGGLY_STEER_FRACTION = 0.6  # an edge is very squiggly if its mean |steering| / (most allowed at that speed) is at least this
IMPROVE_SKIP_FRAMES = 5  # skip a squiggly edge the car reaches within this many display frames
IMPROVE_MIN_GAIN = 0.02  # an improvement must make the cost to the goal at least this fraction cheaper
LOOP_SPAN = 150  # px; a stretch from a node to any node up to this far along the route counts as loopy if...
LOOP_DETOUR_RATIO = 1.4  # ...the distance driven along it is at least this many times the straight line across it,
LOOP_TURN = math.radians(270)  # ...or its heading turns through at least this much in total
# Smoothing the preprocessed paths (smoothPath, once at the end of preprocessing): from every node, drive Hermite-curve
# shortcuts (pure pursuit, every PLAN_DT step through propagate) to route states up to SMOOTH_NODES_AHEAD nodes ahead and
# splice in the cheapest one that rejoins the path within SMOOTH_TOLERANCE_SCALE of the join tolerance
SMOOTH_TOLERANCE_SCALE = 2 / 3  # stricter than the live joins (1): a smoothed splice is never a bigger jump than a live one
SMOOTH_NODES_AHEAD = 2
SMOOTH_LOOP_SPAN = 250  # px; from the start of a loopy stretch (routeLoopy), targets reach this far along the path instead
SMOOTH_TARGET_SPACING = 15  # px between target states tried along the path
SMOOTH_JOIN_SLACK = 30  # route states past the last target a shortcut may still join at
SMOOTH_TANGENTS = (0.3, 0.5, 0.75, 1.0, 1.3)  # Hermite end-tangent lengths, times the straight distance
SMOOTH_LOOKAHEADS = (10, 20, 35)  # px, pure pursuit lookahead
SMOOTH_BUMPS = (0.0, 0.15)  # extra speed mid-shortcut, times the faster end's speed
SMOOTH_STEER_BLEND = 0.3  # over this last fraction of the curve the steering blends toward the target state's
SMOOTH_MIN_SPEED = 20  # px/s, least reference speed mid-shortcut (so one from rest gets going)
SMOOTH_MIN_GAIN = 0.002  # a shortcut must make the cost to the goal at least this fraction cheaper
SMOOTH_RETRIES = 4  # times the same node is tried again after a splice
SMOOTH_SWEEPS = 3  # sweeps over the whole path, stopping early once one changes nothing
# Before a full replan, nudge the current route: re-drive it from the replan start with a lane-change-shaped steering
# offset (NUDGE_VARIANTS sizes each way, up to the most steering allowed) until just past the obstacle
NUDGE_VARIANTS = 5
STOP_FAN_TESTS = 50  # once stopped with no new route: forward tests from rest at slightly different angles...
STOP_FAN_FRACTION = 0.5  # ...with lane-change offsets spread over +-this fraction of the allowed steering; if none work,
                         # the car backs up to the previous node and the replan restarts from there
# Turnaround library: when a replan or improvement node is near a route (TURN_NEAR_RADIUS) but facing almost the opposite
# way (within TURN_HEADING_TOL of opposite) and no faster than TURN_MAX_ENTRY_SPEED, each manoeuvre is tried from it; one
# whose end joins the route becomes a join. Phases: (steering as a fraction of the most allowed at TURN_SPEED, target
# speed, total heading change to stop at). Each phase first stops and sets the wheel in place, then drives; at the end
# the car stops and straightens the wheel in place. (Unwinding the wheel while moving takes ~1.7 s at the steering rate,
# during which a car this short keeps circling.) Steering + is right.
TURN_SPEED = 25  # px/s
TURN_NEAR_RADIUS = 25  # px
TURN_HEADING_TOL = math.radians(35)
TURN_MAX_ENTRY_SPEED = 120  # px/s
TURNAROUND_MANEUVERS = {
    'u_turn_left': ((-1, TURN_SPEED, math.pi),),
    'u_turn_right': ((1, TURN_SPEED, math.pi),),
    'three_point_left': ((-1, TURN_SPEED, math.radians(70)), (1, -TURN_SPEED, math.radians(130)),
                         (-1, TURN_SPEED, math.pi)),
}
NUDGE_SHIFT_LENGTH = 120  # px of route a nudge uses to shift sideways, ending where the obstacle starts...
NUDGE_BACK_LENGTH = 60  # ...and px after the obstacle to shift back onto the route
NUDGE_NODE_FRAMES = 8  # a nudged stretch gets a tree node (for the safe-speed look-ahead) every this many PLAN_DT steps
# The original geometric RRT runs alongside a replan once the car has been stopped for RRT_STOP_DELAY s: each frame it
# gets as much wall-clock time as the SST replan's iterations took. Once the car has stopped and the RRT has a path, RRT_FOCUS_SHARE of the replan's normal-SST samples
# follow that path as a skeleton: RRT_LOOKAHEAD px beyond the furthest point along it that a replan node has reached
# (within RRT_FOCUS_RADIUS of it), so the search starts near the car and moves out as it progresses
RRT_ENABLED = False  # temporarily off: the RRT (and its skeleton focus) didn't seem to help much
RRT_STOP_DELAY = 3  # s the car must have been stopped (waiting on a replan) before the RRT does anything
RRT_RANDOM = random.Random(0)  # the RRT's own random numbers: its time budget is wall-clock, so drawing from the shared
                               # generator made runs unrepeatable
STALL_REPLAN_TIME = 2  # s stopped and unable to move while just following a route before replanning from where it is
RRT_FOCUS_SHARE = 0.6
RRT_FOCUS_RADIUS = 30  # px
RRT_LOOKAHEAD = 40  # px
RRT_SAMPLE_SIGMA = 10  # px
SKELETON_CORNER_RADIUS = 25  # px; skeleton samples slow to what this turn allows at each bend of the RRT path
GLOBAL_PLAN_SEES_LOCAL = False  # temporary: main() plans on a map that already contains every hidden local obstacle
# SST* annealing in planGlobal: once a path exists, every ANNEAL_WINDOW iterations the best path cost is compared with
# the window's start; if it improved by less than ANNEAL_MIN_GAIN, delta_B_N and delta_S shrink by ANNEAL_FACTOR.
ANNEAL_WINDOW = 10000
ANNEAL_MIN_GAIN = 0.10
ANNEAL_FACTOR = 0.8
# Car geometry (pixels)
CAR_LENGTH = 10
CAR_WIDTH = 5
WHEELBASE = 6  # distance between front and rear axles
RS_WHEELBASE = 8  # (unused while stateDistance stands in for Reeds-Shepp) used only by the Reeds-Shepp distance, for a more car-like low-speed turning radius
RS_TURN_SCALE = 0.5  # Reeds-Shepp turns at speed use this fraction of the lateral-limit radius v^2 / MAX_LATERAL_ACCELERATION
WHEEL_LENGTH = 2.5
WHEEL_WIDTH = 1
# Collision footprint: FOOTPRINT_CIRCLES circles in a row along the car's length that together cover its
# CAR_LENGTH x CAR_WIDTH body. Each covers a CAR_LENGTH / FOOTPRINT_CIRCLES slice of it (+1 px for grid rounding).
FOOTPRINT_CIRCLES = 3
FOOTPRINT_RADIUS = math.hypot(CAR_LENGTH / FOOTPRINT_CIRCLES / 2, CAR_WIDTH / 2) + 1
# Circle centers, in px ahead of the rear axle (the body is centered WHEELBASE / 2 ahead of it)
FOOTPRINT_OFFSETS = tuple(WHEELBASE / 2 + (i + 0.5 - FOOTPRINT_CIRCLES / 2) * CAR_LENGTH / FOOTPRINT_CIRCLES
                          for i in range(FOOTPRINT_CIRCLES))
FOOTPRINT_REACH = max(abs(o) for o in FOOTPRINT_OFFSETS)  # farthest circle center from the rear axle
FOOTPRINT_OFFSETS_ARRAY = np.array(FOOTPRINT_OFFSETS, dtype=np.float64)  # (for the compiled clearance checks)
SAFETY_MARGIN = 2  # px of clearance every checked state must have; SDF_CAP must stay >= FOOTPRINT_RADIUS + SAFETY_MARGIN

DISTANCE_TIME_PROP = 1
# Selection and witness radii, in stateDistance units (roughly px). With the Reeds-Shepp distance they were seconds:
# delta_B_N = 1  # s of Reeds-Shepp time; plan: anneal down to 1/20
DELTA_B_N_START = 60
DELTA_B_N_MIN = DELTA_B_N_START / 20  # planGlobal doesn't anneal below this
# planGlobal anneals these; they live on a shared object so every module sees the current values
radii = SimpleNamespace(delta_B_N=DELTA_B_N_START, delta_S=DELTA_B_N_START / 4)  # delta_S = 15 to start
# stateDistance weights: euclidean px + W_A * |heading diff| (rad) + W_V * |speed diff| (px/s) + W_ST * |steering diff| (rad)
W_A = 1 / 4 * 360 / (2 * math.pi)  # 1/4 per degree
W_V = 1 / 5
W_ST = 1 / 10
TIME_PROP = 0.5  # s, longest edge
# Edge cost per second: 1 (time) + COST_ACCEL * |acceleration| + COST_DECEL * braking + COST_STEER_RATE * |steering rate|
#   + COST_STEER_ANGLE * |steering angle|
COST_ACCEL = 0  # per px/s^2; off for now (was 0.005)
COST_STEER_RATE = 0  # per rad/s; off for now (was 1.5)
COST_STEER_ANGLE = 10.8  # per rad: pushes toward driving with the wheel straight
COST_DECEL = 0.01  # per px/s^2 of braking (negative acceleration only)
PLAN_DT = 1 / 80  # s, planner step (a quarter of a 1/20 s frame); edges last a whole number of these
LIDAR_DT = 1 / 20  # s between lidar scans in main()
# Car dynamics
MAX_ACCELERATION = 120  # px/s^2, limit on set_acceleration()
MAX_SPEED = 300  # px/s
MAX_REVERSE_SPEED = 100
MAX_STEER_RATE = math.radians(20)  # rad/s, limit on set_steering_rate()
MAX_STEER = math.radians(35)  # rad, limit on front wheel steering angle
MAX_LATERAL_ACCELERATION = 200
STEER_TARGET_SIGMA = 0.5  # spread of the random target steering angle, as a fraction of the most allowed at the edge's top speed
EARLY_STEER_ITERATIONS = 100  # planGlobal's first iterations sample steering with a narrower spread...
EARLY_STEER_VARIANCE_SCALE = 0.25  # ...this fraction of the normal variance (spread times sqrt of this)
EARLY_ACCEL_MEAN = MAX_ACCELERATION / 2  # ...and acceleration from a normal around this (biased toward speeding up)
# Informed expansion in planGlobal: most iterations expand the active node with the lowest g + h (g = cost so far,
# h = grid Dijkstra distance to the goal / MAX_SPEED) instead of SST's random-sample selection
GREEDY_SHARE = 0.75  # fraction of iterations that expand by g + h; the rest are normal SST iterations
GREEDY_CHAIN_MAX = 15  # a child closer to the goal (lower h) than its parent is expanded next, at most this many in a row
EXPANSION_PENALTY = 0.5  # added to a node's g + h each time it's expanded, so a stuck node stops being picked
TIE_BREAK = 0.05  # random 0..this added to g + h
BLOSSOM_RANDOM = 5  # a node's first expansion tries the PRIMITIVES plus this many random actions, keeping the best g + h
# Sensible actions tried on a node's first expansion: (acceleration as a fraction of MAX_ACCELERATION, target steering
# as a fraction of the most allowed at the edge's top speed, duration in s)
PRIMITIVES = (
    (1.0, 0.0, 0.3),  # speed up, wheel straight
    (0.0, 0.0, 0.3),  # hold speed, wheel straight
    (0.5, 0.3, 0.3),  # speed up gently, gentle right
    (0.5, -0.3, 0.3),  # speed up gently, gentle left
    (-0.5, 0.0, 0.3),  # brake gently, wheel straight
)
H_CELL = 10  # px, cell size of the grid h is computed on
GRID_INFLATION = FOOTPRINT_RADIUS + SAFETY_MARGIN + 2  # px a grid cell's center must clear known obstacles by to count as
                                                      # free in the Dijkstra grids (the car's clearance, the margin, 2 more)
# Turn primitives (no map information), only once the car has been stuck (at rest waiting on a replan) for
# TURN_PRIMITIVE_STUCK_TIME: each node the replanner expands then also gets, once, every collision-free turn as an extra
# child: brake to each TURN_PRIMITIVE_SPEEDS speed and turn left and right by TURN_PRIMITIVE_ANGLE at the most steering
# that speed allows, coming out lined up. A turn only fits where there's room for it (a side street, a wide spot)
TURN_PRIMITIVE_STUCK_TIME = 5  # s
TURN_PRIMITIVE_SPEEDS = (30, 60, 90)  # px/s; turn radii of about 4 (the steering limit), 18 and 40 px (the lateral limit)
TURN_PRIMITIVE_ANGLE = math.radians(90)
TURN_PRIMITIVE_MAX_BRAKE = 0.6  # s; skip a corner speed that takes longer than this to reach
TURN_PRIMITIVE_MAX_FRAMES = 240  # PLAN_DT steps, cap on one turn primitive
# Manoeuvre library (maneuverLibrary, built once and saved to MANEUVER_LIBRARY_FILE): smooth, never-stopping ways for the
# car to get from its state onto a known track inside a known-free circle. Built by exploring every sequence of LIB_DEPTH
# steps of LIB_STEP_FRAMES PLAN_DT frames with full braking / coasting / full throttle and full-left / held / full-right
# steering rate (the car's limits, the lateral one included), from rest and from each LIB_SPEEDS speed with the wheel
# straight, keeping the fastest sequence into each bin of end position, heading, speed and steering, within libRange(v)
MANEUVER_LIBRARY_FILE = os.path.join(ROOT_DIR, 'maneuver_library.npz')
LIB_SPEEDS = tuple(range(0, 160, 10))  # px/s, start speeds the library is built for
LIB_STEP_FRAMES = 20  # PLAN_DT frames per control step (0.25 s)
LIB_DEPTH = 12  # steps (3 s)
LIB_MIN_SPEED = 8  # px/s; manoeuvres never slow below this (they don't stop)
LIB_BINS = (3.0, math.radians(10), 20.0, math.radians(7))  # dedup bins: position px, heading, speed, steering
LIB_KEY_CELL = (3.0, math.radians(10))  # lookup bins of end position and heading
LIB_CANDIDATES = 20  # best predicted entries re-driven from the real state per try
LIB_TARGET_OFFSETS = (-6, -3, 0, 3, 6, 9, 12)  # route states around the chosen one that entries are looked up for
LIB_TRIES_EVERY = 500  # a replan tries the library from a node near a route at most once per this many iterations
# Stuck for GROUP_STUCK_TIME at a stop: take the GROUP_SAMPLES best scoring (most progress: replanH plus a little g)
# distinct branches of the replan tree from where the car is, k-means their end states into GROUP_K groups, pick the
# group with the best median score (at least GROUP_MIN_SIZE branches), drive its best branch, and replan from
# GROUP_BACK_NODES nodes before its end on the way
GROUP_STUCK_TIME = 5  # s
GROUP_SAMPLES = 100
GROUP_K = 4
GROUP_MIN_SIZE = 5
GROUP_HEADING_SCALE = 40  # px per unit of (cos, sin) heading in the end-state features, so headings count beside positions
GROUP_BACK_NODES = 2
GROUP_MIN_DISTANCE = 120  # px; only branches ending at least this far (straight line) from the car are sampled
GROUP_SPEED_SCALE = 1.0  # the committed branch is driven at this fraction of its planned speed
GROUP_RETRY_INTERVAL = 0.25  # s between tries to chain the next branch on while driving a committed one
# Non-stop: while replanning, once the car is within NONSTOP_LEAD s of the replan start without a join, commit to a
# branch instead of braking to the stop point: the best branch past the obstacle if there is one, otherwise the grouping
NONSTOP = True
DEAD_REPLAN_ITERATIONS = 20000  # a replan with fewer than DEAD_REPLAN_NODES active nodes after this many iterations is
DEAD_REPLAN_NODES = 5  # dead (nothing can grow from its start): stopped, the car backs up a node; moving, it stops
GROUP_PAST_FIRST = False  # True: a branch already past the obstacle is taken without grouping; False (testing): always the
                          # k-means grouping's pick, with branches past the obstacle in the pool like any other
NONSTOP_LEAD = 0.5  # s
GROUP_G_WEIGHT = 0.2  # branches are scored by the cost to go from their end (replanH) plus this times their own cost
# Reconnecting: when the route is blocked and a nudge doesn't get past, and again at a stop when the forward tests fail,
# drive smooth shortcuts (shortcutDrive: Hermite curve, pure pursuit, every step through propagate) from the replan start
# to states of any still-valid route (the current one beyond the blockage, or another branch) and splice in the cheapest
# that rejoins within the live join tolerance, before any SST replan
RECONNECT_RADIUS = 250  # px from the start a target may be
RECONNECT_TARGET_SPACING = 15  # px along a route between targets
RECONNECT_MAX_TARGETS = 20  # most promising targets tried (straight-line time there plus the route's cost from there)
RECONNECT_TANGENTS = (0.5, 0.8, 1.1)
RECONNECT_LOOKAHEADS = (15, 30)
RECONNECT_MAX_TURN = math.radians(120)  # skip targets whose heading differs more than this
RECONNECT_JOIN_WINDOW = (10, 30)  # route states before and after a target where the shortcut may join
EARLY_ACCEL_SIGMA = MAX_ACCELERATION / 2 * math.sqrt(0.5)  # ...with this spread (half the variance of MAX_ACCELERATION / 2), instead of uniform
# Most any footprint circle's center moves between the states monteCarloProp returns. Every point between two states
# is within half this of one of them, so while it is <= 2 * SAFETY_MARGIN, a state with SAFETY_MARGIN of clearance
# guarantees the stretch around it is collision free.
CHECK_SPACING = 2 * SAFETY_MARGIN  # px, as large as the guarantee allows
SDF_CAP = 20
STOP_SPEED_BINS = 5  # stopping trajectories: bins over |speed| in [0, MAX_SPEED]
STOP_STEER_BINS = 5  # ... and over steering as a fraction of the most allowed at that speed, in [-1, 1]

# Occupancy grid cell states
FREE, OCCUPIED, UNKNOWN = 0, 1, 2

# Street grid: road centerlines (px); everything between roads is a city block
STREET_X = [250, 620, 980, 1330, 1680, 1950]
STREET_Y = [100, 390, 700]
STREET_WIDTH = 80
BLOCK_CORNER = 15  # rounded block corners, px
# Local obstacles: hidden on the roads until the car senses them
LOCAL_OBSTACLE_COUNT = 60
LOCAL_OBSTACLE_RADIUS = (7, 20)  # px, min/max for circles
LOCAL_BOX_LENGTH = (15, 45)  # px, min/max for boxes
LOCAL_BOX_WIDTH = (8, 20)
BLOCKED_ROADS = 2  # road segments fully blocked by one big obstacle
LOCAL_SEED = None  # set an int for a repeatable layout
LIDAR_BEAMS = 3600  # long range, so fine angular steps: ~1.7 px apart at 1000 px
DIRTY_TILE = 50  # px, SDF updates are grouped on a grid this coarse
DIAGONALS = [((0, 1000), (980, 100)), ((1330, 100), (2000, 760))]  # diagonal road endpoints, cut through the blocks
