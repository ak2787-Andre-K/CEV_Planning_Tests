import bisect
import copy
import time as clock_time
import heapq
import itertools
import math
import os
import random
import numpy as np
from numba import njit
from dataclasses import dataclass
from enum import Enum

import pygame
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
OBSTACLE_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'obstacle_log')
OBSTACLE_LOG_PATCH = 150  # px, half the side of the square of map saved around the blocked stretch
PATH_LIBRARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'path_library.npz')
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
delta_B_N = DELTA_B_N_START  # planGlobal anneals these
delta_S = delta_B_N / 4  # = 15 to start
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
MANEUVER_LIBRARY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maneuver_library.npz')
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

# Obstacles: [x, y] centers and radii
GLOBAL_OBSTACLE_POS = [[574, 109], [1592, 867], [1425, 537], [1579, 85], [438, 146], [1885, 883], [894, 870], [1414, 70], [569, 816], [433, 522], [364, 841], [1515, 785], [1123, 921], [968, 218], [1412, 199], [1614, 521], [1397, 843], [88, 873], [1031, 834], [1128, 503], [1876, 520], [1100, 219], [892, 113], [1073, 81], [608, 518], [70, 91], [1925, 96], [122, 525], [460, 866], [971, 452], [884, 547], [1555, 213], [634, 937], [1512, 450], [361, 60], [129, 202], [1019, 946], [129, 433], [1060, 567], [1343, 454], [1647, 211], [1494, 894], [495, 45], [343, 195], [1665, 786], [449, 779], [1947, 430], [951, 780], [1901, 215], [846, 216], [970, 568], [375, 946], [353, 442], [119, 772], [1412, 436], [645, 39], [536, 235], [1083, 422], [1138, 787], [867, 431], [151, 956], [159, 824], [41, 473], [53, 247], [654, 855], [1854, 434], [1931, 761], [498, 944], [1861, 759], [1605, 783], [1486, 958], [1520, 569], [1158, 159], [663, 425], [628, 222], [1366, 958], [1006, 162], [439, 241], [1854, 38], [1650, 961], [333, 517], [866, 962], [1557, 934], [1850, 210], [382, 249], [660, 173], [42, 943], [439, 46], [1972, 233], [604, 413], [578, 894], [1859, 247], [647, 794], [1662, 907], [839, 799], [1116, 582], [1014, 747], [550, 932], [511, 414], [576, 24], [56, 520], [526, 460], [1340, 109], [351, 577], [172, 77], [1173, 108], [1958, 564], [1498, 165], [670, 94], [1959, 811], [1427, 934], [58, 798], [1975, 151], [332, 751], [965, 909], [149, 36], [868, 751], [77, 580], [1834, 132], [1021, 527], [1486, 64], [1616, 940], [457, 428], [1325, 68], [594, 747], [383, 770], [1361, 752], [1953, 956], [1969, 501], [1668, 25], [176, 481], [1616, 424], [1329, 910], [59, 410], [46, 763], [832, 459], [455, 971], [1030, 239], [47, 184], [661, 134], [329, 125], [1671, 169], [1134, 30], [904, 244], [1488, 123], [1162, 428], [976, 53], [1350, 28], [955, 945], [1860, 593], [1837, 964], [167, 775], [409, 417], [1085, 762], [498, 581], [1339, 157], [1964, 23], [1338, 573], [1338, 255], [994, 120], [154, 117]]

GLOBAL_OBSTACLE_RAD = [64, 45, 63, 73, 57, 71, 61, 46, 54, 62, 46, 47, 56, 45, 63, 59, 74, 56, 60, 56, 46, 46, 69, 63, 70, 52, 53, 48, 49, 47, 52, 50, 49, 48, 50, 54, 41, 34, 27, 32, 29, 28, 29, 27, 25, 34, 28, 28, 31, 35, 27, 35, 40, 29, 26, 26, 33, 25, 36, 27, 27, 25, 30, 26, 31, 31, 32, 30, 26, 28, 28, 32, 31, 27, 40, 29, 16, 15, 17, 15, 16, 22, 23, 15, 17, 12, 14, 20, 15, 17, 13, 14, 19, 18, 14, 18, 14, 13, 15, 14, 13, 20, 25, 13, 17, 17, 24, 13, 12, 14, 13, 19, 12, 21, 14, 16, 21, 17, 21, 22, 17, 15, 23, 13, 16, 21, 15, 21, 16, 15, 12, 19, 17, 15, 13, 14, 17, 12, 14, 14, 17, 14, 13, 17, 13, 21, 12, 12, 19, 14, 14, 12, 18, 16, 18, 16, 12, 13, 17, 16, 19]

LOCAL_OBSTACLE_POS = [[784, 116], [889, 692], [1258, 643], [277, 763], [697, 676], [941, 287], [1200, 663], [1649, 714], [1715, 160], [1780, 697], [1781, 628], [1800, 76], [786, 772], [804, 47], [1002, 614], [611, 290], [1695, 939], [1541, 307], [1194, 881], [1214, 192], [1247, 378], [106, 623], [1194, 584], [1178, 711], [180, 310], [787, 475], [1291, 41], [239, 627], [1886, 380], [343, 647], [1769, 740], [1719, 275], [219, 174], [628, 718], [1420, 304], [725, 403], [1269, 295], [1288, 129], [283, 917], [1206, 292], [856, 286], [487, 283], [1826, 279], [1195, 39], [1717, 674], [1773, 836], [1702, 569], [806, 824], [1709, 828], [1555, 366], [288, 468], [1779, 379], [1451, 624], [1973, 722], [107, 698], [1227, 725], [198, 385], [298, 574], [1016, 387], [214, 563], [1560, 715], [1296, 512], [674, 620], [630, 360], [47, 631], [1659, 280], [305, 359], [727, 66], [216, 494], [1077, 621], [689, 359], [1905, 698], [1793, 952], [1591, 630], [124, 286], [798, 632], [821, 716], [1187, 361], [737, 279], [1778, 539], [228, 292], [954, 710], [1893, 302], [1211, 516], [1707, 79], [1705, 421], [722, 825], [1695, 616], [29, 356], [1950, 633], [288, 146], [1965, 296], [559, 371], [1028, 710], [1769, 223], [230, 706], [426, 723], [296, 708], [210, 938], [701, 550], [1723, 761], [902, 625], [429, 296], [730, 329], [1826, 719], [572, 648], [1311, 374], [273, 257], [1392, 628], [809, 406], [915, 375], [1274, 183], [768, 963], [1298, 871], [1960, 367], [192, 665], [210, 765], [698, 224], [270, 856], [717, 908], [1800, 180], [1808, 464], [297, 204], [1223, 448], [418, 368], [1317, 714], [1479, 282], [776, 880], [1408, 360], [696, 469], [1731, 349], [1292, 821], [694, 136], [447, 614], [675, 303], [1413, 696], [363, 377], [952, 610], [1803, 342], [1279, 597], [793, 351], [1719, 218], [855, 632], [1704, 520], [1846, 621], [1497, 378], [1019, 273], [208, 55], [1217, 974], [1089, 688], [25, 282], [1138, 306], [1111, 726], [352, 290], [726, 642], [167, 619], [1528, 634], [795, 547], [1132, 643], [1285, 459], [1214, 88], [1488, 708], [195, 113], [629, 611], [1327, 636], [1291, 934], [549, 710], [173, 717], [206, 863], [199, 235], [1068, 310], [491, 363], [729, 179], [1902, 638], [1833, 369], [753, 703], [1623, 372], [236, 978], [1362, 317], [1588, 282], [728, 748], [1723, 891], [296, 972], [106, 386], [1214, 810], [273, 409], [1767, 491], [1281, 751], [48, 714], [486, 719], [1098, 372], [283, 86], [691, 973], [211, 436], [505, 624], [1697, 724], [785, 173], [1032, 648], [1786, 786], [1780, 298], [805, 935], [782, 302], [358, 710], [792, 255], [297, 26], [546, 313], [1638, 608], [68, 304], [863, 376], [158, 367], [730, 949], [1726, 32], [1299, 243], [390, 628], [1232, 22], [1273, 695]]

LOCAL_OBSTACLE_RAD = [17, 12, 10, 17, 10, 17, 9, 13, 17, 10, 16, 10, 11, 16, 8, 9, 15, 19, 16, 13, 18, 15, 13, 9, 7, 13, 10, 18, 12, 11, 8, 16, 19, 9, 16, 17, 11, 19, 9, 18, 18, 15, 9, 9, 8, 12, 8, 18, 19, 9, 15, 12, 9, 11, 19, 16, 8, 14, 16, 12, 16, 14, 8, 12, 16, 10, 20, 13, 10, 16, 14, 13, 17, 12, 19, 12, 16, 9, 12, 11, 10, 14, 9, 8, 11, 16, 13, 7, 10, 10, 17, 9, 19, 14, 9, 17, 19, 17, 13, 19, 11, 10, 13, 10, 15, 13, 19, 11, 16, 20, 17, 12, 9, 8, 14, 11, 8, 17, 13, 12, 18, 16, 13, 12, 14, 15, 14, 9, 11, 18, 10, 10, 11, 13, 14, 15, 11, 8, 8, 16, 10, 12, 8, 16, 10, 19, 15, 16, 12, 7, 7, 18, 10, 14, 8, 15, 15, 9, 15, 16, 15, 12, 12, 8, 12, 13, 16, 14, 10, 7, 8, 10, 11, 13, 9, 9, 16, 8, 9, 10, 9, 12, 18, 10, 14, 13, 8, 11, 14, 10, 20, 10, 12, 8, 19, 7, 13, 12, 9, 14, 13, 10, 10, 10, 15, 11, 9, 14, 10, 11, 7, 12, 11, 12, 7, 8]


@dataclass(frozen=True)
class CarState:
    x: float  # px, rear axle center
    y: float  # px
    velocity: float  # px/s, negative = reversing
    heading: float  # rad, 0 = +x, positive = clockwise on screen


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float

    @property
    def reach(self):
        return self.radius

    def bounds(self, margin):
        reach = self.radius + margin
        return self.x - reach, self.x + reach, self.y - reach, self.y + reach

    def signed_distance(self, xs, ys):
        return np.hypot(xs - self.x, ys - self.y) - self.radius


def rounded_rect_distance(u, v, half_w, half_h, corner):
    """Signed distance to a rectangle centered at the origin, for points (u, v) in the rectangle's own frame."""
    r = min(corner, half_w, half_h)
    qx = np.abs(u) - (half_w - r)
    qy = np.abs(v) - (half_h - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - r


@dataclass(frozen=True)
class Box:
    """Rectangle centered at (x, y), `length` along `angle` (rad) and `width` across it, with rounded corners."""
    x: float
    y: float
    length: float
    width: float
    angle: float = 0.0
    corner: float = 0.0

    @property
    def reach(self):
        return math.hypot(self.length, self.width) / 2

    def bounds(self, margin):
        reach = self.reach + margin
        return self.x - reach, self.x + reach, self.y - reach, self.y + reach

    def signed_distance(self, xs, ys):
        c, s = math.cos(self.angle), math.sin(self.angle)
        dx, dy = xs - self.x, ys - self.y
        return rounded_rect_distance(dx * c + dy * s, -dx * s + dy * c, self.length / 2, self.width / 2, self.corner)


@dataclass(frozen=True)
class Block:
    """Axis-aligned rectangle from (x_lo, y_lo) to (x_hi, y_hi) with rounded corners."""
    x_lo: float
    y_lo: float
    x_hi: float
    y_hi: float
    corner: float = BLOCK_CORNER

    @property
    def x(self):
        return (self.x_lo + self.x_hi) / 2

    @property
    def y(self):
        return (self.y_lo + self.y_hi) / 2

    def bounds(self, margin):
        return self.x_lo - margin, self.x_hi + margin, self.y_lo - margin, self.y_hi + margin

    def signed_distance(self, xs, ys):
        half_w, half_h = (self.x_hi - self.x_lo) / 2, (self.y_hi - self.y_lo) / 2
        return rounded_rect_distance(xs - self.x, ys - self.y, half_w, half_h, self.corner)


@dataclass(frozen=True)
class Road:
    """Straight road of `width` from (x0, y0) to (x1, y1), carved out of whatever it crosses."""
    x0: float
    y0: float
    x1: float
    y1: float
    width: float = STREET_WIDTH

    def bounds(self, margin):
        reach = self.width / 2 + margin
        return (min(self.x0, self.x1) - reach, max(self.x0, self.x1) + reach,
                min(self.y0, self.y1) - reach, max(self.y0, self.y1) + reach)

    def signed_distance(self, xs, ys):
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        t = np.clip(((xs - self.x0) * dx + (ys - self.y0) * dy) / (dx * dx + dy * dy), 0, 1)
        return np.hypot(xs - (self.x0 + t * dx), ys - (self.y0 + t * dy)) - self.width / 2


def diagonal_roads():
    return [Road(x0, y0, x1, y1) for (x0, y0), (x1, y1) in DIAGONALS]


def street_blocks():
    """City blocks filling the space between the STREET_X / STREET_Y roads. Outer blocks extend
    past the screen edge so their rounded corners don't leave gaps along the border."""
    half = STREET_WIDTH / 2
    xs = [-BLOCK_CORNER] + [e for c in STREET_X for e in (c - half, c + half)] + [SCREEN_WIDTH + BLOCK_CORNER]
    ys = [-BLOCK_CORNER] + [e for c in STREET_Y for e in (c - half, c + half)] + [SCREEN_HEIGHT + BLOCK_CORNER]
    return [
        Block(x_lo, y_lo, x_hi, y_hi)
        for x_lo, x_hi in zip(xs[::2], xs[1::2])
        for y_lo, y_hi in zip(ys[::2], ys[1::2])
        if x_hi > x_lo and y_hi > y_lo
    ]


OBSTACLES = [Obstacle(pos[0], pos[1], rad) for pos, rad in zip(GLOBAL_OBSTACLE_POS + LOCAL_OBSTACLE_POS, GLOBAL_OBSTACLE_RAD + LOCAL_OBSTACLE_RAD)]
GLOBAL_OBSTACLES = [Obstacle(pos[0], pos[1], rad) for pos, rad in zip(GLOBAL_OBSTACLE_POS, GLOBAL_OBSTACLE_RAD)]


class Car:
    """Kinematic Ackermann model driven by front wheel steering angle. The pose is defined at the rear axle center.

    Control it by setting rates; update() integrates them each frame:
        car.set_acceleration(px_per_s2)    # clamped to +/-MAX_ACCELERATION
        car.set_steering_rate(rad_per_s)  # d(delta)/dt, clamped to +/-MAX_STEER_RATE
        car.get_state() -> CarState
    """

    def __init__(self, x, y, heading=0.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = 0.0
        self.steering = 0.0  # rad, front wheel steering angle delta
        self.acceleration = 0.0  # px/s^2, current speed rate of change
        self.steering_rate = 0.0  # rad/s, current steering angle rate of change

    def get_state(self):
        return CarState(self.x, self.y, self.velocity, self.heading)

    def set_acceleration(self, acceleration):
        """Rate of change of speed in px/s^2. Returns the clamped value actually applied."""
        self.acceleration = clamp(acceleration, -MAX_ACCELERATION, MAX_ACCELERATION)
        return self.acceleration

    def set_steering_rate(self, steering_rate):
        """Rate of change of steering angle in rad/s (positive = right). Returns the clamped value actually applied."""
        self.steering_rate = clamp(steering_rate, -MAX_STEER_RATE, MAX_STEER_RATE)
        return self.steering_rate

    def update(self, dt):
        self.velocity = clamp(self.velocity + self.acceleration * dt, -MAX_SPEED, MAX_SPEED)
        self.steering = clamp(self.steering + self.steering_rate * dt, -MAX_STEER, MAX_STEER)

        self.x += self.velocity * math.cos(self.heading) * dt
        self.y += self.velocity * math.sin(self.heading) * dt
        self.heading += self.velocity * math.tan(self.steering) / WHEELBASE * dt
        self.heading %= 2 * math.pi

    def draw(self, surface, color="red"):
        cos_h, sin_h = math.cos(self.heading), math.sin(self.heading)

        def to_world(forward, left):
            # Body frame (forward along heading, left = toward -y side) -> screen coords
            return (
                self.x + forward * cos_h + left * sin_h,
                self.y + forward * sin_h - left * cos_h,
            )

        # Body: centered between the axles
        overhang = (CAR_LENGTH - WHEELBASE) / 2
        body_center = to_world(WHEELBASE / 2, 0)
        pygame.draw.polygon(surface, color, rotated_rect(body_center, CAR_LENGTH, CAR_WIDTH, self.heading))

        # Wheels: rear follow the body, front use Ackermann geometry so each wheel
        # points around the same turning center (inner wheel turns more than outer)
        half_track = CAR_WIDTH / 2
        tan_steer = math.tan(self.steering)
        for left in (half_track, -half_track):
            wheel_steer = math.atan(WHEELBASE * tan_steer / (WHEELBASE + left * tan_steer))
            for forward, wheel_angle in ((0, self.heading), (WHEELBASE, self.heading + wheel_steer)):
                wheel = rotated_rect(to_world(forward, left), WHEEL_LENGTH, WHEEL_WIDTH, wheel_angle)
                pygame.draw.polygon(surface, "white", wheel)

        # Nose marker so heading is obvious
        pygame.draw.circle(surface, "yellow", to_world(WHEELBASE + overhang, 0), 2)


def clamp(value, low, high):
    return max(low, min(high, value))


def rotated_rect(center, length, width, angle):
    """Corners of a length x width rectangle centered at `center`, rotated by `angle`."""
    cx, cy = center
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    corners = []
    for dx, dy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        fx, fy = dx * length / 2, dy * width / 2
        corners.append((cx + fx * cos_a - fy * sin_a, cy + fx * sin_a + fy * cos_a))
    return corners


def load_font():
    # pygame 2.6.1's font module is broken on Python 3.14; fall back to the window title
    try:
        return pygame.font.SysFont(None, 28)
    except (NotImplementedError, ImportError):
        return None


def draw_hud(surface, font, car):
    text = f"x: {car.x:7.1f}   y: {car.y:7.1f}   heading: {math.degrees(car.heading):6.1f} deg   speed: {car.velocity:6.1f} px/s   steering: {math.degrees(car.steering):6.1f} deg"
    if font:
        surface.blit(font.render(text, True, "white"), (10, 10))
    else:
        pygame.display.set_caption(text)


def rasterize(grid, shapes, value):
    """Set every cell of `grid` whose center lies inside one of `shapes` to `value`."""
    for o in shapes:
        bx_lo, bx_hi, by_lo, by_hi = o.bounds(0)
        x_lo, x_hi = max(0, math.floor(bx_lo)), min(grid.shape[0], math.ceil(bx_hi) + 1)
        y_lo, y_hi = max(0, math.floor(by_lo)), min(grid.shape[1], math.ceil(by_hi) + 1)
        if x_lo >= x_hi or y_lo >= y_hi:
            continue
        inside = o.signed_distance(np.arange(x_lo, x_hi)[:, None], np.arange(y_lo, y_hi)[None, :]) <= 0
        grid[x_lo:x_hi, y_lo:y_hi][inside] = value
    return grid


def capped_distance(mask, cap):
    """Euclidean distance from every cell to the nearest True cell of `mask` (0 on those cells), capped at `cap`.

    Exact separable transform. Pass 1 finds the nearest True cell along x for every cell with running
    max/min accumulations (O(cells)); pass 2 combines those along y over offsets up to `cap` (O(cells * cap)),
    in int16 since squared distances stay below (cap + 1)^2 + cap^2.
    """
    reach = math.ceil(cap)
    n = mask.shape[0]
    idx = np.arange(n, dtype=np.int32)[:, None]
    before = np.where(mask, idx, np.int32(-n - reach - 1))
    np.maximum.accumulate(before, axis=0, out=before)
    after = np.minimum.accumulate(np.where(mask, idx, np.int32(2 * n + reach + 1))[::-1], axis=0)[::-1]
    along_x = np.minimum(idx - before, after - idx)
    np.minimum(along_x, reach + 1, out=along_x)
    along_sq = (along_x * along_x).astype(np.int16)
    best = along_sq.copy()
    shifted = np.empty_like(along_sq)
    for d in range(1, min(reach, mask.shape[1] - 1) + 1):
        np.add(along_sq[:, :-d], d * d, out=shifted[:, :-d])
        np.minimum(best[:, d:], shifted[:, :-d], out=best[:, d:])
        np.add(along_sq[:, d:], d * d, out=shifted[:, d:])
        np.minimum(best[:, :-d], shifted[:, d:], out=best[:, :-d])
    return np.minimum(np.sqrt(best, dtype=np.float32), np.float32(cap))


def changedRects(changed, x0, y0, tile=DIRTY_TILE):
    """Bounding boxes (x_lo, x_hi, y_lo, y_hi) in map coords of the clusters of True cells in `changed`, whose
    [0, 0] cell is at (x0, y0). Clusters are found on a coarse tile grid, so changes scattered across a long
    range scan become several small windows instead of one map-sized one."""
    w, h = changed.shape
    tw, th = -(-w // tile), -(-h // tile)
    padded = np.zeros((tw * tile, th * tile), dtype=bool)
    padded[:w, :h] = changed
    tiles = padded.reshape(tw, tile, th, tile).any(axis=(1, 3))
    seen = np.zeros_like(tiles)
    rects = []
    for i, j in zip(*np.nonzero(tiles)):  # flood fill 8-connected tiles; the tile grid is small
        if seen[i, j]:
            continue
        seen[i, j] = True
        stack, i_lo, i_hi, j_lo, j_hi = [(i, j)], i, i, j, j
        while stack:
            a, b = stack.pop()
            i_lo, i_hi, j_lo, j_hi = min(i_lo, a), max(i_hi, a), min(j_lo, b), max(j_hi, b)
            for na in range(max(0, a - 1), min(tw, a + 2)):
                for nb in range(max(0, b - 1), min(th, b + 2)):
                    if tiles[na, nb] and not seen[na, nb]:
                        seen[na, nb] = True
                        stack.append((na, nb))
        # Tighten to the changed cells inside the cluster's tiles
        sub = changed[i_lo * tile:(i_hi + 1) * tile, j_lo * tile:(j_hi + 1) * tile]
        xs, ys = np.flatnonzero(sub.any(axis=1)), np.flatnonzero(sub.any(axis=0))
        rects.append((x0 + i_lo * tile + xs[0], x0 + i_lo * tile + xs[-1] + 1,
                      y0 + j_lo * tile + ys[0], y0 + j_lo * tile + ys[-1] + 1))
    return rects


class SDFState:
    """The world as an occupancy grid, occupancy[x, y] in FREE / OCCUPIED / UNKNOWN, plus two distance
    fields derived from it (0 on the cells they measure from, capped at SDF_CAP):
        obstacle_sdf[x, y]  distance to the nearest OCCUPIED cell
        unknown_sdf[x, y]   distance to the nearest UNKNOWN cell

    `obstacles` start OCCUPIED, `roads` are cut back out of them, and everything else is `background`.
    For the global map that is UNKNOWN: only cells a scan has confirmed (see merge) become FREE.
    """

    PALETTE = np.array([(0, 0, 0), (0, 0, 255), (45, 45, 45)], dtype=np.uint8)  # indexed by cell state

    def __init__(self, obstacles=(), roads=(), background=UNKNOWN):
        self.occupancy = np.full((SCREEN_WIDTH, SCREEN_HEIGHT), background, dtype=np.int8)
        rasterize(self.occupancy, obstacles, OCCUPIED)
        rasterize(self.occupancy, roads, background)
        self.obstacle_sdf = np.empty((SCREEN_WIDTH, SCREEN_HEIGHT), dtype=np.float32)
        self.unknown_sdf = np.empty((SCREEN_WIDTH, SCREEN_HEIGHT), dtype=np.float32)
        self.has_changed = False
        self.change_coords = []  # (x_lo, y_lo, x_hi, y_hi) of each merge's changed cells
        # (x, y) of every cell a merge newly made OCCUPIED. obstacle_sdf only changes within SDF_CAP of these, so
        # anything farther than SDF_CAP from all of them is unaffected. Consumers read and clear this list.
        self.new_obstacle_points = []
        self.version = 0  # bumped on every change, for anything caching a view of the map
        self.surface = None  # cached render; merges queue the rects that need repainting
        self.redraw = []
        self.recompute(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT)

    def recompute(self, x_lo, x_hi, y_lo, y_hi, obstacles=True, unknown=True):
        """Refresh the chosen distance fields around [x_lo, x_hi) x [y_lo, y_hi) after the occupancy there changed."""
        # Distances are capped, so cells more than SDF_CAP past the edited cells keep their values, and
        # only cells within SDF_CAP of those edits can be the nearest one to them
        m = math.ceil(SDF_CAP)
        x_lo, x_hi = max(0, x_lo - m), min(SCREEN_WIDTH, x_hi + m)
        y_lo, y_hi = max(0, y_lo - m), min(SCREEN_HEIGHT, y_hi + m)
        ax_lo, ax_hi = max(0, x_lo - m), min(SCREEN_WIDTH, x_hi + m)
        ay_lo, ay_hi = max(0, y_lo - m), min(SCREEN_HEIGHT, y_hi + m)
        occ = self.occupancy[ax_lo:ax_hi, ay_lo:ay_hi]
        inner = slice(x_lo - ax_lo, x_hi - ax_lo), slice(y_lo - ay_lo, y_hi - ay_lo)
        if obstacles:
            self.obstacle_sdf[x_lo:x_hi, y_lo:y_hi] = capped_distance(occ == OCCUPIED, SDF_CAP)[inner]
        if unknown:
            self.unknown_sdf[x_lo:x_hi, y_lo:y_hi] = capped_distance(occ == UNKNOWN, SDF_CAP)[inner]

    def merge(self, local, x0, y0):
        """Fold a local occupancy grid (e.g. one lidar scan) whose [0, 0] cell sits at (x0, y0) into this map.

        OCCUPIED always wins, FREE only fills cells that are UNKNOWN, and UNKNOWN leaves the map alone, so a
        merge can add obstacles and confirm free space but never clear an obstacle. Returns whether anything
        changed; merging what the map already knows is a few comparisons and skips the SDF update. The SDFs
        are refreshed only around each cluster of changed cells.
        """
        region = self.occupancy[x0:x0 + local.shape[0], y0:y0 + local.shape[1]]
        new_obstacle = (local == OCCUPIED) & (region != OCCUPIED)
        new_free = (local == FREE) & (region == UNKNOWN)
        changed = new_obstacle | new_free
        if not changed.any():
            return False
        region[new_obstacle] = OCCUPIED
        region[new_free] = FREE
        # Every change shrinks the unknown area, but only new obstacles can move obstacle distances
        for x_lo, x_hi, y_lo, y_hi in changedRects(changed, x0, y0):
            self.recompute(x_lo, x_hi, y_lo, y_hi, obstacles=False)
            self.change_coords.append((x_lo, y_lo, x_hi, y_hi))
            self.redraw.append((x_lo, x_hi, y_lo, y_hi))
        if new_obstacle.any():
            for x_lo, x_hi, y_lo, y_hi in changedRects(new_obstacle, x0, y0):
                self.recompute(x_lo, x_hi, y_lo, y_hi, unknown=False)
            xs, ys = np.nonzero(new_obstacle)
            self.new_obstacle_points.extend(zip((xs + x0).tolist(), (ys + y0).tolist()))
        self.has_changed = True
        self.version += 1
        return True

    def draw(self, surface):
        if self.surface is None:
            self.surface = pygame.surfarray.make_surface(self.PALETTE[self.occupancy])
        elif self.redraw:
            pixels = pygame.surfarray.pixels3d(self.surface)
            for x_lo, x_hi, y_lo, y_hi in self.redraw:
                pixels[x_lo:x_hi, y_lo:y_hi] = self.PALETTE[self.occupancy[x_lo:x_hi, y_lo:y_hi]]
            del pixels  # unlocks the surface
        self.redraw.clear()
        surface.blit(self.surface, (0, 0))


class LocalObstacleMap:
    """The true world the car drives in: the known map's blocks plus random obstacles on the roads that
    the planner doesn't know about. scan() simulates the lidar that will produce local occupancy grids
    in the future, and reveal() merges a scan into the known map.

    Hidden obstacles can be any shape with reach, bounds() and signed_distance().
    """

    def __init__(self, known, count=LOCAL_OBSTACLE_COUNT, blocked_roads=BLOCKED_ROADS, seed=LOCAL_SEED):
        rng = random.Random(seed)
        self.known = known
        self.obstacles = self.roadBlocks(rng, blocked_roads) + self.scatter(rng, count)
        self.hidden = rasterize(np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT), dtype=bool), self.obstacles, True)
        self.truth = self.hidden | (known.occupancy == OCCUPIED)
        self.truth_sdf = capped_distance(self.truth, SDF_CAP)  # lets the simulated rays take big steps
        self.surface, self.drawn_changes = None, 0

    @staticmethod
    def keepsClear(x, y, reach):
        return all(math.hypot(x - px, y - py) > reach + STREET_WIDTH / 2 for px, py in (START, TARGET))

    def roadBlocks(self, rng, n):
        # Barriers across the midpoints of straight road segments between intersections, closing those roads
        mids = [((a + b) / 2, y, math.pi / 2) for y in STREET_Y for a, b in zip(STREET_X, STREET_X[1:])]
        mids += [(x, (a + b) / 2, 0.0) for x in STREET_X for a, b in zip(STREET_Y, STREET_Y[1:])]
        barriers = [Box(x, y, STREET_WIDTH + 10, 12, angle, corner=3) for x, y, angle in mids]
        barriers = [b for b in barriers if self.keepsClear(b.x, b.y, b.reach)]
        return rng.sample(barriers, min(n, len(barriers)))

    def randomShape(self, rng, x, y):
        if rng.random() < 0.5:
            return Obstacle(x, y, rng.uniform(*LOCAL_OBSTACLE_RADIUS))
        width = rng.uniform(*LOCAL_BOX_WIDTH)
        return Box(x, y, rng.uniform(*LOCAL_BOX_LENGTH), width, rng.uniform(0, math.pi), rng.uniform(0, width / 2))

    def scatter(self, rng, n):
        obstacles = []
        for _ in range(n * 50):  # rejection sampling; gives up quietly if the roads are too full
            if len(obstacles) == n:
                break
            x, y = rng.uniform(0, SCREEN_WIDTH), rng.uniform(0, SCREEN_HEIGHT)
            shape = self.randomShape(rng, x, y)
            if self.known.obstacle_sdf[int(x), int(y)] >= 10 and self.keepsClear(x, y, shape.reach):  # on a road, off the kerb
                obstacles.append(shape)
        return obstacles

    def scan(self, x, y):
        """Simulated long range lidar: LIDAR_BEAMS rays from (x, y), each running until it hits something in the
        true world or leaves the map.

        Rays step by the true world's distance to the nearest obstacle (sphere tracing), so open road takes a
        few big steps. The visible area is the polygon through the ray ends, filled by pygame.
        Returns (local, x0, y0), a FREE / OCCUPIED / UNKNOWN grid whose [0, 0] cell is at (x0, y0); anything
        behind the first thing a ray hits is UNKNOWN.
        """
        angles = np.arange(LIDAR_BEAMS) * (2 * math.pi / LIDAR_BEAMS)
        cos, sin = np.cos(angles), np.sin(angles)
        reach = np.zeros(LIDAR_BEAMS)
        hit = np.zeros(LIDAR_BEAMS, dtype=bool)
        live = np.arange(LIDAR_BEAMS)
        while live.size:
            px = np.rint(x + cos[live] * reach[live]).astype(np.intp)
            py = np.rint(y + sin[live] * reach[live]).astype(np.intp)
            on_map = (px >= 0) & (px < SCREEN_WIDTH) & (py >= 0) & (py < SCREEN_HEIGHT)
            clearance = np.zeros(live.size, dtype=np.float32)  # off the map counts as the end of the ray
            clearance[on_map] = self.truth_sdf[px[on_map], py[on_map]]
            hit[live[on_map & (clearance <= 0)]] = True
            going = clearance > 0
            reach[live[going]] += np.maximum(clearance[going], 1)
            live = live[going]
        reach[hit] += 1.5  # so the polygon covers the surface the ray hit

        ex, ey = x + cos * reach, y + sin * reach
        x0, x1 = max(0, math.floor(min(ex.min(), x))), min(SCREEN_WIDTH, math.ceil(max(ex.max(), x)) + 1)
        y0, y1 = max(0, math.floor(min(ey.min(), y))), min(SCREEN_HEIGHT, math.ceil(max(ey.max(), y)) + 1)
        canvas = pygame.Surface((x1 - x0, y1 - y0), depth=32)
        pygame.draw.polygon(canvas, (255, 255, 255), np.column_stack((ex - x0, ey - y0)).tolist())
        seen = pygame.surfarray.pixels_red(canvas) > 0
        truth = self.truth[x0:x1, y0:y1]
        local = np.full(seen.shape, UNKNOWN, dtype=np.int8)
        local[seen & ~truth] = FREE
        local[seen & truth] = OCCUPIED
        return local, x0, y0

    def reveal(self, x, y):
        """Scan from (x, y) and merge the result into the known map. Returns whether the known map changed."""
        return self.known.merge(*self.scan(x, y))

    def draw(self, surface):
        """Overlay the hidden obstacles the known map hasn't picked up yet, in green."""
        changes = self.known.change_coords
        if self.surface is None or self.drawn_changes > len(changes):  # first draw, or someone cleared the log
            self.surface = pygame.surfarray.make_surface(self.overlay(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT))
            self.surface.set_colorkey((0, 0, 0))
        elif self.drawn_changes < len(changes):
            pixels = pygame.surfarray.pixels3d(self.surface)
            for x_lo, y_lo, x_hi, y_hi in changes[self.drawn_changes:]:
                pixels[x_lo:x_hi, y_lo:y_hi] = self.overlay(x_lo, x_hi, y_lo, y_hi)
            del pixels
        self.drawn_changes = len(changes)
        surface.blit(self.surface, (0, 0))

    def overlay(self, x_lo, x_hi, y_lo, y_hi):
        pixels = np.zeros((x_hi - x_lo, y_hi - y_lo, 3), dtype=np.uint8)
        unseen = self.hidden[x_lo:x_hi, y_lo:y_hi] & (self.known.occupancy[x_lo:x_hi, y_lo:y_hi] != OCCUPIED)
        pixels[unseen] = (0, 160, 0)
        return pixels


class NodeStatus(Enum):
    deadEnd = 0  # doesn't lead to the target
    solution = 1  # leads to the target, but isn't on the path being pursued
    following = 2  # on the path being pursued
    orphaned = 3  # cut off from the tree by a new obstacle; ignored by planning


class state:
    
    def __init__(self, x1, x2, a, s, cost = 0, steer = 0, status = NodeStatus.deadEnd, safeVelocity = None, time = 0):
        self.x1 = x1
        self.x2 = x2 
        self.a = a 
        self.s = s
        self.x1bar = self.x1 +self.s * math.cos(self.a)*DISTANCE_TIME_PROP
        self.x2bar = self.x2 + self.s * math.sin(self.a)*DISTANCE_TIME_PROP 
        self.cost = cost #we need to define cost
        self.steer = steer
        self.children =  []
        self.parent = None
        self.status = status
        self.safeVelocity = safeVelocity  # px/s; None until computed
        self.traj = []  # collisionCheckPoints of the edge from parent to this node, set when the tree keeps it
        self.time = time  # s of driving from the root to here (cost can include more than time)
        self.expansions = 0  # times planGlobal's g + h selection has expanded this node
        self.turns_tried = False  # whether turn primitives have been tried from this node (GlobalPlanner.use_turns)
        self.witness = None  # the Witness this node was last made the rep of

    def distance(self, s2):
        return stateDistance(self, s2)  # was calcReedsSheppDist(self, s2)
    
    def subtree(self):
        """This vertex and everything below it (iterative, so long branches don't hit the recursion limit)."""
        stack, branch = [self], []
        while stack:
            v = stack.pop()
            branch.append(v)
            stack.extend(v.children)
        return branch

    def removeUntilActive(self, tree):
        """Prune from this vertex upward: if nothing in its branch (itself included) is active, cut the whole
        branch off its parent, then check the parent the same way, stopping at the first vertex still needed.
        Returns the removed vertices so the caller can drop them from any lookup structures (e.g. the hash map)."""
        removed = []
        node = self
        while node is not None:
            branch = node.subtree()
            if any(v in tree.active for v in branch):
                break
            for v in branch:
                tree.inactive.discard(v)
            removed.extend(branch)
            parent = node.parent
            if parent is not None:
                parent.children.remove(node)
            node.parent = None
            node = parent
        return removed
    
    
class collisionCheckPoints(state):
    """A state along an edge, kept so the edge can be rechecked when new obstacles show up.
    `node` is the tree node the edge was propagated from."""

    def __init__(self, x1, x2, a, s, node, cost=0, steer=0, time=0):
        super().__init__(x1, x2, a, s, cost=cost, steer=steer, time=time)
        self.node = node
        self.edge_end = None  # the tree node this point's edge leads to, set when the tree keeps the edge


class action:
    def __init__(self, da, ds):
        self.da = da  # steering rate, rad/s
        self.ds = ds  # acceleration, px/s^2

class edge:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2


class SSTTree:
    def __init__(self, s1):
        self.active = {s1}

        self.inactive = set()
        self.goal_nodes = []  # kept nodes inside the goal region (TARGET, TARGET_RAD)
    
        self.E = np.array([])
        
    def activate(self, vertex):
        self.inactive.discard(vertex)
        self.active.add(vertex)
    def unActivate(self,vertex):
        self.active.discard(vertex)
        self.inactive.add(vertex)

    def add(self, path, parent, witness_map, traj_map):
        """SST's witness step for a freshly propagated edge.

        `path` is what monteCarloProp returned (its last state is the new node), `parent` the node it was propagated
        from, `witness_map` a WitnessHashMap of every Witness, `traj_map` a proximityHashMap of the check points of
        every edge in the tree.

        If no witness is within delta_S of the new node, the node becomes a new witness and its rep. Otherwise the
        closest witness decides: if the new node is cheaper than its rep, the node takes over as rep, and the old rep
        is made inactive and pruned with removeUntilActive (the check points of everything pruned leave traj_map).
        If the new node isn't cheaper, it's dropped. A kept node is attached to `parent`, made active, and its check
        points go into traj_map. Returns the new node if kept, else None.
        """
        node = path[-1]
        # distanceLowerBound is cheap and never more than the real distance. Skip witnesses whose bound is already past
        # delta_S, then try the rest in order of that bound and stop once it can't beat the closest found so far; most
        # witnesses never need the full distance.
        candidates = []
        for w in witness_map.get_nbhd(node, delta_S):
            bound = distanceLowerBound(w.state, node)
            if bound <= delta_S:
                candidates.append((bound, w))
        candidates.sort(key=lambda c: c[0])
        closest, closest_dist = None, math.inf
        for bound, w in candidates:
            if bound >= closest_dist:
                break
            d = w.state.distance(node)
            if d <= delta_S and d < closest_dist:
                closest, closest_dist = w, d
        if closest is None:
            closest = Witness(node)
            witness_map.add(closest)
        elif closest.rep is not None and node.cost >= closest.rep.cost:
            return None
        old_rep = None if closest.rep is node else closest.rep

        # Attach before pruning, so the old rep's branch still counts the new node if it hangs below it
        node.parent = parent
        parent.children.append(node)
        node.traj = path
        for p in path:
            p.edge_end = node
            traj_map.add(p)
        self.activate(node)
        witness_map.setRep(closest, node)

        # In the goal region: the node and everything above it now lead to the target
        if math.hypot(node.x1 - TARGET[0], node.x2 - TARGET[1]) <= TARGET_RAD:
            self.goal_nodes.append(node)
            v = node
            while v is not None:
                v.status = NodeStatus.solution
                v = v.parent

        if old_rep is not None:
            self.unActivate(old_rep)
            ancestors = []  # old_rep and everything above it; pruning cuts this chain somewhere
            v = old_rep
            while v is not None:
                ancestors.append(v)
                v = v.parent
            removed = old_rep.removeUntilActive(self)
            lost_goal = False
            for v in removed:
                for p in v.traj:
                    traj_map.remove(p)
                if v.status is NodeStatus.solution and v in self.goal_nodes:
                    self.goal_nodes.remove(v)
                    lost_goal = True
            if lost_goal:
                # Walk up from where the pruned branch was cut off: ancestors with no goal node left below them no
                # longer lead to the target
                removed_ids = {id(v) for v in removed}
                goal_ids = {id(g) for g in self.goal_nodes}
                for v in ancestors:
                    if id(v) in removed_ids:
                        continue
                    if any(id(u) in goal_ids for u in v.subtree()):
                        break
                    v.status = NodeStatus.deadEnd
        return node


class proximityHashMap:
    def __init__(self, delta):
        self.rad = delta
        self.map = {}
    def add(self, s):
        x_coord = math.floor(s.x1/self.rad)
        y_coord = math.floor(s.x2/self.rad)
        if x_coord not in self.map:
            self.map[x_coord] = {}
        if y_coord in self.map[x_coord]:
            self.map[x_coord][y_coord].append(s)

        else:
            self.map[x_coord][y_coord] = [s]
            
    def remove(self, s):
        cell = self.map.get(math.floor(s.x1/self.rad), {}).get(math.floor(s.x2/self.rad))
        if cell is not None and s in cell:
            cell.remove(s)

    def get(self, x_coord, y_coord):
        if x_coord in self.map and y_coord in self.map[x_coord]:
            return self.map[x_coord][y_coord]
        return []
            
    def get_nbhd(self, s, delta):
        x_lo = math.floor((s.x1 - delta)/self.rad)
        x_hi = math.floor((s.x1 + delta)/self.rad)
        y_lo = math.floor((s.x2 - delta)/self.rad)
        y_hi = math.floor((s.x2 + delta)/self.rad)
        result = []
        for i in range(x_lo, x_hi + 1):
            for j in range(y_lo, y_hi + 1):
                for other in self.get(i, j):
                    if math.hypot(other.x1 - s.x1, other.x2 - s.x2) <= delta:
                        result.append(other)
                
        return result 




    
    
            
        

class Witness:
    """An SST witness: a fixed point in state space and the node currently representing it.

    `state` is the witness's center and never moves. `rep` is the cheapest node found within delta_S of it, measured
    as distance from `state` to the node. The state a witness is created with is also its first rep.
    """

    def __init__(self, s):
        # The center is a detached copy (no parent, children or tree status), so pruning the original node, e.g.
        # once a new obstacle covers it, leaves the witness intact and doesn't keep the pruned branch alive.
        self.state = state(s.x1, s.x2, s.a, s.s, cost=s.cost, steer=s.steer)
        self.rep = s

    def setRep(self, node):
        self.rep = node


class WitnessHashMap:
    """Witnesses in a position-only hash, plus a numpy row per witness holding its rep, for whole-map searches.

    get_nbhd(s, delta) visits the position cells within distanceBounds(delta)'s position bound and returns the witnesses
    inside all of its bounds (position, speed and heading are cheap to check); the caller still checks the exact
    distance. Cells are as wide as the position bound for the `delta` the map is built for. Searching a smaller radius is
    still correct, so rebuild with the smaller delta once it has shrunk a lot.

    repDistances(s) gives the distance from every witness's rep to s in one numpy pass, for when the hash finds nothing
    nearby. Witnesses are never removed, so rows only grow; change a rep through setRep so its row stays in sync.
    """

    def __init__(self, delta):
        self.cell = distanceBounds(delta)[0]
        self.cells = {}
        self.witnesses = []
        self.reps = np.empty((64, 5))  # per witness: x, y, heading, speed, steering of its rep

    def key(self, s):
        return math.floor(s.x1 / self.cell), math.floor(s.x2 / self.cell)

    def add(self, w):
        if w.rep is not None:
            w.rep.witness = w
        w.index = len(self.witnesses)
        self.witnesses.append(w)
        self.cells.setdefault(self.key(w.state), []).append(w)
        if w.index == len(self.reps):
            self.reps = np.concatenate((self.reps, np.empty_like(self.reps)))
        self.writeRep(w)

    def writeRep(self, w):
        r = w.rep
        # A witness with no rep gets a row infinitely far away, so repDistances never picks it
        self.reps[w.index] = (math.inf, math.inf, 0, 0, 0) if r is None else (r.x1, r.x2, r.a, r.s, r.steer)

    def setRep(self, w, node):
        """Make `node` (or None) w's rep and keep its numpy row and the node's back-link in sync."""
        w.setRep(node)
        if node is not None:
            node.witness = w
        self.writeRep(w)

    def get_nbhd(self, s, delta):
        """Witnesses that could be within distance delta of state s."""
        r_pos, r_speed, r_heading = distanceBounds(delta)
        result = []
        for i in range(math.floor((s.x1 - r_pos) / self.cell), math.floor((s.x1 + r_pos) / self.cell) + 1):
            for j in range(math.floor((s.x2 - r_pos) / self.cell), math.floor((s.x2 + r_pos) / self.cell) + 1):
                for w in self.cells.get((i, j), ()):
                    o = w.state
                    if (math.hypot(o.x1 - s.x1, o.x2 - s.x2) <= r_pos and abs(o.s - s.s) <= r_speed
                            and abs((o.a - s.a + math.pi) % (2 * math.pi) - math.pi) <= r_heading):
                        result.append(w)
        return result

    def repDistances(self, s):
        """stateDistance from every witness's rep to s, as a numpy array in witness order. Mirrors stateDistance, so
        change the two together."""
        r = self.reps[:len(self.witnesses)]
        dh = np.abs((r[:, 2] - s.a + np.pi) % (2 * np.pi) - np.pi)
        return (np.hypot(r[:, 0] - s.x1, r[:, 1] - s.x2) + W_A * dh + W_V * np.abs(r[:, 3] - s.s)
                + W_ST * np.abs(r[:, 4] - s.steer))

    def __iter__(self):
        return iter(self.witnesses)

    def __len__(self):
        return len(self.witnesses)


def makeSampleBiasingGrid():
    """An occupancy grid that steers sampling: getRandomState never samples a cell marked OCCUPIED. Starts all FREE."""
    return np.full((SCREEN_WIDTH, SCREEN_HEIGHT), FREE, dtype=np.int8)


def getRandomState(sdf_state=None, sample_bias=None):
    """Uniform over the map, heading and the full speed range (reverse included), with a steering angle the
    lateral acceleration limit allows at that speed.

    Positions that are OCCUPIED in sdf_state's occupancy grid or in the sample_bias grid are redrawn (unknown space can
    still be sampled). Gives up after 1000 tries and keeps the last position, so a fully blocked grid can't hang it.
    """
    for _ in range(1000):
        x_1_rand = random.uniform(0, SCREEN_WIDTH)
        x_2_rand = random.uniform(0, SCREEN_HEIGHT)
        px, py = min(int(x_1_rand), SCREEN_WIDTH - 1), min(int(x_2_rand), SCREEN_HEIGHT - 1)
        if sdf_state is not None and sdf_state.occupancy[px, py] == OCCUPIED:
            continue
        if sample_bias is not None and sample_bias[px, py] == OCCUPIED:
            continue
        break
    a_rand = random.uniform(0, 2 * math.pi)
    speed_rand = random.uniform(-MAX_REVERSE_SPEED, MAX_SPEED)
    steer_limit = maxSteerAtSpeed(abs(speed_rand))
    steer_rand = random.uniform(-steer_limit, steer_limit)

    return state(x_1_rand, x_2_rand, a_rand, speed_rand, steer=steer_rand)

# Compiled kernels (numba). The car's limits come in as arguments, not module constants (numba would freeze those at
# compile time), so changing a setting at runtime still applies. The functions below them keep their signatures.

@njit(cache=True)
def _max_steer(speed, max_steer, max_lateral, wheelbase):
    if speed == 0:
        return max_steer
    return min(max_steer, math.atan(max_lateral * wheelbase / speed ** 2))


@njit(cache=True)
def _steer_rate_bounds(speed0, steer0, accel, frames, dt, max_rate, max_steer, max_lateral, wheelbase):
    lo, hi = -max_rate, max_rate
    v_prev = speed0
    for k in range(1, frames + 1):
        v = speed0 + accel * k * dt
        limit = _max_steer(max(abs(v_prev), abs(v)), max_steer, max_lateral, wheelbase)
        t = k * dt
        lo = max(lo, (-limit - steer0) / t)
        hi = min(hi, (limit - steer0) / t)
        v_prev = v
    return lo, hi


@njit(cache=True)
def _feasible_accel(speed0, steer0, time, accel, dt, max_accel, max_reverse, max_speed, max_rate, max_steer,
                    max_lateral, wheelbase):
    frames = max(1, int(np.rint(time / dt)))
    accel = min(max(accel, max(-max_accel, (-max_reverse - speed0) / time)), min(max_accel, (max_speed - speed0) / time))
    braking = min(max(-speed0 / time, -max_accel), max_accel)
    for _ in range(20):
        lo, hi = _steer_rate_bounds(speed0, steer0, accel, frames, dt, max_rate, max_steer, max_lateral, wheelbase)
        if lo <= hi:
            return accel, lo, hi
        accel = (accel + braking) / 2
    lo, hi = _steer_rate_bounds(speed0, steer0, braking, frames, dt, max_rate, max_steer, max_lateral, wheelbase)
    return braking, lo, hi


@njit(cache=True)
def _arc_pose(x, y, a, curvature, d):
    if abs(curvature) < 1e-9:
        return x + d * math.cos(a), y + d * math.sin(a), a
    a_end = a + curvature * d
    return x + (math.sin(a_end) - math.sin(a)) / curvature, y - (math.cos(a_end) - math.cos(a)) / curvature, a_end


@njit(cache=True)
def _body_clearance(x, y, a, sdf, offsets, radius):
    c, s = math.cos(a), math.sin(a)
    clearance = np.inf
    for offset in offsets:
        px, py = int(np.rint(x + offset * c)), int(np.rint(y + offset * s))
        if not (0 <= px < sdf.shape[0] and 0 <= py < sdf.shape[1]):
            return 0.0
        clearance = min(clearance, sdf[px, py] - radius)
    return clearance


@njit(cache=True)
def _known_clearance(x, y, a, sdf, unknown_sdf, offsets, radius):
    c, s = math.cos(a), math.sin(a)
    clearance = np.inf
    for offset in offsets:
        px, py = int(np.rint(x + offset * c)), int(np.rint(y + offset * s))
        if not (0 <= px < sdf.shape[0] and 0 <= py < sdf.shape[1]):
            return 0.0
        clearance = min(clearance, min(sdf[px, py], unknown_sdf[px, py]) - radius)
    return clearance


@njit(cache=True)
def _footprint_clearances(points, sdf, unknown_sdf, unknown_too, offsets, radius):
    out = np.empty(points.shape[0])
    for n in range(points.shape[0]):
        if unknown_too:
            out[n] = _known_clearance(points[n, 0], points[n, 1], points[n, 2], sdf, unknown_sdf, offsets, radius)
        else:
            out[n] = _body_clearance(points[n, 0], points[n, 1], points[n, 2], sdf, offsets, radius)
    return out


@njit(cache=True)
def _all_clear(points, sdf, unknown_sdf, unknown_too, offsets, radius, margin):
    """Whether every pose of `points` keeps `margin` (stops at the first that doesn't)."""
    for n in range(points.shape[0]):
        if unknown_too:
            c = _known_clearance(points[n, 0], points[n, 1], points[n, 2], sdf, unknown_sdf, offsets, radius)
        else:
            c = _body_clearance(points[n, 0], points[n, 1], points[n, 2], sdf, offsets, radius)
        if c < margin:
            return False
    return True


@njit(cache=True)
def _simulate_stop(speed, steer, steer_rate, dt, max_accel, wheelbase, reach, spacing, max_steer, max_lateral):
    x = y = a = 0.0
    v = speed
    xs, ys, as_ = [0.0], [0.0], [0.0]
    xs.pop(); ys.pop(); as_.pop()
    while v > 0:
        t_move = min(dt, v / max_accel)
        limit = _max_steer(v, max_steer, max_lateral, wheelbase)
        steer = min(max(steer + steer_rate * dt, -limit), limit)
        curvature = math.tan(steer) / wheelbase
        steps = max(1, int(math.ceil(v * t_move * math.hypot(1.0, reach * curvature) / spacing)))
        for i in range(1, steps + 1):
            t = t_move * i / steps
            px, py, pa = _arc_pose(x, y, a, curvature, v * t - max_accel * t * t / 2)
            xs.append(px); ys.append(py); as_.append(pa)
        x, y, a = xs[-1], ys[-1], as_[-1]
        v -= max_accel * t_move
        if v < 1e-9:
            v = 0.0
    out = np.empty((len(xs), 3))
    for n in range(len(xs)):
        out[n, 0], out[n, 1], out[n, 2] = xs[n], ys[n], as_[n]
    return out


@njit(cache=True)
def _grid_dijkstra(free, dist, cell):
    """In place: Dijkstra over the 8-connected free cells of `dist` (sources already holding their starting cost)."""
    nx, ny = dist.shape
    heap = [(0.0, 0, 0)]
    heap.pop()
    for i in range(nx):
        for j in range(ny):
            if np.isfinite(dist[i, j]):
                heap.append((dist[i, j], i, j))
    if len(heap) == 0:
        return dist
    heapq.heapify(heap)
    diag = cell * math.sqrt(2.0)
    while len(heap) > 0:
        d, i, j = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di == 0 and dj == 0:
                    continue
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and free[a, b]:
                    nd = d + (diag if di != 0 and dj != 0 else cell)
                    if nd < dist[a, b]:
                        dist[a, b] = nd
                        heapq.heappush(heap, (nd, a, b))
    return dist


@njit(cache=True)
def _lib_step(x, y, a, v, s, acc, rate, frames, dt, max_speed, max_steer, max_lateral, wheelbase):
    """In place, per car: `frames` dt frames of (acc, rate), steering within the lateral limit, speed in [0, max_speed]."""
    for n in range(x.shape[0]):
        xn, yn, an, vn, sn = x[n], y[n], a[n], v[n], s[n]
        for _ in range(frames):
            v_end = min(max(vn + acc[n] * dt, 0.0), max_speed)
            top = max(max(vn, v_end), 1e-9)
            limit = min(max_steer, math.atan(max_lateral * wheelbase / top ** 2))
            sn = min(max(sn + rate[n] * dt, -limit), limit)
            k = math.tan(sn) / wheelbase
            d = (vn + v_end) / 2 * dt
            a_end = an + k * d
            if abs(k) < 1e-9:
                xn, yn = xn + d * math.cos(an), yn + d * math.sin(an)
            else:
                xn, yn = xn + (math.sin(a_end) - math.sin(an)) / k, yn - (math.cos(a_end) - math.cos(an)) / k
            an, vn = a_end, v_end
        x[n], y[n], a[n], v[n], s[n] = xn, yn, an, vn, sn


def maxSteerAtSpeed(speed):
    return _max_steer(float(speed), MAX_STEER, MAX_LATERAL_ACCELERATION, WHEELBASE)


def steerRateBounds(x_selected, accel, frames):
    """Steering rates that keep every frame of an edge within the lateral acceleration limit for this acceleration.

    Frame k drives with steering steer0 + rate * k * PLAN_DT at speeds up to that frame's top speed, so each frame
    bounds the rate from both sides and the bounds just intersect. The per-frame limit is at most MAX_STEER, so these
    rates also never hit the steering clamp. Returns (lo, hi); lo > hi means no rate works.
    """
    return _steer_rate_bounds(float(x_selected.s), float(x_selected.steer), float(accel), int(frames), PLAN_DT,
                              MAX_STEER_RATE, MAX_STEER, MAX_LATERAL_ACCELERATION, WHEELBASE)


def accelRange(x, time):
    """Accelerations that keep the speed within its limits over an edge of `time`."""
    return (max(-MAX_ACCELERATION, (-MAX_REVERSE_SPEED - x.s) / time),
            min(MAX_ACCELERATION, (MAX_SPEED - x.s) / time))


def feasibleAccel(x, time, accel):
    """Clamp `accel` to accelRange, then, if it leaves no steering rate that keeps the edge within the lateral limit,
    move it halfway toward braking until one exists: braking toward zero never raises the speed, so with the steering
    held the state's own limit still holds. Returns (accel, rate_lo, rate_hi) with the rates from steerRateBounds."""
    return _feasible_accel(float(x.s), float(x.steer), float(time), float(accel), PLAN_DT, MAX_ACCELERATION,
                           MAX_REVERSE_SPEED, MAX_SPEED, MAX_STEER_RATE, MAX_STEER, MAX_LATERAL_ACCELERATION, WHEELBASE)


def getRandomAction(x_selected, time, steer_sigma=STEER_TARGET_SIGMA, accel_mean=None):
    """Random (steering rate, acceleration) to hold for `time` (a whole number of PLAN_DT frames).

    The acceleration is uniform over what keeps the speed within its limits, or (with accel_mean) a normal around
    accel_mean with spread EARLY_ACCEL_SIGMA, clamped the same way (see feasibleAccel). For steering, a target
    angle is drawn from a normal centered at 0 (spread steer_sigma, STEER_TARGET_SIGMA by default, times the most the lateral limit allows at
    the edge's top speed), and the rate is the one that reaches it at the end of the edge, capped so every frame stays
    within the lateral acceleration limit. Propagation can therefore only fail on collisions.
    """
    # Uniform acceleration, or a normal around accel_mean (spread EARLY_ACCEL_SIGMA) if given; feasibleAccel clamps it
    if accel_mean is None:
        accel = random.uniform(*accelRange(x_selected, time))
    else:
        accel = random.gauss(accel_mean, EARLY_ACCEL_SIGMA)
    accel, lo, hi = feasibleAccel(x_selected, time, accel)
    v_end = x_selected.s + accel * time
    target = random.gauss(0, steer_sigma * maxSteerAtSpeed(max(abs(x_selected.s), abs(v_end))))
    return action(clamp((target - x_selected.steer) / time, lo, max(lo, hi)), accel)


def arcPose(x, y, a, curvature, d):
    """Rear-axle pose after driving signed arc length d from (x, y) at heading a with constant curvature. (Plain Python:
    it's too small for a compiled call to pay off; _arc_pose is the compiled twin used inside the kernels.)"""
    if abs(curvature) < 1e-9:
        return x + d * math.cos(a), y + d * math.sin(a), a
    a_end = a + curvature * d
    return x + (math.sin(a_end) - math.sin(a)) / curvature, y - (math.cos(a_end) - math.cos(a)) / curvature, a_end


def bodyClearance(x, y, a, sdf_state):
    """Gap between the car's footprint circles and the nearest obstacle, rear axle at (x, y) with heading a (the
    smallest over the circles). 0 or less means a collision (or a circle center being off the map)."""
    return _body_clearance(float(x), float(y), float(a), sdf_state.obstacle_sdf, FOOTPRINT_OFFSETS_ARRAY,
                           FOOTPRINT_RADIUS)


def pathIsClear(states, sdf_state):
    """Whether every state keeps SAFETY_MARGIN of clearance. For the states monteCarloProp returns, that means the
    whole edge is collision free; rerun it on stored edges when the map changes to find ones that became invalid."""
    return all(bodyClearance(s.x1, s.x2, s.a, sdf_state) >= SAFETY_MARGIN for s in states)


def monteCarloProp(x_selected, sdf_state, steer_sigma=STEER_TARGET_SIGMA, accel_mean=None):
    """Hold a random (steering rate, acceleration) action from x_selected for a random whole number of PLAN_DT frames.

    Both controls are constant over the edge, the same inputs Car takes. As in Car.update, each frame first
    applies the steering rate and then drives that frame with the new steering. Cost grows by
    (1 + COST_ACCEL * |acceleration| + COST_DECEL * braking + COST_STEER_RATE * |steering rate|
    + COST_STEER_ANGLE * |steering angle|) per second along the edge, with each frame's own steering angle.

    Returns the collisionCheckPoints along the edge (each with node=x_selected), spaced so the car's body center
    moves at most CHECK_SPACING px between them and ending at the new state (x_selected itself is not included),
    so pathIsClear on them validates the whole edge. Returns [] if the edge comes within SAFETY_MARGIN of an
    obstacle or leaves the map; getRandomAction already keeps it within the lateral acceleration limit.
    """
    frames = random.randint(1, max(1, round(TIME_PROP / PLAN_DT)))
    return propagate(x_selected, getRandomAction(x_selected, frames * PLAN_DT, steer_sigma, accel_mean), frames, sdf_state)


@njit(cache=True)
def _propagate(x, y, a, v, steer, cost, time, rate, accel, frames, dt, max_steer, wheelbase, reach, spacing,
               control_rate, cost_steer_angle, sdf, offsets, radius, margin):
    """The simulation behind propagate: one row (x, y, heading mod 2 pi, speed, steering, cost, time) per check point,
    or an empty (0, 7) array if one comes within `margin` of an obstacle. See propagate for the model."""
    # First the number of check points (it only depends on speed and steering), then the poses
    total, s_, v_ = 0, steer, v
    for frame in range(frames):
        s_ = min(max(s_ + rate * dt, -max_steer), max_steer)
        v_end = v_ + accel * dt
        travel = max(abs(v_), abs(v_end)) * dt * math.hypot(1.0, reach * math.tan(s_) / wheelbase)
        total += max(1, int(math.ceil(travel / spacing)))
        v_ = v_end
    out = np.empty((total, 7))
    n = 0
    two_pi = 2 * math.pi
    for frame in range(frames):
        steer = min(max(steer + rate * dt, -max_steer), max_steer)
        cost_rate = control_rate + cost_steer_angle * abs(steer)  # steering is held for the frame
        v_end = v + accel * dt
        curvature = math.tan(steer) / wheelbase
        travel = max(abs(v), abs(v_end)) * dt * math.hypot(1.0, reach * curvature)
        steps = max(1, int(math.ceil(travel / spacing)))
        for i in range(1, steps + 1):
            t = dt * i / steps
            sx, sy, sa = _arc_pose(x, y, a, curvature, v * t + accel * t * t / 2)
            if _body_clearance(sx, sy, sa, sdf, offsets, radius) < margin:  # checked before keeping the point
                return np.empty((0, 7))
            out[n, 0], out[n, 1], out[n, 2] = sx, sy, sa % two_pi
            out[n, 3], out[n, 4] = v + accel * t, steer
            out[n, 5], out[n, 6] = cost + t * cost_rate, time + frame * dt + t
            n += 1
        x, y, a, v = out[n - 1, 0], out[n - 1, 1], out[n - 1, 2], v_end
        cost += dt * cost_rate
    return out


def propagate(x_selected, action, frames, sdf_state):
    """Hold `action` from x_selected for `frames` PLAN_DT steps; the simulation behind monteCarloProp (see there for
    what it returns). The action must already respect the steering-rate and lateral limits (e.g. from getRandomAction).

    Each frame first applies the steering rate (clamped to MAX_STEER), then drives that frame with the new steering in
    equal time steps in which every footprint circle's center moves at most CHECK_SPACING (also right if the car
    reverses mid-frame: a point d ahead of the rear axle moves hypot(1, d * curvature) px per px of axle travel, so the
    farthest circle, FOOTPRINT_REACH, moves the most). Cost grows per second by the action's rate plus
    COST_STEER_ANGLE * |steering| of the frame. The simulation is the compiled _propagate; this builds the states."""
    control_rate = (1 + COST_ACCEL * abs(action.ds) + COST_STEER_RATE * abs(action.da)
                    + COST_DECEL * max(0, -action.ds))  # cost per second from the action
    rows = _propagate(float(x_selected.x1), float(x_selected.x2), float(x_selected.a), float(x_selected.s),
                      float(x_selected.steer), float(x_selected.cost), float(x_selected.time), float(action.da),
                      float(action.ds), int(frames), PLAN_DT, MAX_STEER, WHEELBASE, FOOTPRINT_REACH, CHECK_SPACING,
                      float(control_rate), COST_STEER_ANGLE, sdf_state.obstacle_sdf, FOOTPRINT_OFFSETS_ARRAY,
                      FOOTPRINT_RADIUS, SAFETY_MARGIN)
    return [collisionCheckPoints(px, py, pa, pv, x_selected, steer=ps, cost=pc, time=pt)
            for px, py, pa, pv, ps, pc, pt in rows.tolist()]


def simulateStop(speed, steer, steer_rate):
    """Brake as hard as possible from rest pose (0, 0, heading 0) at forward `speed` and steering `steer`, turning the
    wheel at `steer_rate` but never past the lateral acceleration limit for the current speed (each frame's limit is
    the one at its start speed, the highest in it; the last frame ends early, when the car stops).

    Returns an (n, 3) array of (x, y, heading) points after the start, spaced like monteCarloProp's (every footprint circle
    moves at most CHECK_SPACING between them), the last one where the car stops.
    """
    return _simulate_stop(float(speed), float(steer), float(steer_rate), PLAN_DT, MAX_ACCELERATION, WHEELBASE,
                          FOOTPRINT_REACH, CHECK_SPACING, MAX_STEER, MAX_LATERAL_ACCELERATION).reshape(-1, 3)


class StopTraj:
    """An emergency-stop manoeuvre: full braking while turning the wheel at a fixed steering rate.

    `action` is the control for forward motion (action.da = steering rate, action.ds = -MAX_ACCELERATION); in reverse
    the acceleration flips sign. The stop itself is simulated from the actual state each time, so it's exact.
    """

    def __init__(self, name, steer_rate):
        self.name = name
        self.action = action(steer_rate, -MAX_ACCELERATION)

    def worldPoints(self, s):
        """The points of this stop from state s, as an (n, 3) array of world (x, y, heading), spaced so checking each
        with SAFETY_MARGIN covers the whole stop (empty if s is already stopped). Braking in reverse traces the
        forward stop mirrored front to back, so for s.s < 0 the forward stop is flipped first."""
        points = simulateStop(abs(s.s), s.steer, self.action.da)
        x, y, h = points[:, 0], points[:, 1], points[:, 2]
        if s.s < 0:
            x, h = -x, -h
        c, sn = math.cos(s.a), math.sin(s.a)
        return np.column_stack((s.x1 + x * c - y * sn, s.x2 + x * sn + y * c, (s.a + h) % (2 * math.pi)))


class StopTrajLibrary:
    """Which stopping manoeuvres to try, binned by the starting speed and steering.

    Speed bins split |speed| in [0, MAX_SPEED] evenly; steering bins split steer / maxSteerAtSpeed(|speed|) in
    [-1, 1] evenly, since the steering allowed shrinks with speed. Every bin holds go forward, hard right and hard
    left, sorted by increasing size of steering rate so the gentlest manoeuvre is tried first.
    trajectories[speed_bin][steer_bin] is that bin's list.
    """

    MANEUVERS = (("forward", 0.0), ("hard_right", MAX_STEER_RATE), ("hard_left", -MAX_STEER_RATE))

    def __init__(self):
        maneuvers = sorted((StopTraj(name, rate) for name, rate in self.MANEUVERS), key=lambda t: abs(t.action.da))
        self.trajectories = [[list(maneuvers) for _ in range(STOP_STEER_BINS)] for _ in range(STOP_SPEED_BINS)]

    def bins(self, s):
        """(speed_bin, steer_bin) for state s."""
        speed = abs(s.s)
        speed_bin = min(STOP_SPEED_BINS - 1, int(speed / MAX_SPEED * STOP_SPEED_BINS))
        fraction = clamp(s.steer / maxSteerAtSpeed(speed), -1, 1)
        steer_bin = min(STOP_STEER_BINS - 1, int((fraction + 1) / 2 * STOP_STEER_BINS))
        return speed_bin, steer_bin

    def lookup(self, s):
        """The manoeuvres to try for state s, gentlest first."""
        speed_bin, steer_bin = self.bins(s)
        return self.trajectories[speed_bin][steer_bin]

    def __iter__(self):
        return (t for row in self.trajectories for cell in row for t in cell)

    def __len__(self):
        return sum(len(cell) for row in self.trajectories for cell in row)


STOP_TRAJECTORIES = StopTrajLibrary()


def bestNear(witness_map, s):
    """The cheapest active node (witness rep) within distance delta_B_N of s, measured from the node to s, or
    None if there is none.

    A rep sits within delta_S of its witness, so every rep within delta_B_N of s belongs to a witness the map finds
    within delta_B_N + delta_S. Reps are tried cheapest first, skipping any whose cheap lower bound is already past
    delta_B_N, so the first one actually in range is the answer.
    """
    reps = [w.rep for w in witness_map.get_nbhd(s, delta_B_N + delta_S) if w.rep is not None]
    reps.sort(key=lambda node: node.cost)
    for node in reps:
        if distanceLowerBound(node, s) <= delta_B_N and node.distance(s) <= delta_B_N:
            return node
    return None


def nearestActive(witness_map, s):
    """The active node (witness rep) closest to s, from one numpy pass over every rep."""
    return witness_map.witnesses[int(np.argmin(witness_map.repDistances(s)))].rep  # None only if no witness has a rep


def bestFirstSelectionSST(witness_map, sdf_state=None, sample_bias=None, x_rand=None):
    """SST's best-first selection: sample a random state (see getRandomState for what it skips), unless x_rand is given,
    and pick the cheapest active node within delta_B_N of it, or the nearest active node if none is that close."""
    if x_rand is None:
        x_rand = getRandomState(sdf_state, sample_bias)
    return bestNear(witness_map, x_rand) or nearestActive(witness_map, x_rand)


def stateDistance(s0, s1):
    """Distance between states, standing in for the distance for now (that code is commented out below):
    euclidean px + W_A * heading difference + W_V * speed difference + W_ST * steering difference."""
    dh = abs((s1.a - s0.a + math.pi) % (2 * math.pi) - math.pi)
    return (math.hypot(s1.x1 - s0.x1, s1.x2 - s0.x2) + W_A * dh + W_V * abs(s1.s - s0.s)
            + W_ST * abs(s1.steer - s0.steer))


def distanceLowerBound(s0, s1):
    """A cheap value never more than state.distance, for skipping candidates. stateDistance is already cheap, so this is
    the distance itself (for Reeds-Shepp it was reedsSheppLowerBound)."""
    return stateDistance(s0, s1)


def distanceBounds(delta):
    """(position px, speed px/s, heading rad) that two states within distance delta can differ by at most; the
    WitnessHashMap bins by these. Steering isn't binned. For distance these were
    (MAX_SPEED * delta, MAX_ACCELERATION * delta, MAX_SPEED * delta / (RS_WHEELBASE / tan(MAX_STEER)))."""
    return delta, delta / W_V, delta / W_A


# Reeds-Shepp distance, commented out while stateDistance stands in for it.
# def reedsSheppTurnRadius(s0, s1):
#     """The turning radius calcReedsSheppDist uses for this pair: the tightest the wheel allows, or RS_TURN_SCALE times the
#     tightest the lateral acceleration limit allows at the faster of the two speeds."""
#     v = max(abs(s0.s), abs(s1.s))
#     return max(RS_WHEELBASE / math.tan(MAX_STEER), RS_TURN_SCALE * v * v / MAX_LATERAL_ACCELERATION)
#
#
# def reedsSheppLowerBound(s0, s1):
#     """A cheap value that is never more than calcReedsSheppDist(s0, s1), for skipping far-away candidates.
#
#     The path is at least the straight-line distance d long, and starting at speed |v0| the car covers at most
#     |v0| t + MAX_ACCELERATION t^2 / 2 in time t (likewise backward from |v1|). Changing speed takes |v1 - v0| / MAX_ACCELERATION.
#     Turning by the heading difference at radius r takes a path at least |dh| * r long, driven at most at MAX_SPEED.
#     """
#     d = math.hypot(s1.x1 - s0.x1, s1.x2 - s0.x2)
#     a = MAX_ACCELERATION
#     v_start, v_end = abs(s0.s), abs(s1.s)
#     reach = max(math.sqrt(v_start * v_start + 2 * a * d) - v_start, math.sqrt(v_end * v_end + 2 * a * d) - v_end) / a
#     dh = abs((s1.a - s0.a + math.pi) % (2 * math.pi) - math.pi)
#     return max(reach, abs(s1.s - s0.s) / a, dh * reedsSheppTurnRadius(s0, s1) / MAX_SPEED)
#
#
# def calcReedsSheppDist(s0, s1):
#     pi2 = 2 * math.pi
#     eps = 1e-9
#
#     def mod2pi(x):
#         v = math.fmod(x, pi2)
#         if v < -math.pi:
#             v += pi2
#         elif v > math.pi:
#             v -= pi2
#         return v
#
#     def polar(x, y):
#         return math.hypot(x, y), math.atan2(y, x)
#
#     def tau_omega(u, v, xi, eta, phi):
#         delta = mod2pi(u - v)
#         a = math.sin(u) - math.sin(delta)
#         b = math.cos(u) - math.cos(delta) - 1
#         t1 = math.atan2(eta * a - xi * b, xi * a + eta * b)
#         t2 = 2 * (math.cos(delta) - math.cos(v) - math.cos(u)) + 3
#         tau = mod2pi(t1 + math.pi) if t2 < 0 else mod2pi(t1)
#         return tau, mod2pi(tau - u + v - phi)
#
#     def lsl(x, y, phi):
#         u, t = polar(x - math.sin(phi), y - 1 + math.cos(phi))
#         if t >= -eps:
#             v = mod2pi(phi - t)
#             if v >= -eps:
#                 return [t, u, v]
#         return None
#
#     def lsr(x, y, phi):
#         u1, t1 = polar(x + math.sin(phi), y - 1 - math.cos(phi))
#         u1 = u1 * u1
#         if u1 >= 4:
#             u = math.sqrt(u1 - 4)
#             t = mod2pi(t1 + math.atan2(2, u))
#             v = mod2pi(t - phi)
#             if t >= -eps and v >= -eps:
#                 return [t, u, v]
#         return None
#
#     def lrl(x, y, phi):
#         xi = x - math.sin(phi)
#         eta = y - 1 + math.cos(phi)
#         u1, theta = polar(xi, eta)
#         if u1 <= 4:
#             u = -2 * math.asin(0.25 * u1)
#             t = mod2pi(theta + 0.5 * u + math.pi)
#             v = mod2pi(phi - t + u)
#             if t >= -eps and u <= eps:
#                 return [t, u, v]
#         return None
#
#     def lrlr_a(x, y, phi):
#         xi = x + math.sin(phi)
#         eta = y - 1 - math.cos(phi)
#         rho = 0.25 * (2 + math.hypot(xi, eta))
#         if rho <= 1:
#             u = math.acos(rho)
#             t, v = tau_omega(u, -u, xi, eta, phi)
#             if t >= -eps and v <= eps:
#                 return [t, u, -u, v]
#         return None
#
#     def lrlr_b(x, y, phi):
#         xi = x + math.sin(phi)
#         eta = y - 1 - math.cos(phi)
#         rho = (20 - xi * xi - eta * eta) / 16
#         if 0 <= rho <= 1:
#             u = -math.acos(rho)
#             if u >= -math.pi / 2:
#                 t, v = tau_omega(u, u, xi, eta, phi)
#                 if t >= -eps and v >= -eps:
#                     return [t, u, u, v]
#         return None
#
#     def lrsl(x, y, phi):
#         xi = x - math.sin(phi)
#         eta = y - 1 + math.cos(phi)
#         rho, theta = polar(xi, eta)
#         if rho >= 2:
#             r = math.sqrt(rho * rho - 4)
#             u = 2 - r
#             t = mod2pi(theta + math.atan2(r, -2))
#             v = mod2pi(phi - math.pi / 2 - t)
#             if t >= -eps and u <= eps and v <= eps:
#                 return [t, -math.pi / 2, u, v]
#         return None
#
#     def lrsr(x, y, phi):
#         xi = x + math.sin(phi)
#         eta = y - 1 - math.cos(phi)
#         rho, theta = polar(-eta, xi)
#         if rho >= 2:
#             t = theta
#             u = 2 - rho
#             v = mod2pi(t + math.pi / 2 - phi)
#             if t >= -eps and u <= eps and v <= eps:
#                 return [t, -math.pi / 2, u, v]
#         return None
#
#     def lrslr(x, y, phi):
#         xi = x + math.sin(phi)
#         eta = y - 1 - math.cos(phi)
#         rho, theta = polar(xi, eta)
#         if rho >= 2:
#             u = 4 - math.sqrt(rho * rho - 4)
#             if u <= eps:
#                 t = mod2pi(math.atan2((4 - u) * xi - 2 * eta, -2 * xi + (u - 4) * eta))
#                 v = mod2pi(t - phi)
#                 if t >= -eps and v >= -eps:
#                     return [t, -math.pi / 2, u, -math.pi / 2, v]
#         return None
#
#     def variants(fn, x, y, phi, backwards=False):
#         out = []
#         for sx, sy, sp, flip in ((1, 1, 1, False), (-1, 1, -1, True), (1, -1, -1, False), (-1, -1, 1, True)):
#             seg = fn(sx * x, sy * y, sp * phi)
#             if seg is None:
#                 continue
#             if backwards:
#                 seg = seg[::-1]
#             if flip:
#                 seg = [-s for s in seg]
#             out.append(seg)
#         return out
#
#     def run_time(length, va, vb, vmax):
#         a = MAX_ACCELERATION
#         va = min(va, vmax)
#         vb = min(vb, vmax)
#         if abs(vb * vb - va * va) > 2 * a * length:
#             return float('inf')
#         v_peak = math.sqrt(a * length + (va * va + vb * vb) / 2)
#         if v_peak <= vmax:
#             return (2 * v_peak - va - vb) / a
#         accel_dist = (vmax * vmax - va * va) / (2 * a)
#         decel_dist = (vmax * vmax - vb * vb) / (2 * a)
#         return (2 * vmax - va - vb) / a + (length - accel_dist - decel_dist) / vmax
#
#     # Turning radius (see reedsSheppTurnRadius). The speed-based part goes some way toward the car braking before a
#     # turn, which Reeds-Shepp can't do; without it, turns at speed would count as nearly free.
#     r = reedsSheppTurnRadius(s0, s1)
#     dx, dy = s1.x1 - s0.x1, s1.x2 - s0.x2
#     c, s = math.cos(s0.a), math.sin(s0.a)
#     x = (c * dx + s * dy) / r
#     y = (-s * dx + c * dy) / r
#     phi = mod2pi(s1.a - s0.a)
#     xb = x * math.cos(phi) + y * math.sin(phi)
#     yb = x * math.sin(phi) - y * math.cos(phi)
#
#     candidates = []
#     for fn in (lsl, lsr, lrlr_a, lrlr_b, lrslr):
#         candidates += variants(fn, x, y, phi)
#     for fn in (lrl, lrsl, lrsr):
#         candidates += variants(fn, x, y, phi)
#         candidates += variants(fn, xb, yb, phi, backwards=True)
#     v0, v1 = s0.s, s1.s
#
#     def path_time(segments):
#         runs = []
#         for seg in segments:
#             length = abs(seg) * r
#             if length < eps:
#                 continue
#             direction = 1 if seg > 0 else -1
#             if runs and runs[-1][0] == direction:
#                 runs[-1][1] += length
#             else:
#                 runs.append([direction, length])
#
#         if not runs:
#             return abs(v1 - v0) / MAX_ACCELERATION
#
#         total = 0.0
#         last = len(runs) - 1
#         for i, (direction, length) in enumerate(runs):
#             vmax = MAX_SPEED if direction > 0 else MAX_REVERSE_SPEED
#             va = 0.0
#             vb = 0.0
#             if i == 0:
#                 if v0 * direction >= 0:
#                     va = abs(v0)
#                 else:
#                     total += abs(v0) / MAX_ACCELERATION
#             if i == last:
#                 if v1 * direction >= 0:
#                     vb = abs(v1)
#                 else:
#                     total += abs(v1) / MAX_ACCELERATION
#             total += run_time(length, va, vb, vmax)
#         return total
#
#     return min(path_time(seg) for seg in candidates)


def goalDistanceGrid(sdf_state, cell=H_CELL):
    """Grid Dijkstra distance (px) from every cell to the goal region through free space (gridDistance), starting from
    the free cells whose center is inside the goal circle."""
    nx, ny = SCREEN_WIDTH // cell, SCREEN_HEIGHT // cell
    cx = np.arange(nx) * cell + cell // 2
    cy = np.arange(ny) * cell + cell // 2
    sources = np.hypot(cx[:, None] - TARGET[0], cy[None, :] - TARGET[1]) <= TARGET_RAD
    return gridDistance(sdf_state, sources, cell)


def gridValue(grid, x, y):
    """A gridDistance grid's value at (x, y); if that cell's center is too close to a wall (inf), the best way in
    through one of its neighbors."""
    i = min(max(int(x // H_CELL), 0), grid.shape[0] - 1)
    k = min(max(int(y // H_CELL), 0), grid.shape[1] - 1)
    if math.isfinite(grid[i, k]):
        return float(grid[i, k])
    return min((float(grid[a, b]) + H_CELL * math.hypot(a - i, b - k)
                for a in range(max(i - 1, 0), min(i + 2, grid.shape[0]))
                for b in range(max(k - 1, 0), min(k + 2, grid.shape[1]))), default=math.inf)


def gridDistance(sdf_state, sources, cell=H_CELL):
    """Grid Dijkstra distance (px) from every cell to the nearest `sources` cell through free space, on cells of `cell`
    px. `sources` is a boolean (nx, ny) array (every source starts at 0), or a float one with each source's starting
    cost in px and inf elsewhere (then a cell gets the least of distance plus starting cost over the sources).

    A cell is free if its center is at least GRID_INFLATION from known obstacles (obstacle_sdf only, so unknown space
    counts as free), i.e. obstacles are grown by the car's clearance plus a margin; only free sources count. Moves go to
    the 8 neighbors. Unreachable cells are inf.
    """
    nx, ny = SCREEN_WIDTH // cell, SCREEN_HEIGHT // cell
    cx = np.arange(nx) * cell + cell // 2
    cy = np.arange(ny) * cell + cell // 2
    free = sdf_state.obstacle_sdf[np.ix_(cx, cy)] >= GRID_INFLATION
    initial = np.where(sources, 0.0, np.inf) if sources.dtype == bool else sources.astype(float)
    dist = np.ascontiguousarray(np.where(free, initial, np.inf))
    return _grid_dijkstra(np.ascontiguousarray(free), dist, float(cell))


def primitivePath(node, primitive, sdf_state):
    """Propagate one of the PRIMITIVES from `node`: its acceleration (made feasible by feasibleAccel) and the steering
    rate that reaches its target steering at the end, capped to the allowed rates. Returns propagate's states."""
    accel_frac, steer_frac, duration = primitive
    frames = max(1, round(duration / PLAN_DT))
    time = frames * PLAN_DT
    accel, lo, hi = feasibleAccel(node, time, accel_frac * MAX_ACCELERATION)
    v_end = node.s + accel * time
    target = steer_frac * maxSteerAtSpeed(max(abs(node.s), abs(v_end)))
    return propagate(node, action(clamp((target - node.steer) / time, lo, max(lo, hi)), accel), frames, sdf_state)


def turnPath(node, turn, corner_speed, sdf_state):
    """Brake (or speed up) to corner_speed with the wheel going straight, then turn by `turn` rad (signed) at the most
    steering that speed allows, unwinding the wheel in time to come out lined up. Every PLAN_DT step goes through
    propagate. Returns the states, or [] if it hits something, the node is reversing, or the speed change takes over
    TURN_PRIMITIVE_MAX_BRAKE."""
    if node.s < 0 or abs(node.s - corner_speed) / MAX_ACCELERATION > TURN_PRIMITIVE_MAX_BRAKE:
        return []
    side = 1 if turn > 0 else -1
    x, states = node, []
    goal_heading = node.a + turn
    turning = False
    for _ in range(TURN_PRIMITIVE_MAX_FRAMES):
        accel, lo, hi = feasibleAccel(x, PLAN_DT, (corner_speed - x.s) / PLAN_DT)
        left = side * ((goal_heading - x.a + math.pi) % (2 * math.pi) - math.pi)  # heading still to turn, > 0
        if not turning and abs(x.s - corner_speed) < 2:
            turning = True
        if turning:
            # Unwind once the heading the wheel turns while straightening (about steer^2 v / (2 W rate)) would finish it
            unwind = x.steer ** 2 * max(x.s, 1) / (2 * WHEELBASE * MAX_STEER_RATE)
            target = 0.0 if left <= unwind else side * maxSteerAtSpeed(max(x.s, corner_speed))
            if left <= 0 or (target == 0.0 and abs(x.steer) < 1e-3):
                return states
        else:
            target = 0.0
        seg = propagate(x, action(clamp((target - x.steer) / PLAN_DT, lo, max(lo, hi)), accel), 1, sdf_state)
        if not seg:
            return []
        states += seg
        x = seg[-1]
    return []


def kMeans(features, k, iterations=25):
    """Labels (0..k-1) of k-means on the rows of `features`, started with k-means++ (seeded from `random`)."""
    rng = np.random.default_rng(random.randrange(2 ** 32))
    centers = [features[rng.integers(len(features))]]
    for _ in range(1, k):
        d2 = np.min([((features - c) ** 2).sum(axis=1) for c in centers], axis=0)
        centers.append(features[rng.choice(len(features), p=d2 / d2.sum())] if d2.sum() > 0 else features[rng.integers(len(features))])
    centers = np.array(centers)
    labels = np.zeros(len(features), dtype=int)
    for _ in range(iterations):
        labels = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        moved = np.array([features[labels == g].mean(axis=0) if np.any(labels == g) else centers[g] for g in range(k)])
        if np.allclose(moved, centers):
            break
        centers = moved
    return labels


class GlobalPlanner:
    """SST* with informed expansion on sdf_state from `start`, run in steps so it can be paused, given new obstacles
    (addObstacle) and resumed with the same tree.

    Each step expands one node. With probability GREEDY_SHARE it's the active node with the lowest g + h
    (g = cost, h = goalDistanceGrid distance / MAX_SPEED, plus EXPANSION_PENALTY per earlier expansion and a small
    random tie-break). A node's first expansion "blossoms": the PRIMITIVES and BLOSSOM_RANDOM random actions are tried
    and the child with the best g + h is kept; later expansions try one random action. With use_turns,
    every turn primitive (turnPaths) that fits is also added as an extra child, once per node. If the kept child is closer to
    the goal (lower h) than its parent, it is expanded next, up to GREEDY_CHAIN_MAX in a row. The other steps are
    normal SST (random sample, best-first selection, one random action).

    Once a path exists, every ANNEAL_WINDOW steps the best path cost is compared with the window's start. If it
    improved by less than ANNEAL_MIN_GAIN, delta_B_N and delta_S shrink by ANNEAL_FACTOR (not below DELTA_B_N_MIN) and
    the witness hash is rebuilt for the new size. The radii are reset to their starting values on creation.

    Uses obstacle_sdf only (through monteCarloProp), so unknown space counts as free, as a global planner should.
    Samples skip sdf_state's obstacles and anything OCCUPIED in `sample_bias` (a fresh, empty makeSampleBiasingGrid()
    if none is given).
    """

    def __init__(self, sdf_state, start, sample_bias=None, h_fn=None, sample_fn=None):
        global delta_B_N, delta_S
        delta_B_N = DELTA_B_N_START
        delta_S = delta_B_N / 4
        self.sdf_state = sdf_state
        self.start = start
        self.tree = SSTTree(start)
        self.witness_map = WitnessHashMap(delta_S)
        self.witness_map.add(Witness(start))
        self.traj_map = proximityHashMap(50)
        self.sample_bias = makeSampleBiasingGrid() if sample_bias is None else sample_bias
        self.it = 0
        self.window_start, self.window_cost = None, None  # set once a path exists
        self.h_fn = h_fn  # replaces the goalDistanceGrid h if given
        self.sample_fn = sample_fn  # replaces getRandomState for the normal-SST steps if given
        self.h_grid = goalDistanceGrid(sdf_state) if h_fn is None else None
        self.open_heap = []  # (priority, counter, node) for active nodes; stale entries are skipped when popped
        self.counter = itertools.count()
        self.push(start)
        self.chain_node, self.chain_depth = None, 0
        self.child_hook = None  # called with every node the tree keeps (used by the Driver to detect joins)
        self.edge_filter = None  # if set, an edge (list of states) is only kept when edge_filter(path) is true
        self.use_turns = False  # set (by the Driver, once stuck) to add turn primitives as extra children

    def h(self, n):
        if self.h_fn is not None:
            return self.h_fn(n)
        i = min(max(int(n.x1 // H_CELL), 0), self.h_grid.shape[0] - 1)
        j = min(max(int(n.x2 // H_CELL), 0), self.h_grid.shape[1] - 1)
        return self.h_grid[i, j] / MAX_SPEED

    def turnPaths(self, node):
        """Turn primitives: turnPath left and right by TURN_PRIMITIVE_ANGLE at each TURN_PRIMITIVE_SPEEDS."""
        return [turnPath(node, side * TURN_PRIMITIVE_ANGLE, v, self.sdf_state)
                for side in (-1, 1) for v in TURN_PRIMITIVE_SPEEDS]

    def push(self, n):
        priority = n.cost + self.h(n) + EXPANSION_PENALTY * n.expansions + random.uniform(0, TIE_BREAK)
        if math.isfinite(priority):
            heapq.heappush(self.open_heap, (priority, next(self.counter), n))

    def keep(self, path):
        """Whether an edge may be kept: it exists and passes edge_filter, if there is one."""
        return bool(path) and (self.edge_filter is None or self.edge_filter(path))

    def bestGoal(self):
        return min(self.tree.goal_nodes, key=lambda g: g.cost) if self.tree.goal_nodes else None

    def step(self):
        """One planning iteration (see the class docstring)."""
        global delta_B_N, delta_S
        self.it += 1
        it, tree, sdf_state = self.it, self.tree, self.sdf_state
        early = it <= EARLY_STEER_ITERATIONS
        steer_sigma = STEER_TARGET_SIGMA * (math.sqrt(EARLY_STEER_VARIANCE_SCALE) if early else 1)
        accel_mean = EARLY_ACCEL_MEAN if early else None

        # Pick the node to expand: continue a greedy chain, the lowest g + h, or (otherwise) normal SST selection
        node = None
        if self.chain_node is not None and self.chain_node in tree.active:
            node = self.chain_node
        elif random.random() < GREEDY_SHARE:
            while self.open_heap:
                _, _, candidate = heapq.heappop(self.open_heap)
                if candidate in tree.active:
                    node = candidate
                    break
            self.chain_depth = 0
        self.chain_node = None

        if node is not None:
            if node.expansions == 0:  # blossom: library actions plus a few random ones, keep the best g + h
                paths = [primitivePath(node, prim, sdf_state) for prim in PRIMITIVES]
                paths += [monteCarloProp(node, sdf_state, steer_sigma, accel_mean) for _ in range(BLOSSOM_RANDOM)]
                paths = sorted((p for p in paths if p), key=lambda p: p[-1].cost + self.h(p[-1]))
                path = next((p for p in paths if self.keep(p)), [])
            else:
                path = monteCarloProp(node, sdf_state, steer_sigma, accel_mean)
                path = path if self.keep(path) else []
            node.expansions += 1
            if node in tree.active:
                self.push(node)  # back in line, a little further back for each expansion
            child = tree.add(path, node, self.witness_map, self.traj_map) if path else None
            if child is not None:
                self.push(child)
                if self.child_hook is not None:
                    self.child_hook(child)
                if self.h(child) < self.h(node) and self.chain_depth < GREEDY_CHAIN_MAX:
                    self.chain_node, self.chain_depth = child, self.chain_depth + 1
            if self.use_turns and not node.turns_tried:  # stuck: every turn that fits becomes an extra child
                node.turns_tried = True
                for turn in self.turnPaths(node):
                    if self.keep(turn):
                        extra = tree.add(turn, node, self.witness_map, self.traj_map)
                        if extra is not None:
                            self.push(extra)
                            if self.child_hook is not None:
                                self.child_hook(extra)
        else:
            x_rand = self.sample_fn() if self.sample_fn is not None else None
            if x_rand is not None and getattr(x_rand, 'nearest_only', False):
                parent = nearestActive(self.witness_map, x_rand)  # push the frontier toward it, not the cheapest node
            else:
                parent = bestFirstSelectionSST(self.witness_map, sdf_state, self.sample_bias, x_rand)
            path = monteCarloProp(parent, sdf_state, steer_sigma, accel_mean) if parent is not None else []
            path = path if self.keep(path) else []
            if path:
                child = tree.add(path, parent, self.witness_map, self.traj_map)
                if child is not None:
                    self.push(child)
                    if self.child_hook is not None:
                        self.child_hook(child)

        if not tree.goal_nodes:
            if it % 10000 == 0:
                print(f'planGlobal: iteration {it}, no path yet, {len(tree.active)} active nodes', flush=True)
            return
        best = min(g.cost for g in tree.goal_nodes)
        if self.window_start is None:
            print(f'planGlobal: path found at iteration {it}, cost {best:.2f}', flush=True)
            self.window_start, self.window_cost = it, best
        elif it - self.window_start >= ANNEAL_WINDOW:
            gain = (self.window_cost - best) / self.window_cost
            if gain < ANNEAL_MIN_GAIN and delta_B_N * ANNEAL_FACTOR >= DELTA_B_N_MIN:
                delta_B_N *= ANNEAL_FACTOR
                delta_S *= ANNEAL_FACTOR
                rebuilt = WitnessHashMap(delta_S)
                for w in self.witness_map:
                    rebuilt.add(w)
                self.witness_map = rebuilt
                print(f'planGlobal: iteration {it}, best cost {best:.2f} ({100 * gain:.1f}% better in the last '
                      f'{ANNEAL_WINDOW}), annealed to delta_B_N {delta_B_N:.2f}, delta_S {delta_S:.2f}, '
                      f'{len(tree.active)} active nodes', flush=True)
            else:
                print(f'planGlobal: iteration {it}, best cost {best:.2f} ({100 * gain:.1f}% better in the last '
                      f'{ANNEAL_WINDOW}), {len(tree.active)} active nodes', flush=True)
            self.window_start, self.window_cost = it, best

    def run(self, iterations):
        for _ in range(iterations):
            self.step()

    def runUntilPath(self, max_iterations):
        """Step until a goal node exists; returns whether one does."""
        for _ in range(max_iterations):
            if self.tree.goal_nodes:
                return True
            self.step()
        return bool(self.tree.goal_nodes)

    def orphan(self, node):
        """Cut `node` and everything below it off the tree: they become NodeStatus.orphaned, leave the active and
        inactive sets and the goal list, their check points leave traj_map, and any witness they represent is freed.
        Nothing is deleted: nodes are only dropped once preprocessing is over (main keeps just the planned paths)."""
        if node.status is NodeStatus.orphaned:
            return
        parent = node.parent
        if parent is not None and node in parent.children:
            parent.children.remove(node)
        node.parent = None
        for v in node.subtree():
            v.status = NodeStatus.orphaned
            self.tree.active.discard(v)
            self.tree.inactive.discard(v)
            for p in v.traj:
                self.traj_map.remove(p)
            if v.witness is not None and v.witness.rep is v:
                self.witness_map.setRep(v.witness, None)
            if v in self.tree.goal_nodes:
                self.tree.goal_nodes.remove(v)

    def addObstacle(self, obstacle):
        """Add an obstacle shape to the planning map and repair the tree around it: every edge whose check points now
        lack SAFETY_MARGIN of clearance is cut (orphan), the solution labels are redone from the remaining goal nodes,
        h is recomputed and the g + h queue rebuilt. Returns the number of nodes orphaned."""
        # Stamp it in through merge, so obstacle_sdf is refreshed around it
        bx_lo, bx_hi, by_lo, by_hi = obstacle.bounds(0)
        x0, x1 = max(0, math.floor(bx_lo)), min(SCREEN_WIDTH, math.ceil(bx_hi) + 1)
        y0, y1 = max(0, math.floor(by_lo)), min(SCREEN_HEIGHT, math.ceil(by_hi) + 1)
        local = np.full((x1 - x0, y1 - y0), UNKNOWN, dtype=np.int8)
        inside = obstacle.signed_distance(np.arange(x0, x1)[:, None], np.arange(y0, y1)[None, :]) <= 0
        local[inside] = OCCUPIED
        self.sdf_state.merge(local, x0, y0)

        # Edges whose check points can now touch it: only points within reach of its edge can have lost clearance
        reach = obstacle.reach + FOOTPRINT_REACH + FOOTPRINT_RADIUS + SAFETY_MARGIN + CHECK_SPACING
        cut = {}
        for p in self.traj_map.get_nbhd(state(obstacle.x, obstacle.y, 0, 0), reach):
            if p.edge_end is not None and bodyClearance(p.x1, p.x2, p.a, self.sdf_state) < SAFETY_MARGIN:
                cut[id(p.edge_end)] = p.edge_end
        before = sum(1 for v in self.start.subtree())
        for node in cut.values():
            self.orphan(node)
        orphaned = before - sum(1 for v in self.start.subtree())

        # Redo labels, h and the queue; the next path starts a fresh improvement window
        for v in self.start.subtree():
            v.status = NodeStatus.deadEnd
        for g in self.tree.goal_nodes:
            v = g
            while v is not None:
                v.status = NodeStatus.solution
                v = v.parent
        if self.h_fn is None:
            self.h_grid = goalDistanceGrid(self.sdf_state)
        self.open_heap = []
        for v in self.tree.active:
            self.push(v)
        self.chain_node, self.chain_depth = None, 0
        self.window_start, self.window_cost = None, None
        return orphaned


def pathStates(root, goal):
    """The check points from root (included) to goal along the tree."""
    states = []
    node = goal
    while node.parent is not None:
        states = node.traj + states
        node = node.parent
    return [root] + states


def planGlobal(sdf_state, start, max_iterations=MAX_PLAN_ITERATIONS, sample_bias=None):
    """Run a GlobalPlanner for max_iterations steps. Returns (tree, witness_map, traj_map); tree.goal_nodes is empty if
    no path was found."""
    planner = GlobalPlanner(sdf_state, start, sample_bias)
    planner.run(max_iterations)
    best_goal = planner.bestGoal()
    if best_goal is not None:
        print(f'planGlobal: done after {max_iterations} iterations, best cost {best_goal.cost:.2f}, '
              f'driving time {best_goal.time:.2f} s, delta_B_N {delta_B_N:.2f}, {len(planner.tree.active)} active nodes',
              flush=True)
    return planner.tree, planner.witness_map, planner.traj_map


def replanDemo(plan_map, root):
    """Plan, improve for REPLAN_IMPROVE_ITERATIONS, then block the paths: all the circles from before are removed and
    every path found so far gets a fresh circle of BLOCK_RADIUS around a random node of it (away from the start and
    goal), and the same tree plans on. A round that finds no path within REPLAN_MAX_ITERATIONS redraws the circles the
    same way and goes on. Stops once REPLAN_PATHS paths exist or PREPROCESS_MAX_ITERATIONS planner iterations have run,
    whichever comes first. Returns (planner, paths, blocks): paths as lists of states from root, blocks as
    the Obstacle circles last in place."""
    planner = GlobalPlanner(plan_map, root)
    base = plan_map.occupancy.copy()  # the map without any circle, to take old circles back out
    paths, spots, blocks = [], [], []  # spots: for each path, the (x, y) of its nodes a circle may go around
    while len(paths) < REPLAN_PATHS and planner.it < PREPROCESS_MAX_ITERATIONS:
        left = PREPROCESS_MAX_ITERATIONS - planner.it
        if planner.runUntilPath(min(REPLAN_MAX_ITERATIONS, left)):
            planner.run(min(REPLAN_IMPROVE_ITERATIONS, PREPROCESS_MAX_ITERATIONS - planner.it))
            goal = planner.bestGoal()
            paths.append(pathStates(root, goal))
            print(f'replanDemo: path {len(paths)}: {goal.time:.2f} s (iteration {planner.it})', flush=True)
            if len(paths) == REPLAN_PATHS or planner.it >= PREPROCESS_MAX_ITERATIONS:
                break
            nodes = []
            node = goal
            while node is not None:
                nodes.append((node.x1, node.x2))
                node = node.parent
            spots.append([(x, y) for x, y in nodes if math.hypot(x - START[0], y - START[1]) > BLOCK_RADIUS + 60
                          and math.hypot(x - TARGET[0], y - TARGET[1]) > BLOCK_RADIUS + TARGET_RAD + 40])
        elif planner.it >= PREPROCESS_MAX_ITERATIONS:
            break
        else:
            print(f'replanDemo: no path in {REPLAN_MAX_ITERATIONS} iterations (iteration {planner.it}); redrawing the '
                  'circles', flush=True)
        if not spots:  # nothing to block yet (no path so far): just keep planning
            continue
        # Take the old circles out (the tree cut around them stays cut; the space is free to grow back into)
        for block in blocks:
            bx_lo, bx_hi, by_lo, by_hi = block.bounds(0)
            x0, x1 = max(0, math.floor(bx_lo)), min(SCREEN_WIDTH, math.ceil(bx_hi) + 1)
            y0, y1 = max(0, math.floor(by_lo)), min(SCREEN_HEIGHT, math.ceil(by_hi) + 1)
            plan_map.occupancy[x0:x1, y0:y1] = base[x0:x1, y0:y1]
            plan_map.recompute(x0, x1, y0, y1)
        plan_map.version += 1
        # A fresh circle on every path so far
        blocks = [Obstacle(*random.choice(spot), BLOCK_RADIUS) for spot in spots if spot]
        if not blocks:
            print('replanDemo: no node on any path is far enough from the start and goal to block; stopping', flush=True)
            break
        orphaned = sum(planner.addObstacle(block) for block in blocks)
        print(f'replanDemo: {len(blocks)} circles redrawn at ' + ', '.join(f'({b.x:.0f}, {b.y:.0f})' for b in blocks)
              + f', {orphaned} nodes orphaned, {len(planner.tree.goal_nodes)} goal nodes left', flush=True)
    return planner, paths, blocks


def routeLoopy(route, a):
    """Whether a stretch of `route` starting at node index a and ending at a later node at most LOOP_SPAN px further
    along is a loop or big wiggle: the distance driven along it is at least LOOP_DETOUR_RATIO times the straight line
    between its ends, or its heading turns through at least LOOP_TURN in total."""
    nodes = route.nodes
    k = nodes.index(a)
    p = route.states[a]
    turning = 0.0
    last = a
    for c in nodes[k + 1:]:
        if route.arc[c] - route.arc[a] > LOOP_SPAN:
            break
        turning += sum(abs((route.states[i + 1].a - route.states[i].a + math.pi) % (2 * math.pi) - math.pi)
                       for i in range(last, c))
        last = c
        q = route.states[c]
        if turning >= LOOP_TURN or route.arc[c] - route.arc[a] >= LOOP_DETOUR_RATIO * max(math.hypot(q.x1 - p.x1, q.x2 - p.x2), 1.0):
            return True
    return False


def hermiteCurve(p, q, tangent):
    """Dense (n, 2) points and their arc lengths along the cubic Hermite curve from p's position and heading to q's,
    with both end tangents `tangent` times the straight distance between them."""
    d = math.hypot(q.x1 - p.x1, q.x2 - p.x2)
    t0 = np.array([math.cos(p.a), math.sin(p.a)]) * tangent * d
    t1 = np.array([math.cos(q.a), math.sin(q.a)]) * tangent * d
    u = np.linspace(0, 1, max(20, int(d / 2)))[:, None]
    points = ((2 * u ** 3 - 3 * u ** 2 + 1) * np.array([p.x1, p.x2]) + (u ** 3 - 2 * u ** 2 + u) * t0
              + (-2 * u ** 3 + 3 * u ** 2) * np.array([q.x1, q.x2]) + (u ** 3 - u ** 2) * t1)
    arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))))
    return points, arcs


def routeJoin(n, route, columns, lo, hi, scale):
    """Where state n joins the route between indices lo and hi: the state i such that n lies between states i-1 and i
    along the route, within `scale` times joinTolerance of state i (sideways to the line from i-1 to i, heading,
    steering; same direction, at most JOIN_SPEED_OVER faster). Returns (i, cost of the gap from n on to state i), or
    None. Requiring n to lie between the two states, not just beside the line, stops a join from skipping ahead along
    the route. columns: the route's (x, y, heading, speed, steer) as numpy arrays."""
    side_tol, heading_tol, steer_tol = (scale * t for t in joinTolerance(n.s))
    x, y = columns[0][lo:hi], columns[1][lo:hi]
    close = np.flatnonzero(np.hypot(x - n.x1, y - n.x2) <= side_tol + CHECK_SPACING)
    for k in close:
        i = lo + int(k)
        if i == 0:
            continue
        p, q = route.states[i], route.states[i - 1]
        if n.s * p.s < 0 or abs(n.s) > abs(p.s) + JOIN_SPEED_OVER:
            continue
        if abs((n.a - p.a + math.pi) % (2 * math.pi) - math.pi) > heading_tol or abs(n.steer - p.steer) > steer_tol:
            continue
        dx, dy = p.x1 - q.x1, p.x2 - q.x2
        seg = math.hypot(dx, dy)
        if seg == 0:
            continue
        along = ((n.x1 - q.x1) * dx + (n.x2 - q.x2) * dy) / seg
        side = abs((n.x1 - q.x1) * dy - (n.x2 - q.x2) * dx) / seg
        if 0 <= along <= seg and side <= side_tol:
            gap = (seg - along) / max(abs(p.s), 1.0) * (1 + COST_STEER_ANGLE * abs(p.steer))
            return i, gap
    return None


def shortcutDrive(p, q, route, columns, a, hi, tangent, lookahead, bump, sdf_state, scale=None):
    """Drive from route state p (index a) along a Hermite curve to route state q with pure pursuit (lookahead px ahead
    on the curve), speed going linearly from p's to q's plus bump times the faster of them in the middle, the steering
    blended toward q's over the last SMOOTH_STEER_BLEND of the curve. Every PLAN_DT step goes through propagate, so the
    planner's limits and collision checks apply. Once past half way, each step is checked for a join to the route
    (routeJoin within `scale` of the join tolerance, SMOOTH_TOLERANCE_SCALE by default, indices a+1 to hi). Returns (cost to the goal, states, join index, gap
    cost) for the cheapest join, or None."""
    points, arcs = hermiteCurve(p, q, tangent)
    length = arcs[-1]
    if length < 1:
        return None
    x = state(p.x1, p.x2, p.a, p.s, steer=p.steer)
    top = max(abs(p.s), abs(q.s))
    states, best, k = [], None, 0
    max_steps = int(1.5 * length / max(min(abs(p.s), abs(q.s)), 20) / PLAN_DT) + 40
    for step in range(max_steps):
        # Progress: the nearest curve point, searching forward from the last one
        window = points[k:k + 40]
        k += int(np.argmin(np.hypot(window[:, 0] - x.x1, window[:, 1] - x.x2)))
        f = arcs[k] / length
        if f >= 0.999 and step > 0 and best is not None:
            break
        j = min(int(np.searchsorted(arcs, arcs[k] + lookahead)), len(points) - 1)
        tx, ty = points[j]
        if j == len(points) - 1:  # past the curve: aim along q's heading
            extra = arcs[k] + lookahead - length
            tx, ty = tx + extra * math.cos(q.a), ty + extra * math.sin(q.a)
        alpha = math.atan2(ty - x.x2, tx - x.x1) - x.a
        dist = max(math.hypot(tx - x.x1, ty - x.x2), 1e-6)
        desired = math.atan(2 * math.sin(alpha) * WHEELBASE / dist)
        w = clamp((f - (1 - SMOOTH_STEER_BLEND)) / SMOOTH_STEER_BLEND, 0, 1)
        desired = (1 - w) * desired + w * q.steer
        v_ref = abs(p.s) + (abs(q.s) - abs(p.s)) * f + bump * top * math.sin(math.pi * f)
        v_ref = max(v_ref, SMOOTH_MIN_SPEED if f < 0.9 else abs(q.s))
        accel, lo, hi_rate = feasibleAccel(x, PLAN_DT, (v_ref - x.s) / PLAN_DT)
        limit = maxSteerAtSpeed(max(abs(x.s), abs(x.s + accel * PLAN_DT)))
        desired = clamp(desired, -limit, limit)
        seg = propagate(x, action(clamp((desired - x.steer) / PLAN_DT, lo, max(lo, hi_rate)), accel), 1, sdf_state)
        if not seg:
            return best
        if step % NUDGE_NODE_FRAMES == NUDGE_NODE_FRAMES - 1:
            seg[-1].edge_end = seg[-1]
        states += seg
        x = seg[-1]
        if f >= 0.5:
            join = routeJoin(x, route, columns, a + 1, hi, SMOOTH_TOLERANCE_SCALE if scale is None else scale)
            if join is not None:
                i, gap = join
                total = x.cost + gap + route.remaining[i]
                if best is None or total < best[0]:
                    best = (total, list(states), i, gap)
    return best


def smoothRoute(route, sdf_state):
    """One sweep over the route's nodes: from each, try shortcutDrive to route states up to SMOOTH_NODES_AHEAD nodes
    ahead (every SMOOTH_TARGET_SPACING px, plus the nodes themselves; out to SMOOTH_LOOP_SPAN px if the stretch is loopy)
    with every combination of SMOOTH_TANGENTS, SMOOTH_LOOKAHEADS and SMOOTH_BUMPS, and splice in the cheapest result if
    it's at least SMOOTH_MIN_GAIN cheaper to the goal (then try the same node again). Returns (route, splices made)."""
    made, k = 0, 0
    tries_here = 0
    while k < len(route.nodes) - 1:
        a = route.nodes[k]
        p = route.states[a]
        if p.s < 0:  # reversing: leave it
            k, tries_here = k + 1, 0
            continue
        ahead = route.nodes[k + 1:k + 1 + SMOOTH_NODES_AHEAD]
        end = ahead[-1]
        if routeLoopy(route, a):
            while end + 1 < len(route.states) and route.arc[end + 1] - route.arc[a] <= SMOOTH_LOOP_SPAN:
                end += 1
        targets, last_arc = set(ahead), route.arc[a]
        for i in range(a + 1, end + 1):
            if route.arc[i] - last_arc >= SMOOTH_TARGET_SPACING:
                targets.add(i)
                last_arc = route.arc[i]
        columns = tuple(np.array(c) for c in zip(*((s.x1, s.x2, s.a, s.s, s.steer) for s in route.states)))
        hi = min(len(route.states), end + SMOOTH_JOIN_SLACK)
        best = None
        for t in sorted(targets):
            q = route.states[t]
            if q.s < 0:
                continue
            for tangent in SMOOTH_TANGENTS:
                for lookahead in SMOOTH_LOOKAHEADS:
                    for bump in SMOOTH_BUMPS:
                        found = shortcutDrive(p, q, route, columns, a, hi, tangent, lookahead, bump, sdf_state)
                        if found is not None and (best is None or found[0] < best[0]):
                            best = found
        if best is not None and best[0] < route.remaining[a] * (1 - SMOOTH_MIN_GAIN):
            total, states, j, gap = best
            states[-1].edge_end = states[-1]
            states[-1].cost += gap  # the bit from the shortcut's end on to state j, so the route's costs stay right
            route = spliceRoute(route, a, states, route, j)
            made += 1
            tries_here += 1
            if tries_here < SMOOTH_RETRIES:
                continue  # same node again, on the new route
        k, tries_here = k + 1, 0
    return route, made


def smoothPath(path, sdf_state):
    """Smooth a planned path (states from the root) with smoothRoute sweeps until one makes no change or
    SMOOTH_SWEEPS have run. Returns the new states, copied so their costs (and times) run consistently along the path,
    as Route expects."""
    route, before = Route(path), Route(path)
    for sweep in range(SMOOTH_SWEEPS):
        route, made = smoothRoute(route, sdf_state)
        if not made:
            break
    # Materialize: copies with cost = cost so far (from the remaining costs) and time from speed along the arc
    states, t = [], 0.0
    for i, s in enumerate(route.states):
        c = copy.copy(s)
        if getattr(s, 'edge_end', None) is s:
            c.edge_end = c
        c.cost = route.remaining[0] - route.remaining[i]
        if i:
            ds = route.arc[i] - route.arc[i - 1]
            t += ds / max((abs(s.s) + abs(route.states[i - 1].s)) / 2, 1.0)
        c.time = t
        states.append(c)
    print(f'smoothPath: cost {before.remaining[0]:.2f} -> {route.remaining[0]:.2f}, '
          f'length {before.arc[-1]:.0f} -> {route.arc[-1]:.0f} px, {len(route.states)} states', flush=True)
    return states


def savePathLibrary(paths, filename=PATH_LIBRARY_FILE):
    """Save paths (lists of states) to an .npz file: per path an (n, 8) array of x, y, heading, speed, steering, cost,
    time and whether the state is a tree node (edge end), which is all Route and the Driver use."""
    arrays = {f'path_{k}': np.array([(p.x1, p.x2, p.a, p.s, p.steer, p.cost, p.time,
                                      float(i == 0 or i == len(path) - 1 or getattr(p, 'edge_end', None) is p))
                                     for i, p in enumerate(path)]) for k, path in enumerate(paths)}
    np.savez_compressed(filename, **arrays)


def loadPathLibrary(filename=PATH_LIBRARY_FILE):
    """The paths saved by savePathLibrary, as lists of states (nodes marked as edge ends again), or [] if there is no
    such file."""
    if not os.path.exists(filename):
        return []
    with np.load(filename) as data:
        rows = [data[f'path_{k}'] for k in range(len(data.files))]
    paths = []
    for table in rows:
        path = []
        for x, y, a, speed, steer, cost, t, is_node in table:
            p = state(float(x), float(y), float(a), float(speed), cost=float(cost), steer=float(steer), time=float(t))
            p.edge_end = p if is_node else None
            path.append(p)
        paths.append(path)
    return paths


def knownClearance(x, y, a, sdf_state):
    """Like bodyClearance, but unknown space counts as an obstacle too (min of obstacle_sdf and unknown_sdf)."""
    return _known_clearance(float(x), float(y), float(a), sdf_state.obstacle_sdf, sdf_state.unknown_sdf,
                            FOOTPRINT_OFFSETS_ARRAY, FOOTPRINT_RADIUS)


def footprintClearances(points, sdf_state, unknown_too=False):
    """bodyClearance (or with unknown_too, knownClearance) for each pose of an (n, 3) array of rear-axle poses (x, y,
    heading): the gap between the footprint circles and the nearest obstacle (or unknown space) at each. Off-map is 0."""
    return _footprint_clearances(np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3), sdf_state.obstacle_sdf,
                                 sdf_state.unknown_sdf, unknown_too, FOOTPRINT_OFFSETS_ARRAY, FOOTPRINT_RADIUS)


def posesClear(points, sdf_state, unknown_too=False):
    """Whether every pose of an (n, 3) array keeps SAFETY_MARGIN (footprintClearances >= it), stopping at the first that
    doesn't."""
    return _all_clear(np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3), sdf_state.obstacle_sdf,
                      sdf_state.unknown_sdf, unknown_too, FOOTPRINT_OFFSETS_ARRAY, FOOTPRINT_RADIUS, SAFETY_MARGIN)


def canStopFrom(x, y, a, speed, steer, sdf_state):
    """Whether the car, at this pose, speed (signed) and steering, can brake to a stop keeping SAFETY_MARGIN from
    obstacles and unknown space with one of STOP_TRAJECTORIES' manoeuvres (gentlest first). The pose itself must be
    clear too."""
    if knownClearance(x, y, a, sdf_state) < SAFETY_MARGIN:
        return False
    s = state(x, y, a, speed, steer=steer)
    for maneuver in STOP_TRAJECTORIES.lookup(s):
        if posesClear(maneuver.worldPoints(s), sdf_state, unknown_too=True):
            return True
    return False


def safeSpeed(p, sdf_state):
    """The highest of p's planned speed and SAFE_SPEED_STEP steps below it (down to 0) from which the car can still
    stop before obstacles and unknown space (canStopFrom), found by bisection (slower never stops later). 0 if even
    stopping at p isn't safe, i.e. p is not in known free space."""
    top = abs(p.s)
    speeds = [top] + [v for v in range(int(top // SAFE_SPEED_STEP) * SAFE_SPEED_STEP, -1, -SAFE_SPEED_STEP) if v < top]
    if speeds[-1] != 0:
        speeds.append(0)
    sign = -1 if p.s < 0 else 1
    ok = lambda i: canStopFrom(p.x1, p.x2, p.a, sign * speeds[i], p.steer, sdf_state)
    if not ok(len(speeds) - 1):
        return 0
    lo, hi = 0, len(speeds) - 1  # speeds[hi] is safe; find the first (fastest) safe one
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid + 1
    return speeds[lo]


def canStopClear(node, sdf_state):
    """Whether the car can brake to a stop from `node` with one of STOP_TRAJECTORIES' manoeuvres (gentlest first) while
    keeping SAFETY_MARGIN from known obstacles (unknown space counts as free, as in planning)."""
    for maneuver in STOP_TRAJECTORIES.lookup(node):
        if posesClear(maneuver.worldPoints(node), sdf_state):
            return True
    return False


def runTurnaround(start, phases, sdf_state, max_frames=3000):
    """Drive one of TURNAROUND_MANEUVERS from `start`, one PLAN_DT step at a time through propagate (so the steering-rate,
    lateral and collision limits are the planner's own): for each phase, stop, set the wheel in place, then drive until
    the total heading change reaches the phase's target; finally stop and straighten the wheel in place. Returns the
    states, or [] if it hits something or runs out of frames."""
    x, states, turned, frames = start, [], 0.0, 0

    def step(target_speed, target_steer):
        nonlocal x, turned, frames
        frames += 1
        if frames > max_frames:
            return False
        accel, lo, hi = feasibleAccel(x, PLAN_DT, (target_speed - x.s) / PLAN_DT)
        seg = propagate(x, action(clamp((target_steer - x.steer) / PLAN_DT, lo, max(lo, hi)), accel), 1, sdf_state)
        if not seg:
            return False
        turned += (seg[-1].a - x.a + math.pi) % (2 * math.pi) - math.pi
        if frames % NUDGE_NODE_FRAMES == 0:
            seg[-1].edge_end = seg[-1]  # a node, so the safe-speed look-ahead has points along it
        states.extend(seg)
        x = seg[-1]
        return True

    def stop_and_set_wheel(target_steer):
        while abs(x.s) > 1e-3 or abs(x.steer - target_steer) > 1e-3:
            if not step(0.0, target_steer):
                return False
        return True

    for steer_fraction, speed, until in phases:
        wheel = steer_fraction * maxSteerAtSpeed(abs(speed))
        if not stop_and_set_wheel(wheel):
            return []
        # Stop driving early by the turn the car will still make while braking to a stop at this lock
        while abs(turned) + x.s * x.s / (2 * MAX_ACCELERATION) * abs(math.tan(x.steer)) / WHEELBASE < until:
            if not step(speed, wheel):
                return []
    if not stop_and_set_wheel(0.0):
        return []
    if states:
        states[-1].edge_end = states[-1]
    return states


def turnCircleRadius(speed):
    """Radius of free circle the car needs, entering at speed, to come out at any heading: the fit to the measured
    table (about 23.5 px when slow, then braking distance / 2 + 9 px)."""
    return max(23.5, speed * speed / (4 * MAX_ACCELERATION) + 9)


def libRange(speed):
    """How far (px) from its start the manoeuvre library explores for a start speed: across the circle it needs."""
    return max(60.0, 2 * turnCircleRadius(speed) + 10)


def freeRadius(sdf_state, cx, cy, max_radius):
    """Radius of the known-free circle around (cx, cy): the distance to the nearest cell that isn't known FREE (unknown
    counts as blocked; so does the map's edge), at most max_radius. (obstacle_sdf is capped at SDF_CAP, too small.)"""
    x0, x1 = math.floor(cx - max_radius), math.ceil(cx + max_radius) + 1
    y0, y1 = math.floor(cy - max_radius), math.ceil(cy + max_radius) + 1
    edge = min(cx, cy, SCREEN_WIDTH - 1 - cx, SCREEN_HEIGHT - 1 - cy)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, SCREEN_WIDTH), min(y1, SCREEN_HEIGHT)
    blocked = np.nonzero(sdf_state.occupancy[x0:x1, y0:y1] != FREE)
    if len(blocked[0]) == 0:
        return max(0.0, min(max_radius, edge))
    return max(0.0, min(max_radius, edge, float(np.hypot(blocked[0] + x0 - cx, blocked[1] + y0 - cy).min())))


LIB_ACCELS = np.array([-1.0, 0.0, 1.0]) * MAX_ACCELERATION
LIB_RATES = np.array([-1.0, 0.0, 1.0]) * MAX_STEER_RATE


def libControls(code, depth):
    """The (acceleration, steering rate) of each step of a library sequence, first step first."""
    steps = []
    for _ in range(depth):
        code, c = divmod(code, 9)
        steps.append((LIB_ACCELS[c // 3], LIB_RATES[c % 3]))
    return steps[::-1]


def libStep(x, y, a, v, s, acc, rate):
    """Per car: LIB_STEP_FRAMES PLAN_DT frames of (acc, rate), the steering kept within the lateral limit and the speed
    within [0, MAX_SPEED], integrated like propagate. Returns new arrays (the inputs are left alone)."""
    x, y, a, v, s = (np.array(q, dtype=np.float64) for q in (x, y, a, v, s))
    n = len(x)
    acc = np.broadcast_to(np.asarray(acc, dtype=np.float64), (n,)).copy()
    rate = np.broadcast_to(np.asarray(rate, dtype=np.float64), (n,)).copy()
    _lib_step(x, y, a, v, s, acc, rate, LIB_STEP_FRAMES, PLAN_DT, MAX_SPEED, MAX_STEER, MAX_LATERAL_ACCELERATION,
              WHEELBASE)
    return x, y, a, v, s


def libKey(x, y, a):
    """Lookup key of end positions and headings (numpy arrays or scalars)."""
    xb = np.round(np.asarray(x) / LIB_KEY_CELL[0]).astype(np.int64) + 10000
    yb = np.round(np.asarray(y) / LIB_KEY_CELL[0]).astype(np.int64) + 10000
    ab = np.round((np.asarray(a) % (2 * np.pi)) / LIB_KEY_CELL[1]).astype(np.int64) % round(2 * np.pi / LIB_KEY_CELL[1])
    return (xb * 20001 + yb) * 64 + ab


def buildManeuverLibrary():
    """Explore the library's sequences for every LIB_SPEEDS start (see MANEUVER_LIBRARY_FILE's comment). Returns a dict
    of arrays per start speed: end states (x, y, heading, speed, steering, relative to a start at the origin facing +x
    with the wheel straight), sequence codes (base 9, first step most significant), depths, and lookup keys, sorted by
    key."""
    bx, ba, bv, bs = LIB_BINS
    def dedup_key(x, y, a, v, s):
        return ((((np.round(x / bx).astype(np.int64) + 1000) * 2001 + np.round(y / bx).astype(np.int64) + 1000) * 64
                 + np.round((a % (2 * np.pi)) / ba).astype(np.int64) % 36) * 32 + np.round(v / bv).astype(np.int64)) * 16 \
               + np.round(s / bs).astype(np.int64) + 8
    library = {}
    for v0 in LIB_SPEEDS:
        reach = libRange(v0)
        x, y, a, s = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        v = np.full(1, float(v0))
        code, depth = np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64)
        visited = dedup_key(x, y, a, v, s)
        ends, codes, depths = [], [], []
        for d in range(1, LIB_DEPTH + 1):
            n = len(x)
            if n == 0:
                break
            X, Y, A, V, S = (np.repeat(q, 9) for q in (x, y, a, v, s))
            c = np.tile(np.arange(9), n)
            X, Y, A, V, S = libStep(X, Y, A, V, S, LIB_ACCELS[c // 3], LIB_RATES[c % 3])
            C = np.repeat(code, 9) * 9 + c
            ok = (V >= LIB_MIN_SPEED) & (np.hypot(X, Y) <= reach)
            X, Y, A, V, S, C = X[ok], Y[ok], A[ok], V[ok], S[ok], C[ok]
            K, first = np.unique(dedup_key(X, Y, A, V, S), return_index=True)
            fresh = ~np.isin(K, visited)
            keep = first[fresh]
            x, y, a, v, s, code = X[keep], Y[keep], A[keep], V[keep], S[keep], C[keep]
            visited = np.union1d(visited, K[fresh])
            ends.append(np.stack([x, y, a % (2 * np.pi), v, s], axis=1))
            codes.append(code)
            depths.append(np.full(len(code), d))
        end = np.concatenate(ends) if ends else np.empty((0, 5))
        code = np.concatenate(codes) if codes else np.empty(0, dtype=np.int64)
        depth = np.concatenate(depths) if depths else np.empty(0, dtype=np.int64)
        key = libKey(end[:, 0], end[:, 1], end[:, 2])
        order = np.argsort(key, kind='stable')
        library[v0] = dict(end=end[order].astype(np.float32), code=code[order], depth=depth[order].astype(np.uint8),
                           key=key[order])
        print(f'maneuverLibrary: start speed {v0} px/s: {len(key)} manoeuvres', flush=True)
    return library


_MANEUVER_LIBRARY = None


def maneuverLibrary():
    """The manoeuvre library, loaded from MANEUVER_LIBRARY_FILE, or built (about a minute) and saved there the first
    time."""
    global _MANEUVER_LIBRARY
    if _MANEUVER_LIBRARY is None:
        if os.path.exists(MANEUVER_LIBRARY_FILE):
            with np.load(MANEUVER_LIBRARY_FILE) as data:
                _MANEUVER_LIBRARY = {v0: {f: data[f'v{v0}_{f}'] for f in ('end', 'code', 'depth', 'key')}
                                     for v0 in LIB_SPEEDS}
        else:
            print('maneuverLibrary: building it (once)...', flush=True)
            _MANEUVER_LIBRARY = buildManeuverLibrary()
            np.savez_compressed(MANEUVER_LIBRARY_FILE, **{f'v{v0}_{f}': arr for v0, entry in _MANEUVER_LIBRARY.items()
                                                          for f, arr in entry.items()})
            print(f'maneuverLibrary: saved to {MANEUVER_LIBRARY_FILE}', flush=True)
    return _MANEUVER_LIBRARY


def libraryJoin(start, route, i, sdf_state, columns=None):
    """A library manoeuvre from state `start` (moving forward at most LIB_SPEEDS[-1] + 5 px/s) onto `route` near state i,
    or None. The known-free circle around the midpoint of the two (freeRadius) must be at least turnCircleRadius for
    the speed and hold both ends, and the manoeuvre must stay inside it. Candidates are the entries whose predicted end
    (start rotated in) lies within a lookup bin of route state i + LIB_TARGET_OFFSETS; they're ranked by time plus the
    time later lost accelerating from their end speed to the route's (so joining at the highest safe speed up to the
    route's wins unless it costs more than it saves), then the best LIB_CANDIDATES are re-driven from the real start
    through propagate (steering kept within the lateral limit) until one joins the route within the live join
    tolerance (routeJoin, around those states) at a state the car can stop from. Returns (cost of the manoeuvre to the
    join plus the gap, states, join index) or None."""
    if start.s < 0 or start.s > LIB_SPEEDS[-1] + 5:
        return None
    q = route.states[i]
    dist = math.hypot(q.x1 - start.x1, q.x2 - start.x2)
    reach = libRange(start.s)
    if q.s <= 0 or dist > reach:
        return None
    pad = FOOTPRINT_RADIUS + SAFETY_MARGIN
    cx, cy = (start.x1 + q.x1) / 2, (start.x2 + q.x2) / 2
    radius = freeRadius(sdf_state, cx, cy, reach)
    if radius < max(turnCircleRadius(start.s), dist / 2 + FOOTPRINT_REACH + pad):
        return None
    lib = maneuverLibrary()[min(LIB_SPEEDS, key=lambda v0: abs(v0 - start.s))]
    c, sn = math.cos(start.a), math.sin(start.a)
    cell, acell = LIB_KEY_CELL
    first = max(route.valid_from, 1)
    targets = sorted({t for t in (i + o for o in LIB_TARGET_OFFSETS) if first <= t < len(route.states)})
    picks, wants = [], []
    for t in targets:
        p = route.states[t]
        if p.s <= 0:
            continue
        dx, dy = p.x1 - start.x1, p.x2 - start.x2
        tx, ty, ta = dx * c + dy * sn, -dx * sn + dy * c, (p.a - start.a) % (2 * math.pi)
        for key in {int(libKey(tx + a_ * cell, ty + b_ * cell, ta + k_ * acell))
                    for a_ in (-1, 0, 1) for b_ in (-1, 0, 1) for k_ in (-1, 0, 1)}:
            lo, hi = np.searchsorted(lib['key'], key), np.searchsorted(lib['key'], key, side='right')
            picks.extend(range(lo, hi))
            wants.extend([(tx, ty, ta, p.s, p.steer)] * (hi - lo))
    if not picks:
        return None
    picks, wants = np.array(picks), np.array(wants)
    end = lib['end'][picks].astype(float)
    steer_tol = np.array([2 * joinTolerance(v)[2] for v in wants[:, 3]]) + LIB_BINS[3]
    ok = (end[:, 3] <= wants[:, 3] + JOIN_SPEED_OVER) & (np.abs(end[:, 4] - wants[:, 4]) <= steer_tol)
    if not ok.any():
        return None
    picks, wants, end = picks[ok], wants[ok], end[ok]
    duration = lib['depth'][picks].astype(float) * LIB_STEP_FRAMES * PLAN_DT
    lost = np.maximum(wants[:, 3] - end[:, 3], 0) ** 2 / (2 * MAX_ACCELERATION * np.maximum(wants[:, 3], 1.0))
    miss = (np.hypot(end[:, 0] - wants[:, 0], end[:, 1] - wants[:, 1]) / MAX_SPEED
            + np.abs((end[:, 2] - wants[:, 2] + np.pi) % (2 * np.pi) - np.pi) * 0.1)
    _, unique = np.unique(picks, return_index=True)  # an entry found for two targets is tried once
    order = unique[np.argsort((duration + lost + miss)[unique])][:LIB_CANDIDATES]
    if columns is None:
        columns = tuple(np.array(col) for col in zip(*((p.x1, p.x2, p.a, p.s, p.steer) for p in route.states)))
    lo_i = max(targets[0] - RECONNECT_JOIN_WINDOW[0], first) - 1
    hi_i = min(len(route.states), targets[-1] + RECONNECT_JOIN_WINDOW[1])
    best = None
    for k in order:
        steps = libControls(int(lib['code'][picks[k]]), int(lib['depth'][picks[k]]))
        x = state(start.x1, start.x2, start.a, start.s, cost=start.cost, steer=start.steer, time=start.time)
        states, found, frames = [], None, 0
        total_frames = len(steps) * LIB_STEP_FRAMES
        for acc, rate in steps:
            for _ in range(LIB_STEP_FRAMES):
                frames += 1
                acc_f = clamp(acc, -x.s / PLAN_DT, (MAX_SPEED - x.s) / PLAN_DT)
                limit = maxSteerAtSpeed(max(x.s, x.s + acc_f * PLAN_DT))
                rate_f = clamp(rate, (-limit - x.steer) / PLAN_DT, (limit - x.steer) / PLAN_DT)
                seg = propagate(x, action(rate_f, acc_f), 1, sdf_state)
                if not seg:
                    break
                p = seg[-1]
                if any(math.hypot(p.x1 + o * math.cos(p.a) - cx, p.x2 + o * math.sin(p.a) - cy) > radius - pad
                       for o in FOOTPRINT_OFFSETS):
                    seg = []
                    break
                if frames % NUDGE_NODE_FRAMES == 0:
                    p.edge_end = p
                states += seg
                x = p
                if frames >= total_frames // 2:
                    join = routeJoin(x, route, columns, lo_i, hi_i, JOIN_TOLERANCE_SCALE)
                    if join is not None and canStopClear(x, sdf_state):
                        found = join
                        break
            if not seg or found is not None:
                break
        if found is None:
            continue
        j, gap = found
        cost = x.cost - start.cost + gap
        if best is None or cost + route.remaining[j] < best[0] + route.remaining[best[2]]:
            best = (cost, states, j, gap)
    if best is None:
        return None
    cost, states, j, gap = best
    states[-1].edge_end = states[-1]
    states[-1].cost += gap
    return cost, states, j


def joinTolerance(speed):
    """How far a replanned state may be from a route and still join it, so an MPC can take over without leaving the
    SAFETY_MARGIN: (sideways px, heading rad, steering rad). Correcting a heading error theta at the sharpest allowed
    curvature drifts the car about theta^2 / (2 curvature) sideways; a steering error takes |error| / MAX_STEER_RATE
    to remove, during which the heading drifts about speed * error^2 / (2 WHEELBASE MAX_STEER_RATE)."""
    v = max(abs(speed), 1e-6)
    curvature = min(math.tan(MAX_STEER) / WHEELBASE, MAX_LATERAL_ACCELERATION / (v * v))
    heading = min(math.pi, math.sqrt(2 * curvature * SAFETY_MARGIN))
    steering = math.sqrt(heading * WHEELBASE * MAX_STEER_RATE / v)
    return SAFETY_MARGIN, heading, steering


class Route:
    """A drivable path: states (the start, then check points) into the goal region, with which of them are tree nodes
    (edge ends), the arc length along it, and the remaining cost from each state to its end."""

    def __init__(self, states, remaining=None):
        self.states = states
        n = len(states)
        self.nodes = [i for i, p in enumerate(states) if i == 0 or i == n - 1 or getattr(p, 'edge_end', None) is p]
        self.arc = [0.0]
        for a, b in zip(states, states[1:]):
            self.arc.append(self.arc[-1] + math.hypot(b.x1 - a.x1, b.x2 - a.x2))
        self.remaining = remaining if remaining is not None else [states[-1].cost - p.cost for p in states]
        self.valid_from = 0  # states before this are cut off by an obstacle; the rest still reach the goal

    def blocked(self, sdf_state, unknown_too=False):
        """Boolean array: which states now lack SAFETY_MARGIN of clearance from obstacles (vectorized bodyClearance), or
        with unknown_too from obstacles or unknown space (vectorized knownClearance)."""
        x = np.array([p.x1 for p in self.states])
        y = np.array([p.x2 for p in self.states])
        a = np.array([p.a for p in self.states])
        offsets = np.array(FOOTPRINT_OFFSETS)
        px = np.rint(x[:, None] + offsets[None, :] * np.cos(a)[:, None]).astype(int)
        py = np.rint(y[:, None] + offsets[None, :] * np.sin(a)[:, None]).astype(int)
        on_map = (px >= 0) & (px < SCREEN_WIDTH) & (py >= 0) & (py < SCREEN_HEIGHT)
        clearance = np.zeros(px.shape)
        field = np.minimum(sdf_state.obstacle_sdf, sdf_state.unknown_sdf) if unknown_too else sdf_state.obstacle_sdf
        clearance[on_map] = field[px[on_map], py[on_map]] - FOOTPRINT_RADIUS
        return (clearance < SAFETY_MARGIN).any(axis=1)

    def indexAt(self, s):
        """Index of the last state at or before arc length s."""
        return max(0, bisect.bisect_right(self.arc, s) - 1)

    def poseAt(self, s):
        """(x, y, heading, planned speed, steering) at arc length s, interpolated."""
        i = min(self.indexAt(s), len(self.states) - 2)
        a, b = self.states[i], self.states[i + 1]
        length = self.arc[i + 1] - self.arc[i]
        f = 0 if length == 0 else clamp((s - self.arc[i]) / length, 0, 1)
        dh = (b.a - a.a + math.pi) % (2 * math.pi) - math.pi
        return (a.x1 + f * (b.x1 - a.x1), a.x2 + f * (b.x2 - a.x2), a.a + f * dh,
                a.s + f * (b.s - a.s), a.steer + f * (b.steer - a.steer))


def spliceRoute(prefix, upto, new_states, join_route, join_index):
    """A Route that follows prefix.states[:upto + 1], then new_states (planned from prefix.states[upto], costs counted
    from 0), then join_route's states after join_index (or nothing if join_route is None: new_states reach the goal)."""
    tail = join_route.states[join_index + 1:] if join_route is not None else []
    tail_rem = join_route.remaining[join_index + 1:] if join_route is not None else []
    join_rem = join_route.remaining[join_index] if join_route is not None else 0
    new_end = new_states[-1].cost if new_states else 0
    new_rem = [new_end - p.cost + join_rem for p in new_states]
    head = prefix.states[:upto + 1]
    head_rem = [prefix.remaining[i] - prefix.remaining[upto] + new_end + join_rem for i in range(upto + 1)]
    return Route(head + new_states + tail, head_rem + new_rem + list(tail_rem))


class Driver:
    """Drives the ghost car along the best Route while the lidar (scanned from the ghost) reveals the world.

    Speed: the next SAFE_SPEED_NODES tree nodes each get a safeSpeed (cached per map version); the car drives at
    min(planned speed, min over those nodes of sqrt(safe^2 + 2 a distance)), so it can always brake to their safe speeds
    in time, and goes back up to the planned speed whenever that allows. It can also always stop before the first route
    state ahead in an obstacle or unknown space.

    When a scan adds obstacles, every route's states are re-checked. If the current route is blocked ahead, replanning
    starts right away: from R (REPLAN_BACK_NODES nodes before L, the last node before the blocked part, or the node just
    before L if the car is already past that) if the car hasn't reached it,
    with the car able to stop at L meanwhile; otherwise from the point where the car will stop, at rest. If the car
    reaches R without a new route, replanning restarts at once from L at rest while the car brakes there.

    The replanner gets REPLAN_ITERATIONS_PER_FRAME iterations per frame. Its h is the smaller of the straight-line
    distance to the goal region and the weighted state distance to the nearest joinable route state (/ MAX_SPEED), and
    its normal-SST samples come from a circle around the car that grows while nothing is found. A new node joins a
    route's still-valid tail (including this route beyond the obstacle) when it's within JOIN_TOLERANCE_SCALE times
    joinTolerance of it and not more than JOIN_SPEED_OVER faster; reaching the goal region counts too. The first time
    any join exists, the cheapest is spliced in and the car jumps onto it (after coming to rest first, if the plan
    started from rest).

    While nothing needs replanning, the same machinery improves the route ahead: from the start of the next very
    squiggly edge (squiggly), it looks for a way on that is at least IMPROVE_MIN_GAIN cheaper, and splices it in if
    found before the car gets there.
    """

    def __init__(self, sdf_state, routes):
        self.sdf_state = sdf_state
        self.routes = routes
        self.route = min(routes, key=lambda r: r.remaining[0])
        self.s, self.v = 0.0, 0.0  # arc length along self.route, speed
        self.mode = 'follow'  # 'follow', 'replan' (from self.replan_index), 'stop' (planning from rest at self.stop_index), 'done'
        self.replanner, self.joins = None, []
        self.replan_index = self.stop_index = None
        self.fan_tried = False  # whether the forward tests have been tried at the current stop
        self.hold_index = None  # after joining a plan made from rest: the car comes to rest here before going on
        self.hold_route = None  # the route hold_index is an index into (a hold on any other route is dropped)
        self.safe_cache = {}  # id(state) -> (map version, safe speed)
        self.seen_version = None
        self.bad_index = None  # first blocked state on the current route that triggered the current replan/stop
        self.bad_end = None  # last state of that blocked stretch; route states after it are past the obstacle
        self.past_nodes = []  # heap of (h, order, replanner node, nearby route index): nodes that got past the obstacle
        self.unsafe = []  # per map version: indices of route states in an obstacle or unknown space
        self.join_points = np.empty((0, 5))  # (x, y, heading, speed, steering) of every join candidate, for h
        self.near_nodes = []  # heap of (h, order, planner node, route, index): near misses of the current planner
        self.heap_order = itertools.count()  # breaks h ties by arrival, like the list scans these heaps replaced
        self.improver, self.improve_joins = None, []  # planner looking for a cheaper way past a squiggly edge
        self.improve_index = None  # route node it plans from
        self.improve_route = None  # the route improve_index is an index into
        self.no_squiggles = None  # the route last found to have no squiggly edge ahead (skip searching it again)
        self.rrt = None  # the original geometric RRT, grown alongside each replan for comparison (not used for driving)
        self.rrt_time = self.sst_time = 0.0  # wall-clock seconds each has used on the current replan
        self.rrt_skeleton = None  # (points every 5 px, arc length, heading) along the RRT's path, once it has one
        self.stopped_time = 0.0  # s the car has been at rest while waiting on a replan
        self.stalled_time = 0.0  # s the car has been at rest while just following its route
        self.rrt_progress = 0.0  # furthest arc length along the skeleton a replan node has reached
        self.join_grid, self.h_version = None, 0  # replanH's distances around obstacles, and its cache version
        self.group_tried = False  # whether the stuck-branch grouping has been tried during the current stop
        self.group_end = None  # (route, index) of the end of the branch groupCommit last drove: no backing up from there
        self.lib_tried_at = -LIB_TRIES_EVERY  # replanner iteration of the last library try (tryJoin)
        self.group_retry = 0.0  # s until the next try to chain a new branch onto the committed one, while driving it
        self.event = None  # the obstacle episode being recorded (openEvent), if OBSTACLE_LOG
        self.event_dir, self.event_count = None, 0
        self.resolution = None  # how the latest blockage got resolved, for the obstacle log
        self.trace = []  # every motion substep: (duration, x, y, heading, speed, steering), for the replay

    def pose(self):
        return self.route.poseAt(self.s)

    def safeAt(self, p):
        cached = self.safe_cache.get(id(p))
        if cached is None or cached[0] != self.sdf_state.version:
            cached = (self.sdf_state.version, safeSpeed(p, self.sdf_state))
            self.safe_cache[id(p)] = cached
        return cached[1]

    def speedLimit(self):
        """The fastest the car may go now (speedLimitAt the car, with the current stop point)."""
        stop_at = self.stop_index if self.mode in ('replan', 'stop') else self.hold_index
        return self.speedLimitAt(self.s, stop_at)

    def hardStopArc(self):
        """Arc length the car must not drive past this frame: the last clear state before the first one ahead in an
        obstacle or unknown space, or the stop point while replanning, stopping or holding (if it's ahead)."""
        limit = self.route.arc[-1]
        here = self.route.indexAt(self.s)
        unsafe_ahead = next((i for i in self.unsafe if i > here), None)
        if unsafe_ahead is not None:
            limit = min(limit, max(self.s, self.route.arc[unsafe_ahead - 1]))
        stop_at = self.stop_index if self.mode in ('replan', 'stop') else self.hold_index
        if stop_at is not None and self.route.arc[stop_at] >= self.s:
            limit = min(limit, self.route.arc[stop_at])
        return limit

    def speedLimitAt(self, s, stop_at):
        """The fastest the car may go at arc length s of the route: it must be able to brake to the safe speed of each
        of the next SAFE_SPEED_NODES nodes by the time it reaches them, to a stop before the first route state ahead
        that's in an obstacle or unknown space, and (if stop_at is a route index) to a stop at that point."""
        limit = MAX_SPEED
        ahead = [k for k in self.route.nodes if self.route.arc[k] > s][:SAFE_SPEED_NODES]
        for k in ahead:
            d = self.route.arc[k] - s
            limit = min(limit, math.sqrt(self.safeAt(self.route.states[k]) ** 2 + 2 * MAX_ACCELERATION * d))
        here = self.route.indexAt(s)
        unsafe_ahead = next((i for i in self.unsafe if i > here), None)
        if unsafe_ahead is not None:
            # Stop at the state before it: that one was still clear
            d = max(0, self.route.arc[unsafe_ahead - 1] - s)
            limit = min(limit, math.sqrt(2 * MAX_ACCELERATION * d))
        if stop_at is not None:
            limit = min(limit, math.sqrt(2 * MAX_ACCELERATION * max(0, self.route.arc[stop_at] - s)))
        return limit

    def checkRoutes(self):
        """After obstacles were added: re-check every route, and react if the current one is blocked ahead."""
        for r in self.routes:
            blocked = np.flatnonzero(r.blocked(self.sdf_state))
            r.valid_from = int(blocked[-1]) + 1 if len(blocked) else 0
        self.routes = [r for r in self.routes if r.valid_from < len(r.states)]
        blocked = np.flatnonzero(self.route.blocked(self.sdf_state))
        here = self.route.indexAt(self.s)
        ahead = [int(i) for i in blocked if i > here]
        if not ahead or self.mode == 'done':
            return
        bad = ahead[0]
        blocked_set = set(int(i) for i in blocked)
        if self.mode in ('replan', 'stop') and self.bad_index is not None:
            if bad >= self.bad_index:
                # More of the same obstacle (or one further on): keep the current replan, just refresh where it can join
                if self.replanner is not None:
                    self.buildJoinMap()
                return
            # More of the obstacle showed up closer: keep planning from where the first replan started, if that's still
            # ahead of (or, when stopping, at) the car and before the obstacle; only the fallback stop point moves
            keep_from = self.replan_index if self.mode == 'replan' else self.stop_index
            usable = (keep_from is not None and keep_from < bad
                      and (keep_from > here if self.mode == 'replan' else keep_from >= here))
            if usable:
                self.bad_index, self.bad_end = bad, bad
                while self.bad_end + 1 in blocked_set:
                    self.bad_end += 1
                if self.mode == 'replan':
                    self.stop_index = [k for k in self.route.nodes if k < bad][-1]
                if self.replanner is not None:
                    self.buildJoinMap()
                return
        self.bad_index = bad
        self.bad_end = bad
        while self.bad_end + 1 in blocked_set:
            self.bad_end += 1
        before = [k for k in self.route.nodes if k < bad]
        last = before[-1]  # L
        # R: REPLAN_BACK_NODES nodes before L, or (as before) the node just before L if the car is already past that one
        replan_from = next((before[-1 - back] for back in (REPLAN_BACK_NODES, 1)
                            if back < len(before) and before[-1 - back] > here), None)
        self.openEvent(replan_from, last)
        if replan_from is not None and self.tryNudge(replan_from):
            print('Driver: route blocked ahead; a small nudge gets past it', flush=True)
            self.resolution = 'nudge'
            return
        if replan_from is not None:
            p = self.route.states[replan_from]
            speed = math.copysign(min(abs(p.s), self.speedLimitAt(self.route.arc[replan_from], last)), p.s)
            if self.tryReconnect(replan_from, speed):
                print('Driver: route blocked ahead; reconnected to a route nearby', flush=True)
                self.resolution = 'reconnect'
                return
        if replan_from is not None:
            self.mode, self.replan_index, self.stop_index = 'replan', replan_from, last
            p = self.route.states[replan_from]
            # Start at the speed the car can actually have at R, not the route's planned one: the speed limit (stopping
            # before unknown space and at L) holds the car below it, and a start too fast to stop before the obstacle
            # makes every edge fail canStopClear, so the replan can't grow at all (world 19 sat at 1 node at 251 px/s)
            speed = math.copysign(min(abs(p.s), self.speedLimitAt(self.route.arc[replan_from], last)), p.s)
            self.startReplan(state(p.x1, p.x2, p.a, speed, steer=p.steer))
        else:
            self.beginStop(last if last > here else None)
        print(f'Driver: route blocked ahead, {self.mode} ({len(self.routes)} routes still reach the goal)', flush=True)

    def beginStop(self, at_index):
        """Stop at route state at_index (or, if None, as soon as the car can) and start planning from there, at rest,
        right away."""
        here = self.route.indexAt(self.s)
        if at_index is None:
            stop_arc = self.s + self.v * self.v / (2 * MAX_ACCELERATION)
            at_index = min(bisect.bisect_left(self.route.arc, stop_arc), len(self.route.states) - 1)
            unsafe_ahead = next((i for i in self.unsafe if i > here), None)
            if unsafe_ahead is not None:
                at_index = min(at_index, unsafe_ahead - 1)
            at_index = max(at_index, here)
        self.mode, self.stop_index, self.fan_tried = 'stop', at_index, False
        p = self.route.states[at_index]
        self.startReplan(state(p.x1, p.x2, p.a, 0.0, steer=p.steer))

    def tryForwardFan(self):
        """Stopped at stop_index with no new route yet: try STOP_FAN_TESTS forward nudges from rest with lane-change
        offsets spread over +-STOP_FAN_FRACTION of the allowed steering, until PAST_BEYOND past the blocked stretch. If
        any ends within join tolerance of a route beyond it (or the goal), splice in the cheapest and return True."""
        if self.bad_end is None or self.bad_end <= self.stop_index:
            return False
        self.buildJoinMap()
        best = None
        for k in range(STOP_FAN_TESTS):
            fraction = -STOP_FAN_FRACTION + 2 * STOP_FAN_FRACTION * k / (STOP_FAN_TESTS - 1)
            states = self.nudge(self.stop_index, fraction, from_rest=True)
            if not states:
                continue
            states[-1].edge_end = states[-1]
            joins = self.joinsFor(states[-1], after=self.bad_end)
            if joins:
                join = min(joins, key=lambda j: j[0])
                if best is None or join[0] < best[0][0]:
                    best = (join, states)
        if best is None:
            return False
        (_, _, r, i), states = best
        self.route = spliceRoute(self.route, self.stop_index, states, r, i)
        self.routes.append(self.route)
        self.mode, self.replanner, self.rrt, self.joins = 'follow', None, None, []
        self.bad_index = self.bad_end = self.replan_index = self.stop_index = None
        self.unsafe = np.flatnonzero(self.route.blocked(self.sdf_state, unknown_too=True)).tolist()
        return True

    def backUp(self):
        """The forward tests failed: back up along the route to the previous node (already driven, so clear) and
        restart the replan from there at rest right away."""
        previous = [k for k in self.route.nodes if k < self.stop_index]
        if not previous:
            return
        self.stop_index = previous[-1]
        p = self.route.states[self.stop_index]
        self.startReplan(state(p.x1, p.x2, p.a, 0.0, steer=p.steer))

    def nudge(self, start_index, fraction, from_rest=False):
        """Re-drive the current route from start_index past the blocked stretch (bad_index to bad_end), dodging it:
        follow the route, then shift sideways over the NUDGE_SHIFT_LENGTH px ending where the obstacle starts, stay
        shifted alongside it, and shift back over NUDGE_BACK_LENGTH px after it. Each shift steers like the route plus a
        lane-change-shaped offset (one way, then the other, so the heading comes back), at most `fraction` of the
        steering allowed at that speed; `fraction`'s sign picks the side. Speed follows the route's planned speed
        (starting from 0 with from_rest). Each PLAN_DT step goes through propagate, so the limits and collision checks
        are the planner's own. Returns the states, or [] if it hits something (or the obstacle isn't ahead of it)."""
        r = self.route
        base = r.arc[start_index]
        obstacle_start, obstacle_end = r.arc[self.bad_index] - base, r.arc[self.bad_end] - base
        if obstacle_end <= 0:
            return []
        shift_start = max(0.0, obstacle_start - NUDGE_SHIFT_LENGTH)
        shift_end = max(obstacle_start, shift_start + 20)
        back_start = max(obstacle_end, shift_end)
        length = back_start + NUDGE_BACK_LENGTH

        def lane_change(w):  # steering offset shape over one sideways shift, w in [0, 1]
            return 4 * w if w < 0.25 else (2 - 4 * w if w < 0.75 else 4 * w - 4)

        p = r.states[start_index]
        x = state(p.x1, p.x2, p.a, 0.0 if from_rest else p.s, steer=p.steer)
        states, travelled = [], 0.0
        for step in range(4000):
            if travelled >= length:
                return states
            if shift_start <= travelled < shift_end:
                shape = lane_change((travelled - shift_start) / (shift_end - shift_start))
            elif travelled >= back_start:
                shape = -lane_change((travelled - back_start) / NUDGE_BACK_LENGTH)
            else:
                shape = 0.0
            _, _, _, ref_speed, ref_steer = r.poseAt(base + travelled)
            limit = maxSteerAtSpeed(max(abs(x.s), abs(ref_speed)))
            desired = clamp(ref_steer + fraction * limit * shape, -limit, limit)
            accel, lo, hi = feasibleAccel(x, PLAN_DT, (ref_speed - x.s) / PLAN_DT)
            seg = propagate(x, action(clamp((desired - x.steer) / PLAN_DT, lo, max(lo, hi)), accel), 1, self.sdf_state)
            if not seg:
                return []
            travelled += math.hypot(seg[-1].x1 - x.x1, seg[-1].x2 - x.x2)
            if step % NUDGE_NODE_FRAMES == NUDGE_NODE_FRAMES - 1:
                seg[-1].edge_end = seg[-1]  # a node, so the safe-speed look-ahead has points along it
            states += seg
            x = seg[-1]
        return []

    def libraryJoinsFor(self, node, max_targets=3):
        """Library manoeuvres (libraryJoin) from `node` onto the most promising join candidates within libRange of it
        (straight-line time there plus the route's remaining cost; at most one per route). Returns (joins as
        (total cost, node, route, index, manoeuvre states), whether any manoeuvre was actually tried)."""
        if not 0 <= node.s <= LIB_SPEEDS[-1] + 5:
            return [], False
        best = {}
        for jp in self.join_map.get_nbhd(node, libRange(node.s)):
            r, i = jp.route, jp.index
            score = math.hypot(jp.x1 - node.x1, jp.x2 - node.x2) / max(node.s, 20.0) + r.remaining[i]
            if id(r) not in best or score < best[id(r)][0]:
                best[id(r)] = (score, r, i)
        if not best:
            return [], False
        joins = []
        for _, r, i in sorted(best.values(), key=lambda t: t[0])[:max_targets]:
            found = libraryJoin(node, r, i, self.sdf_state)
            if found is not None:
                cost, states, j = found
                joins.append((node.cost + cost + r.remaining[j], node, r, j, states))
        return joins, True

    def tryLibraryAt(self, start_index):
        """At rest at route state start_index: a library manoeuvre onto a known route nearby (libraryJoinsFor), spliced
        in if one works. Returns whether one was."""
        self.buildJoinMap()
        p = self.route.states[start_index]
        start = state(p.x1, p.x2, p.a, 0.0, steer=p.steer)
        joins, _ = self.libraryJoinsFor(start)
        if not joins:
            return False
        _, _, r, j, states = min(joins, key=lambda t: t[0])
        self.adoptSplice(start_index, states, r, j, from_rest=True)
        return True

    def adoptSplice(self, start_index, states, r, j, from_rest):
        """Follow the current route up to start_index, then `states`, then route r after index j."""
        self.route = spliceRoute(self.route, start_index, states, r, j)
        self.routes.append(self.route)
        if from_rest:
            self.hold_index, self.hold_route = start_index, self.route  # planned from rest: rest at the splice first
        self.mode, self.replanner, self.joins = 'follow', None, []
        self.bad_index = self.bad_end = self.replan_index = self.stop_index = None
        self.unsafe = np.flatnonzero(self.route.blocked(self.sdf_state, unknown_too=True)).tolist()

    def tryReconnect(self, start_index, speed):
        """From the current route's state at start_index, at `speed` (what the car can actually have there; 0 from a
        stop), try shortcutDrive to states of every still-valid route within RECONNECT_RADIUS (beyond the blocked
        stretch on the current route), every RECONNECT_TARGET_SPACING px. Targets the car can't reach without reversing
        (inside a tightest-turn circle at the start, Dubins style) or facing more than RECONNECT_MAX_TURN away are
        skipped; the RECONNECT_MAX_TARGETS most promising are tried with each RECONNECT_TANGENTS x RECONNECT_LOOKAHEADS
        variant. The cheapest one that rejoins within the live join tolerance and ends where the car can stop
        (canStopClear) is spliced in. Returns whether one was."""
        p = self.route.states[start_index]
        if p.s < 0 or speed < 0:
            return False
        start = state(p.x1, p.x2, p.a, speed, steer=clamp(p.steer, -maxSteerAtSpeed(speed), maxSteerAtSpeed(speed)))
        c, sn = math.cos(p.a), math.sin(p.a)
        targets = []
        for r in self.routes:
            first = max(r.valid_from, 1)
            if r is self.route:
                first = max(first, (self.bad_end if self.bad_end is not None else start_index) + 1)
            last_arc = -math.inf
            for i in range(first, len(r.states)):
                q = r.states[i]
                if r.arc[i] - last_arc < RECONNECT_TARGET_SPACING or q.s <= 0:
                    continue
                dx, dy = q.x1 - p.x1, q.x2 - p.x2
                dist = math.hypot(dx, dy)
                if dist > RECONNECT_RADIUS or dist < 20:
                    continue
                last_arc = r.arc[i]
                fwd, left = dx * c + dy * sn, -dx * sn + dy * c
                if abs((q.a - p.a + math.pi) % (2 * math.pi) - math.pi) > RECONNECT_MAX_TURN:
                    continue
                # Dubins-style reachability: not inside either tightest-turn circle at the slower end's speed
                v = min(speed, q.s)
                radius = max(WHEELBASE / math.tan(MAX_STEER), v * v / MAX_LATERAL_ACCELERATION)
                if math.hypot(fwd, abs(left) - radius) < radius:
                    continue
                targets.append((dist / max(speed, q.s, 1.0) + r.remaining[i], r, i))
        targets.sort(key=lambda t: t[0])
        best, columns_of = None, {}
        for _, r, i in targets[:RECONNECT_MAX_TARGETS]:
            if id(r) not in columns_of:
                columns_of[id(r)] = tuple(np.array(col) for col in zip(*((s.x1, s.x2, s.a, s.s, s.steer) for s in r.states)))
            lo = max(i - RECONNECT_JOIN_WINDOW[0], max(r.valid_from, 1)) - 1
            hi = min(len(r.states), i + RECONNECT_JOIN_WINDOW[1])
            for tangent in RECONNECT_TANGENTS:
                for lookahead in RECONNECT_LOOKAHEADS:
                    found = shortcutDrive(start, r.states[i], r, columns_of[id(r)], lo, hi, tangent, lookahead, 0.0,
                                          self.sdf_state, scale=JOIN_TOLERANCE_SCALE)
                    if found is None or (best is not None and found[0] >= best[0][0]):
                        continue
                    if canStopClear(found[1][-1], self.sdf_state):
                        best = (found, r)
        if best is None:
            return False
        (_, states, j, gap), r = best
        states[-1].edge_end = states[-1]
        states[-1].cost += gap
        self.adoptSplice(start_index, states, r, j, from_rest=speed == 0.0)
        return True

    def tryNudge(self, start_index):
        """Before a full replan: try NUDGE_VARIANTS nudges each way (nudge, which dodges right at the obstacle) from
        start_index. If any ends within join tolerance of the route beyond the obstacle (or another route, or the goal),
        splice in the cheapest and return True."""
        self.buildJoinMap()
        best = None
        for k in range(1, NUDGE_VARIANTS + 1):
            for side in (1, -1):
                states = self.nudge(start_index, side * k / NUDGE_VARIANTS)
                if not states:
                    continue
                states[-1].edge_end = states[-1]
                joins = self.joinsFor(states[-1], after=self.bad_end)
                if joins:
                    join = min(joins, key=lambda j: j[0])
                    if best is None or join[0] < best[0][0]:
                        best = (join, states)
        if best is None:
            return False
        (_, _, r, i), states = best
        self.route = spliceRoute(self.route, start_index, states, r, i)
        self.routes.append(self.route)
        self.bad_index = self.bad_end = None
        self.unsafe = np.flatnonzero(self.route.blocked(self.sdf_state, unknown_too=True)).tolist()
        return True

    def makePlanner(self, start, center, hook):
        """A GlobalPlanner from `start` aimed by replanH, sampling (normal-SST steps) around center() in a circle that
        grows while nothing is found, reporting every kept node to hook."""
        planner = GlobalPlanner(self.sdf_state, start, h_fn=self.replanH)
        planner.sample_fn = lambda: self.sampleAround(*center(), planner.it)
        planner.child_hook = hook
        return planner

    def startReplan(self, start):
        self.improver = None  # a real replan takes over
        self.replanner = self.makePlanner(start, lambda: self.pose()[:2], self.tryJoin)
        planner = self.replanner
        planner.sample_fn = lambda: self.sampleReplan(planner)
        # Replans only keep nodes the car can still stop from: doomed nodes (too fast to make the next corner) would
        # otherwise soak up the search, like the 270 px/s nodes that stalled world 15 at a corner
        planner.edge_filter = lambda path: canStopClear(path[-1], self.sdf_state)
        self.joins, self.past_nodes, self.near_nodes, self.turn_tried = [], [], [], set()
        self.lib_tried_at = -LIB_TRIES_EVERY
        self.buildJoinMap()
        self.rrt = None  # only started once the car has been stopped for RRT_STOP_DELAY s
        self.rrt_time = self.sst_time = 0.0
        self.rrt_skeleton, self.rrt_progress = None, 0.0

    def groupCommit(self, at_index):
        """Stuck at the stop point, or driving a committed branch with no join yet: drive the most promising partial
        branch of the replan tree (rooted at route index at_index: the stop point, or the moving replan start 2 nodes
        before the committed branch's end) instead of waiting, spliced in at at_index so the car carries straight on.

        Only branches ending no further from the target than the car (goalDistanceGrid, around known obstacles) count;
        if there are none, the car stays put. Takes the GROUP_SAMPLES active tree nodes with the lowest score (replanH, the estimated cost to the goal from
        there, plus GROUP_G_WEIGHT times their own cost) at least GROUP_BACK_NODES + 1 nodes deep and GROUP_MIN_DISTANCE
        from the car (straight line) whose branches don't contain one another, k-means (k-means++ start) their end states (x, y, and heading as
        GROUP_HEADING_SCALE times (cos, sin)) into GROUP_K groups, picks the group of at least GROUP_MIN_SIZE with the
        lowest median score, and takes its lowest scoring branch. The route becomes the car's route up to here plus that
        branch at GROUP_SPEED_SCALE of its planned speed, ending in a stop at its end (the speed limit keeps the car able
        to stop there and short of unknown space), and a new replan starts GROUP_BACK_NODES nodes before its end while the car drives. Returns whether it
        committed to a branch."""
        planner = self.replanner
        root = planner.start
        # Progress: the estimated cost to the goal from the branch's end, with only a little of its own cost (scored by
        # g + h, every branch around a blockage has about the same h, so the shortest ones always won and the car crept)
        score = lambda n: self.replanH(n) + GROUP_G_WEIGHT * n.cost
        # Never further from the target: the branch must end no further (Dijkstra around known obstacles) than the car is
        target_grid = goalDistanceGrid(self.sdf_state)
        here = gridValue(target_grid, root.x1, root.x2)
        past = {entry[2] for entry in self.past_nodes}  # nodes tryJoin found past the obstacle
        candidates, past_candidates = [], []
        for n in planner.tree.active:
            is_past = n in past
            if not is_past and math.hypot(n.x1 - root.x1, n.x2 - root.x2) < GROUP_MIN_DISTANCE:
                continue
            if not gridValue(target_grid, n.x1, n.x2) <= here:
                continue
            depth, v = 0, n
            while v.parent is not None:
                depth, v = depth + 1, v.parent
            if v is root and depth > GROUP_BACK_NODES and math.isfinite(score(n)):
                (past_candidates if is_past else candidates).append(n)
        if past_candidates and GROUP_PAST_FIRST:  # a branch already gets past the obstacle: the best of those, no grouping
            return self.commitBranch(min(past_candidates, key=score), at_index, 'the best branch past the obstacle')
        candidates += past_candidates
        candidates.sort(key=score)
        picked, picked_set, ancestors = [], set(), set()
        for n in candidates:
            if n in ancestors:  # a picked branch already runs through it
                continue
            chain, v = [], n.parent
            while v is not None:
                chain.append(v)
                v = v.parent
            if any(v in picked_set for v in chain):  # it extends a picked branch
                continue
            picked.append(n)
            picked_set.add(n)
            ancestors.update(chain)
            if len(picked) == GROUP_SAMPLES:
                break
        if len(picked) < GROUP_K * GROUP_MIN_SIZE // 2:
            return False
        features = np.array([(n.x1, n.x2, GROUP_HEADING_SCALE * math.cos(n.a), GROUP_HEADING_SCALE * math.sin(n.a))
                             for n in picked])
        labels = kMeans(features, min(GROUP_K, len(picked)))  # (scores below: lower is better)
        scores = np.array([score(n) for n in picked])
        groups = [np.flatnonzero(labels == g) for g in range(labels.max() + 1)]
        groups = [g for g in groups if len(g) >= GROUP_MIN_SIZE] or [max(groups, key=len)]
        best_group = min(groups, key=lambda g: float(np.median(scores[g])))
        chosen = picked[int(best_group[np.argmin(scores[best_group])])]
        return self.commitBranch(chosen, at_index, f'grouped {len(picked)} branches into '
                                 f'{sorted((len(g) for g in groups), reverse=True)}, the best of the best group '
                                 f'(median score {float(np.median(scores[best_group])):.2f})')

    def commitBranch(self, chosen, at_index, what):
        """Drive the replan tree's branch to `chosen` (see groupCommit), spliced in at route index at_index (where the
        tree's root is), and start the next replan GROUP_BACK_NODES nodes before its end. Returns whether it did."""
        planner = self.replanner
        root = planner.start
        new_states = pathStates(root, chosen)[1:]
        if not pathIsClear(new_states, self.sdf_state):
            return False
        # Driven at GROUP_SPEED_SCALE of the planned speed (copies, so the tree's states are left alone): the same
        # geometry slower only needs less grip
        slowed = []
        for p in new_states:
            c = copy.copy(p)
            c.s = p.s * GROUP_SPEED_SCALE
            if getattr(p, 'edge_end', None) is p:
                c.edge_end = c
            slowed.append(c)
        new_states = slowed
        # The car's route up to the replan start (at_index), then the branch; remaining costs guess the rest with replanH.
        # The route is the same up to there, so the car (at or before it) carries on without a jump
        spliced = spliceRoute(self.route, at_index, new_states, None, None)
        route = Route(spliced.states, [r + self.replanH(chosen) for r in spliced.remaining])
        nodes_after = [k for k in route.nodes if k > at_index]
        if len(nodes_after) <= GROUP_BACK_NODES:
            return False
        if self.event is not None:
            self.event['commits'] += 1
        how = (f'stuck {self.stopped_time:.1f} s' if self.mode == 'stop'
               else 'nearing the replan start without a join')
        print(f'Driver: {how}; driving {what} and replanning on the way', flush=True)
        self.route = route
        self.mode = 'replan'
        self.replan_index, self.stop_index = nodes_after[-1 - GROUP_BACK_NODES], len(route.states) - 1
        self.group_end = (route, self.stop_index)  # stopping there later replans from there, without backing up
        self.bad_index = self.bad_end = None
        self.unsafe = np.flatnonzero(self.route.blocked(self.sdf_state, unknown_too=True)).tolist()
        p = route.states[self.replan_index]
        speed = math.copysign(min(abs(p.s), self.speedLimitAt(route.arc[self.replan_index], self.stop_index)), p.s)
        use_turns = planner.use_turns
        self.startReplan(state(p.x1, p.x2, p.a, speed, steer=p.steer))
        self.replanner.use_turns = use_turns  # still stuck-ish: keep the turn primitives on
        return True

    def openEvent(self, replan_from, last):
        """Start recording an obstacle episode (OBSTACLE_LOG): the known map around the blocked stretch (occupancy and
        both distance fields), the car, every still-valid route, and the decision (blocked stretch, replan start R,
        stop point L). Any episode still open is closed as superseded first."""
        if not OBSTACLE_LOG:
            return
        if self.event is not None:
            self.closeEvent('superseded by a new blockage')
        self.resolution = None
        b = self.route.states[self.bad_index]
        x0, x1 = max(0, int(b.x1) - OBSTACLE_LOG_PATCH), min(SCREEN_WIDTH, int(b.x1) + OBSTACLE_LOG_PATCH)
        y0, y1 = max(0, int(b.x2) - OBSTACLE_LOG_PATCH), min(SCREEN_HEIGHT, int(b.x2) + OBSTACLE_LOG_PATCH)
        x, y, a, _, steer = self.pose()
        table = lambda r: np.array([(p.x1, p.x2, p.a, p.s, p.steer, r.remaining[i],
                                     float(i == 0 or i == len(r.states) - 1 or getattr(p, 'edge_end', None) is p))
                                    for i, p in enumerate(r.states)], dtype=np.float32)
        self.event = dict(
            patch_origin=np.array([x0, y0]), occupancy=self.sdf_state.occupancy[x0:x1, y0:y1].copy(),
            obstacle_sdf=self.sdf_state.obstacle_sdf[x0:x1, y0:y1].astype(np.float16),
            unknown_sdf=self.sdf_state.unknown_sdf[x0:x1, y0:y1].astype(np.float16),
            car=np.array([x, y, a, self.v, steer, self.s]), car_mode=self.mode,
            blocked=np.array([self.bad_index, self.bad_end]),
            blocked_xy=np.array([[b.x1, b.x2], [self.route.states[self.bad_end].x1, self.route.states[self.bad_end].x2]]),
            replan_from=-1 if replan_from is None else replan_from, stop_at=last,
            route_current=table(self.route), routes_valid_from=np.array([r.valid_from for r in self.routes]),
            duration=0.0, stopped=False, stop_time=0.0, iterations=0, commits=0)
        for k, r in enumerate(self.routes):
            self.event[f'route_{k}'] = table(r)

    def closeEvent(self, resolution):
        """Finish the open obstacle episode with how it was resolved and save it (OBSTACLE_LOG_DIR/<run>/event_###.npz)."""
        if self.event is None:
            return
        e, self.event = self.event, None
        e['resolution'] = resolution
        if self.event_dir is None:
            self.event_dir = os.path.join(OBSTACLE_LOG_DIR, clock_time.strftime('%Y%m%d-%H%M%S') + f'-{os.getpid()}')
            os.makedirs(self.event_dir, exist_ok=True)
        self.event_count += 1
        np.savez_compressed(os.path.join(self.event_dir, f'event_{self.event_count:03d}.npz'), **e)

    def setSkeleton(self):
        """Resample the RRT's solution path (start to target) every 5 px, with arc length and heading along it."""
        corners = []
        v = self.rrt.solved_vertex
        while v is not None:
            corners.append((v.x1, v.x2))
            v = v.parent
        corners.reverse()
        points, arcs, headings, total = [], [], [], 0.0
        for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
            length = math.hypot(x1 - x0, y1 - y0)
            heading = math.atan2(y1 - y0, x1 - x0)
            for k in range(max(1, math.ceil(length / 5))):
                f = k * 5 / length if length else 0
                points.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
                arcs.append(total + f * length)
                headings.append(heading)
            total += length
        points.append(corners[-1]); arcs.append(total); headings.append(headings[-1] if headings else 0.0)
        # Speed along it: slow enough at each bend for a SKELETON_CORNER_RADIUS turn, lowered upstream so the car can
        # brake in time, and capped from the start by accelerating from rest
        speeds = [MAX_SPEED] * len(points)
        corner_speed = math.sqrt(MAX_LATERAL_ACCELERATION * SKELETON_CORNER_RADIUS)
        for i in range(1, len(points)):
            if abs((headings[i] - headings[i - 1] + math.pi) % (2 * math.pi) - math.pi) > math.radians(10):
                speeds[i - 1] = speeds[i] = min(speeds[i], corner_speed)
        for i in range(len(points) - 2, -1, -1):
            speeds[i] = min(speeds[i], math.sqrt(speeds[i + 1] ** 2 + 2 * MAX_ACCELERATION * (arcs[i + 1] - arcs[i])))
        for i in range(len(points)):
            speeds[i] = min(speeds[i], math.sqrt(2 * MAX_ACCELERATION * arcs[i]) + 10)
        self.rrt_skeleton = (np.array(points), np.array(arcs), np.array(headings))
        self.rrt_speeds = speeds
        self.rrt_progress = 0.0

    def sampleSkeleton(self):
        """A sample on the RRT skeleton RRT_LOOKAHEAD px beyond the replan's progress along it (position noise
        RRT_SAMPLE_SIGMA, heading along the skeleton, its speed profile's speed there, wheel straight), marked so the
        planner extends the nearest node to it rather than the cheapest one nearby."""
        points, arcs, headings = self.rrt_skeleton
        k = min(int(np.searchsorted(arcs, self.rrt_progress + RRT_LOOKAHEAD)), len(arcs) - 1)
        speed = clamp(random.gauss(self.rrt_speeds[k], 10), 5, MAX_SPEED)
        sample = state(clamp(random.gauss(points[k, 0], RRT_SAMPLE_SIGMA), 0, SCREEN_WIDTH - 1),
                       clamp(random.gauss(points[k, 1], RRT_SAMPLE_SIGMA), 0, SCREEN_HEIGHT - 1),
                       random.gauss(headings[k], 0.2) % (2 * math.pi), speed, steer=0.0)
        # Extend the node nearest to it (the one at the frontier), not the cheapest nearby: those sit behind the
        # frontier, and growing from them kept re-covering ground the tree already had
        sample.nearest_only = True
        return sample

    @staticmethod
    def bestActive(heap, planner):
        """The lowest-h entry of a near/past heap whose node is still active in `planner`, dropping inactive ones off
        the top (nodes don't become active again), or None."""
        while heap and heap[0][2] not in planner.tree.active:
            heapq.heappop(heap)
        return heap[0] if heap else None

    def recordNear(self, near):
        for n, r, i in near:
            heapq.heappush(self.near_nodes, (self.replanH(n), next(self.heap_order), n, r, i))

    def sampleNear(self, planner):
        """If a near miss (within NEAR_JOIN_SCALE times the join tolerances of a route) is still active in `planner`,
        NEAR_FOCUS_SHARE of the time: a sample on that route NEAR_LOOKAHEAD states ahead of the best one (lowest h),
        lined up with it (position noise NEAR_SAMPLE_SIGMA, the route's heading, speed and steering). Otherwise None."""
        best = self.bestActive(self.near_nodes, planner)
        if best is None or random.random() >= NEAR_FOCUS_SHARE:
            return None
        _, _, _, r, i = best
        t = r.states[min(i + NEAR_LOOKAHEAD, len(r.states) - 1)]
        limit = maxSteerAtSpeed(abs(t.s))
        return state(clamp(random.gauss(t.x1, NEAR_SAMPLE_SIGMA), 0, SCREEN_WIDTH - 1),
                     clamp(random.gauss(t.x2, NEAR_SAMPLE_SIGMA), 0, SCREEN_HEIGHT - 1),
                     random.gauss(t.a, 0.05) % (2 * math.pi), t.s, steer=clamp(t.steer, -limit, limit))

    def sampleReplan(self, planner):
        """Replan sampling: if a node that got past the obstacle is still active, PAST_FOCUS_SHARE of the time sample
        near the route PAST_LOOKAHEAD states ahead of the best one (lowest h), lined up with the route there; otherwise
        in the growing circle around the car."""
        near = self.sampleNear(planner)
        if near is not None:
            return near
        # Stopped with an RRT path: follow it as a skeleton, near the car first and further out as the tree progresses
        if (self.rrt_skeleton is not None and self.mode == 'stop' and abs(self.v) < 1
                and random.random() < RRT_FOCUS_SHARE):
            return self.sampleSkeleton()
        best = self.bestActive(self.past_nodes, planner)
        if best is not None and random.random() < PAST_FOCUS_SHARE:
            i = best[3]
            t = self.route.states[min(i + PAST_LOOKAHEAD, len(self.route.states) - 1)]
            speed = clamp(random.gauss(t.s, 15), -MAX_REVERSE_SPEED, MAX_SPEED)
            limit = maxSteerAtSpeed(abs(speed))
            return state(clamp(random.gauss(t.x1, PAST_SAMPLE_SIGMA), 0, SCREEN_WIDTH - 1),
                         clamp(random.gauss(t.x2, PAST_SAMPLE_SIGMA), 0, SCREEN_HEIGHT - 1),
                         random.gauss(t.a, 0.1) % (2 * math.pi), speed, steer=clamp(t.steer, -limit, limit))
        return self.sampleAround(*self.pose()[:2], planner.it)

    def buildJoinMap(self):
        """Join candidates: the still-valid tails of every route, filed by position (for tryJoin) and as numpy rows
        (for replanH)."""
        self.join_map = proximityHashMap(10)
        rows, remaining = [], []
        for r in self.routes:
            for i in range(max(r.valid_from, 1), len(r.states)):
                p = r.states[i]
                self.join_map.add(_JoinPoint(p.x1, p.x2, r, i))
                rows.append((p.x1, p.x2, p.a, p.s, p.steer))
                remaining.append(r.remaining[i])
        self.join_points = np.array(rows) if rows else np.empty((0, 5))
        self.join_remaining = np.array(remaining) if remaining else np.empty(0)
        self.join_grid = None  # gridDistance to these join points (and the goal), built when replanH first needs it
        self.h_version += 1  # replanH values cached on nodes before this are stale

    def joinGrid(self):
        """Estimated cost to the goal from every cell, in px: the distance around known obstacles (gridDistance) to a
        join candidate plus that route's remaining cost from it (times MAX_SPEED), or to the goal region, whichever is
        least. So it points the way to the best way home, not just to the nearest route."""
        if self.join_grid is None:
            nx, ny = SCREEN_WIDTH // H_CELL, SCREEN_HEIGHT // H_CELL
            cx = np.arange(nx) * H_CELL + H_CELL // 2
            cy = np.arange(ny) * H_CELL + H_CELL // 2
            sources = np.where(np.hypot(cx[:, None] - TARGET[0], cy[None, :] - TARGET[1]) <= TARGET_RAD, 0.0, np.inf)
            if len(self.join_points):
                i = np.clip((self.join_points[:, 0] // H_CELL).astype(int), 0, nx - 1)
                j = np.clip((self.join_points[:, 1] // H_CELL).astype(int), 0, ny - 1)
                np.minimum.at(sources, (i, j), self.join_remaining * MAX_SPEED)
            self.join_grid = gridDistance(self.sdf_state, sources)
        return self.join_grid

    def replanH(self, n):
        """Replanner h, an estimate of the cost to the goal (px over MAX_SPEED): the larger of
        - joinGrid: around known obstacles to a join candidate plus that route's remaining cost, or to the goal region,
          so the tree isn't drawn into an obstacle, and heads for the best way home rather than the nearest route, and
        - the smaller of the straight-line distance to the goal region and, over the join candidates, the weighted
          state distance (stateDistance) plus the route's remaining cost, which near a join also asks for the right
          heading, speed and steering.
        Cached on the node (until buildJoinMap changes the candidates)."""
        cached = getattr(n, 'replan_h', None)
        if cached is not None and cached[0] == self.h_version:
            return cached[1]
        d = max(0.0, math.hypot(n.x1 - TARGET[0], n.x2 - TARGET[1]) - TARGET_RAD)
        if len(self.join_points):
            j = self.join_points
            dh = np.abs((j[:, 2] - n.a + np.pi) % (2 * np.pi) - np.pi)
            dist = (np.hypot(j[:, 0] - n.x1, j[:, 1] - n.x2) + W_A * dh + W_V * np.abs(j[:, 3] - n.s)
                    + W_ST * np.abs(j[:, 4] - n.steer) + self.join_remaining * MAX_SPEED)
            d = min(d, float(dist.min()))
        h = max(d, gridValue(self.joinGrid(), n.x1, n.x2)) / MAX_SPEED
        n.replan_h = (self.h_version, h)
        return h

    def sampleAround(self, cx, cy, iterations):
        """A random state in a circle around (cx, cy) (avoiding known obstacles), REPLAN_SAMPLE_RADIUS to start and
        REPLAN_SAMPLE_GROWTH times bigger every REPLAN_SAMPLE_GROW_EVERY iterations of the planner using it."""
        growths = min(iterations // REPLAN_SAMPLE_GROW_EVERY, 30)  # (capped: 1.5 ** 30 is already far past the map)
        radius = min(REPLAN_SAMPLE_RADIUS * REPLAN_SAMPLE_GROWTH ** growths, SCREEN_WIDTH + SCREEN_HEIGHT)
        for _ in range(50):
            r, t = radius * math.sqrt(random.random()), random.uniform(0, 2 * math.pi)
            x = clamp(cx + r * math.cos(t), 0, SCREEN_WIDTH - 1)
            y = clamp(cy + r * math.sin(t), 0, SCREEN_HEIGHT - 1)
            if self.sdf_state.occupancy[int(x), int(y)] != OCCUPIED:
                break
        speed = random.uniform(-MAX_REVERSE_SPEED, MAX_SPEED)
        limit = maxSteerAtSpeed(abs(speed))
        return state(x, y, random.uniform(0, 2 * math.pi), speed, steer=random.uniform(-limit, limit))

    def turnaroundJoins(self, node, after=None):
        """If `node` is within TURN_NEAR_RADIUS of a joinable route state, facing within TURN_HEADING_TOL of the opposite
        way and no faster than TURN_MAX_ENTRY_SPEED, try every TURNAROUND_MANEUVERS from it; those whose end joins a route
        give joins with the manoeuvre's states attached: (cost, node, route, index, extra states). Each spot (10 px cell,
        30 degree heading bin) is tried once per planner."""
        if abs(node.s) > TURN_MAX_ENTRY_SPEED:
            return []
        facing_back = False
        for jp in self.join_map.get_nbhd(node, TURN_NEAR_RADIUS):
            if after is not None and jp.route is self.route and jp.index <= after:
                continue
            p = jp.route.states[jp.index]
            if abs((node.a - p.a + math.pi) % (2 * math.pi) - math.pi) >= math.pi - TURN_HEADING_TOL:
                facing_back = True
                break
        if not facing_back:
            return []
        spot = (int(node.x1 // 10), int(node.x2 // 10), int((node.a % (2 * math.pi)) // math.radians(30)))
        if spot in self.turn_tried:
            return []
        self.turn_tried.add(spot)
        found = []
        for phases in TURNAROUND_MANEUVERS.values():
            states = runTurnaround(node, phases, self.sdf_state)
            if states:
                found += [(c, node, r, i, states) for c, _, r, i in self.joinsFor(states[-1], after=after)]
        return found

    def tryJoin(self, node):
        near = []
        self.joins.extend(self.joinsFor(node, near=near))
        self.recordNear(near)
        self.joins.extend(self.turnaroundJoins(node))
        # Close to a known route in a free circle: a library manoeuvre onto it (at most once per LIB_TRIES_EVERY)
        planner = self.replanner
        if planner is not None and planner.it - self.lib_tried_at >= LIB_TRIES_EVERY:
            joins, tried = self.libraryJoinsFor(node)
            if tried:
                self.lib_tried_at = planner.it
            self.joins.extend(joins)
        # How far along the RRT skeleton the replan has got
        if self.rrt_skeleton is not None:
            points, arcs, _ = self.rrt_skeleton
            d = np.hypot(points[:, 0] - node.x1, points[:, 1] - node.x2)
            k = int(np.argmin(d))
            if d[k] <= RRT_FOCUS_RADIUS:
                self.rrt_progress = max(self.rrt_progress, float(arcs[k]))
        # Did it get past the obstacle? (near the current route, beyond the blocked stretch)
        if self.bad_end is not None:
            for jp in self.join_map.get_nbhd(node, PAST_RADIUS):
                if jp.route is self.route and self.route.arc[jp.index] >= self.route.arc[self.bad_end] + PAST_BEYOND:
                    heapq.heappush(self.past_nodes, (self.replanH(node), next(self.heap_order), node, jp.index))
                    break

    def joinsFor(self, node, after=None, near=None):
        """Every way `node` can join: (total cost to the goal, node, route, index), with route None for the goal region
        itself. States of the current route at or before index `after` are skipped. If `near` is a list, every
        (node, route, index) within NEAR_JOIN_SCALE times the tolerances (a near miss) is appended to it too."""
        if math.hypot(node.x1 - TARGET[0], node.x2 - TARGET[1]) <= TARGET_RAD:
            return [(node.cost, node, None, None)]
        found = []
        side_tol, heading_tol, steer_tol = (JOIN_TOLERANCE_SCALE * t for t in joinTolerance(node.s))
        scale = NEAR_JOIN_SCALE if near is not None else 1
        for jp in self.join_map.get_nbhd(node, scale * side_tol + CHECK_SPACING):
            r, i = jp.route, jp.index
            if after is not None and r is self.route and i <= after:
                continue
            p = r.states[i]
            if node.s * p.s < 0 or abs(node.s) > abs(p.s) + JOIN_SPEED_OVER:
                continue
            heading = abs((node.a - p.a + math.pi) % (2 * math.pi) - math.pi)
            steer = abs(node.steer - p.steer)
            if heading > scale * heading_tol or steer > scale * steer_tol:
                continue
            # Sideways distance: to the route's line through this state's neighbors
            q = r.states[i - 1]
            dx, dy = p.x1 - q.x1, p.x2 - q.x2
            seg = math.hypot(dx, dy)
            side = math.hypot(node.x1 - p.x1, node.x2 - p.x2) if seg == 0 else abs((node.x1 - q.x1) * dy - (node.x2 - q.x2) * dx) / seg
            if side > scale * side_tol:
                continue
            if side <= side_tol and heading <= heading_tol and steer <= steer_tol:
                found.append((node.cost + r.remaining[i], node, r, i))
            elif near is not None:
                near.append((node, r, i))
        return found

    def squiggly(self, a, b):
        """Whether the route's edge from node index a to node index b is very squiggly: the mean of
        |steering| / maxSteerAtSpeed(|speed|) over its states is at least SQUIGGLY_STEER_FRACTION."""
        states = self.route.states[a + 1:b + 1]
        if not states:
            return False
        return sum(abs(p.steer) / maxSteerAtSpeed(abs(p.s)) for p in states) / len(states) >= SQUIGGLY_STEER_FRACTION

    def loopy(self, a):
        """Whether the stretch of the current route from node a is a loop or big wiggle (routeLoopy)."""
        return routeLoopy(self.route, a)

    def startImprove(self):
        """Plan from the start of the next very squiggly edge (squiggly) or loopy stretch (loopy) ahead, on whatever
        route the car is on (skipping one the car reaches within IMPROVE_SKIP_FRAMES frames), looking for a cheaper way
        on."""
        self.improver, self.improve_index = None, None
        if self.no_squiggles is self.route:
            return
        here = self.route.indexAt(self.s)
        nodes = self.route.nodes
        skip_time = IMPROVE_SKIP_FRAMES / FPS
        for a, b in zip(nodes, nodes[1:]):
            if a <= here or (self.route.arc[a] - self.s) / max(self.v, 1) <= skip_time:
                continue
            if not (self.squiggly(a, b) or self.loopy(a)):
                continue
            p = self.route.states[a]
            self.improve_index, self.improve_route = a, self.route
            self.improve_joins, self.near_nodes, self.turn_tried = [], [], set()
            self.buildJoinMap()
            self.improver = self.makePlanner(state(p.x1, p.x2, p.a, p.s, steer=p.steer), lambda: (p.x1, p.x2),
                                             self.tryImproveJoin)
            improver = self.improver
            improver.sample_fn = lambda: self.sampleNear(improver) or self.sampleAround(p.x1, p.x2, improver.it)
            return
        self.no_squiggles = self.route

    def tryImproveJoin(self, node):
        threshold = self.route.remaining[self.improve_index] * (1 - IMPROVE_MIN_GAIN)
        near = []
        joins = self.joinsFor(node, after=self.improve_index, near=near)
        self.recordNear(near)
        joins += self.turnaroundJoins(node, after=self.improve_index)
        self.improve_joins.extend(j for j in joins if j[0] < threshold)

    def finishImprove(self):
        """Splice in the cheapest improvement found, if it's still clear."""
        join = min(self.improve_joins, key=lambda j: j[0])
        _, node, r, i = join[:4]
        new_states = pathStates(self.improver.start, node)[1:] + (join[4] if len(join) > 4 else [])
        if pathIsClear(new_states, self.sdf_state):
            self.route = spliceRoute(self.route, self.improve_index, new_states, r, i)
            self.routes.append(self.route)
            self.unsafe = np.flatnonzero(self.route.blocked(self.sdf_state, unknown_too=True)).tolist()
            print(f'Driver: improved the route, {self.route.remaining[self.route.indexAt(self.s)]:.2f} to go', flush=True)
        self.improver, self.improve_joins, self.improve_index = None, [], None

    def finishReplan(self):
        """Splice in the cheapest join found and follow it."""
        join = min(self.joins, key=lambda j: j[0])
        _, node, r, i = join[:4]
        new_states = pathStates(self.replanner.start, node)[1:] + (join[4] if len(join) > 4 else [])
        if not pathIsClear(new_states, self.sdf_state):  # the map changed under it; plan again from the same start
            self.startReplan(self.replanner.start)
            return
        upto = self.replan_index if self.mode == 'replan' else self.stop_index
        self.route = spliceRoute(self.route, upto, new_states, r, i)
        # A plan made from rest starts from rest: the car comes to a stop at the splice point first
        self.hold_index = upto if self.mode == 'stop' else None
        self.hold_route = self.route
        self.routes.append(self.route)
        self.resolution = 'replan join' + (' with a manoeuvre (library or turnaround)' if len(join) > 4 else '')
        self.mode, self.replanner, self.joins, self.bad_index = 'follow', None, [], None
        self.replan_index = self.stop_index = self.bad_end = None
        self.past_nodes = []
        self.unsafe = np.flatnonzero(self.route.blocked(self.sdf_state, unknown_too=True)).tolist()
        print(f'Driver: joined {"the goal" if r is None else "a route"}, {self.route.remaining[self.route.indexAt(self.s)]:.2f} to go', flush=True)

    def update(self, dt):
        if self.mode == 'done':
            return
        if self.hold_index is not None and self.hold_route is not self.route:
            self.hold_index = None  # the route was replaced since (an improvement, a reconnect...): the hold went with it
        if self.sdf_state.version != self.seen_version:
            self.seen_version = self.sdf_state.version
            self.unsafe = np.flatnonzero(self.route.blocked(self.sdf_state, unknown_too=True)).tolist()
            if self.sdf_state.new_obstacle_points:
                self.sdf_state.new_obstacle_points = []
                self.checkRoutes()

        self.stopped_time = self.stopped_time + dt if (self.mode == 'stop' and abs(self.v) < 1) else 0.0
        if self.stopped_time == 0.0:
            self.group_tried = False
        self.stalled_time = self.stalled_time + dt if (self.mode == 'follow' and abs(self.v) < 1) else 0.0
        if self.replanner is not None and self.stopped_time >= TURN_PRIMITIVE_STUCK_TIME:
            self.replanner.use_turns = True

        # Planning budget for this frame: a replan if one is running, otherwise improving a squiggly stretch ahead
        if self.replanner is not None:
            started = clock_time.perf_counter()
            joined = False
            for _ in range(REPLAN_ITERATIONS_PER_FRAME):
                self.replanner.step()
                if self.event is not None:
                    self.event['iterations'] += 1
                if self.joins:
                    joined = True
                    break
            spent = clock_time.perf_counter() - started
            self.sst_time += spent
            # Stopped long enough: the RRT starts (from the replan's start) and gets the same wall-clock time this frame,
            # until it has a solution
            if RRT_ENABLED and self.rrt is None and self.stopped_time >= RRT_STOP_DELAY:
                start = self.replanner.start
                self.rrt = RRTGraph(start.x1, start.x2, self.sdf_state)
            if self.rrt is not None and self.rrt.solved_vertex is None:
                started = clock_time.perf_counter()
                while clock_time.perf_counter() - started < spent and self.rrt.solved_vertex is None:
                    self.rrt.add(RRT_RANDOM.uniform(0, SCREEN_WIDTH), RRT_RANDOM.uniform(0, SCREEN_HEIGHT))
                self.rrt_time += clock_time.perf_counter() - started
                if self.rrt.solved_vertex is not None:
                    self.setSkeleton()
                    print(f'RRT: solved after {self.rrt_time:.3f} s ({len(self.rrt.vertices)} vertices); the SST replan '
                          f'had used {self.sst_time:.3f} s and {"had" if joined else "had not"} joined yet', flush=True)
            if joined:
                rrt_note = (f'RRT solved after {self.rrt_time:.3f} s' if self.rrt is not None and self.rrt.solved_vertex is not None
                            else f'RRT not solved after {self.rrt_time:.3f} s')
                print(f'SST: joined after {self.sst_time:.3f} s; {rrt_note}', flush=True)
                self.finishReplan()
                if self.replanner is None:  # (a replan restarted by finishReplan keeps its fresh RRT)
                    self.rrt = None
        elif self.mode == 'follow':
            if (self.improver is None or self.improve_route is not self.route  # (a nudge may have replaced the route)
                    or self.improve_index <= self.route.indexAt(self.s)
                    or (self.route.arc[self.improve_index] - self.s) / max(self.v, 1) <= IMPROVE_SKIP_FRAMES / FPS):
                self.startImprove()
            if self.improver is not None:
                for _ in range(REPLAN_ITERATIONS_PER_FRAME):
                    self.improver.step()
                    if self.improve_joins:
                        self.finishImprove()
                        break

        # Speed and motion along the route (backwards while backing up to a stop point behind the car), in substeps of
        # at most MOTION_DT with the speed limit re-checked each one: a slow frame (heavy planning) then still brakes in
        # time, where one big step would carry the car past where it had to stop
        hard_stop = self.hardStopArc()
        steps = max(1, math.ceil(dt / MOTION_DT))
        h = dt / steps
        for _ in range(steps):
            limit = self.speedLimit()
            if self.mode == 'stop' and self.s > self.route.arc[self.stop_index] + 0.5:
                behind = self.s - self.route.arc[self.stop_index]
                target = -min(MAX_REVERSE_SPEED, math.sqrt(2 * MAX_ACCELERATION * behind))
                self.v += clamp(target - self.v, -MAX_ACCELERATION * h, MAX_ACCELERATION * h)
                self.s = max(self.s + self.v * h, self.route.arc[self.stop_index])
                if self.s <= self.route.arc[self.stop_index] + 0.5:
                    self.v = 0.0
            else:
                planned = abs(self.pose()[3])
                target = min(planned, limit)
                if limit > CREEP_SPEED:
                    target = max(target, CREEP_SPEED)
                self.v += clamp(target - self.v, -MAX_ACCELERATION * h, MAX_ACCELERATION * h)
                self.s = min(self.s + self.v * h, self.route.arc[-1])
                if self.s > hard_stop:  # backstop for rounding: never past the last clear state or the stop point
                    self.s, self.v = hard_stop, 0.0
            x, y, a, _, steer = self.pose()
            self.trace.append((h, x, y, a, self.v, steer))

        # Stuck at the stop point for GROUP_STUCK_TIME: drive the most promising partial branch (groupCommit)
        if (self.mode == 'stop' and self.replanner is not None and not self.group_tried
                and self.stopped_time >= GROUP_STUCK_TIME and abs(self.s - self.route.arc[self.stop_index]) <= 1):
            self.group_tried = True
            self.groupCommit(self.stop_index)
        # Driving a committed branch with no join yet: every GROUP_RETRY_INTERVAL, group the branches of the replan
        # growing from 2 nodes before its end (still ahead of the car) and, if one qualifies, splice it in there, so the
        # car goes straight on into the next branch instead of stopping at the end of this one
        # Non-stop (any replan, NONSTOP): the same once the car is within NONSTOP_LEAD s of the replan start
        committed = self.group_end is not None and self.group_end[0] is self.route
        lead = (self.route.arc[self.replan_index] - self.s) / max(self.v, 1.0) if self.mode == 'replan' else math.inf
        if (self.mode == 'replan' and self.replanner is not None and self.route.indexAt(self.s) < self.replan_index
                and (committed or (NONSTOP and lead <= NONSTOP_LEAD))):
            self.group_retry -= dt
            if self.group_retry <= 0:
                self.group_retry = GROUP_RETRY_INTERVAL
                self.groupCommit(self.replan_index)
        # Stopped at the stop point with no new route yet: forward tests at slightly different angles, else back up
        if (self.mode == 'stop' and not self.fan_tried and abs(self.v) < 1
                and abs(self.s - self.route.arc[self.stop_index]) <= 1):
            self.fan_tried = True
            if self.tryForwardFan():
                print('Driver: stopped; a forward test at a slightly different angle gets past', flush=True)
                self.resolution = 'forward fan'
            elif self.tryLibraryAt(self.stop_index):
                print('Driver: stopped; a library manoeuvre gets onto a route nearby', flush=True)
                self.resolution = 'library (at a stop)'
            elif self.tryReconnect(self.stop_index, 0.0):
                print('Driver: stopped; reconnected to a route nearby', flush=True)
                self.resolution = 'reconnect (at a stop)'
            elif self.groupCommit(self.stop_index):
                pass  # committed to a branch of the replan from here: on we go, no backing up
            elif self.group_end is not None and self.group_end[0] is self.route and self.group_end[1] == self.stop_index:
                pass  # at the end of a branch groupCommit drove: no backing up, the replan just goes on from here
            else:
                self.backUp()
                print('Driver: stopped; no forward test got past, backing up a node and replanning from there', flush=True)

        # At the replan start without a join: one last try to commit to a branch from here (the car is at most a frame
        # past it, so the route it's on hardly changes) before braking to the stop point
        if (NONSTOP and self.mode == 'replan' and self.replanner is not None
                and self.route.indexAt(self.s) >= self.replan_index
                and self.s - self.route.arc[self.replan_index] <= max(self.v * dt, 2.0)):
            self.groupCommit(self.replan_index)
        # Out of time to replan from R: replan from L at rest right away, while braking to stop there
        if self.mode == 'replan' and self.route.indexAt(self.s) >= self.replan_index:
            print('Driver: reached the replanning point without a new route; replanning from the last safe node', flush=True)
            self.beginStop(self.stop_index)
        # Stuck short of the stop point (nothing safe ahead): replan from right here instead
        if (self.mode == 'stop' and self.v < 1 and limit < CREEP_SPEED
                and self.s < self.route.arc[self.stop_index] - 1):
            self.v = 0.0
            self.beginStop(self.route.indexAt(self.s))
            print('Driver: stopped short of the planned stop; replanning from here', flush=True)
        # Stuck while just following the route (nothing ahead it can safely drive into): replan from here
        if self.mode == 'follow' and self.stalled_time >= STALL_REPLAN_TIME and limit < CREEP_SPEED:
            self.stalled_time = 0.0
            self.v = 0.0
            self.bad_index = self.bad_end = None
            self.beginStop(self.route.indexAt(self.s))
            print('Driver: stalled while following the route; replanning from here', flush=True)
        # Came to rest at the start of a plan made from rest: go on
        if self.hold_index is not None and self.v < 1 and self.s >= self.route.arc[self.hold_index] - 1:
            self.hold_index = None
        if self.s >= self.route.arc[-1] - 1e-6 and self.mode == 'follow':
            self.mode = 'done'
        # Dead replan (its start can't grow, e.g. the end of a committed branch facing a wall): don't wait on it forever
        if (self.replanner is not None and self.mode in ('replan', 'stop')
                and self.replanner.it >= DEAD_REPLAN_ITERATIONS and len(self.replanner.tree.active) < DEAD_REPLAN_NODES):
            if self.mode == 'replan':
                print('Driver: the replan can\'t grow from its start; stopping and replanning from the stop point', flush=True)
                self.beginStop(self.stop_index)
            elif abs(self.v) < 1:
                print('Driver: the replan can\'t grow from here; backing up a node and replanning from there', flush=True)
                self.group_end = None
                self.backUp()
        # Obstacle log: time, stops, and the end of the episode once the car is following a route again
        if self.event is not None:
            self.event['duration'] += dt
            if self.mode == 'stop' and abs(self.v) < 1:
                self.event['stopped'] = True
                self.event['stop_time'] += dt
            if self.mode in ('follow', 'done'):
                self.closeEvent(self.resolution or ('reached the goal' if self.mode == 'done' else 'unknown'))
            print('Driver: reached the goal', flush=True)

    def draw(self, surface, ghost):
        for r in self.routes:
            if r is not self.route and len(r.states) > 1:
                pygame.draw.lines(surface, "magenta", False, [(p.x1, p.x2) for p in r.states], 2)
        for planner, color in ((self.replanner, (90, 90, 90)), (self.improver, (60, 60, 90))):
            if planner is not None:
                for v in planner.tree.active | planner.tree.inactive:
                    if v.parent is not None:
                        pygame.draw.line(surface, color, (v.parent.x1, v.parent.x2), (v.x1, v.x2), 1)
        pygame.draw.lines(surface, "cyan", False, [(p.x1, p.x2) for p in self.route.states], 3)
        if self.rrt is not None:
            for v in self.rrt.vertices:
                if v.parent is not None:
                    pygame.draw.line(surface, (110, 0, 0), (v.parent.x1, v.parent.x2), (v.x1, v.x2), 1)
            if self.rrt.solved_vertex is not None:
                path = []
                v = self.rrt.solved_vertex
                while v is not None:
                    path.append((v.x1, v.x2))
                    v = v.parent
                pygame.draw.lines(surface, (255, 0, 0), False, path, 4)
        ghost.x, ghost.y, ghost.heading, _, ghost.steering = self.pose()
        ghost.velocity = self.v
        ghost.draw(surface, color="orange")


class Vertex:
    def __init__(self, x1, x2, parent=None, child=None, cost=0):
        self.x1 = x1
        self.x2 = x2
        self.parent = parent
        self.child = [] if child is None else child
        self.cost = cost


class RRTGraph:
    def __init__(self, x1, x2, sdf_state):
        self.vertices = [Vertex(x1, x2)]
        self.solved_vertex = None
        self.sdf_state = sdf_state

    def getClosestVertex(self, x1, x2):
        closest = None
        min_dist = float('inf')
        for v in self.vertices:
            dist = math.hypot(v.x1 - x1, v.x2 - x2)
            if dist < min_dist:
                min_dist = dist
                closest = v
        return closest

    def add(self, x1, x2):
        closest = self.getClosestVertex(x1, x2)
        if not getValidityOfPath(closest.x1, closest.x2, x1, x2, self.sdf_state):
            return False
        new_vertex = Vertex(x1, x2, parent=closest)
        closest.child.append(new_vertex)
        self.vertices.append(new_vertex)
        if math.hypot(x1 - TARGET[0], x2 - TARGET[1]) <= TARGET_RAD:
            self.solved_vertex = new_vertex
        return True


def getValidityOfPath(s1x1, s1x2, s2x1, s2x2, sdf_state):
    """Whether the straight segment between the two points keeps half the car's width plus SAFETY_MARGIN from the known
    obstacles, sampled every px along it (the map is an SDF now, rather than the old circle lists)."""
    length = math.hypot(s2x1 - s1x1, s2x2 - s1x2)
    steps = max(1, math.ceil(length))
    for k in range(steps + 1):
        t = k / steps
        px, py = round(s1x1 + t * (s2x1 - s1x1)), round(s1x2 + t * (s2x2 - s1x2))
        if not (0 <= px < SCREEN_WIDTH and 0 <= py < SCREEN_HEIGHT):
            return False
        if sdf_state.obstacle_sdf[px, py] < CAR_WIDTH / 2 + SAFETY_MARGIN:
            return False
    return True


class _JoinPoint:
    """A route state filed in the Driver's join map (proximityHashMap needs x1/x2)."""

    def __init__(self, x1, x2, route, index):
        self.x1, self.x2, self.route, self.index = x1, x2, route, index


def drawTree(tree, root, surface, color=(90, 90, 90)):
    """Draw every edge of the tree onto `surface` through its check points."""
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        if node.parent is not None:
            pygame.draw.lines(surface, color, False, [(node.parent.x1, node.parent.x2)] + [(p.x1, p.x2) for p in node.traj])


def replayPose(states, t):
    """(x, y, heading, speed, steering) at time t along `states` (each with .time, in increasing order), interpolated
    between the two states around t."""
    times = [p.time for p in states]
    i = min(max(1, bisect.bisect_right(times, t)), len(states) - 1)
    a, b = states[i - 1], states[i]
    f = 0 if b.time == a.time else clamp((t - a.time) / (b.time - a.time), 0, 1)
    dh = (b.a - a.a + math.pi) % (2 * math.pi) - math.pi
    return (a.x1 + f * (b.x1 - a.x1), a.x2 + f * (b.x2 - a.x2), a.a + f * dh,
            a.s + f * (b.s - a.s), a.steer + f * (b.steer - a.steer))


class Replay:
    """Plays a Driver's trace back in real time, with no planning: what the car actually drove, so you can judge if
    it's realistic. Each stop (at rest while waiting on a plan) is cut to at most REPLAY_MAX_STOP."""

    def __init__(self, trace):
        times, poses, t, stopped = [], [], 0.0, 0.0
        for h, x, y, a, v, steer in trace:
            still = bool(poses) and abs(v) < 1 and math.hypot(x - poses[-1][0], y - poses[-1][1]) < 1e-3
            stopped = stopped + h if still else 0.0
            if still and stopped > REPLAY_MAX_STOP:
                continue
            if poses:
                t += h
            times.append(t)
            poses.append((x, y, a, v, steer))
        self.times = np.array(times)
        self.poses = poses
        self.t = 0.0

    def done(self):
        return not self.poses or self.t >= self.times[-1]

    def pose(self):
        """(x, y, heading, speed, steering) at the current replay time, interpolated."""
        i = min(max(1, int(np.searchsorted(self.times, self.t))), len(self.poses) - 1)
        p, q = self.poses[i - 1], self.poses[i]
        span = self.times[i] - self.times[i - 1]
        f = 0.0 if span <= 0 else clamp((self.t - self.times[i - 1]) / span, 0, 1)
        dh = (q[2] - p[2] + math.pi) % (2 * math.pi) - math.pi
        return (p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1]), p[2] + f * dh,
                p[3] + f * (q[3] - p[3]), p[4] + f * (q[4] - p[4]))

    def draw(self, surface, ghost):
        if len(self.poses) > 1:
            pygame.draw.lines(surface, "orange", False, [(p[0], p[1]) for p in self.poses], 2)
        ghost.x, ghost.y, ghost.heading, ghost.velocity, ghost.steering = self.pose()
        ghost.draw(surface, color="orange")


def main():
    sdf_state = SDFState(street_blocks(), diagonal_roads())
    local_map = LocalObstacleMap(sdf_state)

    # Global plan on a copy of the prior map, before any lidar scans; the replanning demo's blocks only go into this
    # copy, so they're gone from the map the car actually drives on
    def planningMap():
        plan_map = SDFState(street_blocks(), diagonal_roads())
        if GLOBAL_PLAN_SEES_LOCAL:
            # The hidden obstacles are added after the roads are carved (carving would erase them)
            rasterize(plan_map.occupancy, local_map.obstacles, OCCUPIED)
            plan_map.recompute(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT)
        return plan_map

    root = state(*START, START_HEADING, 0.0)
    library = loadPathLibrary()
    if PATH_LIBRARY_MODE == 'use' and library:
        paths = library
        print(f'main: using the path library ({len(paths)} paths from {PATH_LIBRARY_FILE})', flush=True)
    elif REPLAN_DEMO:
        # PREPROCESS_BATCHES independent batches, each from scratch on a fresh map (so the previous batch's blocks are
        # gone), keeping every batch's paths
        paths = []
        for batch in range(PREPROCESS_BATCHES):
            print(f'main: preprocessing batch {batch + 1} of {PREPROCESS_BATCHES}', flush=True)
            _, batch_paths, _ = replanDemo(planningMap(), root)
            paths += batch_paths
        print(f'main: {len(paths)} paths kept', flush=True)
        smooth_map = planningMap()  # without any batch's blocks: shortcuts are local, so the paths stay distinct
        paths = [smoothPath(path, smooth_map) for path in paths]
        paths = library + paths
        savePathLibrary(paths)
        print(f'main: path library now has {len(paths)} paths ({len(library)} from before), saved to {PATH_LIBRARY_FILE}',
              flush=True)
    else:
        plan_map = planningMap()
        tree, _, _ = planGlobal(plan_map, root)
        best = min(tree.goal_nodes, key=lambda g: g.cost) if tree.goal_nodes else None
        paths = [pathStates(root, best)] if best else []
    # Only the planned paths are kept (everything else in the planning tree is dropped); the ghost car drives the best
    driver = Driver(sdf_state, [Route(path) for path in paths]) if paths else None
    restart_timer = None  # counts down once the ghost arrives; then the drive starts over

    def restart(new_obstacles=False):
        """Start the drive over: a fresh known map (roads unknown again), the same hidden obstacles (or, with
        new_obstacles, a new random set), the same planned routes, the ghost back at the start."""
        nonlocal sdf_state, driver, local_map
        sdf_state = SDFState(street_blocks(), diagonal_roads())
        if new_obstacles:
            local_map = LocalObstacleMap(sdf_state)
        else:
            local_map.known = sdf_state
            local_map.surface, local_map.drawn_changes = None, 0
        driver = Driver(sdf_state, [Route(path) for path in paths])
    show_local = True  # hidden local obstacles drawn in green; L toggles
    scan_timer = 0.0  # lidar runs every LIDAR_DT, not every display frame

    pygame.init()
    window = pygame.display.set_mode((int(SCREEN_WIDTH * SCALE), int(SCREEN_HEIGHT * SCALE)))
    screen = pygame.Surface(SCREEN_SIZE)
    clock = pygame.time.Clock()
    font = load_font()
    car = Car(*START, heading=START_HEADING)  # driven with WASD; just for you, it doesn't scan
    ghost = Car(*START, heading=START_HEADING)

    dt = 0.0
    replay = None  # a Replay while the last live drive is being played back
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                show_local = not show_local
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r and paths:  # new random hidden obstacles
                restart(new_obstacles=True)
                restart_timer, scan_timer, replay = None, 0.0, None

        keys = pygame.key.get_pressed()
        throttle = keys[pygame.K_w] - keys[pygame.K_s]
        steer = keys[pygame.K_d] - keys[pygame.K_a]
        car.set_acceleration(throttle * MAX_ACCELERATION)
        car.set_steering_rate(steer * MAX_STEER_RATE)
        car.update(dt)
        # Live drive (planning as it goes), then a replay of what it drove at full speed, then a new live drive...
        if replay is not None:
            replay.t += dt
            if replay.done():
                restart_timer = RESTART_DELAY if restart_timer is None else restart_timer - dt
                if restart_timer <= 0:
                    restart()
                    restart_timer, scan_timer, replay = None, 0.0, None
        elif driver is not None:
            scan_timer -= dt
            if scan_timer <= 0:  # the lidar rides on the ghost car
                gx, gy = driver.pose()[:2]
                local_map.reveal(gx, gy)
                scan_timer = LIDAR_DT
            driver.update(dt)
            if driver.mode == 'done':
                restart_timer = RESTART_DELAY if restart_timer is None else restart_timer - dt
                if restart_timer <= 0:
                    replay = Replay(driver.trace)
                    restart_timer = None

        screen.fill("black")
        sdf_state.draw(screen)
        if show_local:
            local_map.draw(screen)
        if replay is not None:
            pygame.draw.lines(screen, "cyan", False, [(p.x1, p.x2) for p in driver.route.states], 3)
            replay.draw(screen, ghost)
        elif driver is not None:
            driver.draw(screen, ghost)
        car.draw(screen)
        draw_hud(screen, font, car)
        if font and driver is not None:
            label = (f'REPLAY of the last drive (stops cut to {REPLAY_MAX_STOP} s)' if replay is not None
                     else 'LIVE: planning while driving')
            screen.blit(font.render(label, True, "orange" if replay is not None else "white"), (10, 30))

        pygame.draw.circle(screen, "PURPLE", TARGET, TARGET_RAD)
        window.blit(pygame.transform.smoothscale(screen, window.get_size()), (0, 0))
        pygame.display.flip()

        dt = clock.tick(FPS) / 1000
    pygame.quit()


if __name__ == "__main__":
    main()
