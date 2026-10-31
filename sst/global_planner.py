"""The global SST planner, its grid heuristic, motion primitives and the replanning demo."""
import heapq
import itertools
import math
import random

import numpy as np
from numba import njit

from .car import clamp, maxSteerAtSpeed
from .config import (
    ANNEAL_FACTOR, ANNEAL_MIN_GAIN, ANNEAL_WINDOW, BLOCK_RADIUS, BLOSSOM_RANDOM, CHECK_SPACING, DELTA_B_N_MIN,
    DELTA_B_N_START, EARLY_ACCEL_MEAN, EARLY_STEER_ITERATIONS, EARLY_STEER_VARIANCE_SCALE, EXPANSION_PENALTY,
    FOOTPRINT_RADIUS, FOOTPRINT_REACH, GREEDY_CHAIN_MAX, GREEDY_SHARE, GRID_INFLATION, H_CELL, MAX_ACCELERATION,
    MAX_PLAN_ITERATIONS, MAX_SPEED, MAX_STEER_RATE, OCCUPIED, PLAN_DT, PREPROCESS_MAX_ITERATIONS, PRIMITIVES, radii,
    REPLAN_IMPROVE_ITERATIONS, REPLAN_MAX_ITERATIONS, REPLAN_PATHS, SAFETY_MARGIN, SCREEN_HEIGHT, SCREEN_WIDTH,
    START, STEER_TARGET_SIGMA, TARGET, TARGET_RAD, TIE_BREAK, TURN_PRIMITIVE_ANGLE, TURN_PRIMITIVE_MAX_BRAKE,
    TURN_PRIMITIVE_MAX_FRAMES, TURN_PRIMITIVE_SPEEDS, UNKNOWN, WHEELBASE,
)
from .kinematics import bodyClearance, feasibleAccel, monteCarloProp, propagate
from .streets import Obstacle
from .tree import (
    action, bestFirstSelectionSST, makeSampleBiasingGrid, nearestActive, NodeStatus, proximityHashMap, SSTTree,
    state, Witness, WitnessHashMap,
)


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
        radii.delta_B_N = DELTA_B_N_START
        radii.delta_S = radii.delta_B_N / 4
        self.sdf_state = sdf_state
        self.start = start
        self.tree = SSTTree(start)
        self.witness_map = WitnessHashMap(radii.delta_S)
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
            if gain < ANNEAL_MIN_GAIN and radii.delta_B_N * ANNEAL_FACTOR >= DELTA_B_N_MIN:
                radii.delta_B_N *= ANNEAL_FACTOR
                radii.delta_S *= ANNEAL_FACTOR
                rebuilt = WitnessHashMap(radii.delta_S)
                for w in self.witness_map:
                    rebuilt.add(w)
                self.witness_map = rebuilt
                print(f'planGlobal: iteration {it}, best cost {best:.2f} ({100 * gain:.1f}% better in the last '
                      f'{ANNEAL_WINDOW}), annealed to delta_B_N {radii.delta_B_N:.2f}, delta_S {radii.delta_S:.2f}, '
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
              f'driving time {best_goal.time:.2f} s, delta_B_N {radii.delta_B_N:.2f}, {len(planner.tree.active)} active nodes',
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
