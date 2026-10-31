"""Shortcutting planned paths into smoother ones, and the saved path library."""
import copy
import math
import os

import numpy as np

from .car import clamp, maxSteerAtSpeed
from .config import (
    LOOP_DETOUR_RATIO, LOOP_SPAN, LOOP_TURN, NUDGE_NODE_FRAMES, PATH_LIBRARY_FILE, PLAN_DT, SMOOTH_BUMPS,
    SMOOTH_JOIN_SLACK, SMOOTH_LOOKAHEADS, SMOOTH_LOOP_SPAN, SMOOTH_MIN_GAIN, SMOOTH_MIN_SPEED, SMOOTH_NODES_AHEAD,
    SMOOTH_RETRIES, SMOOTH_STEER_BLEND, SMOOTH_SWEEPS, SMOOTH_TANGENTS, SMOOTH_TARGET_SPACING,
    SMOOTH_TOLERANCE_SCALE, WHEELBASE,
)
from .kinematics import feasibleAccel, propagate
from .route import Route, routeJoin, spliceRoute
from .tree import action, state


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
