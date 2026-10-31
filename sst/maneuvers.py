"""Precomputed maneuvers: turnarounds and the maneuver library used to rejoin a route."""
import math
import os

import numpy as np

from .car import clamp, maxSteerAtSpeed
from .config import (
    FOOTPRINT_OFFSETS, FOOTPRINT_RADIUS, FOOTPRINT_REACH, FREE, JOIN_SPEED_OVER, JOIN_TOLERANCE_SCALE, LIB_BINS,
    LIB_CANDIDATES, LIB_DEPTH, LIB_KEY_CELL, LIB_MIN_SPEED, LIB_SPEEDS, LIB_STEP_FRAMES, LIB_TARGET_OFFSETS,
    MANEUVER_LIBRARY_FILE, MAX_ACCELERATION, MAX_LATERAL_ACCELERATION, MAX_SPEED, MAX_STEER, MAX_STEER_RATE,
    NUDGE_NODE_FRAMES, PLAN_DT, RECONNECT_JOIN_WINDOW, SAFETY_MARGIN, SCREEN_HEIGHT, SCREEN_WIDTH, WHEELBASE,
)
from .kinematics import _lib_step, canStopClear, feasibleAccel, propagate
from .route import joinTolerance, routeJoin
from .tree import action, edge, state


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
