"""Car motion: compiled propagation and clearance checks, random actions, and braking to a stop."""
import math
import random

import numpy as np
from numba import njit

from .car import _max_steer, clamp, maxSteerAtSpeed
from .config import (
    CHECK_SPACING, COST_ACCEL, COST_DECEL, COST_STEER_ANGLE, COST_STEER_RATE, EARLY_ACCEL_SIGMA,
    FOOTPRINT_OFFSETS_ARRAY, FOOTPRINT_RADIUS, FOOTPRINT_REACH, MAX_ACCELERATION, MAX_LATERAL_ACCELERATION,
    MAX_REVERSE_SPEED, MAX_SPEED, MAX_STEER, MAX_STEER_RATE, PLAN_DT, SAFE_SPEED_STEP, SAFETY_MARGIN,
    STEER_TARGET_SIGMA, STOP_SPEED_BINS, STOP_STEER_BINS, TIME_PROP, WHEELBASE,
)
from .tree import action, collisionCheckPoints, state


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
