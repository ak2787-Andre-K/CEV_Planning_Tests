"""Routes the car follows, and joining or splicing new states onto them."""
import bisect
import math

import numpy as np

from .car import clamp
from .config import (
    CHECK_SPACING, COST_STEER_ANGLE, FOOTPRINT_OFFSETS, FOOTPRINT_RADIUS, JOIN_SPEED_OVER, MAX_LATERAL_ACCELERATION,
    MAX_STEER, MAX_STEER_RATE, SAFETY_MARGIN, SCREEN_HEIGHT, SCREEN_WIDTH, WHEELBASE,
)


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
