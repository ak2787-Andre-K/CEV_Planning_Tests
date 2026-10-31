"""The Driver: follows a route, replans around obstacles the lidar finds, and the RRT fallback."""
import bisect
import copy
import heapq
import itertools
import math
import os
import random
import time as clock_time

import numpy as np
import pygame

from .car import clamp, maxSteerAtSpeed
from .config import (
    CAR_WIDTH, CHECK_SPACING, CREEP_SPEED, DEAD_REPLAN_ITERATIONS, DEAD_REPLAN_NODES, FPS, GROUP_BACK_NODES,
    GROUP_G_WEIGHT, GROUP_HEADING_SCALE, GROUP_K, GROUP_MIN_DISTANCE, GROUP_MIN_SIZE, GROUP_PAST_FIRST,
    GROUP_RETRY_INTERVAL, GROUP_SAMPLES, GROUP_SPEED_SCALE, GROUP_STUCK_TIME, H_CELL, IMPROVE_MIN_GAIN,
    IMPROVE_SKIP_FRAMES, JOIN_SPEED_OVER, JOIN_TOLERANCE_SCALE, LIB_SPEEDS, LIB_TRIES_EVERY, MAX_ACCELERATION,
    MAX_LATERAL_ACCELERATION, MAX_REVERSE_SPEED, MAX_SPEED, MAX_STEER, MOTION_DT, NEAR_FOCUS_SHARE, NEAR_JOIN_SCALE,
    NEAR_LOOKAHEAD, NEAR_SAMPLE_SIGMA, NONSTOP, NONSTOP_LEAD, NUDGE_BACK_LENGTH, NUDGE_NODE_FRAMES,
    NUDGE_SHIFT_LENGTH, NUDGE_VARIANTS, OBSTACLE_LOG, OBSTACLE_LOG_DIR, OBSTACLE_LOG_PATCH, OCCUPIED, PAST_BEYOND,
    PAST_FOCUS_SHARE, PAST_LOOKAHEAD, PAST_RADIUS, PAST_SAMPLE_SIGMA, PLAN_DT, RECONNECT_JOIN_WINDOW,
    RECONNECT_LOOKAHEADS, RECONNECT_MAX_TARGETS, RECONNECT_MAX_TURN, RECONNECT_RADIUS, RECONNECT_TANGENTS,
    RECONNECT_TARGET_SPACING, REPLAN_BACK_NODES, REPLAN_ITERATIONS_PER_FRAME, REPLAN_SAMPLE_GROW_EVERY,
    REPLAN_SAMPLE_GROWTH, REPLAN_SAMPLE_RADIUS, RRT_ENABLED, RRT_FOCUS_RADIUS, RRT_FOCUS_SHARE, RRT_LOOKAHEAD,
    RRT_RANDOM, RRT_SAMPLE_SIGMA, RRT_STOP_DELAY, SAFE_SPEED_NODES, SAFETY_MARGIN, SCREEN_HEIGHT, SCREEN_WIDTH,
    SKELETON_CORNER_RADIUS, SQUIGGLY_STEER_FRACTION, STALL_REPLAN_TIME, STOP_FAN_FRACTION, STOP_FAN_TESTS, TARGET,
    TARGET_RAD, TURN_HEADING_TOL, TURN_MAX_ENTRY_SPEED, TURN_NEAR_RADIUS, TURN_PRIMITIVE_STUCK_TIME,
    TURNAROUND_MANEUVERS, W_A, W_ST, W_V, WHEELBASE,
)
from .global_planner import GlobalPlanner, goalDistanceGrid, gridDistance, gridValue, kMeans, pathStates
from .kinematics import canStopClear, feasibleAccel, pathIsClear, propagate, safeSpeed
from .maneuvers import libRange, libraryJoin, runTurnaround
from .route import joinTolerance, Route, spliceRoute
from .smoothing import routeLoopy, shortcutDrive
from .tree import action, proximityHashMap, state


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
            print('Driver: reached the goal', flush=True)
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
