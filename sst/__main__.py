"""Runs the SST demo window: python -m sst"""
import bisect
import math

import numpy as np
import pygame

from .car import Car, clamp, draw_hud, load_font
from .config import (
    FPS, GLOBAL_PLAN_SEES_LOCAL, LIDAR_DT, MAX_ACCELERATION, MAX_STEER_RATE, OCCUPIED, PATH_LIBRARY_FILE,
    PATH_LIBRARY_MODE, PREPROCESS_BATCHES, REPLAN_DEMO, REPLAY_MAX_STOP, RESTART_DELAY, SCALE, SCREEN_HEIGHT,
    SCREEN_SIZE, SCREEN_WIDTH, START, START_HEADING, TARGET, TARGET_RAD,
)
from .driver import Driver
from .global_planner import pathStates, planGlobal, replanDemo
from .route import Route
from .smoothing import loadPathLibrary, savePathLibrary, smoothPath
from .streets import diagonal_roads, LocalObstacleMap, rasterize, SDFState, street_blocks
from .tree import state


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
