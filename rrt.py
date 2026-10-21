"""Geometric RRT: a point robot grows a tree in straight steps, then the car follows the path."""
import math
import random

import numpy as np

import cev_sim
from world import MAX_ACCELERATION, MAX_STEER, MAX_STEER_RATE, Car, angle_diff, clamp, dist

ENVIRONMENT = "first-demo"
VEHICLE = "merceves"
STEP_SIZE = 0.4  # m
GOAL_BIAS = 0.05
MAX_ITERS = 5000
GOAL_RADIUS = 0.6  # m
# path following
FOLLOW_SPEED = 2.0  # m/s, cruising speed along the path
LOOKAHEAD = 1.5  # m, aim at the first path point at least this far ahead
K_HEADING = 4  # heading error (rad) -> desired heading rate (rad/s)
K_STEER = 10  # heading rate error -> steering rate
K_SPEED = 2  # speed error -> acceleration

# Filled in by prepare() from the cev_sim world.
GRID = None
BOUNDS = None
START = None
TARGET = None
WHEELBASE = None
MAX_ANGLE = None
MAX_SPEED = None


class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent


def prepare(world):
    """Bounds, start, and goal from a cev_sim PlanningWorld."""
    global GRID, BOUNDS, START, TARGET, WHEELBASE, MAX_ANGLE, MAX_SPEED
    GRID = world.occupancy_grid(0.25, inflate_m=world.vehicle.width_m * 0.5)
    rows, cols = GRID.cells.shape
    BOUNDS = (
        GRID.origin_x,
        GRID.origin_z,
        GRID.origin_x + cols * GRID.resolution_m,
        GRID.origin_z + rows * GRID.resolution_m,
    )
    START = free_endpoint(world.start.x, world.start.z, world.route_polyline)
    TARGET = free_endpoint(world.goal.x, world.goal.z, world.route_polyline[::-1])
    WHEELBASE = world.vehicle.wheelbase_m
    MAX_ANGLE = world.vehicle.max_steering_rad
    MAX_SPEED = world.vehicle.max_speed_mps


def cell_free(x, y):
    row, col = GRID.world_to_grid(x, y)
    rows, cols = GRID.cells.shape
    if row < 0 or col < 0 or row >= rows or col >= cols:
        return False
    return GRID.cells[row, col] == 0


def free_endpoint(x, y, polyline):
    """The pose itself when it is free, otherwise the first free point along the route."""
    if cell_free(x, y):
        return (x, y)
    for point in polyline:
        if cell_free(float(point[0]), float(point[1])):
            return (float(point[0]), float(point[1]))
    raise RuntimeError(f"{VEHICLE} does not fit on the {ENVIRONMENT} road.")


def collision_free(x1, y1, x2, y2):
    samples = int(dist(x1, y1, x2, y2) / GRID.resolution_m) + 1
    for i in range(samples + 1):
        t = i / samples
        if not cell_free(x1 + t * (x2 - x1), y1 + t * (y2 - y1)):
            return False
    return True


def reached_goal(x, y):
    return dist(x, y, TARGET[0], TARGET[1]) <= GOAL_RADIUS


def sample_point():
    if random.random() < GOAL_BIAS:
        return TARGET
    else:
        return (random.uniform(BOUNDS[0], BOUNDS[2]), random.uniform(BOUNDS[1], BOUNDS[3]))
    
def nearest_node(tree, x, y):
    best = tree[0]
    for node in tree:
        if dist(node.x, node.y, x, y) < dist(best.x, best.y, x, y):
            best = node
    return best

def steer(n_near, x, y):
    if dist(n_near.x, n_near.y, x, y) <= STEP_SIZE:
        return x, y
    theta = math.atan2(y - n_near.y, x - n_near.x)
    return n_near.x + STEP_SIZE * math.cos(theta), n_near.y + STEP_SIZE * math.sin(theta)


def get_path(node):
    path = []
    while node is not None:
        path.append((node.x, node.y))
        node = node.parent
    path.reverse()
    return path

def rrt(start):
    tree = [Node(start[0], start[1])]
    for _ in range(MAX_ITERS):
        x, y = sample_point()
        n_near = nearest_node(tree, x , y)
        next_x, next_y = steer(n_near, x, y)

        if collision_free(n_near.x, n_near.y, next_x, next_y):
            new_node = Node(next_x, next_y, n_near)
            tree.append(new_node)
            if reached_goal(next_x, next_y):
                return tree, get_path(new_node)

    return tree, None


def follow_path(car, path, index):
    """Steer the car toward the path point LOOKAHEAD px ahead. Returns the updated waypoint index."""
    # Skip past waypoints that are already close, so the car always aims a bit ahead of itself
    while index < len(path) - 1 and dist(car.x, car.y, *path[index]) < LOOKAHEAD:
        index += 1
    tx, ty = path[index]

    # Heading: turn toward the waypoint, faster the further off we are
    heading_error = angle_diff(math.atan2(ty - car.y, tx - car.x), car.heading)
    desired_rate = clamp(K_HEADING * heading_error, -MAX_STEER, MAX_STEER)
    car.set_steering_rate(K_STEER * (desired_rate - car.steering))

    # Speed: cruise, slow down in sharp turns, and stop once inside the target
    target_speed = 0 if reached_goal(car.x, car.y) else FOLLOW_SPEED * max(0.3, math.cos(heading_error))
    car.set_acceleration(K_SPEED * (target_speed - car.velocity))
    return index


def sync(car, sim):
    """Copy the sim pose. follow_path's heading is atan2(dz, dx), which is -yaw."""
    pose = sim.pose()
    velocity = sim.velocity()
    yaw = float(pose[2])
    car.x, car.y, car.heading = float(pose[0]), float(pose[1]), -yaw
    car.steering = -float(velocity[2])
    return yaw


def step_car(car, sim, dt):
    """Integrate the heading-rate command, then send the steering fraction."""
    car.velocity = clamp(car.velocity + car.acceleration * dt, -MAX_SPEED, MAX_SPEED)
    omega = clamp(car.steering + car.steering_rate * dt, -MAX_STEER, MAX_STEER)
    yaw = float(sim.pose()[2])
    velocity = sim.velocity()
    v = float(velocity[0]) * math.cos(yaw) + float(velocity[1]) * (-math.sin(yaw))
    # shortcut: yaw rate is undefined below 0.3 m/s; a creep reading also flips sign, so use the command
    if abs(v) < 0.3:
        v_map = 0.3 if car.velocity >= 0.0 else -0.3
    else:
        v_map = v
    steer = math.atan(omega * WHEELBASE / v_map) / max(MAX_ANGLE, 1e-6)
    action = np.array(
        [np.clip(car.velocity / MAX_SPEED, -1.0, 1.0), np.clip(steer, -1.0, 1.0)],
        dtype=np.float32,
    )
    sim.step(action)
    sync(car, sim)
    return action


def main():
    import matplotlib.pyplot as plt

    config = cev_sim.configure(environment=ENVIRONMENT, vehicle=VEHICLE, goal=None)
    sim = None
    try:
        prepare(config.world)
        tree, path = rrt(START)
        if path:
            print(f"RRT reached the goal: {len(tree)} nodes, path of {len(path)} points")
        else:
            print(f"RRT did not reach the goal in {MAX_ITERS} iterations ({len(tree)} nodes)")

        car = Car(START[0], START[1])
        sim = cev_sim.launch(config, render_mode="human")
        sim.reset(seed=1)
        sync(car, sim)
        if path:
            sim.set_overlay(path)
        held = set()

        def on_press(event):
            if event.key:
                held.add(event.key.lower())

        def on_release(event):
            if event.key:
                held.discard(event.key.lower())

        figure = plt.gcf()
        figure.canvas.mpl_connect("key_press_event", on_press)
        figure.canvas.mpl_connect("key_release_event", on_release)
        waypoint = 0
        while sim.loop() and plt.fignum_exists(figure.number):
            throttle = ("w" in held) - ("s" in held)
            steer = ("d" in held) - ("a" in held)
            if path and not (throttle or steer):
                waypoint = follow_path(car, path, waypoint)  # autopilot along the RRT path
            else:
                car.set_acceleration(throttle * MAX_ACCELERATION)  # WASD takes over while held
                car.set_steering_rate(steer * MAX_STEER_RATE)
            step_car(car, sim, sim.clock.dt())
    finally:
        if sim is not None:
            sim.close()
        config.close()


if __name__ == "__main__":
    main()
