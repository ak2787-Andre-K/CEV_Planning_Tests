"""Geometric RRT: a point robot grows a tree in straight steps, then the car follows the path."""
import math
import random

import pygame

from world import (
    FPS, MAX_ACCELERATION, MAX_STEER, MAX_STEER_RATE, SCREEN_HEIGHT, SCREEN_SIZE, SCREEN_WIDTH,
    START, TARGET, WINDOW_SIZE, Car, angle_diff, clamp, collision_free, dist, draw_hud,
    draw_world, load_font, reached_goal,
)

STEP_SIZE = 20
GOAL_BIAS = 0.05
MAX_ITERS = 5000
# path following
FOLLOW_SPEED = 80  # px/s, cruising speed along the path
LOOKAHEAD = 25  # px, aim at the first path point at least this far ahead
K_HEADING = 4  # heading error (rad) -> desired heading rate (rad/s)
K_STEER = 10  # heading rate error -> steering rate
K_SPEED = 2  # speed error -> acceleration


class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent


def sample_point():
    if random.random() < GOAL_BIAS:
        return TARGET
    else:
        return (random.uniform(0, SCREEN_WIDTH), random.uniform(0, SCREEN_HEIGHT))
    
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


def draw_rrt(surface, tree, path):
    for node in tree:
        if node.parent is not None:
            pygame.draw.line(surface, "gray", (node.parent.x, node.parent.y), (node.x, node.y), 2)
    if path:
        pygame.draw.lines(surface, "yellow", False, path, 4)


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


def main():
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    world = pygame.Surface(SCREEN_SIZE)  # full-size map; shrunk onto the window each frame
    clock = pygame.time.Clock()
    font = load_font()

    tree, path = rrt(START)
    if path:
        print(f"RRT reached the goal: {len(tree)} nodes, path of {len(path)} points")
    else:
        print(f"RRT did not reach the goal in {MAX_ITERS} iterations ({len(tree)} nodes)")

    # Start the car at the RRT start, pointing along the first path segment
    heading = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0]) if path else 0.0
    car = Car(START[0], START[1], heading)
    waypoint = 0

    dt = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        throttle = keys[pygame.K_w] - keys[pygame.K_s]
        steer = keys[pygame.K_d] - keys[pygame.K_a]
        if path and not (throttle or steer):
            waypoint = follow_path(car, path, waypoint)  # autopilot along the RRT path
        else:
            car.set_acceleration(throttle * MAX_ACCELERATION)  # WASD takes over while held
            car.set_steering_rate(steer * MAX_STEER_RATE)
        car.update(dt)

        world.fill("black")
        draw_world(world)
        draw_rrt(world, tree, path)
        car.draw(world)

        screen.blit(pygame.transform.smoothscale(world, WINDOW_SIZE), (0, 0))
        draw_hud(screen, font, car)
        pygame.display.flip()

        dt = clock.tick(FPS) / 1000
    pygame.quit()


if __name__ == "__main__":
    main()
