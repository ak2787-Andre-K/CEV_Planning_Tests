"""The car model, its steering limit, and drawing it."""
import math
from dataclasses import dataclass

import pygame
from numba import njit

from .config import (
    CAR_LENGTH, CAR_WIDTH, MAX_ACCELERATION, MAX_LATERAL_ACCELERATION, MAX_SPEED, MAX_STEER, MAX_STEER_RATE,
    WHEEL_LENGTH, WHEEL_WIDTH, WHEELBASE,
)


@dataclass(frozen=True)
class CarState:
    x: float  # px, rear axle center
    y: float  # px
    velocity: float  # px/s, negative = reversing
    heading: float  # rad, 0 = +x, positive = clockwise on screen


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

# Compiled kernels (numba). The car's limits come in as arguments, not module constants (numba would freeze those at
# compile time), so changing a setting at runtime still applies. The functions below them keep their signatures.

@njit(cache=True)
def _max_steer(speed, max_steer, max_lateral, wheelbase):
    if speed == 0:
        return max_steer
    return min(max_steer, math.atan(max_lateral * wheelbase / speed ** 2))


def maxSteerAtSpeed(speed):
    return _max_steer(float(speed), MAX_STEER, MAX_LATERAL_ACCELERATION, WHEELBASE)
