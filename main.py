import math
import numpy as np
from dataclasses import dataclass

import pygame
SCREEN_WIDTH = 2000
SCREEN_HEIGHT = 1000
SCREEN_SIZE = (2000, 1000)
FPS = 60
TARGET = (1950, 700)
TARGET_RAD = (70)
TARGET_SPEED = 5
# Car geometry (pixels)
CAR_LENGTH = 20
CAR_WIDTH = 10
WHEELBASE = 14  # distance between front and rear axles
WHEEL_LENGTH = 5
WHEEL_WIDTH = 2

DISTANCE_TIME_PROP = 1
delta_B_N = 7 
delta_S = 7 
# Car dynamics
MAX_ACCELERATION = 120  # px/s^2, limit on set_acceleration()
MAX_SPEED = 300  # px/s
MAX_STEER_RATE = math.radians(360)  # rad/s^2, limit on set_steering_rate()
MAX_STEER = math.radians(180)  # rad/s, limit on heading rate (d theta/dt)

# Obstacles: [x, y] centers and radii
GLOBAL_OBSTACLE_POS = [[574, 109], [1592, 867], [1425, 537], [1579, 85], [438, 146], [1885, 883], [894, 870], [1414, 70], [569, 816], [433, 522], [364, 841], [1515, 785], [1123, 921], [968, 218], [1412, 199], [1614, 521], [1397, 843], [88, 873], [1031, 834], [1128, 503], [1876, 520], [1100, 219], [892, 113], [1073, 81], [608, 518], [70, 91], [1925, 96], [122, 525], [460, 866], [971, 452], [884, 547], [1555, 213], [634, 937], [1512, 450], [361, 60], [129, 202], [1019, 946], [129, 433], [1060, 567], [1343, 454], [1647, 211], [1494, 894], [495, 45], [343, 195], [1665, 786], [449, 779], [1947, 430], [951, 780], [1901, 215], [846, 216], [970, 568], [375, 946], [353, 442], [119, 772], [1412, 436], [645, 39], [536, 235], [1083, 422], [1138, 787], [867, 431], [151, 956], [159, 824], [41, 473], [53, 247], [654, 855], [1854, 434], [1931, 761], [498, 944], [1861, 759], [1605, 783], [1486, 958], [1520, 569], [1158, 159], [663, 425], [628, 222], [1366, 958], [1006, 162], [439, 241], [1854, 38], [1650, 961], [333, 517], [866, 962], [1557, 934], [1850, 210], [382, 249], [660, 173], [42, 943], [439, 46], [1972, 233], [604, 413], [578, 894], [1859, 247], [647, 794], [1662, 907], [839, 799], [1116, 582], [1014, 747], [550, 932], [511, 414], [576, 24], [56, 520], [526, 460], [1340, 109], [351, 577], [172, 77], [1173, 108], [1958, 564], [1498, 165], [670, 94], [1959, 811], [1427, 934], [58, 798], [1975, 151], [332, 751], [965, 909], [149, 36], [868, 751], [77, 580], [1834, 132], [1021, 527], [1486, 64], [1616, 940], [457, 428], [1325, 68], [594, 747], [383, 770], [1361, 752], [1953, 956], [1969, 501], [1668, 25], [176, 481], [1616, 424], [1329, 910], [59, 410], [46, 763], [832, 459], [455, 971], [1030, 239], [47, 184], [661, 134], [329, 125], [1671, 169], [1134, 30], [904, 244], [1488, 123], [1162, 428], [976, 53], [1350, 28], [955, 945], [1860, 593], [1837, 964], [167, 775], [409, 417], [1085, 762], [498, 581], [1339, 157], [1964, 23], [1338, 573], [1338, 255], [994, 120], [154, 117]]

GLOBAL_OBSTACLE_RAD = [64, 45, 63, 73, 57, 71, 61, 46, 54, 62, 46, 47, 56, 45, 63, 59, 74, 56, 60, 56, 46, 46, 69, 63, 70, 52, 53, 48, 49, 47, 52, 50, 49, 48, 50, 54, 41, 34, 27, 32, 29, 28, 29, 27, 25, 34, 28, 28, 31, 35, 27, 35, 40, 29, 26, 26, 33, 25, 36, 27, 27, 25, 30, 26, 31, 31, 32, 30, 26, 28, 28, 32, 31, 27, 40, 29, 16, 15, 17, 15, 16, 22, 23, 15, 17, 12, 14, 20, 15, 17, 13, 14, 19, 18, 14, 18, 14, 13, 15, 14, 13, 20, 25, 13, 17, 17, 24, 13, 12, 14, 13, 19, 12, 21, 14, 16, 21, 17, 21, 22, 17, 15, 23, 13, 16, 21, 15, 21, 16, 15, 12, 19, 17, 15, 13, 14, 17, 12, 14, 14, 17, 14, 13, 17, 13, 21, 12, 12, 19, 14, 14, 12, 18, 16, 18, 16, 12, 13, 17, 16, 19]

LOCAL_OBSTACLE_POS = [[784, 116], [889, 692], [1258, 643], [277, 763], [697, 676], [941, 287], [1200, 663], [1649, 714], [1715, 160], [1780, 697], [1781, 628], [1800, 76], [786, 772], [804, 47], [1002, 614], [611, 290], [1695, 939], [1541, 307], [1194, 881], [1214, 192], [1247, 378], [106, 623], [1194, 584], [1178, 711], [180, 310], [787, 475], [1291, 41], [239, 627], [1886, 380], [343, 647], [1769, 740], [1719, 275], [219, 174], [628, 718], [1420, 304], [725, 403], [1269, 295], [1288, 129], [283, 917], [1206, 292], [856, 286], [487, 283], [1826, 279], [1195, 39], [1717, 674], [1773, 836], [1702, 569], [806, 824], [1709, 828], [1555, 366], [288, 468], [1779, 379], [1451, 624], [1973, 722], [107, 698], [1227, 725], [198, 385], [298, 574], [1016, 387], [214, 563], [1560, 715], [1296, 512], [674, 620], [630, 360], [47, 631], [1659, 280], [305, 359], [727, 66], [216, 494], [1077, 621], [689, 359], [1905, 698], [1793, 952], [1591, 630], [124, 286], [798, 632], [821, 716], [1187, 361], [737, 279], [1778, 539], [228, 292], [954, 710], [1893, 302], [1211, 516], [1707, 79], [1705, 421], [722, 825], [1695, 616], [29, 356], [1950, 633], [288, 146], [1965, 296], [559, 371], [1028, 710], [1769, 223], [230, 706], [426, 723], [296, 708], [210, 938], [701, 550], [1723, 761], [902, 625], [429, 296], [730, 329], [1826, 719], [572, 648], [1311, 374], [273, 257], [1392, 628], [809, 406], [915, 375], [1274, 183], [768, 963], [1298, 871], [1960, 367], [192, 665], [210, 765], [698, 224], [270, 856], [717, 908], [1800, 180], [1808, 464], [297, 204], [1223, 448], [418, 368], [1317, 714], [1479, 282], [776, 880], [1408, 360], [696, 469], [1731, 349], [1292, 821], [694, 136], [447, 614], [675, 303], [1413, 696], [363, 377], [952, 610], [1803, 342], [1279, 597], [793, 351], [1719, 218], [855, 632], [1704, 520], [1846, 621], [1497, 378], [1019, 273], [208, 55], [1217, 974], [1089, 688], [25, 282], [1138, 306], [1111, 726], [352, 290], [726, 642], [167, 619], [1528, 634], [795, 547], [1132, 643], [1285, 459], [1214, 88], [1488, 708], [195, 113], [629, 611], [1327, 636], [1291, 934], [549, 710], [173, 717], [206, 863], [199, 235], [1068, 310], [491, 363], [729, 179], [1902, 638], [1833, 369], [753, 703], [1623, 372], [236, 978], [1362, 317], [1588, 282], [728, 748], [1723, 891], [296, 972], [106, 386], [1214, 810], [273, 409], [1767, 491], [1281, 751], [48, 714], [486, 719], [1098, 372], [283, 86], [691, 973], [211, 436], [505, 624], [1697, 724], [785, 173], [1032, 648], [1786, 786], [1780, 298], [805, 935], [782, 302], [358, 710], [792, 255], [297, 26], [546, 313], [1638, 608], [68, 304], [863, 376], [158, 367], [730, 949], [1726, 32], [1299, 243], [390, 628], [1232, 22], [1273, 695]]

LOCAL_OBSTACLE_RAD = [17, 12, 10, 17, 10, 17, 9, 13, 17, 10, 16, 10, 11, 16, 8, 9, 15, 19, 16, 13, 18, 15, 13, 9, 7, 13, 10, 18, 12, 11, 8, 16, 19, 9, 16, 17, 11, 19, 9, 18, 18, 15, 9, 9, 8, 12, 8, 18, 19, 9, 15, 12, 9, 11, 19, 16, 8, 14, 16, 12, 16, 14, 8, 12, 16, 10, 20, 13, 10, 16, 14, 13, 17, 12, 19, 12, 16, 9, 12, 11, 10, 14, 9, 8, 11, 16, 13, 7, 10, 10, 17, 9, 19, 14, 9, 17, 19, 17, 13, 19, 11, 10, 13, 10, 15, 13, 19, 11, 16, 20, 17, 12, 9, 8, 14, 11, 8, 17, 13, 12, 18, 16, 13, 12, 14, 15, 14, 9, 11, 18, 10, 10, 11, 13, 14, 15, 11, 8, 8, 16, 10, 12, 8, 16, 10, 19, 15, 16, 12, 7, 7, 18, 10, 14, 8, 15, 15, 9, 15, 16, 15, 12, 12, 8, 12, 13, 16, 14, 10, 7, 8, 10, 11, 13, 9, 9, 16, 8, 9, 10, 9, 12, 18, 10, 14, 13, 8, 11, 14, 10, 20, 10, 12, 8, 19, 7, 13, 12, 9, 14, 13, 10, 10, 10, 15, 11, 9, 14, 10, 11, 7, 12, 11, 12, 7, 8]


@dataclass(frozen=True)
class CarState:
    x: float  # px, rear axle center
    y: float  # px
    velocity: float  # px/s, negative = reversing
    heading: float  # rad, 0 = +x, positive = clockwise on screen


class Car:
    """Kinematic model driven by heading rate. The pose is defined at the rear axle center.

    Control it by setting rates; update() integrates them each frame:
        car.set_acceleration(px_per_s2)    # clamped to +/-MAX_ACCELERATION
        car.set_steering_rate(rad_per_s2)  # d^2(theta)/dt^2, clamped to +/-MAX_STEER_RATE
        car.get_state() -> CarState
    """

    def __init__(self, x, y, heading=0.0):
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = 0.0
        self.steering = 0.0  # rad/s, heading rate d(theta)/dt
        self.acceleration = 0.0  # px/s^2, current speed rate of change
        self.steering_rate = 0.0  # rad/s^2, current heading rate of change

    def get_state(self):
        return CarState(self.x, self.y, self.velocity, self.heading)

    def set_acceleration(self, acceleration):
        """Rate of change of speed in px/s^2. Returns the clamped value actually applied."""
        self.acceleration = clamp(acceleration, -MAX_ACCELERATION, MAX_ACCELERATION)
        return self.acceleration

    def set_steering_rate(self, steering_rate):
        """Rate of change of heading rate in rad/s^2 (positive = right). Returns the clamped value actually applied."""
        self.steering_rate = clamp(steering_rate, -MAX_STEER_RATE, MAX_STEER_RATE)
        return self.steering_rate

    def update(self, dt):
        self.velocity = clamp(self.velocity + self.acceleration * dt, -MAX_SPEED, MAX_SPEED)
        self.steering = clamp(self.steering + self.steering_rate * dt, -MAX_STEER, MAX_STEER)

        self.x += self.velocity * math.cos(self.heading) * dt
        self.y += self.velocity * math.sin(self.heading) * dt
        self.heading += self.steering * dt
        self.heading %= 2 * math.pi

    def draw(self, surface):
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
        pygame.draw.polygon(surface, "red", rotated_rect(body_center, CAR_LENGTH, CAR_WIDTH, self.heading))

        # Wheels: rear follow the body, front show the wheel angle that would produce
        # the current heading rate (tan(delta) = steering * WHEELBASE / velocity)
        half_track = CAR_WIDTH / 2
        direction = math.copysign(1, self.velocity)
        wheel_steer = math.atan2(self.steering * WHEELBASE * direction, abs(self.velocity))
        for forward, wheel_angle in ((0, self.heading), (WHEELBASE, self.heading + wheel_steer)):
            for left in (half_track, -half_track):
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


def draw_obstacles(surface):
    for pos, rad in zip(GLOBAL_OBSTACLE_POS, GLOBAL_OBSTACLE_RAD):
        pygame.draw.circle(surface, "blue", pos, rad)
    for pos, rad in zip(LOCAL_OBSTACLE_POS, LOCAL_OBSTACLE_RAD):
        pygame.draw.circle(surface, "green", pos, rad)


def load_font():
    # pygame 2.6.1's font module is broken on Python 3.14; fall back to the window title
    try:
        return pygame.font.SysFont(None, 28)
    except (NotImplementedError, ImportError):
        return None


def draw_hud(surface, font, car):
    text = f"speed: {car.velocity:6.1f} px/s   heading rate: {math.degrees(car.steering):6.1f} deg/s"
    if font:
        surface.blit(font.render(text, True, "white"), (10, 10))
    else:
        pygame.display.set_caption(text)

class state:
    
    def __init__(self, x1, x2, a, s):
        self.x1 = x1
        self.x2 = x2 
        self.a = a 
        self.s = s
        self.x1bar = self.x0 + self.s * math.cos(self.a)*DISTANCE_TIME_PROP
        self.x2bar = self.x2 + self.s * math.sin(self.a)*DISTANCE_TIME_PROP 

    def distance(self, s2):
        return (math.sqrt((self.x1bar - s2.x1bar)**2 + (self.x2bar - s2.x2bar)))
    
    
class action:
    def __init__(self, da, ds):
        self.da = da 
        self.ds = ds 

class edge:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2


class SSTTree:
    def __init__(self, s1):
        self.active = np.array([s1])

        self.inactive = np.array([])
    
        self.E = np.array([])
def getRandomState():
    x_1_rand = random.unif(0, SCREEN_WIDTH) 
    x_2_rand = random.unif(0, SCREEN_HEIGHT)
    a_rand = random.unif(0, 2*math.pi())
    speed_rand = random.unif(0, 10) #ask claude how we should define this 

    return state(x_1_rand, x_2_rand, a_rand, speed_rand)

def near(nodeList, s1):
    x_near = np.array([])
    for i in range(len(nodeList)):
        if tree.active[i] <= delta_B_N:
            x_near.append(nodeList[i])

    return x_near, min(nodeList.distance(s1))
        


def bestFirstSelectionSST(t):
    x_rand = getRandomState()
    x_near = near(t, x_rand)

    return 



def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()
    font = load_font()
    car = Car(100, 100)

    dt = 0.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        throttle = keys[pygame.K_w] - keys[pygame.K_s]
        steer = keys[pygame.K_d] - keys[pygame.K_a]
        car.set_acceleration(throttle * MAX_ACCELERATION)
        car.set_steering_rate(steer * MAX_STEER_RATE)
        car.update(dt)

        screen.fill("black")
        draw_obstacles(screen)
        car.draw(screen)
        draw_hud(screen, font, car)

        pygame.draw.circle(screen, "PURPLE", TARGET, TARGET_RAD)
        pygame.display.flip()

        dt = clock.tick(FPS) / 1000
    pygame.quit()


if __name__ == "__main__":
    main()
