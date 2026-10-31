"""The street map: blocks, roads and obstacles, the signed distance field, and the hidden obstacles the lidar reveals."""
import math
import random
from dataclasses import dataclass

import numpy as np
import pygame

from .config import (
    BLOCK_CORNER, BLOCKED_ROADS, DIAGONALS, DIRTY_TILE, FREE, LIDAR_BEAMS, LOCAL_BOX_LENGTH, LOCAL_BOX_WIDTH,
    LOCAL_OBSTACLE_COUNT, LOCAL_OBSTACLE_RADIUS, LOCAL_SEED, OCCUPIED, SCREEN_HEIGHT, SCREEN_WIDTH, SDF_CAP, START,
    STREET_WIDTH, STREET_X, STREET_Y, TARGET, UNKNOWN,
)


# Obstacles: [x, y] centers and radii
GLOBAL_OBSTACLE_POS = [[574, 109], [1592, 867], [1425, 537], [1579, 85], [438, 146], [1885, 883], [894, 870], [1414, 70], [569, 816], [433, 522], [364, 841], [1515, 785], [1123, 921], [968, 218], [1412, 199], [1614, 521], [1397, 843], [88, 873], [1031, 834], [1128, 503], [1876, 520], [1100, 219], [892, 113], [1073, 81], [608, 518], [70, 91], [1925, 96], [122, 525], [460, 866], [971, 452], [884, 547], [1555, 213], [634, 937], [1512, 450], [361, 60], [129, 202], [1019, 946], [129, 433], [1060, 567], [1343, 454], [1647, 211], [1494, 894], [495, 45], [343, 195], [1665, 786], [449, 779], [1947, 430], [951, 780], [1901, 215], [846, 216], [970, 568], [375, 946], [353, 442], [119, 772], [1412, 436], [645, 39], [536, 235], [1083, 422], [1138, 787], [867, 431], [151, 956], [159, 824], [41, 473], [53, 247], [654, 855], [1854, 434], [1931, 761], [498, 944], [1861, 759], [1605, 783], [1486, 958], [1520, 569], [1158, 159], [663, 425], [628, 222], [1366, 958], [1006, 162], [439, 241], [1854, 38], [1650, 961], [333, 517], [866, 962], [1557, 934], [1850, 210], [382, 249], [660, 173], [42, 943], [439, 46], [1972, 233], [604, 413], [578, 894], [1859, 247], [647, 794], [1662, 907], [839, 799], [1116, 582], [1014, 747], [550, 932], [511, 414], [576, 24], [56, 520], [526, 460], [1340, 109], [351, 577], [172, 77], [1173, 108], [1958, 564], [1498, 165], [670, 94], [1959, 811], [1427, 934], [58, 798], [1975, 151], [332, 751], [965, 909], [149, 36], [868, 751], [77, 580], [1834, 132], [1021, 527], [1486, 64], [1616, 940], [457, 428], [1325, 68], [594, 747], [383, 770], [1361, 752], [1953, 956], [1969, 501], [1668, 25], [176, 481], [1616, 424], [1329, 910], [59, 410], [46, 763], [832, 459], [455, 971], [1030, 239], [47, 184], [661, 134], [329, 125], [1671, 169], [1134, 30], [904, 244], [1488, 123], [1162, 428], [976, 53], [1350, 28], [955, 945], [1860, 593], [1837, 964], [167, 775], [409, 417], [1085, 762], [498, 581], [1339, 157], [1964, 23], [1338, 573], [1338, 255], [994, 120], [154, 117]]

GLOBAL_OBSTACLE_RAD = [64, 45, 63, 73, 57, 71, 61, 46, 54, 62, 46, 47, 56, 45, 63, 59, 74, 56, 60, 56, 46, 46, 69, 63, 70, 52, 53, 48, 49, 47, 52, 50, 49, 48, 50, 54, 41, 34, 27, 32, 29, 28, 29, 27, 25, 34, 28, 28, 31, 35, 27, 35, 40, 29, 26, 26, 33, 25, 36, 27, 27, 25, 30, 26, 31, 31, 32, 30, 26, 28, 28, 32, 31, 27, 40, 29, 16, 15, 17, 15, 16, 22, 23, 15, 17, 12, 14, 20, 15, 17, 13, 14, 19, 18, 14, 18, 14, 13, 15, 14, 13, 20, 25, 13, 17, 17, 24, 13, 12, 14, 13, 19, 12, 21, 14, 16, 21, 17, 21, 22, 17, 15, 23, 13, 16, 21, 15, 21, 16, 15, 12, 19, 17, 15, 13, 14, 17, 12, 14, 14, 17, 14, 13, 17, 13, 21, 12, 12, 19, 14, 14, 12, 18, 16, 18, 16, 12, 13, 17, 16, 19]

LOCAL_OBSTACLE_POS = [[784, 116], [889, 692], [1258, 643], [277, 763], [697, 676], [941, 287], [1200, 663], [1649, 714], [1715, 160], [1780, 697], [1781, 628], [1800, 76], [786, 772], [804, 47], [1002, 614], [611, 290], [1695, 939], [1541, 307], [1194, 881], [1214, 192], [1247, 378], [106, 623], [1194, 584], [1178, 711], [180, 310], [787, 475], [1291, 41], [239, 627], [1886, 380], [343, 647], [1769, 740], [1719, 275], [219, 174], [628, 718], [1420, 304], [725, 403], [1269, 295], [1288, 129], [283, 917], [1206, 292], [856, 286], [487, 283], [1826, 279], [1195, 39], [1717, 674], [1773, 836], [1702, 569], [806, 824], [1709, 828], [1555, 366], [288, 468], [1779, 379], [1451, 624], [1973, 722], [107, 698], [1227, 725], [198, 385], [298, 574], [1016, 387], [214, 563], [1560, 715], [1296, 512], [674, 620], [630, 360], [47, 631], [1659, 280], [305, 359], [727, 66], [216, 494], [1077, 621], [689, 359], [1905, 698], [1793, 952], [1591, 630], [124, 286], [798, 632], [821, 716], [1187, 361], [737, 279], [1778, 539], [228, 292], [954, 710], [1893, 302], [1211, 516], [1707, 79], [1705, 421], [722, 825], [1695, 616], [29, 356], [1950, 633], [288, 146], [1965, 296], [559, 371], [1028, 710], [1769, 223], [230, 706], [426, 723], [296, 708], [210, 938], [701, 550], [1723, 761], [902, 625], [429, 296], [730, 329], [1826, 719], [572, 648], [1311, 374], [273, 257], [1392, 628], [809, 406], [915, 375], [1274, 183], [768, 963], [1298, 871], [1960, 367], [192, 665], [210, 765], [698, 224], [270, 856], [717, 908], [1800, 180], [1808, 464], [297, 204], [1223, 448], [418, 368], [1317, 714], [1479, 282], [776, 880], [1408, 360], [696, 469], [1731, 349], [1292, 821], [694, 136], [447, 614], [675, 303], [1413, 696], [363, 377], [952, 610], [1803, 342], [1279, 597], [793, 351], [1719, 218], [855, 632], [1704, 520], [1846, 621], [1497, 378], [1019, 273], [208, 55], [1217, 974], [1089, 688], [25, 282], [1138, 306], [1111, 726], [352, 290], [726, 642], [167, 619], [1528, 634], [795, 547], [1132, 643], [1285, 459], [1214, 88], [1488, 708], [195, 113], [629, 611], [1327, 636], [1291, 934], [549, 710], [173, 717], [206, 863], [199, 235], [1068, 310], [491, 363], [729, 179], [1902, 638], [1833, 369], [753, 703], [1623, 372], [236, 978], [1362, 317], [1588, 282], [728, 748], [1723, 891], [296, 972], [106, 386], [1214, 810], [273, 409], [1767, 491], [1281, 751], [48, 714], [486, 719], [1098, 372], [283, 86], [691, 973], [211, 436], [505, 624], [1697, 724], [785, 173], [1032, 648], [1786, 786], [1780, 298], [805, 935], [782, 302], [358, 710], [792, 255], [297, 26], [546, 313], [1638, 608], [68, 304], [863, 376], [158, 367], [730, 949], [1726, 32], [1299, 243], [390, 628], [1232, 22], [1273, 695]]

LOCAL_OBSTACLE_RAD = [17, 12, 10, 17, 10, 17, 9, 13, 17, 10, 16, 10, 11, 16, 8, 9, 15, 19, 16, 13, 18, 15, 13, 9, 7, 13, 10, 18, 12, 11, 8, 16, 19, 9, 16, 17, 11, 19, 9, 18, 18, 15, 9, 9, 8, 12, 8, 18, 19, 9, 15, 12, 9, 11, 19, 16, 8, 14, 16, 12, 16, 14, 8, 12, 16, 10, 20, 13, 10, 16, 14, 13, 17, 12, 19, 12, 16, 9, 12, 11, 10, 14, 9, 8, 11, 16, 13, 7, 10, 10, 17, 9, 19, 14, 9, 17, 19, 17, 13, 19, 11, 10, 13, 10, 15, 13, 19, 11, 16, 20, 17, 12, 9, 8, 14, 11, 8, 17, 13, 12, 18, 16, 13, 12, 14, 15, 14, 9, 11, 18, 10, 10, 11, 13, 14, 15, 11, 8, 8, 16, 10, 12, 8, 16, 10, 19, 15, 16, 12, 7, 7, 18, 10, 14, 8, 15, 15, 9, 15, 16, 15, 12, 12, 8, 12, 13, 16, 14, 10, 7, 8, 10, 11, 13, 9, 9, 16, 8, 9, 10, 9, 12, 18, 10, 14, 13, 8, 11, 14, 10, 20, 10, 12, 8, 19, 7, 13, 12, 9, 14, 13, 10, 10, 10, 15, 11, 9, 14, 10, 11, 7, 12, 11, 12, 7, 8]


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float

    @property
    def reach(self):
        return self.radius

    def bounds(self, margin):
        reach = self.radius + margin
        return self.x - reach, self.x + reach, self.y - reach, self.y + reach

    def signed_distance(self, xs, ys):
        return np.hypot(xs - self.x, ys - self.y) - self.radius


def rounded_rect_distance(u, v, half_w, half_h, corner):
    """Signed distance to a rectangle centered at the origin, for points (u, v) in the rectangle's own frame."""
    r = min(corner, half_w, half_h)
    qx = np.abs(u) - (half_w - r)
    qy = np.abs(v) - (half_h - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - r


@dataclass(frozen=True)
class Box:
    """Rectangle centered at (x, y), `length` along `angle` (rad) and `width` across it, with rounded corners."""
    x: float
    y: float
    length: float
    width: float
    angle: float = 0.0
    corner: float = 0.0

    @property
    def reach(self):
        return math.hypot(self.length, self.width) / 2

    def bounds(self, margin):
        reach = self.reach + margin
        return self.x - reach, self.x + reach, self.y - reach, self.y + reach

    def signed_distance(self, xs, ys):
        c, s = math.cos(self.angle), math.sin(self.angle)
        dx, dy = xs - self.x, ys - self.y
        return rounded_rect_distance(dx * c + dy * s, -dx * s + dy * c, self.length / 2, self.width / 2, self.corner)


@dataclass(frozen=True)
class Block:
    """Axis-aligned rectangle from (x_lo, y_lo) to (x_hi, y_hi) with rounded corners."""
    x_lo: float
    y_lo: float
    x_hi: float
    y_hi: float
    corner: float = BLOCK_CORNER

    @property
    def x(self):
        return (self.x_lo + self.x_hi) / 2

    @property
    def y(self):
        return (self.y_lo + self.y_hi) / 2

    def bounds(self, margin):
        return self.x_lo - margin, self.x_hi + margin, self.y_lo - margin, self.y_hi + margin

    def signed_distance(self, xs, ys):
        half_w, half_h = (self.x_hi - self.x_lo) / 2, (self.y_hi - self.y_lo) / 2
        return rounded_rect_distance(xs - self.x, ys - self.y, half_w, half_h, self.corner)


@dataclass(frozen=True)
class Road:
    """Straight road of `width` from (x0, y0) to (x1, y1), carved out of whatever it crosses."""
    x0: float
    y0: float
    x1: float
    y1: float
    width: float = STREET_WIDTH

    def bounds(self, margin):
        reach = self.width / 2 + margin
        return (min(self.x0, self.x1) - reach, max(self.x0, self.x1) + reach,
                min(self.y0, self.y1) - reach, max(self.y0, self.y1) + reach)

    def signed_distance(self, xs, ys):
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        t = np.clip(((xs - self.x0) * dx + (ys - self.y0) * dy) / (dx * dx + dy * dy), 0, 1)
        return np.hypot(xs - (self.x0 + t * dx), ys - (self.y0 + t * dy)) - self.width / 2


def diagonal_roads():
    return [Road(x0, y0, x1, y1) for (x0, y0), (x1, y1) in DIAGONALS]


def street_blocks():
    """City blocks filling the space between the STREET_X / STREET_Y roads. Outer blocks extend
    past the screen edge so their rounded corners don't leave gaps along the border."""
    half = STREET_WIDTH / 2
    xs = [-BLOCK_CORNER] + [e for c in STREET_X for e in (c - half, c + half)] + [SCREEN_WIDTH + BLOCK_CORNER]
    ys = [-BLOCK_CORNER] + [e for c in STREET_Y for e in (c - half, c + half)] + [SCREEN_HEIGHT + BLOCK_CORNER]
    return [
        Block(x_lo, y_lo, x_hi, y_hi)
        for x_lo, x_hi in zip(xs[::2], xs[1::2])
        for y_lo, y_hi in zip(ys[::2], ys[1::2])
        if x_hi > x_lo and y_hi > y_lo
    ]


OBSTACLES = [Obstacle(pos[0], pos[1], rad) for pos, rad in zip(GLOBAL_OBSTACLE_POS + LOCAL_OBSTACLE_POS, GLOBAL_OBSTACLE_RAD + LOCAL_OBSTACLE_RAD)]
GLOBAL_OBSTACLES = [Obstacle(pos[0], pos[1], rad) for pos, rad in zip(GLOBAL_OBSTACLE_POS, GLOBAL_OBSTACLE_RAD)]


def rasterize(grid, shapes, value):
    """Set every cell of `grid` whose center lies inside one of `shapes` to `value`."""
    for o in shapes:
        bx_lo, bx_hi, by_lo, by_hi = o.bounds(0)
        x_lo, x_hi = max(0, math.floor(bx_lo)), min(grid.shape[0], math.ceil(bx_hi) + 1)
        y_lo, y_hi = max(0, math.floor(by_lo)), min(grid.shape[1], math.ceil(by_hi) + 1)
        if x_lo >= x_hi or y_lo >= y_hi:
            continue
        inside = o.signed_distance(np.arange(x_lo, x_hi)[:, None], np.arange(y_lo, y_hi)[None, :]) <= 0
        grid[x_lo:x_hi, y_lo:y_hi][inside] = value
    return grid


def capped_distance(mask, cap):
    """Euclidean distance from every cell to the nearest True cell of `mask` (0 on those cells), capped at `cap`.

    Exact separable transform. Pass 1 finds the nearest True cell along x for every cell with running
    max/min accumulations (O(cells)); pass 2 combines those along y over offsets up to `cap` (O(cells * cap)),
    in int16 since squared distances stay below (cap + 1)^2 + cap^2.
    """
    reach = math.ceil(cap)
    n = mask.shape[0]
    idx = np.arange(n, dtype=np.int32)[:, None]
    before = np.where(mask, idx, np.int32(-n - reach - 1))
    np.maximum.accumulate(before, axis=0, out=before)
    after = np.minimum.accumulate(np.where(mask, idx, np.int32(2 * n + reach + 1))[::-1], axis=0)[::-1]
    along_x = np.minimum(idx - before, after - idx)
    np.minimum(along_x, reach + 1, out=along_x)
    along_sq = (along_x * along_x).astype(np.int16)
    best = along_sq.copy()
    shifted = np.empty_like(along_sq)
    for d in range(1, min(reach, mask.shape[1] - 1) + 1):
        np.add(along_sq[:, :-d], d * d, out=shifted[:, :-d])
        np.minimum(best[:, d:], shifted[:, :-d], out=best[:, d:])
        np.add(along_sq[:, d:], d * d, out=shifted[:, d:])
        np.minimum(best[:, :-d], shifted[:, d:], out=best[:, :-d])
    return np.minimum(np.sqrt(best, dtype=np.float32), np.float32(cap))


def changedRects(changed, x0, y0, tile=DIRTY_TILE):
    """Bounding boxes (x_lo, x_hi, y_lo, y_hi) in map coords of the clusters of True cells in `changed`, whose
    [0, 0] cell is at (x0, y0). Clusters are found on a coarse tile grid, so changes scattered across a long
    range scan become several small windows instead of one map-sized one."""
    w, h = changed.shape
    tw, th = -(-w // tile), -(-h // tile)
    padded = np.zeros((tw * tile, th * tile), dtype=bool)
    padded[:w, :h] = changed
    tiles = padded.reshape(tw, tile, th, tile).any(axis=(1, 3))
    seen = np.zeros_like(tiles)
    rects = []
    for i, j in zip(*np.nonzero(tiles)):  # flood fill 8-connected tiles; the tile grid is small
        if seen[i, j]:
            continue
        seen[i, j] = True
        stack, i_lo, i_hi, j_lo, j_hi = [(i, j)], i, i, j, j
        while stack:
            a, b = stack.pop()
            i_lo, i_hi, j_lo, j_hi = min(i_lo, a), max(i_hi, a), min(j_lo, b), max(j_hi, b)
            for na in range(max(0, a - 1), min(tw, a + 2)):
                for nb in range(max(0, b - 1), min(th, b + 2)):
                    if tiles[na, nb] and not seen[na, nb]:
                        seen[na, nb] = True
                        stack.append((na, nb))
        # Tighten to the changed cells inside the cluster's tiles
        sub = changed[i_lo * tile:(i_hi + 1) * tile, j_lo * tile:(j_hi + 1) * tile]
        xs, ys = np.flatnonzero(sub.any(axis=1)), np.flatnonzero(sub.any(axis=0))
        rects.append((x0 + i_lo * tile + xs[0], x0 + i_lo * tile + xs[-1] + 1,
                      y0 + j_lo * tile + ys[0], y0 + j_lo * tile + ys[-1] + 1))
    return rects


class SDFState:
    """The world as an occupancy grid, occupancy[x, y] in FREE / OCCUPIED / UNKNOWN, plus two distance
    fields derived from it (0 on the cells they measure from, capped at SDF_CAP):
        obstacle_sdf[x, y]  distance to the nearest OCCUPIED cell
        unknown_sdf[x, y]   distance to the nearest UNKNOWN cell

    `obstacles` start OCCUPIED, `roads` are cut back out of them, and everything else is `background`.
    For the global map that is UNKNOWN: only cells a scan has confirmed (see merge) become FREE.
    """

    PALETTE = np.array([(0, 0, 0), (0, 0, 255), (45, 45, 45)], dtype=np.uint8)  # indexed by cell state

    def __init__(self, obstacles=(), roads=(), background=UNKNOWN):
        self.occupancy = np.full((SCREEN_WIDTH, SCREEN_HEIGHT), background, dtype=np.int8)
        rasterize(self.occupancy, obstacles, OCCUPIED)
        rasterize(self.occupancy, roads, background)
        self.obstacle_sdf = np.empty((SCREEN_WIDTH, SCREEN_HEIGHT), dtype=np.float32)
        self.unknown_sdf = np.empty((SCREEN_WIDTH, SCREEN_HEIGHT), dtype=np.float32)
        self.has_changed = False
        self.change_coords = []  # (x_lo, y_lo, x_hi, y_hi) of each merge's changed cells
        # (x, y) of every cell a merge newly made OCCUPIED. obstacle_sdf only changes within SDF_CAP of these, so
        # anything farther than SDF_CAP from all of them is unaffected. Consumers read and clear this list.
        self.new_obstacle_points = []
        self.version = 0  # bumped on every change, for anything caching a view of the map
        self.surface = None  # cached render; merges queue the rects that need repainting
        self.redraw = []
        self.recompute(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT)

    def recompute(self, x_lo, x_hi, y_lo, y_hi, obstacles=True, unknown=True):
        """Refresh the chosen distance fields around [x_lo, x_hi) x [y_lo, y_hi) after the occupancy there changed."""
        # Distances are capped, so cells more than SDF_CAP past the edited cells keep their values, and
        # only cells within SDF_CAP of those edits can be the nearest one to them
        m = math.ceil(SDF_CAP)
        x_lo, x_hi = max(0, x_lo - m), min(SCREEN_WIDTH, x_hi + m)
        y_lo, y_hi = max(0, y_lo - m), min(SCREEN_HEIGHT, y_hi + m)
        ax_lo, ax_hi = max(0, x_lo - m), min(SCREEN_WIDTH, x_hi + m)
        ay_lo, ay_hi = max(0, y_lo - m), min(SCREEN_HEIGHT, y_hi + m)
        occ = self.occupancy[ax_lo:ax_hi, ay_lo:ay_hi]
        inner = slice(x_lo - ax_lo, x_hi - ax_lo), slice(y_lo - ay_lo, y_hi - ay_lo)
        if obstacles:
            self.obstacle_sdf[x_lo:x_hi, y_lo:y_hi] = capped_distance(occ == OCCUPIED, SDF_CAP)[inner]
        if unknown:
            self.unknown_sdf[x_lo:x_hi, y_lo:y_hi] = capped_distance(occ == UNKNOWN, SDF_CAP)[inner]

    def merge(self, local, x0, y0):
        """Fold a local occupancy grid (e.g. one lidar scan) whose [0, 0] cell sits at (x0, y0) into this map.

        OCCUPIED always wins, FREE only fills cells that are UNKNOWN, and UNKNOWN leaves the map alone, so a
        merge can add obstacles and confirm free space but never clear an obstacle. Returns whether anything
        changed; merging what the map already knows is a few comparisons and skips the SDF update. The SDFs
        are refreshed only around each cluster of changed cells.
        """
        region = self.occupancy[x0:x0 + local.shape[0], y0:y0 + local.shape[1]]
        new_obstacle = (local == OCCUPIED) & (region != OCCUPIED)
        new_free = (local == FREE) & (region == UNKNOWN)
        changed = new_obstacle | new_free
        if not changed.any():
            return False
        region[new_obstacle] = OCCUPIED
        region[new_free] = FREE
        # Every change shrinks the unknown area, but only new obstacles can move obstacle distances
        for x_lo, x_hi, y_lo, y_hi in changedRects(changed, x0, y0):
            self.recompute(x_lo, x_hi, y_lo, y_hi, obstacles=False)
            self.change_coords.append((x_lo, y_lo, x_hi, y_hi))
            self.redraw.append((x_lo, x_hi, y_lo, y_hi))
        if new_obstacle.any():
            for x_lo, x_hi, y_lo, y_hi in changedRects(new_obstacle, x0, y0):
                self.recompute(x_lo, x_hi, y_lo, y_hi, unknown=False)
            xs, ys = np.nonzero(new_obstacle)
            self.new_obstacle_points.extend(zip((xs + x0).tolist(), (ys + y0).tolist()))
        self.has_changed = True
        self.version += 1
        return True

    def draw(self, surface):
        if self.surface is None:
            self.surface = pygame.surfarray.make_surface(self.PALETTE[self.occupancy])
        elif self.redraw:
            pixels = pygame.surfarray.pixels3d(self.surface)
            for x_lo, x_hi, y_lo, y_hi in self.redraw:
                pixels[x_lo:x_hi, y_lo:y_hi] = self.PALETTE[self.occupancy[x_lo:x_hi, y_lo:y_hi]]
            del pixels  # unlocks the surface
        self.redraw.clear()
        surface.blit(self.surface, (0, 0))


class LocalObstacleMap:
    """The true world the car drives in: the known map's blocks plus random obstacles on the roads that
    the planner doesn't know about. scan() simulates the lidar that will produce local occupancy grids
    in the future, and reveal() merges a scan into the known map.

    Hidden obstacles can be any shape with reach, bounds() and signed_distance().
    """

    def __init__(self, known, count=LOCAL_OBSTACLE_COUNT, blocked_roads=BLOCKED_ROADS, seed=LOCAL_SEED):
        rng = random.Random(seed)
        self.known = known
        self.obstacles = self.roadBlocks(rng, blocked_roads) + self.scatter(rng, count)
        self.hidden = rasterize(np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT), dtype=bool), self.obstacles, True)
        self.truth = self.hidden | (known.occupancy == OCCUPIED)
        self.truth_sdf = capped_distance(self.truth, SDF_CAP)  # lets the simulated rays take big steps
        self.surface, self.drawn_changes = None, 0

    @staticmethod
    def keepsClear(x, y, reach):
        return all(math.hypot(x - px, y - py) > reach + STREET_WIDTH / 2 for px, py in (START, TARGET))

    def roadBlocks(self, rng, n):
        # Barriers across the midpoints of straight road segments between intersections, closing those roads
        mids = [((a + b) / 2, y, math.pi / 2) for y in STREET_Y for a, b in zip(STREET_X, STREET_X[1:])]
        mids += [(x, (a + b) / 2, 0.0) for x in STREET_X for a, b in zip(STREET_Y, STREET_Y[1:])]
        barriers = [Box(x, y, STREET_WIDTH + 10, 12, angle, corner=3) for x, y, angle in mids]
        barriers = [b for b in barriers if self.keepsClear(b.x, b.y, b.reach)]
        return rng.sample(barriers, min(n, len(barriers)))

    def randomShape(self, rng, x, y):
        if rng.random() < 0.5:
            return Obstacle(x, y, rng.uniform(*LOCAL_OBSTACLE_RADIUS))
        width = rng.uniform(*LOCAL_BOX_WIDTH)
        return Box(x, y, rng.uniform(*LOCAL_BOX_LENGTH), width, rng.uniform(0, math.pi), rng.uniform(0, width / 2))

    def scatter(self, rng, n):
        obstacles = []
        for _ in range(n * 50):  # rejection sampling; gives up quietly if the roads are too full
            if len(obstacles) == n:
                break
            x, y = rng.uniform(0, SCREEN_WIDTH), rng.uniform(0, SCREEN_HEIGHT)
            shape = self.randomShape(rng, x, y)
            if self.known.obstacle_sdf[int(x), int(y)] >= 10 and self.keepsClear(x, y, shape.reach):  # on a road, off the kerb
                obstacles.append(shape)
        return obstacles

    def scan(self, x, y):
        """Simulated long range lidar: LIDAR_BEAMS rays from (x, y), each running until it hits something in the
        true world or leaves the map.

        Rays step by the true world's distance to the nearest obstacle (sphere tracing), so open road takes a
        few big steps. The visible area is the polygon through the ray ends, filled by pygame.
        Returns (local, x0, y0), a FREE / OCCUPIED / UNKNOWN grid whose [0, 0] cell is at (x0, y0); anything
        behind the first thing a ray hits is UNKNOWN.
        """
        angles = np.arange(LIDAR_BEAMS) * (2 * math.pi / LIDAR_BEAMS)
        cos, sin = np.cos(angles), np.sin(angles)
        reach = np.zeros(LIDAR_BEAMS)
        hit = np.zeros(LIDAR_BEAMS, dtype=bool)
        live = np.arange(LIDAR_BEAMS)
        while live.size:
            px = np.rint(x + cos[live] * reach[live]).astype(np.intp)
            py = np.rint(y + sin[live] * reach[live]).astype(np.intp)
            on_map = (px >= 0) & (px < SCREEN_WIDTH) & (py >= 0) & (py < SCREEN_HEIGHT)
            clearance = np.zeros(live.size, dtype=np.float32)  # off the map counts as the end of the ray
            clearance[on_map] = self.truth_sdf[px[on_map], py[on_map]]
            hit[live[on_map & (clearance <= 0)]] = True
            going = clearance > 0
            reach[live[going]] += np.maximum(clearance[going], 1)
            live = live[going]
        reach[hit] += 1.5  # so the polygon covers the surface the ray hit

        ex, ey = x + cos * reach, y + sin * reach
        x0, x1 = max(0, math.floor(min(ex.min(), x))), min(SCREEN_WIDTH, math.ceil(max(ex.max(), x)) + 1)
        y0, y1 = max(0, math.floor(min(ey.min(), y))), min(SCREEN_HEIGHT, math.ceil(max(ey.max(), y)) + 1)
        canvas = pygame.Surface((x1 - x0, y1 - y0), depth=32)
        pygame.draw.polygon(canvas, (255, 255, 255), np.column_stack((ex - x0, ey - y0)).tolist())
        seen = pygame.surfarray.pixels_red(canvas) > 0
        truth = self.truth[x0:x1, y0:y1]
        local = np.full(seen.shape, UNKNOWN, dtype=np.int8)
        local[seen & ~truth] = FREE
        local[seen & truth] = OCCUPIED
        return local, x0, y0

    def reveal(self, x, y):
        """Scan from (x, y) and merge the result into the known map. Returns whether the known map changed."""
        return self.known.merge(*self.scan(x, y))

    def draw(self, surface):
        """Overlay the hidden obstacles the known map hasn't picked up yet, in green."""
        changes = self.known.change_coords
        if self.surface is None or self.drawn_changes > len(changes):  # first draw, or someone cleared the log
            self.surface = pygame.surfarray.make_surface(self.overlay(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT))
            self.surface.set_colorkey((0, 0, 0))
        elif self.drawn_changes < len(changes):
            pixels = pygame.surfarray.pixels3d(self.surface)
            for x_lo, y_lo, x_hi, y_hi in changes[self.drawn_changes:]:
                pixels[x_lo:x_hi, y_lo:y_hi] = self.overlay(x_lo, x_hi, y_lo, y_hi)
            del pixels
        self.drawn_changes = len(changes)
        surface.blit(self.surface, (0, 0))

    def overlay(self, x_lo, x_hi, y_lo, y_hi):
        pixels = np.zeros((x_hi - x_lo, y_hi - y_lo, 3), dtype=np.uint8)
        unseen = self.hidden[x_lo:x_hi, y_lo:y_hi] & (self.known.occupancy[x_lo:x_hi, y_lo:y_hi] != OCCUPIED)
        pixels[unseen] = (0, 160, 0)
        return pixels
