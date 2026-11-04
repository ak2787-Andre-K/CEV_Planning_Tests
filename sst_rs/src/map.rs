//! The street map as an occupancy grid plus the distance field to its obstacles (`sst/streets.py`).
//!
//! The planner only looks at the known obstacles (unknown space counts as free); the Driver also keeps out of unknown
//! space, and lidar scans are merged in as it drives.
use serde::{Deserialize, Serialize};

use crate::config::{
    BLOCK_CORNER, DIAGONALS, DIRTY_TILE, FREE, OCCUPIED, SCREEN_HEIGHT, SCREEN_WIDTH, SDF_CAP, STREET_WIDTH, STREET_X,
    STREET_Y, UNKNOWN,
};

/// A shape that can be stamped into the grid: anything with bounds and a signed distance (negative inside).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Shape {
    /// Circle at (x, y).
    Circle { x: f64, y: f64, radius: f64 },
    /// Rectangle centered at (x, y), `length` along `angle` (rad) and `width` across it, with rounded corners.
    Box { x: f64, y: f64, length: f64, width: f64, #[serde(default)] angle: f64, #[serde(default)] corner: f64 },
    /// Axis-aligned rectangle from (x_lo, y_lo) to (x_hi, y_hi) with rounded corners (a city block).
    Block { x_lo: f64, y_lo: f64, x_hi: f64, y_hi: f64, corner: f64 },
    /// Straight road of `width` from (x0, y0) to (x1, y1).
    Road { x0: f64, y0: f64, x1: f64, y1: f64, width: f64 },
}

/// Signed distance to a rectangle centered at the origin, for a point (u, v) in the rectangle's own frame.
fn rounded_rect_distance(u: f64, v: f64, half_w: f64, half_h: f64, corner: f64) -> f64 {
    let r = corner.min(half_w).min(half_h);
    let qx = u.abs() - (half_w - r);
    let qy = v.abs() - (half_h - r);
    qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - r
}

impl Shape {
    /// (x_lo, x_hi, y_lo, y_hi) the shape lies inside.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        match *self {
            Shape::Circle { x, y, radius } => (x - radius, x + radius, y - radius, y + radius),
            Shape::Box { x, y, length, width, .. } => {
                let reach = length.hypot(width) / 2.0;
                (x - reach, x + reach, y - reach, y + reach)
            }
            Shape::Block { x_lo, y_lo, x_hi, y_hi, .. } => (x_lo, x_hi, y_lo, y_hi),
            Shape::Road { x0, y0, x1, y1, width } => {
                let reach = width / 2.0;
                (x0.min(x1) - reach, x0.max(x1) + reach, y0.min(y1) - reach, y0.max(y1) + reach)
            }
        }
    }

    pub fn signed_distance(&self, px: f64, py: f64) -> f64 {
        match *self {
            Shape::Circle { x, y, radius } => (px - x).hypot(py - y) - radius,
            Shape::Box { x, y, length, width, angle, corner } => {
                let (s, c) = angle.sin_cos();
                let (dx, dy) = (px - x, py - y);
                rounded_rect_distance(dx * c + dy * s, -dx * s + dy * c, length / 2.0, width / 2.0, corner)
            }
            Shape::Block { x_lo, y_lo, x_hi, y_hi, corner } => rounded_rect_distance(
                px - (x_lo + x_hi) / 2.0, py - (y_lo + y_hi) / 2.0, (x_hi - x_lo) / 2.0, (y_hi - y_lo) / 2.0, corner,
            ),
            Shape::Road { x0, y0, x1, y1, width } => {
                let (dx, dy) = (x1 - x0, y1 - y0);
                let t = (((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
                (px - (x0 + t * dx)).hypot(py - (y0 + t * dy)) - width / 2.0
            }
        }
    }
}

pub fn diagonal_roads() -> Vec<Shape> {
    DIAGONALS.iter().map(|&((x0, y0), (x1, y1))| Shape::Road { x0, y0, x1, y1, width: STREET_WIDTH }).collect()
}

/// City blocks filling the space between the STREET_X / STREET_Y roads. Outer blocks extend past the screen edge so
/// their rounded corners don't leave gaps along the border.
pub fn street_blocks() -> Vec<Shape> {
    let half = STREET_WIDTH / 2.0;
    let edges = |centers: &[f64], size: f64| {
        let mut e = vec![-BLOCK_CORNER];
        for &c in centers {
            e.extend([c - half, c + half]);
        }
        e.push(size + BLOCK_CORNER);
        e
    };
    let xs = edges(&STREET_X, SCREEN_WIDTH as f64);
    let ys = edges(&STREET_Y, SCREEN_HEIGHT as f64);
    let mut blocks = Vec::new();
    for x in xs.chunks(2) {
        for y in ys.chunks(2) {
            if x[1] > x[0] && y[1] > y[0] {
                blocks.push(Shape::Block { x_lo: x[0], y_lo: y[0], x_hi: x[1], y_hi: y[1], corner: BLOCK_CORNER });
            }
        }
    }
    blocks
}

/// Index of cell (x, y) in the x-major grids (the same layout as the Python `occupancy[x, y]`).
#[inline]
pub fn cell(x: usize, y: usize) -> usize {
    x * SCREEN_HEIGHT + y
}

/// Set every cell whose center lies inside one of `shapes` to `value`.
fn rasterize<T: Copy>(grid: &mut [T], shapes: &[Shape], value: T) {
    for shape in shapes {
        let (bx_lo, bx_hi, by_lo, by_hi) = shape.bounds();
        let x_lo = bx_lo.floor().max(0.0) as usize;
        let x_hi = ((bx_hi.ceil() + 1.0).max(0.0) as usize).min(SCREEN_WIDTH);
        let y_lo = by_lo.floor().max(0.0) as usize;
        let y_hi = ((by_hi.ceil() + 1.0).max(0.0) as usize).min(SCREEN_HEIGHT);
        for x in x_lo..x_hi {
            for y in y_lo..y_hi {
                if shape.signed_distance(x as f64, y as f64) <= 0.0 {
                    grid[cell(x, y)] = value;
                }
            }
        }
    }
}

/// Euclidean distance from every cell of a w x h x-major grid to the nearest true cell of `mask` (0 on those cells),
/// capped at `cap`.
///
/// Exact separable transform: pass 1 finds the nearest true cell along x for every cell, pass 2 combines those along y
/// over offsets up to `cap`.
pub fn capped_distance(mask: &[bool], w: usize, h: usize, cap: f64) -> Vec<f32> {
    let reach = cap.ceil() as i64;
    let (w, h) = (w as i64, h as i64);
    let mut along_sq = vec![0i64; mask.len()];
    for y in 0..h {
        let mut last = -w - reach - 1;
        for x in 0..w {
            if mask[(x * h + y) as usize] {
                last = x;
            }
            along_sq[(x * h + y) as usize] = x - last;
        }
        let mut next = 2 * w + reach + 1;
        for x in (0..w).rev() {
            let i = (x * h + y) as usize;
            if mask[i] {
                next = x;
            }
            let d = along_sq[i].min(next - x).min(reach + 1);
            along_sq[i] = d * d;
        }
    }
    let mut out = vec![0f32; mask.len()];
    for x in 0..w {
        let column = &along_sq[(x * h) as usize..((x + 1) * h) as usize];
        for y in 0..h {
            let mut best = column[y as usize];
            for d in 1..=reach.min(h - 1) {
                if y - d >= 0 {
                    best = best.min(column[(y - d) as usize] + d * d);
                }
                if y + d < h {
                    best = best.min(column[(y + d) as usize] + d * d);
                }
            }
            out[(x * h + y) as usize] = (best as f32).sqrt().min(cap as f32);
        }
    }
    out
}

/// Bounding boxes (x_lo, x_hi, y_lo, y_hi), in map coordinates, of the clusters of true cells in a w x h x-major grid
/// whose [0, 0] cell is at (x0, y0). Clusters are found on a coarse DIRTY_TILE grid, so changes scattered across a
/// long range scan become several small windows instead of one map-sized one.
fn changed_rects(changed: &[bool], w: usize, h: usize, x0: usize, y0: usize) -> Vec<(usize, usize, usize, usize)> {
    let (tw, th) = (w.div_ceil(DIRTY_TILE), h.div_ceil(DIRTY_TILE));
    let mut tiles = vec![false; tw * th];
    for x in 0..w {
        for y in 0..h {
            if changed[x * h + y] {
                tiles[(x / DIRTY_TILE) * th + y / DIRTY_TILE] = true;
            }
        }
    }
    let mut seen = vec![false; tw * th];
    let mut rects = Vec::new();
    for start in 0..tw * th {
        if !tiles[start] || seen[start] {
            continue;
        }
        // Flood fill 8-connected tiles; the tile grid is small
        seen[start] = true;
        let (mut stack, mut i_lo, mut i_hi, mut j_lo, mut j_hi) = (vec![start], tw, 0, th, 0);
        while let Some(t) = stack.pop() {
            let (a, b) = (t / th, t % th);
            (i_lo, i_hi, j_lo, j_hi) = (i_lo.min(a), i_hi.max(a), j_lo.min(b), j_hi.max(b));
            for na in a.saturating_sub(1)..(a + 2).min(tw) {
                for nb in b.saturating_sub(1)..(b + 2).min(th) {
                    let n = na * th + nb;
                    if tiles[n] && !seen[n] {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
        }
        // Tighten to the changed cells inside the cluster's tiles
        let (mut x_lo, mut x_hi, mut y_lo, mut y_hi) = (usize::MAX, 0, usize::MAX, 0);
        for x in i_lo * DIRTY_TILE..((i_hi + 1) * DIRTY_TILE).min(w) {
            for y in j_lo * DIRTY_TILE..((j_hi + 1) * DIRTY_TILE).min(h) {
                if changed[x * h + y] {
                    (x_lo, x_hi, y_lo, y_hi) = (x_lo.min(x), x_hi.max(x + 1), y_lo.min(y), y_hi.max(y + 1));
                }
            }
        }
        rects.push((x0 + x_lo, x0 + x_hi, y0 + y_lo, y0 + y_hi));
    }
    rects
}

/// A local occupancy grid (one lidar scan): FREE / OCCUPIED / UNKNOWN cells, x-major, its [0, 0] cell at (x0, y0).
pub struct LocalGrid {
    pub x0: usize,
    pub y0: usize,
    pub w: usize,
    pub h: usize,
    pub cells: Vec<u8>,
}

/// The known map as an occupancy grid, occupancy[cell(x, y)] in FREE / OCCUPIED / UNKNOWN, plus two distance fields
/// derived from it (0 on the cells they measure from, capped at SDF_CAP): to the nearest OCCUPIED cell, and to the
/// nearest UNKNOWN cell.
#[derive(Clone)]
pub struct Map {
    pub occupancy: Vec<u8>,
    pub obstacle_sdf: Vec<f32>,
    pub unknown_sdf: Vec<f32>,
    /// Bumped on every change, for anything caching a view of the map.
    pub version: u64,
    /// Set by a merge that made new cells OCCUPIED; whoever reacts to new obstacles clears it.
    pub new_obstacles: bool,
}

impl Map {
    /// `obstacles` start occupied, `roads` are cut back out of them, then `extra` shapes are added on top. Everything
    /// else is `background`; for the map the car drives on that is UNKNOWN, so only scanned cells become FREE.
    pub fn new(obstacles: &[Shape], roads: &[Shape], extra: &[Shape], background: u8) -> Map {
        let mut occupancy = vec![background; SCREEN_WIDTH * SCREEN_HEIGHT];
        rasterize(&mut occupancy, obstacles, OCCUPIED);
        rasterize(&mut occupancy, roads, background);
        rasterize(&mut occupancy, extra, OCCUPIED);
        let n = occupancy.len();
        let mut map = Map {
            occupancy, obstacle_sdf: vec![0.0; n], unknown_sdf: vec![0.0; n], version: 0, new_obstacles: false,
        };
        map.recompute(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT, true, true);
        map
    }

    /// The street map `sst` plans and drives on: city blocks with the diagonal roads cut through them, plus `extra`
    /// obstacles, with everything off the blocks unknown.
    pub fn streets(extra: &[Shape]) -> Map {
        Map::new(&street_blocks(), &diagonal_roads(), extra, UNKNOWN)
    }

    pub fn occupied(&self, x: usize, y: usize) -> bool {
        self.occupancy[cell(x, y)] == OCCUPIED
    }

    /// Refresh the chosen distance fields around [x_lo, x_hi) x [y_lo, y_hi) after the occupancy there changed.
    /// Distances are capped, so only cells within SDF_CAP of the edits can change, and only cells within SDF_CAP of
    /// those can be the nearest one to them.
    pub fn recompute(&mut self, x_lo: usize, x_hi: usize, y_lo: usize, y_hi: usize, obstacles: bool, unknown: bool) {
        let m = SDF_CAP.ceil() as usize;
        let (x_lo, x_hi) = (x_lo.saturating_sub(m), (x_hi + m).min(SCREEN_WIDTH));
        let (y_lo, y_hi) = (y_lo.saturating_sub(m), (y_hi + m).min(SCREEN_HEIGHT));
        let (ax_lo, ax_hi) = (x_lo.saturating_sub(m), (x_hi + m).min(SCREEN_WIDTH));
        let (ay_lo, ay_hi) = (y_lo.saturating_sub(m), (y_hi + m).min(SCREEN_HEIGHT));
        let (w, h) = (ax_hi - ax_lo, ay_hi - ay_lo);
        for (on, value) in [(obstacles, OCCUPIED), (unknown, UNKNOWN)] {
            if !on {
                continue;
            }
            let mut mask = vec![false; w * h];
            for x in 0..w {
                for y in 0..h {
                    mask[x * h + y] = self.occupancy[cell(ax_lo + x, ay_lo + y)] == value;
                }
            }
            let field = capped_distance(&mask, w, h, SDF_CAP);
            let target = if value == OCCUPIED { &mut self.obstacle_sdf } else { &mut self.unknown_sdf };
            for x in x_lo..x_hi {
                for y in y_lo..y_hi {
                    target[cell(x, y)] = field[(x - ax_lo) * h + (y - ay_lo)];
                }
            }
        }
    }

    /// Fold a local occupancy grid (e.g. one lidar scan) into this map. OCCUPIED always wins, FREE only fills cells
    /// that are UNKNOWN, and UNKNOWN leaves the map alone, so a merge can add obstacles and confirm free space but never
    /// clear an obstacle. Returns whether anything changed. The distance fields are refreshed only around each
    /// cluster of changed cells.
    pub fn merge(&mut self, local: &LocalGrid) -> bool {
        let (w, h) = (local.w, local.h);
        let mut new_obstacle = vec![false; w * h];
        let mut changed = vec![false; w * h];
        let mut any = false;
        let mut any_obstacle = false;
        for x in 0..w {
            for y in 0..h {
                let (k, here) = (x * h + y, cell(local.x0 + x, local.y0 + y));
                let (seen, known) = (local.cells[k], self.occupancy[here]);
                if seen == OCCUPIED && known != OCCUPIED {
                    self.occupancy[here] = OCCUPIED;
                    (new_obstacle[k], changed[k], any, any_obstacle) = (true, true, true, true);
                } else if seen == FREE && known == UNKNOWN {
                    self.occupancy[here] = FREE;
                    (changed[k], any) = (true, true);
                }
            }
        }
        if !any {
            return false;
        }
        // Every change shrinks the unknown area, but only new obstacles can move obstacle distances
        for (x_lo, x_hi, y_lo, y_hi) in changed_rects(&changed, w, h, local.x0, local.y0) {
            self.recompute(x_lo, x_hi, y_lo, y_hi, false, true);
        }
        if any_obstacle {
            for (x_lo, x_hi, y_lo, y_hi) in changed_rects(&new_obstacle, w, h, local.x0, local.y0) {
                self.recompute(x_lo, x_hi, y_lo, y_hi, true, false);
            }
            self.new_obstacles = true;
        }
        self.version += 1;
        true
    }

    /// Obstacle distance at (x, y), or None off the map.
    #[inline]
    pub fn sdf(&self, x: i64, y: i64) -> Option<f32> {
        self.on_map(x, y).then(|| self.obstacle_sdf[cell(x as usize, y as usize)])
    }

    /// Distance to the nearest obstacle or unknown cell at (x, y), or None off the map.
    #[inline]
    pub fn known_sdf(&self, x: i64, y: i64) -> Option<f32> {
        self.on_map(x, y).then(|| {
            let c = cell(x as usize, y as usize);
            self.obstacle_sdf[c].min(self.unknown_sdf[c])
        })
    }

    #[inline]
    fn on_map(&self, x: i64, y: i64) -> bool {
        0 <= x && x < SCREEN_WIDTH as i64 && 0 <= y && y < SCREEN_HEIGHT as i64
    }
}
