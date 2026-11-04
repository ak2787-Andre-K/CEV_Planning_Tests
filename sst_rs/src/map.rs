//! The street map as an occupancy grid plus the distance field to its obstacles (`sst/streets.py`).
//!
//! The global planner only needs the known obstacles: everything else (roads and unknown space) counts as free.
use serde::{Deserialize, Serialize};

use crate::config::{
    BLOCK_CORNER, DIAGONALS, SCREEN_HEIGHT, SCREEN_WIDTH, SDF_CAP, STREET_WIDTH, STREET_X, STREET_Y,
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
fn rasterize(grid: &mut [bool], shapes: &[Shape], value: bool) {
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

/// Euclidean distance from every cell to the nearest true cell of `mask` (0 on those cells), capped at `cap`.
///
/// Exact separable transform: pass 1 finds the nearest true cell along x for every cell, pass 2 combines those along y
/// over offsets up to `cap`.
fn capped_distance(mask: &[bool], cap: f64) -> Vec<f32> {
    let reach = cap.ceil() as i64;
    let (w, h) = (SCREEN_WIDTH as i64, SCREEN_HEIGHT as i64);
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

/// The known map: which cells are obstacles, and the distance from every cell to the nearest one.
pub struct Map {
    pub occupied: Vec<bool>,
    pub obstacle_sdf: Vec<f32>,
}

impl Map {
    /// `obstacles` start occupied, `roads` are cut back out of them, then `extra` shapes are added on top.
    pub fn new(obstacles: &[Shape], roads: &[Shape], extra: &[Shape]) -> Map {
        let mut occupied = vec![false; SCREEN_WIDTH * SCREEN_HEIGHT];
        rasterize(&mut occupied, obstacles, true);
        rasterize(&mut occupied, roads, false);
        rasterize(&mut occupied, extra, true);
        let obstacle_sdf = capped_distance(&occupied, SDF_CAP);
        Map { occupied, obstacle_sdf }
    }

    /// The street map `sst` plans on: city blocks with the diagonal roads cut through them, plus `extra` obstacles.
    pub fn streets(extra: &[Shape]) -> Map {
        Map::new(&street_blocks(), &diagonal_roads(), extra)
    }

    /// Obstacle distance at (x, y), or None off the map.
    #[inline]
    pub fn sdf(&self, x: i64, y: i64) -> Option<f32> {
        if 0 <= x && x < SCREEN_WIDTH as i64 && 0 <= y && y < SCREEN_HEIGHT as i64 {
            Some(self.obstacle_sdf[cell(x as usize, y as usize)])
        } else {
            None
        }
    }
}
