//! The true world the car drives in (`LocalObstacleMap` in `sst/streets.py`): the known map's blocks plus random
//! obstacles on the roads the planner doesn't know about, and a simulated lidar that reveals them.
use std::f64::consts::{PI, TAU};

use crate::config::{
    BLOCKED_ROADS, FREE, LIDAR_BEAMS, LOCAL_BOX_LENGTH, LOCAL_BOX_WIDTH, LOCAL_OBSTACLE_COUNT, LOCAL_OBSTACLE_RADIUS,
    OCCUPIED, SCREEN_HEIGHT, SCREEN_WIDTH, SDF_CAP, START, STREET_WIDTH, STREET_X, STREET_Y, TARGET, UNKNOWN,
};
use crate::map::{LocalGrid, Map, Shape, capped_distance, cell};
use crate::rng::Random;

pub struct World {
    pub obstacles: Vec<Shape>, // the hidden ones
    truth: Vec<bool>,          // hidden obstacles or the known map's
    truth_sdf: Vec<f32>,       // lets the simulated rays take big steps
}

fn keeps_clear(x: f64, y: f64, reach: f64) -> bool {
    [START, TARGET].iter().all(|&(px, py)| (x - px).hypot(y - py) > reach + STREET_WIDTH / 2.0)
}

fn reach(shape: &Shape) -> f64 {
    match *shape {
        Shape::Circle { radius, .. } => radius,
        Shape::Box { length, width, .. } => length.hypot(width) / 2.0,
        _ => 0.0,
    }
}

impl World {
    /// `known` plus `count` random circles and boxes on its roads and `blocked_roads` barriers closing whole road
    /// segments, none near the start or the goal.
    pub fn new(known: &Map, count: usize, blocked_roads: usize, seed: u64) -> World {
        let mut rng = Random::seeded(seed);
        let mut obstacles = road_blocks(&mut rng, blocked_roads);
        obstacles.extend(scatter(&mut rng, known, count));
        let mut truth: Vec<bool> = known.occupancy.iter().map(|&c| c == OCCUPIED).collect();
        for shape in &obstacles {
            let (bx_lo, bx_hi, by_lo, by_hi) = shape.bounds();
            let x_lo = bx_lo.floor().max(0.0) as usize;
            let x_hi = ((bx_hi.ceil() + 1.0).max(0.0) as usize).min(SCREEN_WIDTH);
            let y_lo = by_lo.floor().max(0.0) as usize;
            let y_hi = ((by_hi.ceil() + 1.0).max(0.0) as usize).min(SCREEN_HEIGHT);
            for x in x_lo..x_hi {
                for y in y_lo..y_hi {
                    if shape.signed_distance(x as f64, y as f64) <= 0.0 {
                        truth[cell(x, y)] = true;
                    }
                }
            }
        }
        let truth_sdf = capped_distance(&truth, SCREEN_WIDTH, SCREEN_HEIGHT, SDF_CAP);
        World { obstacles, truth, truth_sdf }
    }

    /// The default layout: LOCAL_OBSTACLE_COUNT obstacles and BLOCKED_ROADS blocked roads.
    pub fn random(known: &Map, seed: u64) -> World {
        World::new(known, LOCAL_OBSTACLE_COUNT, BLOCKED_ROADS, seed)
    }

    /// Simulated long range lidar: LIDAR_BEAMS rays from (x, y), each running until it hits something in the true
    /// world or leaves the map, stepping by the true world's distance to the nearest obstacle. The visible area is
    /// the polygon through the ray ends. Anything behind the first thing a ray hits stays UNKNOWN.
    pub fn scan(&self, x: f64, y: f64) -> LocalGrid {
        let mut ends = Vec::with_capacity(LIDAR_BEAMS);
        for k in 0..LIDAR_BEAMS {
            let (s, c) = (k as f64 * TAU / LIDAR_BEAMS as f64).sin_cos();
            let mut reach = 0.0;
            let hit = loop {
                let (px, py) = ((x + c * reach).round() as i64, (y + s * reach).round() as i64);
                if px < 0 || py < 0 || px >= SCREEN_WIDTH as i64 || py >= SCREEN_HEIGHT as i64 {
                    break false; // off the map ends the ray
                }
                let clearance = self.truth_sdf[cell(px as usize, py as usize)] as f64;
                if clearance <= 0.0 {
                    break true;
                }
                reach += clearance.max(1.0);
            };
            if hit {
                reach += 1.5; // so the polygon covers the surface the ray hit
            }
            ends.push((x + c * reach, y + s * reach));
        }
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (x, x, y, y);
        for &(ex, ey) in &ends {
            (min_x, max_x, min_y, max_y) = (min_x.min(ex), max_x.max(ex), min_y.min(ey), max_y.max(ey));
        }
        let x0 = min_x.floor().max(0.0) as usize;
        let x1 = ((max_x.ceil() + 1.0) as usize).min(SCREEN_WIDTH);
        let y0 = min_y.floor().max(0.0) as usize;
        let y1 = ((max_y.ceil() + 1.0) as usize).min(SCREEN_HEIGHT);
        let (w, h) = (x1 - x0, y1 - y0);
        let mut cells = vec![UNKNOWN; w * h];
        let mut see = |px: i64, py: i64| {
            if 0 <= px && px < w as i64 && 0 <= py && py < h as i64 {
                let (gx, gy) = (x0 + px as usize, y0 + py as usize);
                cells[(px as usize) * h + py as usize] = if self.truth[cell(gx, gy)] { OCCUPIED } else { FREE };
            }
        };
        // Fill the ray-end polygon the way pygame's polygon fill does (as the Python lidar uses): vertices snapped
        // to whole pixels, crossings in integer math, each span filled including both ends, and flat edges drawn
        let points: Vec<(i64, i64)> = ends.iter().map(|&(ex, ey)| ((ex - x0 as f64) as i64, (ey - y0 as f64) as i64)).collect();
        let (min_y, max_y) = (points.iter().map(|p| p.1).min().unwrap(), points.iter().map(|p| p.1).max().unwrap());
        let mut crossings = Vec::new();
        for py in min_y..=max_y {
            crossings.clear();
            for i in 0..points.len() {
                let (prev, here) = (points[if i == 0 { points.len() - 1 } else { i - 1 }], points[i]);
                let ((x1, y1), (x2, y2)) = match prev.1.cmp(&here.1) {
                    std::cmp::Ordering::Less => (prev, here),
                    std::cmp::Ordering::Greater => (here, prev),
                    std::cmp::Ordering::Equal => continue,
                };
                if (py >= y1 && py < y2) || (py == max_y && y2 == max_y) {
                    crossings.push((py - y1) * (x2 - x1) / (y2 - y1) + x1);
                }
            }
            crossings.sort();
            for span in crossings.as_chunks::<2>().0 { // inside between each pair of crossings
                for px in span[0]..=span[1] {
                    see(px, py);
                }
            }
        }
        for i in 0..points.len() {
            let (prev, here) = (points[if i == 0 { points.len() - 1 } else { i - 1 }], points[i]);
            if prev.1 == here.1 && min_y < here.1 && here.1 < max_y {
                for px in prev.0.min(here.0)..=prev.0.max(here.0) {
                    see(px, here.1);
                }
            }
        }
        LocalGrid { x0, y0, w, h, cells }
    }

    /// Scan from (x, y) and merge the result into the known map. Returns whether the known map changed.
    pub fn reveal(&self, known: &mut Map, x: f64, y: f64) -> bool {
        known.merge(&self.scan(x, y))
    }
}

/// Barriers across the midpoints of straight road segments between intersections, closing those roads.
fn road_blocks(rng: &mut Random, n: usize) -> Vec<Shape> {
    let mut mids = Vec::new();
    for &y in &STREET_Y {
        for x in STREET_X.windows(2) {
            mids.push(((x[0] + x[1]) / 2.0, y, PI / 2.0));
        }
    }
    for &x in &STREET_X {
        for y in STREET_Y.windows(2) {
            mids.push((x, (y[0] + y[1]) / 2.0, 0.0));
        }
    }
    let mut barriers: Vec<Shape> = mids
        .into_iter()
        .map(|(x, y, angle)| Shape::Box { x, y, length: STREET_WIDTH + 10.0, width: 12.0, angle, corner: 3.0 })
        .filter(|b| match *b {
            Shape::Box { x, y, .. } => keeps_clear(x, y, reach(b)),
            _ => false,
        })
        .collect();
    // A random n of them (partial Fisher-Yates)
    let n = n.min(barriers.len());
    for k in 0..n {
        let j = rng.randint(k, barriers.len() - 1);
        barriers.swap(k, j);
    }
    barriers.truncate(n);
    barriers
}

fn random_shape(rng: &mut Random, x: f64, y: f64) -> Shape {
    if rng.random() < 0.5 {
        return Shape::Circle { x, y, radius: rng.uniform(LOCAL_OBSTACLE_RADIUS.0, LOCAL_OBSTACLE_RADIUS.1) };
    }
    let width = rng.uniform(LOCAL_BOX_WIDTH.0, LOCAL_BOX_WIDTH.1);
    let length = rng.uniform(LOCAL_BOX_LENGTH.0, LOCAL_BOX_LENGTH.1);
    Shape::Box { x, y, length, width, angle: rng.uniform(0.0, PI), corner: rng.uniform(0.0, width / 2.0) }
}

/// `n` random shapes on the roads (at least 10 px from the curb), by rejection sampling; gives up quietly if the
/// roads are too full.
fn scatter(rng: &mut Random, known: &Map, n: usize) -> Vec<Shape> {
    let mut obstacles = Vec::new();
    for _ in 0..n * 50 {
        if obstacles.len() == n {
            break;
        }
        let (x, y) = (rng.uniform(0.0, SCREEN_WIDTH as f64), rng.uniform(0.0, SCREEN_HEIGHT as f64));
        let shape = random_shape(rng, x, y);
        let (px, py) = ((x as usize).min(SCREEN_WIDTH - 1), (y as usize).min(SCREEN_HEIGHT - 1));
        if known.obstacle_sdf[cell(px, py)] >= 10.0 && keeps_clear(x, y, reach(&shape)) {
            obstacles.push(shape);
        }
    }
    obstacles
}
