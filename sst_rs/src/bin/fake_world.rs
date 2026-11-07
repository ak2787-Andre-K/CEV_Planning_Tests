//! Stands in for the sim until it's on Zenoh: publishes a made-up 30 x 15 m lot as an occupancy grid on MAP_KEY, the
//! car's pose on POSE_KEY and a goal on GOAL_KEY, once a second, and draws every path that comes back on SIM_PATH_KEY
//! (with the map, the start, the goal and the car's outline along it) to `fake_world.png`.
//! `fake_world [--once] [--out <file.png>]`: --once exits after the first path.
use std::io::Write;
use std::time::Duration;

use cev_sst::sim::{
    CAR_LENGTH_M, CAR_WIDTH_M, GOAL_KEY, Goal, MAP_KEY, OccupancyGrid, POSE_KEY, PlannedPath, Pose, SIM_PATH_KEY,
    WHEELBASE_M,
};
use cev_sst::zenoh_config;
use flate2::Compression;
use flate2::write::ZlibEncoder;
use zenoh::Wait;

const RESOLUTION: f64 = 0.05; // m per cell
const SIZE: (f64, f64) = (30.0, 15.0); // m
const START: Pose = Pose { x: 2.0, y: 2.0, heading: 0.0, speed: 0.0 };
const GOAL: Goal = Goal { x: 28.0, y: 12.0, radius: 1.0 };
const PIXELS_PER_M: f64 = 40.0; // picture scale

/// The lot: walls around it, two blocks that force an S-shaped route (gaps sized for the car's ~3.4 m turning
/// radius), and a few small boxes. Rectangles as (x_lo, y_lo, x_hi, y_hi) in meters.
fn lot() -> OccupancyGrid {
    let rects = [
        (0.0, 0.0, 30.0, 0.3), (0.0, 14.7, 30.0, 15.0), (0.0, 0.0, 0.3, 15.0), (29.7, 0.0, 30.0, 15.0), // walls
        (8.0, 0.0, 11.0, 8.5),  // block A: go over it
        (21.0, 6.5, 24.0, 15.0), // block B: go under it
        (15.0, 13.0, 16.0, 14.0), (12.5, 2.6, 13.3, 3.4), (4.0, 12.0, 5.0, 13.0), // small boxes
    ];
    let (width, height) = ((SIZE.0 / RESOLUTION) as usize, (SIZE.1 / RESOLUTION) as usize);
    let mut data = vec![0i8; width * height];
    for row in 0..height {
        for col in 0..width {
            let (x, y) = ((col as f64 + 0.5) * RESOLUTION, (row as f64 + 0.5) * RESOLUTION);
            if rects.iter().any(|&(x0, y0, x1, y1)| x0 <= x && x < x1 && y0 <= y && y < y1) {
                data[row * width + col] = 100;
            }
        }
    }
    OccupancyGrid { resolution: RESOLUTION, width, height, origin: (0.0, 0.0), data }
}

/// An RGB picture, y up like the world.
struct Picture {
    w: usize,
    h: usize,
    rgb: Vec<u8>,
}

impl Picture {
    fn set(&mut self, x: f64, y: f64, color: [u8; 3]) {
        let (px, py) = ((x * PIXELS_PER_M) as i64, self.h as i64 - 1 - (y * PIXELS_PER_M) as i64);
        if 0 <= px && px < self.w as i64 && 0 <= py && py < self.h as i64 {
            let i = (py as usize * self.w + px as usize) * 3;
            self.rgb[i..i + 3].copy_from_slice(&color);
        }
    }

    /// A line `thick` pixels wide.
    fn line(&mut self, (x0, y0): (f64, f64), (x1, y1): (f64, f64), thick: f64, color: [u8; 3]) {
        let steps = ((x1 - x0).hypot(y1 - y0) * PIXELS_PER_M * 2.0).ceil().max(1.0) as usize;
        let r = thick / 2.0 / PIXELS_PER_M;
        for i in 0..=steps {
            let f = i as f64 / steps as f64;
            let (x, y) = (x0 + f * (x1 - x0), y0 + f * (y1 - y0));
            let mut d = -r;
            while d <= r {
                let mut e = -r;
                while e <= r {
                    self.set(x + d, y + e, color);
                    e += 0.5 / PIXELS_PER_M;
                }
                d += 0.5 / PIXELS_PER_M;
            }
        }
    }

    /// The car's outline with its rear axle at `p` (the body centered between the axles, as the planner assumes).
    fn car(&mut self, p: (f64, f64, f64), color: [u8; 3]) {
        let (x, y, a) = p;
        let (s, c) = a.sin_cos();
        let center = WHEELBASE_M / 2.0;
        let corners: Vec<(f64, f64)> = [(1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)]
            .iter()
            .map(|&(f, l)| {
                let (u, v) = (center + f * CAR_LENGTH_M / 2.0, l * CAR_WIDTH_M / 2.0);
                (x + u * c - v * s, y + u * s + v * c)
            })
            .collect();
        for k in 0..4 {
            self.line(corners[k], corners[(k + 1) % 4], 2.0, color);
        }
    }

    fn save_png(&self, file: &str) -> std::io::Result<()> {
        fn crc(bytes: &[u8]) -> u32 {
            let mut c = 0xFFFF_FFFFu32;
            for &b in bytes {
                c ^= b as u32;
                for _ in 0..8 {
                    c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
                }
            }
            !c
        }
        fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
            out.extend((data.len() as u32).to_be_bytes());
            let mut body = kind.to_vec();
            body.extend(data);
            out.extend(&body);
            out.extend(crc(&body).to_be_bytes());
        }
        let mut raw = Vec::with_capacity((self.w * 3 + 1) * self.h);
        for row in self.rgb.chunks(self.w * 3) {
            raw.push(0); // no filter
            raw.extend(row);
        }
        let mut z = ZlibEncoder::new(Vec::new(), Compression::default());
        z.write_all(&raw)?;
        let mut header = (self.w as u32).to_be_bytes().to_vec();
        header.extend((self.h as u32).to_be_bytes());
        header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        chunk(&mut png, b"IHDR", &header);
        chunk(&mut png, b"IDAT", &z.finish()?);
        chunk(&mut png, b"IEND", &[]);
        std::fs::write(file, png)
    }
}

/// The lot with the path drawn on it: walls and boxes dark, the path blue, the car's outline every second along it
/// orange, the start green and the goal red.
fn draw(grid: &OccupancyGrid, path: &PlannedPath, file: &str) -> std::io::Result<()> {
    let (w, h) = ((SIZE.0 * PIXELS_PER_M) as usize, (SIZE.1 * PIXELS_PER_M) as usize);
    let mut pic = Picture { w, h, rgb: vec![0; w * h * 3] };
    for py in 0..h {
        for px in 0..w {
            let (x, y) = ((px as f64 + 0.5) / PIXELS_PER_M, (h - 1 - py) as f64 / PIXELS_PER_M);
            let color = match grid.at(x, y) {
                Some(v) if v >= 50 => [60, 60, 60],
                Some(v) if v >= 0 => [245, 245, 245],
                _ => [180, 180, 180],
            };
            pic.rgb[(py * w + px) * 3..][..3].copy_from_slice(&color);
        }
    }
    for k in 0..48 {
        let (a, b) = (k as f64 / 48.0 * std::f64::consts::TAU, (k + 1) as f64 / 48.0 * std::f64::consts::TAU);
        let r = GOAL.radius;
        pic.line((GOAL.x + r * a.cos(), GOAL.y + r * a.sin()), (GOAL.x + r * b.cos(), GOAL.y + r * b.sin()), 3.0, [210, 40, 40]);
    }
    for pair in path.path.windows(2) {
        pic.line((pair[0].x, pair[0].y), (pair[1].x, pair[1].y), 3.0, [30, 100, 230]);
    }
    let mut next = 0.0;
    for p in &path.path {
        if p.time >= next {
            pic.car((p.x, p.y, p.heading), [240, 140, 20]);
            next += 1.0;
        }
    }
    pic.car((START.x, START.y, START.heading), [40, 170, 60]);
    pic.save_png(file)
}

fn main() -> zenoh::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let once = args.iter().any(|a| a == "--once");
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned()
        .unwrap_or_else(|| "fake_world.png".into());
    let session = zenoh::open(zenoh_config()?).wait()?;
    let paths = session.declare_subscriber(SIM_PATH_KEY).wait()?;
    let grid = lot();
    let (map_json, pose_json, goal_json) =
        (serde_json::to_string(&grid)?, serde_json::to_string(&START)?, serde_json::to_string(&GOAL)?);
    println!("fake world: a {} x {} m lot; car at ({}, {}) m, goal ({}, {}) m; publishing every second",
             SIZE.0, SIZE.1, START.x, START.y, GOAL.x, GOAL.y);
    loop {
        session.put(MAP_KEY, map_json.as_str()).wait()?;
        session.put(POSE_KEY, pose_json.as_str()).wait()?;
        session.put(GOAL_KEY, goal_json.as_str()).wait()?;
        if let Some(sample) = paths.recv_timeout(Duration::from_secs(1))? {
            let path: PlannedPath = serde_json::from_slice(&sample.payload().to_bytes())?;
            println!("fake world: got a path: found={}, {:.1} s of driving, {} points; drawn to {out}",
                     path.found, path.time, path.path.len());
            draw(&grid, &path, &out)?;
            if once {
                return Ok(());
            }
        }
    }
}
