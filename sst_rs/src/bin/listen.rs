//! Prints every Zenoh message it hears, with its key, so you can see what other nodes publish:
//! `listen [<key expression>] [--full]`. The key expression defaults to `**` (everything). The planning and map
//! topics (see `sim.rs`) are decoded from protobuf; other payloads are shown as text if they are text (cut to 300
//! characters unless `--full`), otherwise as their size and first bytes in hex.
use std::time::Instant;

use cev_sst::sim::{
    GLOBAL_PATH_KEY, GOAL_KEY, Goal, MAP_KEY, OccupancyGrid, Path, Pose2, STATE_KEY, VehicleState, xy_heading,
};
use cev_sst::zenoh_config;
use prost::Message;
use zenoh::Wait;

fn pose(p: Option<&Pose2>) -> String {
    let (x, y, h) = xy_heading(p);
    format!("({x:.2}, {y:.2}) m, heading {h:.2} rad")
}

/// A one-line summary of a known protobuf message, or None for other keys (or bytes that don't decode).
fn decode(key: &str, bytes: &[u8]) -> Option<String> {
    match key {
        MAP_KEY => OccupancyGrid::decode(bytes).ok().map(|g| {
            let count = |f: fn(u8) -> bool| g.data.iter().filter(|&&v| f(v)).count();
            format!("OccupancyGrid {} x {} cells at {} m, origin ({}, {}): {} free, {} occupied, {} unknown",
                    g.size_x, g.size_y, g.resolution_m, g.origin_x, g.origin_y, count(|v| v == 0),
                    count(|v| (253..255).contains(&v)), count(|v| v == 255))
        }),
        STATE_KEY => VehicleState::decode(bytes).ok().map(|s| {
            format!("VehicleState {}, speed {:.2} m/s, steering {:.3} rad", pose(s.pose.as_ref()), s.speed_mps,
                    s.steering_rad)
        }),
        GOAL_KEY => Goal::decode(bytes).ok().map(|g| format!("Goal {}, tolerance {} m", pose(g.pose.as_ref()), g.tolerance_m)),
        GLOBAL_PATH_KEY => Path::decode(bytes).ok().map(|p| {
            let (first, last) = (p.points.first(), p.points.last());
            format!("Path reaches_goal={}, cost {:.2}, {} points, {:.1} s: {} -> {}", p.reaches_goal, p.cost,
                    p.points.len(), last.map_or(0.0, |l| l.time_s), pose(first.and_then(|f| f.pose.as_ref())),
                    pose(last.and_then(|l| l.pose.as_ref())))
        }),
        _ => None,
    }
}

fn main() -> zenoh::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let full = args.iter().any(|a| a == "--full");
    // The first argument that isn't a flag (or a --connect / --listen endpoint) is the key expression
    let mut key = "**".to_string();
    let mut skip = false;
    for a in &args {
        if skip {
            skip = false;
        } else if a == "--connect" || a == "--listen" {
            skip = true;
        } else if !a.starts_with("--") {
            key = a.clone();
            break;
        }
    }
    let session = zenoh::open(zenoh_config()?).wait()?;
    let subscriber = session.declare_subscriber(&key).wait()?;
    println!("Listening on '{key}' (Ctrl+C to stop)...");
    let start = Instant::now();
    while let Ok(sample) = subscriber.recv() {
        let bytes = sample.payload().to_bytes();
        let shown = match decode(sample.key_expr().as_str(), &bytes) {
            Some(decoded) => decoded,
            None => match std::str::from_utf8(&bytes) {
                Ok(text) if full || text.len() <= 300 => text.to_string(),
                Ok(text) => format!("{}... ({} bytes)", &text[..text.floor_char_boundary(300)], text.len()),
                Err(_) => {
                    let head: Vec<String> = bytes.iter().take(24).map(|b| format!("{b:02x}")).collect();
                    let more = if bytes.len() > 24 { " ..." } else { "" };
                    format!("<{} bytes of binary: {}{more}>", bytes.len(), head.join(" "))
                }
            },
        };
        println!("[{:7.2} s] {}: {shown}", start.elapsed().as_secs_f64(), sample.key_expr());
    }
    Ok(())
}
