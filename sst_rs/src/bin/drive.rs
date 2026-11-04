//! Drives without a window, like the Python demo: hidden obstacles, a lidar scan from the car every LIDAR_DT, and the
//! Driver on the saved path library. `drive [--seeds <n>] [--first <seed>] [--seconds <s>] [--library <file>]
//! [--verbose]` drives once per world seed and prints one JSON line per drive.
use std::collections::BTreeMap;
use std::time::Instant;

use cev_sst::config::LIDAR_DT;
use cev_sst::driver::{Driver, Mode};
use cev_sst::library::load_path_library;
use cev_sst::map::Map;
use cev_sst::world::World;

const DT: f64 = 1.0 / 60.0;

/// The kind of event a Driver message reports, for counting.
fn kind(event: &str) -> &'static str {
    let kinds = [
        ("nudge gets past", "nudge"), ("blocked ahead; reconnected", "reconnect"), ("blocked ahead, replan", "replan"),
        ("blocked ahead, stop", "stop"), ("joined", "replan_join"), ("improved the route", "improve"),
        ("forward test", "forward_fan"), ("library maneuver", "library"), ("stopped; reconnected", "reconnect_at_stop"),
        ("library manoeuvre", "library"), ("driving", "commit_branch"), ("can't grow", "dead_replan"),
        ("backing up", "back_up"), ("reached the replanning point", "late_stop"), ("stopped short", "stopped_short"),
        ("stalled", "stalled"), ("reached the goal", "goal"),
    ];
    kinds.iter().find(|(text, _)| event.contains(text)).map_or("other", |(_, k)| k)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let seeds: u64 = value("--seeds").and_then(|v| v.parse().ok()).unwrap_or(1);
    let first: u64 = value("--first").and_then(|v| v.parse().ok()).unwrap_or(0);
    let seconds: f64 = value("--seconds").and_then(|v| v.parse().ok()).unwrap_or(120.0);
    let library = value("--library").unwrap_or_else(|| format!("{}/../path_library.npz", env!("CARGO_MANIFEST_DIR")));
    let verbose = args.iter().any(|a| a == "--verbose");
    let paths = load_path_library(&library).unwrap_or_else(|e| panic!("can't read the path library: {e}"));
    cev_sst::maneuvers::warm_up();
    for seed in first..first + seeds {
        let known = Map::streets(&[]);
        let world = World::random(&known, seed);
        let mut driver = Driver::new(known, paths.clone(), seed);
        driver.set_verbose(verbose);
        let (timer, mut scan, mut frames) = (Instant::now(), 0.0, 0);
        while frames < (seconds / DT) as usize && driver.mode() != Mode::Done {
            scan -= DT;
            if scan <= 0.0 {
                let (x, y, ..) = driver.pose();
                world.reveal(&mut driver.map, x, y);
                scan = LIDAR_DT;
            }
            driver.update(DT);
            frames += 1;
        }
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for event in driver.events() {
            *counts.entry(kind(event)).or_default() += 1;
        }
        let (x, y, ..) = driver.pose();
        println!(
            "{}",
            serde_json::json!({
                "seed": seed, "done": driver.mode() == Mode::Done, "frames": frames, "time": frames as f64 * DT,
                "end": [x.round(), y.round()], "wall_ms": timer.elapsed().as_millis(), "events": counts,
            })
        );
    }
}
