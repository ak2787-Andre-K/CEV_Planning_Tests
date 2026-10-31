//! Plans without zenoh, for testing: `plan [--seeds <n>] [--iterations <n>] [--no-smooth] [--out <file>]` plans once
//! per seed 0..n and prints one JSON line of results per seed. `--out` also writes the last plan's full response.
use cev_sst::map::Map;
use cev_sst::{PlanRequest, plan};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let seeds: u64 = value("--seeds").and_then(|v| v.parse().ok()).unwrap_or(1);
    let iterations = value("--iterations").and_then(|v| v.parse().ok());
    let smooth = args.iter().any(|a| a == "--no-smooth").then_some(false);
    let streets = Map::streets(&[]);
    for seed in 0..seeds {
        let request = PlanRequest { id: seed, seed: Some(seed), iterations, smooth, ..Default::default() };
        let response = plan(&streets, &request, seeds == 1);
        println!(
            "{}",
            serde_json::json!({
                "seed": seed, "found": response.found, "found_at": response.found_at, "cost": response.cost,
                "raw_cost": response.raw_cost, "time": response.time, "points": response.path.len(),
                "active_nodes": response.active_nodes, "plan_ms": response.plan_ms, "smooth_ms": response.smooth_ms,
            })
        );
        if seed + 1 == seeds
            && let Some(out) = value("--out") {
                std::fs::write(&out, serde_json::to_string(&response).unwrap()).unwrap();
            }
    }
}
