//! Test client: sends one `PlanRequest` to the planner node over zenoh and prints the path it gets back.
//! `--iterations <n>`, `--seed <n>` and `--no-smooth` set those fields of the request.
use std::time::{Duration, Instant};

use cev_sst::{PATH_KEY, PlanRequest, PlanResponse, REQUEST_KEY, zenoh_config};
use zenoh::Wait;

const TIMEOUT: Duration = Duration::from_secs(120); // a full 100,000-iteration plan takes a while

fn main() -> zenoh::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)?.parse().ok());
    let session = zenoh::open(zenoh_config()?).wait()?;
    // Subscribe before publishing so the response can't arrive before we're listening
    let subscriber = session.declare_subscriber(PATH_KEY).wait()?;
    let id = rand::random::<u32>() as u64;
    let request = PlanRequest {
        id,
        iterations: value("--iterations").map(|n: u64| n as usize),
        seed: value("--seed"),
        smooth: args.iter().any(|a| a == "--no-smooth").then_some(false),
        ..Default::default()
    };

    // Resend every second until the planner answers, since it may not have been discovered yet (it skips repeats)
    let deadline = Instant::now() + TIMEOUT;
    while Instant::now() < deadline {
        session.put(REQUEST_KEY, serde_json::to_string(&request)?).wait()?;
        let Some(sample) = subscriber.recv_timeout(Duration::from_secs(1))? else { continue };
        let response: PlanResponse = serde_json::from_slice(&sample.payload().to_bytes())?;
        if response.id != id {
            continue; // someone else's answer
        }
        println!(
            "Got path over zenoh: found={}, cost {:.2}, {:.2} s of driving, {} points, planned in {:.0} ms",
            response.found, response.cost, response.time, response.path.len(), response.plan_ms
        );
        if let (Some(first), Some(last)) = (response.path.first(), response.path.last()) {
            println!("  ({:.0}, {:.0}) -> ({:.0}, {:.0})", first.x, first.y, last.x, last.y);
        }
        return Ok(());
    }
    Err(format!("no response on '{PATH_KEY}' within {TIMEOUT:?}; is the planner running?").into())
}
