//! Zenoh planner node: plans an SST path for each JSON `PlanRequest` on REQUEST_KEY and publishes the
//! `PlanResponse` on PATH_KEY. `--verbose` prints the planner's progress.
use std::collections::HashSet;

use cev_sst::map::Map;
use cev_sst::{PATH_KEY, PlanRequest, REQUEST_KEY, plan, zenoh_config};
use zenoh::Wait;

fn main() -> zenoh::Result<()> {
    let verbose = std::env::args().any(|a| a == "--verbose");
    let session = zenoh::open(zenoh_config()?).wait()?;
    let subscriber = session.declare_subscriber(REQUEST_KEY).wait()?;
    let publisher = session.declare_publisher(PATH_KEY).wait()?;
    let streets = Map::streets(&[]); // built once; requests with extra obstacles get their own copy
    let mut answered = HashSet::new(); // clients resend until they hear back, so skip repeats of a nonzero id
    println!("SST planner listening on '{REQUEST_KEY}', publishing on '{PATH_KEY}'");

    while let Ok(sample) = subscriber.recv() {
        let request: PlanRequest = match serde_json::from_slice(&sample.payload().to_bytes()) {
            Ok(request) => request,
            Err(err) => {
                eprintln!("Ignoring bad request: {err}");
                continue;
            }
        };
        if request.id != 0 && !answered.insert(request.id) {
            continue;
        }
        let response = plan(&streets, &request, verbose);
        println!(
            "Request {}: found={}, cost {:.2} ({:.2} before smoothing), {:.2} s of driving, {} path points, \
             {} iterations in {:.0} ms, smoothed in {:.0} ms",
            response.id, response.found, response.cost, response.raw_cost, response.time, response.path.len(),
            response.iterations, response.plan_ms, response.smooth_ms
        );
        publisher.put(serde_json::to_string(&response)?).wait()?;
    }
    Ok(())
}
