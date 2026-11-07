//! Planner node for the sim (or `fake_world` standing in for it): listens for the map on MAP_KEY, the car's pose on
//! POSE_KEY and the goal on GOAL_KEY, and whenever the goal or the map changes, plans from the car's latest pose and
//! publishes the path on SIM_PATH_KEY. Everything on the wire is in meters (see `sim.rs`).
//! `sim_planner [--iterations <n>] [--verbose]`; build it with `--features merceves` so it plans for that car.
use std::time::Duration;

use cev_sst::map::Map;
use cev_sst::sim::{
    GOAL_KEY, Goal, MAP_KEY, OccupancyGrid, POSE_KEY, PathPoint, PlannedPath, Pose, SIM_PATH_KEY, to_units,
};
use cev_sst::{PlanRequest, plan_on, zenoh_config};
use zenoh::Wait;

fn main() -> zenoh::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose");
    let iterations = args.iter().position(|a| a == "--iterations").and_then(|i| args.get(i + 1)?.parse().ok());
    let session = zenoh::open(zenoh_config()?).wait()?;
    let maps = session.declare_subscriber(MAP_KEY).wait()?;
    let poses = session.declare_subscriber(POSE_KEY).wait()?;
    let goals = session.declare_subscriber(GOAL_KEY).wait()?;
    let publisher = session.declare_publisher(SIM_PATH_KEY).wait()?;
    println!("sim planner: listening on '{MAP_KEY}', '{POSE_KEY}', '{GOAL_KEY}'; publishing on '{SIM_PATH_KEY}'");

    let (mut map, mut map_bytes, mut pose, mut goal): (Option<Map>, Vec<u8>, Option<Pose>, Option<Goal>) =
        (None, Vec::new(), None, None);
    let mut changed = false;
    loop {
        // Take everything that has arrived; a map identical to the last one changes nothing
        while let Some(sample) = maps.try_recv()? {
            let bytes = sample.payload().to_bytes().to_vec();
            if bytes != map_bytes {
                match serde_json::from_slice::<OccupancyGrid>(&bytes) {
                    Ok(grid) => {
                        map = Some(grid.to_map());
                        map_bytes = bytes;
                        changed = true;
                        println!("sim planner: new map, {} x {} cells at {} m", grid.width, grid.height, grid.resolution);
                    }
                    Err(err) => eprintln!("sim planner: ignoring a bad map: {err}"),
                }
            }
        }
        while let Some(sample) = poses.try_recv()? {
            match serde_json::from_slice::<Pose>(&sample.payload().to_bytes()) {
                Ok(p) => pose = Some(p),
                Err(err) => eprintln!("sim planner: ignoring a bad pose: {err}"),
            }
        }
        while let Some(sample) = goals.try_recv()? {
            match serde_json::from_slice::<Goal>(&sample.payload().to_bytes()) {
                Ok(g) => {
                    let new = goal.is_none_or(|old| (old.x, old.y, old.radius) != (g.x, g.y, g.radius));
                    if new {
                        println!("sim planner: new goal ({:.2}, {:.2}) m, radius {:.2} m", g.x, g.y, g.radius);
                        changed = true;
                    }
                    goal = Some(g);
                }
                Err(err) => eprintln!("sim planner: ignoring a bad goal: {err}"),
            }
        }

        if let (true, Some(map), Some(pose), Some(goal)) = (changed, &map, pose, goal) {
            changed = false;
            let request = PlanRequest {
                start: Some((to_units(pose.x), to_units(pose.y))),
                start_heading: Some(pose.heading),
                goal: Some((to_units(goal.x), to_units(goal.y))),
                goal_radius: Some(to_units(goal.radius)),
                iterations,
                seed: Some(rand::random()),
                ..Default::default()
            };
            let response = plan_on(map, &request, verbose);
            let path = PlannedPath {
                found: response.found,
                cost: response.cost,
                time: response.time,
                plan_ms: response.plan_ms + response.smooth_ms,
                path: response.path.iter().map(PathPoint::from).collect(),
            };
            println!(
                "sim planner: found={}, {:.1} s of driving, {} points, planned in {:.0} ms",
                path.found, path.time, path.path.len(), path.plan_ms
            );
            publisher.put(serde_json::to_string(&path)?).wait()?;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
