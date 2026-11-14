//! Planner node for the sim (or `fake_world` standing in for it): listens for the lidar's global map on MAP_KEY, the
//! car's state on STATE_KEY and the goal on GOAL_KEY, and whenever the goal or the map changes, plans from the car's
//! latest state and publishes the path on GLOBAL_PATH_KEY. All protobuf, from cev-interfaces (see `sim.rs`).
//! `sim_planner [--iterations <n>] [--verbose]`; build it with `--features merceves` so it plans for that car.
use std::time::Duration;

use cev_sst::map::Map;
use cev_sst::sim::{
    GLOBAL_PATH_KEY, GOAL_KEY, Goal, MAP_KEY, OccupancyGrid, Path, PathPoint, STATE_KEY, VehicleState, header_now,
    to_units, xy_heading,
};
use cev_sst::{PlanRequest, plan_on, zenoh_config};
use prost::Message;
use zenoh::Wait;

fn main() -> zenoh::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose");
    let iterations = args.iter().position(|a| a == "--iterations").and_then(|i| args.get(i + 1)?.parse().ok());
    let session = zenoh::open(zenoh_config()?).wait()?;
    let maps = session.declare_subscriber(MAP_KEY).wait()?;
    let states = session.declare_subscriber(STATE_KEY).wait()?;
    let goals = session.declare_subscriber(GOAL_KEY).wait()?;
    let publisher = session.declare_publisher(GLOBAL_PATH_KEY).wait()?;
    println!("sim planner: listening on '{MAP_KEY}', '{STATE_KEY}', '{GOAL_KEY}'; publishing on '{GLOBAL_PATH_KEY}'");

    let (mut map, mut map_data, mut state, mut goal): (Option<Map>, Vec<u8>, Option<VehicleState>, Option<Goal>) =
        (None, Vec::new(), None, None);
    let mut changed = false;
    let mut seq = 0;
    loop {
        // Take everything that has arrived; a map with the same cells as the last one changes nothing
        while let Some(sample) = maps.try_recv()? {
            match OccupancyGrid::decode(sample.payload().to_bytes().as_ref()) {
                Ok(grid) => {
                    let mut key = grid.data.clone();
                    key.extend(format!("{} {} {} {} {}", grid.size_x, grid.size_y, grid.resolution_m, grid.origin_x,
                                       grid.origin_y).bytes());
                    if key != map_data {
                        println!("sim planner: new map, {} x {} cells at {} m", grid.size_x, grid.size_y,
                                 grid.resolution_m);
                        map = Some(grid.to_map());
                        map_data = key;
                        changed = true;
                    }
                }
                Err(err) => eprintln!("sim planner: ignoring a bad map: {err}"),
            }
        }
        while let Some(sample) = states.try_recv()? {
            match VehicleState::decode(sample.payload().to_bytes().as_ref()) {
                Ok(s) => state = Some(s),
                Err(err) => eprintln!("sim planner: ignoring a bad vehicle state: {err}"),
            }
        }
        while let Some(sample) = goals.try_recv()? {
            match Goal::decode(sample.payload().to_bytes().as_ref()) {
                Ok(g) => {
                    let same = goal.as_ref().is_some_and(|old| (old.pose, old.tolerance_m) == (g.pose, g.tolerance_m));
                    if !same {
                        let (x, y, _) = xy_heading(g.pose.as_ref());
                        println!("sim planner: new goal ({x:.2}, {y:.2}) m, tolerance {:.2} m", g.tolerance_m);
                        changed = true;
                    }
                    goal = Some(g);
                }
                Err(err) => eprintln!("sim planner: ignoring a bad goal: {err}"),
            }
        }

        if let (true, Some(map), Some(state), Some(goal)) = (changed, &map, &state, &goal) {
            changed = false;
            let (sx, sy, heading) = xy_heading(state.pose.as_ref());
            let (gx, gy, _) = xy_heading(goal.pose.as_ref());
            let request = PlanRequest {
                start: Some((to_units(sx), to_units(sy))),
                start_heading: Some(heading),
                goal: Some((to_units(gx), to_units(gy))),
                goal_radius: Some(to_units(goal.tolerance_m)),
                iterations,
                seed: Some(rand::random()),
                ..Default::default()
            };
            let response = plan_on(map, &request, verbose);
            seq += 1;
            let path = Path {
                header: Some(cev_sst::sim::Header { seq, ..header_now("world") }),
                points: response.path.iter().map(PathPoint::from).collect(),
                reaches_goal: response.found,
                cost: response.cost,
            };
            println!(
                "sim planner: reaches_goal={}, {:.1} s of driving, {} points, planned in {:.0} ms",
                path.reaches_goal, response.time, path.points.len(), response.plan_ms + response.smooth_ms
            );
            publisher.put(path.encode_to_vec()).wait()?;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}
