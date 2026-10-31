use cev_sst::config::{CHECK_SPACING, SAFETY_MARGIN, START, TARGET, TARGET_RAD};
use cev_sst::kinematics::body_clearance;
use cev_sst::map::{Map, Shape, cell};
use cev_sst::{PlanRequest, PlanResponse, plan};

fn request(seed: u64) -> PlanRequest {
    PlanRequest { seed: Some(seed), iterations: Some(30_000), ..Default::default() }
}

fn assert_valid(map: &Map, response: &PlanResponse) {
    assert!(response.found);
    let (first, last) = (response.path.first().unwrap(), response.path.last().unwrap());
    assert_eq!((first.x, first.y), START);
    // A shortcut that joins at the path's last state ends up to the join tolerance short of it (as in Python's
    // spliceRoute), so a smoothed path can stop just outside the goal circle
    let end = (last.x - TARGET.0).hypot(last.y - TARGET.1);
    assert!(end <= TARGET_RAD + SAFETY_MARGIN + CHECK_SPACING, "path ends {end:.1} px from the goal");
    for s in &response.path {
        assert!(body_clearance(map, s.x, s.y, s.a) >= SAFETY_MARGIN, "state {s:?} is too close to an obstacle");
    }
    assert!(response.cost <= response.raw_cost, "smoothing made the path more expensive");
}

#[test]
fn street_map_has_roads_and_blocks() {
    let map = Map::streets(&[]);
    assert!(!map.occupied[cell(START.0 as usize, START.1 as usize)]); // the start is on a road
    assert!(map.occupied[cell(430, 240)]); // the middle of a city block
    assert_eq!(map.obstacle_sdf[cell(430, 240)], 0.0);
    assert_eq!(map.obstacle_sdf[cell(START.0 as usize, START.1 as usize)], 20.0); // capped, 40 px from the curbs
}

#[test]
fn plans_collision_free_paths_to_the_goal() {
    let map = Map::streets(&[]);
    let found: Vec<PlanResponse> = (0..4).map(|seed| plan(&map, &request(seed), false)).filter(|r| r.found).collect();
    assert!(found.len() >= 2, "only {} of 4 seeds found a path", found.len());
    for response in &found {
        assert_valid(&map, response);
    }
}

#[test]
fn plans_around_extra_obstacles() {
    let streets = Map::streets(&[]);
    let first = (0..4).map(|seed| plan(&streets, &request(seed), false)).find(|r| r.found).unwrap();
    // Block the road at the middle of the path found without the obstacle
    let mid = first.path[first.path.len() / 2];
    let obstacles = vec![Shape::Circle { x: mid.x, y: mid.y, radius: 15.0 }];
    let blocked = Map::streets(&obstacles);
    let replanned = (0..8)
        .map(|seed| plan(&streets, &PlanRequest { obstacles: obstacles.clone(), ..request(seed) }, false))
        .find(|r| r.found)
        .expect("no path around the obstacle");
    assert_valid(&blocked, &replanned);
}

#[test]
fn parses_the_request_example_in_the_docs() {
    let doc = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../docs/planner.md")).unwrap();
    let example = doc.split("```json\n").nth(1).unwrap().split("```").next().unwrap();
    let request: PlanRequest = serde_json::from_str(example).unwrap();
    assert_eq!(request.id, 7);
    assert_eq!(request.obstacles.len(), 2);
    assert!(matches!(request.obstacles[1], Shape::Box { angle, corner, .. } if angle == 0.5 && corner == 0.0));
}
