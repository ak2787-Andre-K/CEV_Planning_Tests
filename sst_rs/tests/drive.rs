use cev_sst::config::{FREE, OCCUPIED, START, UNKNOWN};
use cev_sst::kinematics::{can_stop_from, safe_speed};
use cev_sst::library::load_path_library;
use cev_sst::map::{Map, cell};
use cev_sst::tree::State;
use cev_sst::world::World;

fn library() -> Vec<Vec<State>> {
    load_path_library(concat!(env!("CARGO_MANIFEST_DIR"), "/../path_library.npz")).unwrap()
}

#[test]
fn loads_the_python_path_library() {
    let paths = library();
    assert_eq!(paths.iter().map(Vec::len).collect::<Vec<_>>(), [593, 2118]);
    let first = &paths[0][0];
    assert_eq!((first.x, first.y, first.s, first.cost), (START.0, START.1, 0.0, 0.0));
    assert!(paths[0][0].node && paths[0].last().unwrap().node);
    assert!(paths[0].windows(2).all(|w| w[1].cost >= w[0].cost), "costs run up along the path");
}

#[test]
fn lidar_reveals_free_road_and_hidden_obstacles() {
    let mut map = Map::streets(&[]);
    let (x, y) = START;
    assert_eq!(map.occupancy[cell(x as usize + 30, y as usize)], UNKNOWN); // the road ahead starts unknown
    let world = World::new(&map, 0, 0, 1);
    assert!(world.reveal(&mut map, x, y));
    assert_eq!(map.occupancy[cell(x as usize + 30, y as usize)], FREE);
    assert_eq!(map.unknown_sdf[cell(x as usize + 30, y as usize)], 20.0); // well inside the seen area
    assert!(!world.reveal(&mut map, x, y), "scanning the same spot again changes nothing");
    assert!(!map.new_obstacles, "no hidden obstacles in this world");

    // A world with hidden obstacles: scanning from along a road eventually finds one
    let mut map = Map::streets(&[]);
    let world = World::random(&map, 3);
    assert!(world.obstacles.len() >= 50);
    let found = (0..17).any(|k| {
        world.reveal(&mut map, START.0 + 100.0 * k as f64, START.1);
        map.new_obstacles
    });
    assert!(found);
    assert!(map.occupancy.iter().filter(|&&c| c == OCCUPIED).count() > 0);
}

#[test]
fn safe_speed_needs_known_free_space() {
    let mut map = Map::streets(&[]);
    let p = State::new(START.0, START.1, 0.0, 200.0);
    assert_eq!(safe_speed(&map, &p), 0.0, "nothing is known free before a scan");
    World::new(&map, 0, 0, 1).reveal(&mut map, START.0, START.1);
    assert!(can_stop_from(&map, p.x, p.y, p.a, 0.0, 0.0));
    assert!(safe_speed(&map, &p) > 0.0, "after a scan the car can drive down the road");
}
