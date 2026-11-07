use cev_sst::config::{FREE, OCCUPIED, UNKNOWN};
use cev_sst::map::cell;
use cev_sst::sim::{OccupancyGrid, PlannedPath, to_meters, to_units};

#[test]
fn grid_messages_become_planner_maps() {
    // 4 x 2 cells of 10 m: row 0 = free, occupied, unknown, free; row 1 all free (default build: 1 unit = 1 m)
    let grid = OccupancyGrid {
        resolution: 10.0, width: 4, height: 2, origin: (0.0, 0.0), data: vec![0, 100, -1, 0, 0, 0, 0, 0],
    };
    assert_eq!(grid.at(15.0, 5.0), Some(100));
    assert_eq!(grid.at(45.0, 5.0), None);
    let map = grid.to_map();
    assert_eq!(map.occupancy[cell(5, 5)], FREE);
    assert_eq!(map.occupancy[cell(15, 5)], OCCUPIED);
    assert_eq!(map.occupancy[cell(25, 5)], UNKNOWN);
    assert_eq!(map.occupancy[cell(100, 100)], UNKNOWN, "off the grid is unknown");
    assert_eq!(map.obstacle_sdf[cell(15, 5)], 0.0);
    assert_eq!(to_meters(to_units(3.5)), 3.5);
}

#[test]
fn path_messages_round_trip() {
    let json = r#"{"found":true,"cost":1.5,"time":2.0,"plan_ms":10.0,
                   "path":[{"x":1.0,"y":2.0,"heading":0.5,"speed":1.0,"steer":0.1,"time":0.0}]}"#;
    let path: PlannedPath = serde_json::from_str(json).unwrap();
    assert_eq!((path.path[0].x, path.path[0].heading), (1.0, 0.5));
}
