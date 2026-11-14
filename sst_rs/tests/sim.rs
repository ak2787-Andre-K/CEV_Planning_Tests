use cev_sst::config::{FREE, OCCUPIED, UNKNOWN};
use cev_sst::map::cell;
use cev_sst::sim::{
    NAV2_FREE, NAV2_INSCRIBED, NAV2_LETHAL, NAV2_UNKNOWN, OccupancyGrid, Path, PathPoint, pose2, to_meters, to_units,
    xy_heading,
};
use cev_sst::tree::State;
use prost::Message;

#[test]
fn occupancy_grids_become_planner_maps() {
    // 5 x 2 cells of 10 m (default build: 1 unit = 1 m). Row 0: free, lethal, unknown, inscribed, a cost (free)
    let grid = OccupancyGrid {
        header: None, size_x: 5, size_y: 2, resolution_m: 10.0, origin_x: 0.0, origin_y: 0.0,
        data: vec![NAV2_FREE, NAV2_LETHAL, NAV2_UNKNOWN, NAV2_INSCRIBED, 100, 0, 0, 0, 0, 0],
    };
    // Through the wire and back: same bytes layout as cev-interfaces' lidar.proto
    let grid = OccupancyGrid::decode(grid.encode_to_vec().as_slice()).unwrap();
    assert_eq!(grid.at(15.0, 5.0), Some(NAV2_LETHAL));
    assert_eq!(grid.at(55.0, 5.0), None);
    let map = grid.to_map();
    assert_eq!(map.occupancy[cell(5, 5)], FREE);
    assert_eq!(map.occupancy[cell(15, 5)], OCCUPIED);
    assert_eq!(map.occupancy[cell(25, 5)], UNKNOWN);
    assert_eq!(map.occupancy[cell(35, 5)], OCCUPIED, "inscribed counts as an obstacle");
    assert_eq!(map.occupancy[cell(45, 5)], FREE, "costs below inscribed are free; the planner keeps its own margin");
    assert_eq!(map.occupancy[cell(100, 100)], UNKNOWN, "off the grid is unknown");
    assert_eq!(to_meters(to_units(3.5)), 3.5);
}

#[test]
fn path_points_wrap_heading_and_round_trip() {
    let state = State::new(30.0, 40.0, 3.0 * std::f64::consts::PI / 2.0, 2.0);
    let point = PathPoint::from(&state);
    let (x, y, heading) = xy_heading(point.pose.as_ref());
    assert_eq!((x, y), (to_meters(30.0), to_meters(40.0)));
    assert!((heading + std::f64::consts::FRAC_PI_2).abs() < 1e-12, "3/2 pi wraps to -pi/2, as Pose2 wants");
    let path = Path { header: None, points: vec![point], reaches_goal: true, cost: 1.5 };
    let back = Path::decode(path.encode_to_vec().as_slice()).unwrap();
    assert_eq!(back, path);
    assert_eq!(xy_heading(Some(&pose2(1.0, 2.0, 0.5))), (1.0, 2.0, 0.5));
}
