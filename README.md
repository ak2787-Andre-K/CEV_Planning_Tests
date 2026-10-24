# CEV-planner-hw

Toy homework where we are working on building up a global planner. Currently working with basic rrt and learning sst

## Files

- `rrt.py`: geometric RRT, with the car following the path (`python rrt.py`)
- `world.py`: map, obstacles, car model, collision checking and drawing used by `rrt.py`
- `sst.py`: Andre's SST planner on the street map, with replanning around local obstacles (`python sst.py`).
  Self-contained; it has its own map and car model and doesn't use `world.py`.
- `path_library.npz`: saved paths that `sst.py` loads

Requirements: `pygame` and `numpy` for both, plus `numba` for `sst.py`.

See [docs/planner.md](docs/planner.md) for how both planners work.
