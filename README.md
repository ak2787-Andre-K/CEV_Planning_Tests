# CEV-planner-hw

Toy homework where we are working on building up a global planner. Currently working with basic rrt and learning sst

## Files

- `rrt.py`: geometric RRT, with the car following the path (`python rrt.py`)
- `world.py`: map, obstacles, car model, collision checking and drawing used by `rrt.py`
- `sst/`: Andre's SST planner on the street map, with replanning around local obstacles (`python -m sst`).
  Self-contained; it has its own map and car model and doesn't use `world.py`.
- `path_library.npz`: saved paths that `sst` loads
- `sst_rs/`: the SST global planner in Rust, run as a zenoh node. `planner` takes JSON requests on
  `cev/planner/sst/request` and publishes the path on `cev/planner/sst/path`; `request` is a test client and
  `plan` plans without zenoh. In `sst_rs/`, run `cargo run --release --bin planner` in one terminal and
  `cargo run --release --bin request` in another.
  Add `--listen tcp/0.0.0.0:7447` / `--connect tcp/<ip>:7447` if multicast discovery doesn't work on your network.

Requirements: `pygame` and `numpy` for both, plus `numba` for `sst`, and Rust (`cargo`) for `sst_rs`.

See [docs/planner.md](docs/planner.md) for how both planners work, and Part 3 there for the Rust version and its zenoh messages.
