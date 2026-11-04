# CEV Planner Notes

*Patrick, Andre*

## What we're doing

We have a car that needs to drive from a start point to a purple target circle (radius 70 px at
(1950, 700)) on a 2000 × 1000 pixel map, without hitting anything.

We have two planners, and right now they each use their own map:

- **RRT** (`rrt.py`) runs on the original map: 377 circles scattered everywhere, 161 bigger blue ones
  ("global") and 216 smaller green ones ("local"). It plans for a point, then a controller drives the
  car along the path.
- **SST** (`sst/`) runs on a street map: city blocks with roads between them, plus hidden obstacles
  on the roads that the car only finds out about when its lidar sees them. It plans with the car's
  actual motion, and replans while driving when the road turns out to be blocked.

The code:

| File | What's in it | Run it |
|---|---|---|
| `rrt.py` | RRT and the path-following controller | `python rrt.py` |
| `world.py` | The circle map, the car model, collision checking and drawing, used by `rrt.py` | (not run directly) |
| `sst/` | Andre's SST: street map, lidar, planner, replanning and the driver. Self-contained, doesn't use `world.py` | `python -m sst` |
| `path_library.npz` | Saved paths that `sst` starts from | |
| `sst_rs/` | The SST global planner in Rust, run as a zenoh node (see Part 3) | `cargo run --release --bin planner` |

You need `pygame` and `numpy` for both, and `numba` for `sst`.

`sst/` is split by job:

| Module | What's in it |
|---|---|
| `config.py` | Every setting: map, car, planner, replanning and driver constants |
| `streets.py` | Blocks, roads and obstacles, the signed distance field, hidden obstacles and the lidar |
| `car.py` | The car model, its steering limit, and drawing it |
| `tree.py` | States, edges, the witness and proximity maps, node selection and state distances |
| `kinematics.py` | Propagation and clearance checks (compiled with numba), random actions, braking to a stop |
| `global_planner.py` | `GlobalPlanner`, `planGlobal`, the grid heuristic, motion primitives, the replanning demo |
| `route.py` | `Route`, and joining or splicing new states onto one |
| `smoothing.py` | Shortcutting paths, and saving and loading `path_library.npz` |
| `maneuvers.py` | Turnarounds and the maneuver library used to rejoin a route |
| `driver.py` | `Driver`: follows a route and replans around what the lidar finds; the RRT fallback |
| `__main__.py` | The demo window |

The selection and witness radii that `planGlobal` anneals live on `config.radii`, so every module sees the current values.

---

## Part 1: RRT

### The map and the car

The car starts at (20, 140). We originally had it at (100, 100), but that turned out to be inside one
of the obstacles, so nothing could ever grow out of it. RRT treats all 377 circles (blue and green) as
obstacles.

The car's position is the middle of its rear axle. It has a position $(x, y)$, heading $\theta$, speed
$v$ and turning rate $\omega$. We control it with an acceleration $u_a$ and a steering rate $u_\omega$
(how fast the turning rate changes):

$$
\dot x = v\cos\theta, \quad \dot y = v\sin\theta, \quad \dot\theta = \omega, \quad
\dot v = u_a, \quad \dot\omega = u_\omega
$$

Every frame we step these forward by `dt`. Speed is capped at 300 px/s, acceleration at 120 px/s²,
the turning rate at 180°/s and the steering rate at 360°/s².

The important thing is that the car can't change how sharply it's turning instantly, since we only
control how fast the turning rate changes. That ends up mattering a lot when we try to follow RRT's path.

The window is shrunk to 70% (`WINDOW_SCALE`) so it fits on a laptop screen.

### Collision checking

A point hits an obstacle if it's within the obstacle's radius plus a 5 px margin (half the car's width)
of its centre. Going off the edge of the map also counts.

To check a whole line segment we walk along it and check a point every 2 px or so. Only checking the end
of a segment isn't enough, because a 20 px step can jump right over one of the small green obstacles.

Checking every point against all 377 obstacles is slow, so we sort the obstacles by $x$ once at the start.
An obstacle can only matter if its centre's $x$ is within 79 px of the segment (the biggest radius, 74,
plus the 5 px margin), so we use binary search (`bisect`) to grab just the obstacles in that range. That's
around 32 obstacles per check instead of 377, and in our test it was about 14 times faster with exactly the
same results.

### How RRT works

RRT builds a tree from the start. Each round we pick a random point on the map, find the closest node
already in the tree, and take a small step from that node towards the random point. If the step doesn't
hit anything it becomes a new node. Since the random points are spread all over the map, the tree spreads
out quickly in every direction.

We treat the car as a point that can move in any direction, so the tree is made of straight lines. Each
node just remembers its position and its parent (the node it grew from). The tree is a plain Python list,
and the parent links are what actually make it a tree.

```
tree = [start]
repeat up to 5000 times:
    q_rand = random point (or the target 5% of the time)
    q_near = closest node in the tree to q_rand
    q_new  = step 20 px from q_near towards q_rand
    if the line q_near -> q_new is collision free:
        add q_new to the tree with parent q_near
        if q_new is inside the target:
            return the path from start to q_new
return no path found
```

**Goal bias.** 5% of the time we use the target itself instead of a random point. Without it the tree
gets there eventually, but random points almost never land in the target so it takes a while. Set it too
high and the tree gets greedy and keeps running into obstacles between it and the goal.

**Taking a step.** We get the direction with $\phi = \operatorname{atan2}(\Delta y, \Delta x)$ and move
20 px that way:

$$
q_\text{new} = q_\text{near} + 20\,(\cos\phi, \sin\phi)
$$

If $q_\text{rand}$ is already closer than 20 px we just go straight to it. Small steps make the tree grow
gradually instead of drawing huge lines across the map.

**Getting the path.** Once a node lands in the target we follow the parent links back to the start and
flip the list so it goes start to goal.

**How it does.** Over 30 runs with different random seeds it found the goal in 29. A typical run builds
around 500 nodes, ends up with a path of about 128 points, and takes about a tenth of a second. You get a
different tree every time since it's random.

### Making the car follow the path

RRT only gives us a list of points, so we still need to drive the car along them. Every frame the
controller:

1. Picks a point on the path about 25 px ahead of the car to aim at.
2. Works out how far off the car's heading is from that point ($e_\theta$) and turns towards it. The
   bigger the error, the harder it turns ($\omega_\text{want} = 4\,e_\theta$, and the steering rate
   pushes the actual turning rate towards that).
3. Drives at about 80 px/s, slows down for sharp turns, and stops once it's inside the target.

Holding W, A, S or D takes over, and letting go hands control back to the autopilot.

**The problem.** The car makes it to the goal every time (about 40 seconds), but it cuts corners and clips
obstacles by up to about 11 px on sharp turns. RRT's path is a zig-zag of straight lines hugging the
obstacles, and the car can't turn on the spot. Tuning the controller to follow more tightly just made the
car wobble. Giving the path more room from the obstacles didn't work either: at 8 px of margin or more RRT
couldn't find any path, because the gaps are too narrow. This is really a limit of planning for a point and
then driving a car along it, and it's a big part of why we moved on to SST.

---

## Part 2: SST

SST stands for Stable Sparse RRT (Li, Littlefield and Bekris, 2016). Instead of straight lines, every
branch of the tree comes from actually simulating the car with some controls, so any path it finds is one
the car can really drive.

Compared to plain RRT it changes two things:

- **Best-first selection.** When picking which node to grow from, it looks at the nodes near a random
  sample and takes the one with the lowest cost so far, not just the closest one.
- **Keeping the tree sparse.** The space is covered by "witness" points. Each witness only keeps the
  cheapest node near it, and worse nodes get pruned. That keeps the tree small and lets the paths get
  cheaper over time.

### The street map

The SST side uses a different world from RRT:

- **Known map:** a grid of city blocks (rounded rectangles) with 80 px wide roads between them, plus two
  diagonal roads cut through. The car starts at (250, 100) facing right.
- **Hidden obstacles:** 60 random circles and boxes on the roads, plus 2 road segments that are completely
  blocked. The planner doesn't know about these at the start.
- **Lidar:** 20 times a second the car does a 3600-beam scan, and anything it sees gets added to the map.
  Press L to show or hide the hidden obstacles, and R to get a new random set.

The map is stored as an occupancy grid where each pixel is free, occupied or unknown. From it we keep a
distance field (how far each pixel is from the nearest obstacle), which makes collision checks quick: you
just look up a pixel. The car's body is covered by 3 small circles in a row, and a pose is safe if each
circle's centre is at least its radius plus a 2 px safety margin away from anything.

The old circle obstacle lists are still in `sst/streets.py`, but nothing uses them.

### The car

`sst/car.py` has its own car model, closer to a real car than the one in `world.py`. It's a bicycle-style
model where we steer the front wheels:

$$
\dot x = v\cos\theta, \quad \dot y = v\sin\theta, \quad \dot\theta = \frac{v\tan\delta}{L}, \quad
\dot v = u_a, \quad \dot\delta = u_\delta
$$

Here $\delta$ is the front wheel angle and $L = 6$ px is the wheelbase. The controls are acceleration
$u_a$ and how fast the wheel turns, $u_\delta$. The car is small (10 × 5 px). Limits: 300 px/s forwards,
100 px/s in reverse, 120 px/s² acceleration, a 35° wheel angle, the wheel turning at 20°/s at most, and a
lateral acceleration limit of 200 px/s² so it can't take sharp turns at high speed.

### Growing a branch (`monteCarloProp`)

This is SST's Monte Carlo propagation: pick a random duration up to 0.5 s and a random action, hold it,
and see where the car ends up:

$$
x_\text{new} = x_\text{prop} + \int_0^t f(x(\tau), u)\, d\tau
$$

- **Duration** is a whole number of planner steps of 1/80 s.
- **Acceleration** is random within what keeps the speed in its limits.
- **Steering** picks a random target wheel angle (mostly near straight) and turns the wheel at the rate that
  gets there by the end of the branch, capped so every step stays within the lateral acceleration limit.
  So a branch can only fail by hitting something.

Instead of one big formula, the simulation steps through each 1/80 s frame. Within a frame the wheel angle
is fixed, so the car drives along an exact circular arc. It saves "check points" along the way, spaced so
no part of the car moves more than 4 px between them. If every check point keeps the 2 px safety margin,
the whole branch is collision free. The check points are kept so a branch can be rechecked later if the
lidar finds a new obstacle on it. This part is compiled with `numba` because it runs a huge number of times.

(Our first version did the integral by hand into a closed form, but it divided by the turn rate and used a
simpler car model, so Andre replaced it with this.)

**Cost** is mostly driving time, plus a penalty for driving with the wheel turned (10.8 per radian per
second) and a small one for braking. So the planner prefers routes that are quick and fairly straight.

**Distance between states** is the straight-line distance plus weighted differences in heading (a quarter
per degree), speed (a fifth per px/s) and wheel angle. Two states count as close only if they're near each
other, facing the same way and going a similar speed.

### The global planner (`GlobalPlanner`)

This is SST with some extras to make it find paths faster:

- **Heading for the goal.** 75% of iterations don't use a random sample. They grow the node with the
  lowest "cost so far + estimated cost to go". The estimate comes from a shortest-path search over a 10 px
  grid of the known map, divided by the top speed. If the new child is closer to the goal than its parent,
  it gets grown next, up to 15 in a row. A node gets a small penalty each time it's grown, so one that's
  stuck stops getting picked.
- **Normal SST** for the other 25%: random sample, best-first selection, one random action.
- **Sensible first moves.** The first time a node is grown it tries 5 simple actions (speed up straight,
  hold speed, gentle left, gentle right, brake gently) plus 5 random ones, and keeps the best.
- **Shrinking radii (SST\*).** Once there's a path, every 10,000 iterations it checks whether the best
  path got at least 10% cheaper. If not, the selection and witness radii shrink by 20%, so the search gets
  finer.

The global planner treats unknown space as free, since it can't know about the hidden obstacles yet.

### The path library

Planning from scratch takes a while, so `sst` normally starts from saved paths in `path_library.npz`
(`PATH_LIBRARY_MODE = 'use'`). When building the library it plans a path, improves it, then drops random
60 px blocks on the paths found so far and keeps planning on the same tree. That way the library ends up
with several different routes, not just small variations of one. The paths are smoothed at the end and
saved.

### Driving and replanning (`Driver`)

The car drives the cheapest route while the lidar scans. The main ideas:

- **Safe speed.** The car looks at the next 3 tree nodes and only goes as fast as it could still brake in
  time for them. It can always stop before the first bit of route that's blocked or unknown.
- **When a scan shows the route is blocked,** it tries the cheapest fixes first:
  1. **Nudge:** re-drive the route with a small sideways shift around the obstacle.
  2. **Reconnect:** drive a smooth curve to a nearby route that's still clear (this one past the obstacle,
     or another one).
  3. **Replan with SST** from a bit before the blockage, with 500 iterations per frame. Samples start in a
     circle around the car and the circle grows if nothing turns up. A new node can "join" a still-valid
     route if it's close enough in position, heading and wheel angle, and the cheapest join gets spliced in.
- **If it's running out of room** before the replan finds a join, it doesn't just stop. It groups the most
  promising branches (k-means on where they end up), drives the best one from the best group, and keeps
  replanning on the way.
- **If it's actually stuck,** it tries 90° turns into side streets, or backs up a node and replans from
  there. If a replan ends up near a route but facing the wrong way, it tries U-turns and three-point turns
  to get onto it.
- **Improving as it goes.** When nothing is blocked, the planner looks for a cheaper way past the next
  very wiggly bit of route and swaps it in if it finds one. That's the "improved the route, … to go"
  messages.

### Running it

```
python -m sst
```

- The first run takes a bit longer because `numba` compiles its functions. The first time a replan needs
  it, the program also builds `maneuver_library.npz` (about a minute) and saves it for next time.
- After each drive reaches the goal, it replays the drive at full speed, then starts again. So it keeps
  going until you close the window.
- W, A, S, D drive a second car around by hand. It's just for fun, it doesn't scan or plan.
- The terminal prints what the driver is doing ("route blocked ahead, replan", "joined a route", "reached
  the goal", and so on).

A few things to know:

- The window is the full 2000 × 1000 (`SCALE = 1`), so on a laptop it may run off the screen. Lowering
  `SCALE` fixes that.
- Andre's code also has the geometric RRT built in, to run alongside a replan, but it's switched off
  (`RRT_ENABLED = False`) because it didn't seem to help. That's why the log says "RRT not solved after
  0.000 s".
- Every new blockage gets logged to an `obstacle_log/` folder (`OBSTACLE_LOG = True`). It's in
  `.gitignore`.

## Part 3: SST in Rust

`sst_rs/` is the SST **global planner** from `sst/` rewritten in Rust: the street map, the car model,
the tree, `GlobalPlanner` and `smoothPath`, with the same settings. It runs as a zenoh node: send it a
start (and optionally a goal and some extra obstacles) and it sends back a smoothed path.

The Driver (live replanning, the lidar, nudges, the maneuver library) is **not** ported yet. It's about
half the Python code, it runs frame by frame instead of answering one request, and it's much harder to
check, so it's a separate step.

### Same behavior, different numbers

Rust can't draw the same random numbers as Python, so the same seed gives a different path. What
matters is that the planner behaves the same, so we compared both at 30,000 iterations over many seeds
(40 seeds in Python, 300 in Rust):

| | Python | Rust |
|---|---|---|
| Found a path | 85% | 93% |
| Iteration of the first path (median) | 10,055 | 12,288 |
| Cost when found early (first 10,000 iterations, median) | 9.61 | 9.62 |
| Cost after smoothing (median) | 8.22 | 8.25 |
| Driving time after smoothing (median) | 8.03 s | 8.03 s |
| Time for 30,000 iterations | ~2 s | ~0.25 s |

So paths come out as good, and Rust is about 8× faster (100,000 iterations take about 2 s). We also
checked a Rust path with Python's own collision check: its tightest spot is 5.9 px from an obstacle,
against the 2 px safety margin.

One quirk carried over from Python: when smoothing's last shortcut joins at the path's final state, the
path ends where the shortcut ends, which can be a few px short of it, so just outside the goal circle
(we've seen 73 px from the center with a 70 px radius). Python's `spliceRoute` does the same.

### The code

| File | What's in it | Python version |
|---|---|---|
| `config.rs` | The settings the planner uses | `config.py` |
| `map.rs` | Blocks, roads and extra obstacles, and the distance field to them | `streets.py` |
| `kinematics.rs` | Steering limits, random actions, `propagate` with its collision checks | `car.py`, `kinematics.py` |
| `tree.rs` | States, the tree with SST's witness step and pruning, the witness map, selection | `tree.py` |
| `planner.rs` | The grid heuristic, primitives, `Planner` (g + h expansion, annealing) | `global_planner.py` |
| `route.rs` | `Route`, joining and splicing | `route.py` |
| `smoothing.rs` | Hermite shortcuts and `smooth_path` | `smoothing.py` |
| `rng.rs` | The random draws, named like Python's `random` | |
| `lib.rs` | `plan()`, the zenoh messages and the zenoh config | |

Tree nodes live in one list and point to each other by index, instead of Python objects pointing to
each other. The annealed radii are plain fields on the planner, since there's only one planner.

### Running it

In `sst_rs/`:

```
cargo run --release --bin planner                    # the zenoh node; add --verbose to see its progress
cargo run --release --bin request                    # test client: one request, prints the answer
cargo run --release --bin plan -- --seeds 20         # no zenoh: plan with seeds 0..19, one JSON line each
cargo test --release
```

`request` takes `--seed <n>`, `--iterations <n>` and `--no-smooth`. `plan` takes `--seeds`, `--iterations`,
`--no-smooth`, and `--out <file>` to save the last plan's full answer.

If the two nodes can't find each other (multicast is blocked on some networks), start the planner with
`--listen tcp/0.0.0.0:7447` and the client with `--connect tcp/<planner ip>:7447`.

### Talking to it over zenoh

The planner listens on **`cev/planner/sst/request`** and answers on **`cev/planner/sst/path`**, in JSON.
Every field of a request is optional; anything left out uses the `sst` default:

```json
{
  "id": 7,
  "start": [250, 100],
  "start_heading": 0.0,
  "goal": [1950, 700],
  "goal_radius": 70,
  "iterations": 100000,
  "seed": 3,
  "smooth": true,
  "obstacles": [
    {"type": "circle", "x": 980, "y": 400, "radius": 15},
    {"type": "box", "x": 620, "y": 250, "length": 30, "width": 12, "angle": 0.5}
  ]
}
```

- `id` comes back in the answer, so a client can pick out its own. The planner skips repeats of a
  nonzero `id` it has already answered, so a client can resend until it hears back (`request` does).
- `obstacles` are added to the street map for this request only. Shapes are `circle`, `box` (rotated
  rectangle, optional `angle` and `corner`), `block` and `road`, in map pixels.
- Headings are in radians, 0 = facing +x, positive = clockwise on screen (y points down).

The answer:

```json
{
  "id": 7, "found": true, "cost": 7.92, "raw_cost": 8.91, "time": 7.72,
  "path": [{"x": 250.0, "y": 100.0, "heading": 0.0, "speed": 0.0, "steer": 0.0,
            "cost": 0.0, "time": 0.0, "node": true}, "..."],
  "iterations": 100000, "found_at": 15039, "active_nodes": 25361, "plan_ms": 2138.0, "smooth_ms": 227.0
}
```

- `path` runs from the start into the goal region, one state per collision check point (a few px
  apart), with `node` marking the tree nodes. It's empty when `found` is false.
- `cost` is after smoothing, `raw_cost` before; `time` is seconds of driving.
- These are the same fields `path_library.npz` stores per state, so a path from Rust has everything
  the Python Driver needs.

---

## RRT vs. SST

| | RRT (`rrt.py`) | SST (`sst/`) |
|---|---|---|
| Robot | a point that can move any way | the actual car |
| Branches | straight 20 px lines | simulated driving with random controls |
| Picks which node to grow | closest to a random point | cheapest nearby, or lowest cost + estimate |
| Path | jagged, car has to cut corners | drivable as is |
| Gets better over time | no, stops at the first path | yes |
| Knows about obstacles | all of them up front | finds the hidden ones while driving |
