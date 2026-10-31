"""SST tree pieces: states, edges, the witness and proximity maps, node selection and state distances."""
import math
import random
from enum import Enum

import numpy as np

from .car import maxSteerAtSpeed
from .config import (
    DISTANCE_TIME_PROP, FREE, MAX_REVERSE_SPEED, MAX_SPEED, OCCUPIED, radii, SCREEN_HEIGHT, SCREEN_WIDTH, TARGET,
    TARGET_RAD, W_A, W_ST, W_V,
)


class NodeStatus(Enum):
    deadEnd = 0  # doesn't lead to the target
    solution = 1  # leads to the target, but isn't on the path being pursued
    following = 2  # on the path being pursued
    orphaned = 3  # cut off from the tree by a new obstacle; ignored by planning


class state:
    
    def __init__(self, x1, x2, a, s, cost = 0, steer = 0, status = NodeStatus.deadEnd, safeVelocity = None, time = 0):
        self.x1 = x1
        self.x2 = x2 
        self.a = a 
        self.s = s
        self.x1bar = self.x1 +self.s * math.cos(self.a)*DISTANCE_TIME_PROP
        self.x2bar = self.x2 + self.s * math.sin(self.a)*DISTANCE_TIME_PROP 
        self.cost = cost #we need to define cost
        self.steer = steer
        self.children =  []
        self.parent = None
        self.status = status
        self.safeVelocity = safeVelocity  # px/s; None until computed
        self.traj = []  # collisionCheckPoints of the edge from parent to this node, set when the tree keeps it
        self.time = time  # s of driving from the root to here (cost can include more than time)
        self.expansions = 0  # times planGlobal's g + h selection has expanded this node
        self.turns_tried = False  # whether turn primitives have been tried from this node (GlobalPlanner.use_turns)
        self.witness = None  # the Witness this node was last made the rep of

    def distance(self, s2):
        return stateDistance(self, s2)  # was calcReedsSheppDist(self, s2)
    
    def subtree(self):
        """This vertex and everything below it (iterative, so long branches don't hit the recursion limit)."""
        stack, branch = [self], []
        while stack:
            v = stack.pop()
            branch.append(v)
            stack.extend(v.children)
        return branch

    def removeUntilActive(self, tree):
        """Prune from this vertex upward: if nothing in its branch (itself included) is active, cut the whole
        branch off its parent, then check the parent the same way, stopping at the first vertex still needed.
        Returns the removed vertices so the caller can drop them from any lookup structures (e.g. the hash map)."""
        removed = []
        node = self
        while node is not None:
            branch = node.subtree()
            if any(v in tree.active for v in branch):
                break
            for v in branch:
                tree.inactive.discard(v)
            removed.extend(branch)
            parent = node.parent
            if parent is not None:
                parent.children.remove(node)
            node.parent = None
            node = parent
        return removed
    
    
class collisionCheckPoints(state):
    """A state along an edge, kept so the edge can be rechecked when new obstacles show up.
    `node` is the tree node the edge was propagated from."""

    def __init__(self, x1, x2, a, s, node, cost=0, steer=0, time=0):
        super().__init__(x1, x2, a, s, cost=cost, steer=steer, time=time)
        self.node = node
        self.edge_end = None  # the tree node this point's edge leads to, set when the tree keeps the edge


class action:
    def __init__(self, da, ds):
        self.da = da  # steering rate, rad/s
        self.ds = ds  # acceleration, px/s^2

class edge:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2


class SSTTree:
    def __init__(self, s1):
        self.active = {s1}

        self.inactive = set()
        self.goal_nodes = []  # kept nodes inside the goal region (TARGET, TARGET_RAD)
    
        self.E = np.array([])
        
    def activate(self, vertex):
        self.inactive.discard(vertex)
        self.active.add(vertex)
    def unActivate(self,vertex):
        self.active.discard(vertex)
        self.inactive.add(vertex)

    def add(self, path, parent, witness_map, traj_map):
        """SST's witness step for a freshly propagated edge.

        `path` is what monteCarloProp returned (its last state is the new node), `parent` the node it was propagated
        from, `witness_map` a WitnessHashMap of every Witness, `traj_map` a proximityHashMap of the check points of
        every edge in the tree.

        If no witness is within delta_S of the new node, the node becomes a new witness and its rep. Otherwise the
        closest witness decides: if the new node is cheaper than its rep, the node takes over as rep, and the old rep
        is made inactive and pruned with removeUntilActive (the check points of everything pruned leave traj_map).
        If the new node isn't cheaper, it's dropped. A kept node is attached to `parent`, made active, and its check
        points go into traj_map. Returns the new node if kept, else None.
        """
        node = path[-1]
        # distanceLowerBound is cheap and never more than the real distance. Skip witnesses whose bound is already past
        # delta_S, then try the rest in order of that bound and stop once it can't beat the closest found so far; most
        # witnesses never need the full distance.
        candidates = []
        for w in witness_map.get_nbhd(node, radii.delta_S):
            bound = distanceLowerBound(w.state, node)
            if bound <= radii.delta_S:
                candidates.append((bound, w))
        candidates.sort(key=lambda c: c[0])
        closest, closest_dist = None, math.inf
        for bound, w in candidates:
            if bound >= closest_dist:
                break
            d = w.state.distance(node)
            if d <= radii.delta_S and d < closest_dist:
                closest, closest_dist = w, d
        if closest is None:
            closest = Witness(node)
            witness_map.add(closest)
        elif closest.rep is not None and node.cost >= closest.rep.cost:
            return None
        old_rep = None if closest.rep is node else closest.rep

        # Attach before pruning, so the old rep's branch still counts the new node if it hangs below it
        node.parent = parent
        parent.children.append(node)
        node.traj = path
        for p in path:
            p.edge_end = node
            traj_map.add(p)
        self.activate(node)
        witness_map.setRep(closest, node)

        # In the goal region: the node and everything above it now lead to the target
        if math.hypot(node.x1 - TARGET[0], node.x2 - TARGET[1]) <= TARGET_RAD:
            self.goal_nodes.append(node)
            v = node
            while v is not None:
                v.status = NodeStatus.solution
                v = v.parent

        if old_rep is not None:
            self.unActivate(old_rep)
            ancestors = []  # old_rep and everything above it; pruning cuts this chain somewhere
            v = old_rep
            while v is not None:
                ancestors.append(v)
                v = v.parent
            removed = old_rep.removeUntilActive(self)
            lost_goal = False
            for v in removed:
                for p in v.traj:
                    traj_map.remove(p)
                if v.status is NodeStatus.solution and v in self.goal_nodes:
                    self.goal_nodes.remove(v)
                    lost_goal = True
            if lost_goal:
                # Walk up from where the pruned branch was cut off: ancestors with no goal node left below them no
                # longer lead to the target
                removed_ids = {id(v) for v in removed}
                goal_ids = {id(g) for g in self.goal_nodes}
                for v in ancestors:
                    if id(v) in removed_ids:
                        continue
                    if any(id(u) in goal_ids for u in v.subtree()):
                        break
                    v.status = NodeStatus.deadEnd
        return node


class proximityHashMap:
    def __init__(self, delta):
        self.rad = delta
        self.map = {}
    def add(self, s):
        x_coord = math.floor(s.x1/self.rad)
        y_coord = math.floor(s.x2/self.rad)
        if x_coord not in self.map:
            self.map[x_coord] = {}
        if y_coord in self.map[x_coord]:
            self.map[x_coord][y_coord].append(s)

        else:
            self.map[x_coord][y_coord] = [s]
            
    def remove(self, s):
        cell = self.map.get(math.floor(s.x1/self.rad), {}).get(math.floor(s.x2/self.rad))
        if cell is not None and s in cell:
            cell.remove(s)

    def get(self, x_coord, y_coord):
        if x_coord in self.map and y_coord in self.map[x_coord]:
            return self.map[x_coord][y_coord]
        return []
            
    def get_nbhd(self, s, delta):
        x_lo = math.floor((s.x1 - delta)/self.rad)
        x_hi = math.floor((s.x1 + delta)/self.rad)
        y_lo = math.floor((s.x2 - delta)/self.rad)
        y_hi = math.floor((s.x2 + delta)/self.rad)
        result = []
        for i in range(x_lo, x_hi + 1):
            for j in range(y_lo, y_hi + 1):
                for other in self.get(i, j):
                    if math.hypot(other.x1 - s.x1, other.x2 - s.x2) <= delta:
                        result.append(other)
                
        return result 




    
    
            
        

class Witness:
    """An SST witness: a fixed point in state space and the node currently representing it.

    `state` is the witness's center and never moves. `rep` is the cheapest node found within delta_S of it, measured
    as distance from `state` to the node. The state a witness is created with is also its first rep.
    """

    def __init__(self, s):
        # The center is a detached copy (no parent, children or tree status), so pruning the original node, e.g.
        # once a new obstacle covers it, leaves the witness intact and doesn't keep the pruned branch alive.
        self.state = state(s.x1, s.x2, s.a, s.s, cost=s.cost, steer=s.steer)
        self.rep = s

    def setRep(self, node):
        self.rep = node


class WitnessHashMap:
    """Witnesses in a position-only hash, plus a numpy row per witness holding its rep, for whole-map searches.

    get_nbhd(s, delta) visits the position cells within distanceBounds(delta)'s position bound and returns the witnesses
    inside all of its bounds (position, speed and heading are cheap to check); the caller still checks the exact
    distance. Cells are as wide as the position bound for the `delta` the map is built for. Searching a smaller radius is
    still correct, so rebuild with the smaller delta once it has shrunk a lot.

    repDistances(s) gives the distance from every witness's rep to s in one numpy pass, for when the hash finds nothing
    nearby. Witnesses are never removed, so rows only grow; change a rep through setRep so its row stays in sync.
    """

    def __init__(self, delta):
        self.cell = distanceBounds(delta)[0]
        self.cells = {}
        self.witnesses = []
        self.reps = np.empty((64, 5))  # per witness: x, y, heading, speed, steering of its rep

    def key(self, s):
        return math.floor(s.x1 / self.cell), math.floor(s.x2 / self.cell)

    def add(self, w):
        if w.rep is not None:
            w.rep.witness = w
        w.index = len(self.witnesses)
        self.witnesses.append(w)
        self.cells.setdefault(self.key(w.state), []).append(w)
        if w.index == len(self.reps):
            self.reps = np.concatenate((self.reps, np.empty_like(self.reps)))
        self.writeRep(w)

    def writeRep(self, w):
        r = w.rep
        # A witness with no rep gets a row infinitely far away, so repDistances never picks it
        self.reps[w.index] = (math.inf, math.inf, 0, 0, 0) if r is None else (r.x1, r.x2, r.a, r.s, r.steer)

    def setRep(self, w, node):
        """Make `node` (or None) w's rep and keep its numpy row and the node's back-link in sync."""
        w.setRep(node)
        if node is not None:
            node.witness = w
        self.writeRep(w)

    def get_nbhd(self, s, delta):
        """Witnesses that could be within distance delta of state s."""
        r_pos, r_speed, r_heading = distanceBounds(delta)
        result = []
        for i in range(math.floor((s.x1 - r_pos) / self.cell), math.floor((s.x1 + r_pos) / self.cell) + 1):
            for j in range(math.floor((s.x2 - r_pos) / self.cell), math.floor((s.x2 + r_pos) / self.cell) + 1):
                for w in self.cells.get((i, j), ()):
                    o = w.state
                    if (math.hypot(o.x1 - s.x1, o.x2 - s.x2) <= r_pos and abs(o.s - s.s) <= r_speed
                            and abs((o.a - s.a + math.pi) % (2 * math.pi) - math.pi) <= r_heading):
                        result.append(w)
        return result

    def repDistances(self, s):
        """stateDistance from every witness's rep to s, as a numpy array in witness order. Mirrors stateDistance, so
        change the two together."""
        r = self.reps[:len(self.witnesses)]
        dh = np.abs((r[:, 2] - s.a + np.pi) % (2 * np.pi) - np.pi)
        return (np.hypot(r[:, 0] - s.x1, r[:, 1] - s.x2) + W_A * dh + W_V * np.abs(r[:, 3] - s.s)
                + W_ST * np.abs(r[:, 4] - s.steer))

    def __iter__(self):
        return iter(self.witnesses)

    def __len__(self):
        return len(self.witnesses)


def makeSampleBiasingGrid():
    """An occupancy grid that steers sampling: getRandomState never samples a cell marked OCCUPIED. Starts all FREE."""
    return np.full((SCREEN_WIDTH, SCREEN_HEIGHT), FREE, dtype=np.int8)


def getRandomState(sdf_state=None, sample_bias=None):
    """Uniform over the map, heading and the full speed range (reverse included), with a steering angle the
    lateral acceleration limit allows at that speed.

    Positions that are OCCUPIED in sdf_state's occupancy grid or in the sample_bias grid are redrawn (unknown space can
    still be sampled). Gives up after 1000 tries and keeps the last position, so a fully blocked grid can't hang it.
    """
    for _ in range(1000):
        x_1_rand = random.uniform(0, SCREEN_WIDTH)
        x_2_rand = random.uniform(0, SCREEN_HEIGHT)
        px, py = min(int(x_1_rand), SCREEN_WIDTH - 1), min(int(x_2_rand), SCREEN_HEIGHT - 1)
        if sdf_state is not None and sdf_state.occupancy[px, py] == OCCUPIED:
            continue
        if sample_bias is not None and sample_bias[px, py] == OCCUPIED:
            continue
        break
    a_rand = random.uniform(0, 2 * math.pi)
    speed_rand = random.uniform(-MAX_REVERSE_SPEED, MAX_SPEED)
    steer_limit = maxSteerAtSpeed(abs(speed_rand))
    steer_rand = random.uniform(-steer_limit, steer_limit)

    return state(x_1_rand, x_2_rand, a_rand, speed_rand, steer=steer_rand)


def bestNear(witness_map, s):
    """The cheapest active node (witness rep) within distance delta_B_N of s, measured from the node to s, or
    None if there is none.

    A rep sits within delta_S of its witness, so every rep within delta_B_N of s belongs to a witness the map finds
    within delta_B_N + delta_S. Reps are tried cheapest first, skipping any whose cheap lower bound is already past
    delta_B_N, so the first one actually in range is the answer.
    """
    reps = [w.rep for w in witness_map.get_nbhd(s, radii.delta_B_N + radii.delta_S) if w.rep is not None]
    reps.sort(key=lambda node: node.cost)
    for node in reps:
        if distanceLowerBound(node, s) <= radii.delta_B_N and node.distance(s) <= radii.delta_B_N:
            return node
    return None


def nearestActive(witness_map, s):
    """The active node (witness rep) closest to s, from one numpy pass over every rep."""
    return witness_map.witnesses[int(np.argmin(witness_map.repDistances(s)))].rep  # None only if no witness has a rep


def bestFirstSelectionSST(witness_map, sdf_state=None, sample_bias=None, x_rand=None):
    """SST's best-first selection: sample a random state (see getRandomState for what it skips), unless x_rand is given,
    and pick the cheapest active node within delta_B_N of it, or the nearest active node if none is that close."""
    if x_rand is None:
        x_rand = getRandomState(sdf_state, sample_bias)
    return bestNear(witness_map, x_rand) or nearestActive(witness_map, x_rand)


def stateDistance(s0, s1):
    """Distance between states, standing in for the distance for now (that code is commented out below):
    euclidean px + W_A * heading difference + W_V * speed difference + W_ST * steering difference."""
    dh = abs((s1.a - s0.a + math.pi) % (2 * math.pi) - math.pi)
    return (math.hypot(s1.x1 - s0.x1, s1.x2 - s0.x2) + W_A * dh + W_V * abs(s1.s - s0.s)
            + W_ST * abs(s1.steer - s0.steer))


def distanceLowerBound(s0, s1):
    """A cheap value never more than state.distance, for skipping candidates. stateDistance is already cheap, so this is
    the distance itself (for Reeds-Shepp it was reedsSheppLowerBound)."""
    return stateDistance(s0, s1)


def distanceBounds(delta):
    """(position px, speed px/s, heading rad) that two states within distance delta can differ by at most; the
    WitnessHashMap bins by these. Steering isn't binned. For distance these were
    (MAX_SPEED * delta, MAX_ACCELERATION * delta, MAX_SPEED * delta / (RS_WHEELBASE / tan(MAX_STEER)))."""
    return delta, delta / W_V, delta / W_A
