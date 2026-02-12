import argparse
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np


# =========================
# Helpers
# =========================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_angle(a):
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

def segment_intersects_rect(p0, p1, rect):
    xmin, ymin, xmax, ymax = rect

    # quick reject: both points on same outside side
    if (p0[0] < xmin and p1[0] < xmin) or (p0[0] > xmax and p1[0] > xmax) or \
       (p0[1] < ymin and p1[1] < ymin) or (p0[1] > ymax and p1[1] > ymax):
        return False

    # endpoints inside rect => intersect
    if xmin <= p0[0] <= xmax and ymin <= p0[1] <= ymax:
        return True
    if xmin <= p1[0] <= xmax and ymin <= p1[1] <= ymax:
        return True

    edges = [
        ((xmin, ymin), (xmax, ymin)),
        ((xmax, ymin), (xmax, ymax)),
        ((xmax, ymax), (xmin, ymax)),
        ((xmin, ymax), (xmin, ymin)),
    ]
    for a, b in edges:
        if segments_intersect(p0, p1, a, b):
            return True
    return False

def in_fov(robot_xy, robot_theta, obj_xy, fov_range, fov_half_angle_deg):
    dx = obj_xy[0] - robot_xy[0]
    dy = obj_xy[1] - robot_xy[1]
    r = math.hypot(dx, dy)
    if r > fov_range:
        return False
    ang = math.atan2(dy, dx)
    da = wrap_angle(ang - robot_theta)
    return abs(da) <= math.radians(fov_half_angle_deg)

def visible(robot_xy, robot_theta, obj_xy, obstacles, fov_range, fov_half_angle_deg):
    if not in_fov(robot_xy, robot_theta, obj_xy, fov_range, fov_half_angle_deg):
        return False
    for rect in obstacles:
        if segment_intersects_rect(robot_xy, obj_xy, rect):
            return False
    return True


# =========================
# World
# =========================

@dataclass
class Obj:
    oid: int
    xy: np.ndarray  # (2,)
    v: np.ndarray   # (2,)

def step_world(objects: List[Obj], dt: float, world_size: float, accel_std: float, friction: float):
    for o in objects:
        if accel_std > 0.0:
            a = np.random.randn(2).astype(np.float32) * accel_std
            o.v = friction * o.v + a * dt
        else:
            o.v = friction * o.v

        o.xy = o.xy + o.v * dt

        # bounce off walls
        for k in range(2):
            if o.xy[k] < 0:
                o.xy[k] = 0
                o.v[k] *= -1.0
            if o.xy[k] > world_size:
                o.xy[k] = world_size
                o.v[k] *= -1.0


# =========================
# Kalman Track
# =========================

@dataclass
class Track:
    tid: int
    mu: np.ndarray          # (4,) = [px,py,vx,vy]
    Sigma: np.ndarray       # (4,4)
    age: int = 0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False

    def predict(self, F, Q):
        self.mu = F @ self.mu
        self.Sigma = F @ self.Sigma @ F.T + Q
        self.age += 1

    def innovation(self, z, H, R):
        zhat = H @ self.mu
        S = H @ self.Sigma @ H.T + R
        y = z - zhat
        return y, S

    def update(self, z, H, R):
        y, S = self.innovation(z, H, R)
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ y
        I = np.eye(4, dtype=np.float32)
        self.Sigma = (I - K @ H) @ self.Sigma

        self.hits += 1
        self.misses = 0
        if self.hits >= 3:
            self.confirmed = True


# =========================
# --- D56-B CHANGE ---
# Hungarian assignment (minimal, no scipy)
#
# Returns assignment for a square cost matrix:
#   assignment[row] = col
#
# Implementation: classic Hungarian algorithm with potentials.
# This is O(n^3), fine for our tiny n.
# =========================

def hungarian_min_cost(cost: np.ndarray) -> List[int]:
    """
    Solve min-cost assignment for square cost matrix (n x n).
    Returns list 'p' where p[i] = assigned column for row i.
    """
    n = cost.shape[0]
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int32)   # matching for columns: p[j] = i
    way = np.zeros(n + 1, dtype=np.int32)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=np.bool_)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        # augmenting
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # convert column->row matching p[j] to row->column assignment
    assignment = [-1] * n
    for j in range(1, n + 1):
        i = p[j]
        if i != 0:
            assignment[i - 1] = j - 1
    return assignment


# =========================
# Multi-Object Tracker
# =========================

@dataclass
class MOT:
    next_id: int = 1
    tracks: List[Track] = field(default_factory=list)

    def spawn(self, z_xy, init_vel_std=0.8, init_pos_std=1.0):
        mu = np.array([z_xy[0], z_xy[1],
                       np.random.randn() * init_vel_std,
                       np.random.randn() * init_vel_std], dtype=np.float32)
        Sigma = np.diag([init_pos_std**2, init_pos_std**2,
                         (2.0 * init_vel_std)**2, (2.0 * init_vel_std)**2]).astype(np.float32)
        t = Track(tid=self.next_id, mu=mu, Sigma=Sigma)
        self.next_id += 1
        self.tracks.append(t)

    def prune(self, max_misses):
        self.tracks = [t for t in self.tracks if t.misses <= max_misses]

    def step(self,
             Z: List[np.ndarray],
             F, Q, H, R,
             gate_maha: float,
             max_misses: int,
             spawn_unmatched: bool = True,
             use_hungarian: bool = True,
             track_no_match_cost: float = 20.0,
             meas_no_match_cost: float = 20.0):
        """
        D56-B:
          - use_hungarian=True uses a global min-cost assignment with explicit no-match options.
          - gate_maha still applies: beyond gate => treat as essentially impossible match.
        """
        # predict
        for t in self.tracks:
            t.predict(F, Q)

        # if no tracks yet, spawn from measurements
        if len(self.tracks) == 0:
            if spawn_unmatched:
                for z in Z:
                    self.spawn(z)
            return

        T = len(self.tracks)
        M = len(Z)

        # If no measurements, everything coasts
        if M == 0:
            for tr in self.tracks:
                tr.misses += 1
            self.prune(max_misses)
            return

        # Build Mahalanobis cost matrix C (T x M)
        C = np.full((T, M), 1e9, dtype=np.float64)
        for i, tr in enumerate(self.tracks):
            for j, z in enumerate(Z):
                y, S = tr.innovation(z, H, R)
                try:
                    d2 = float(y.T @ np.linalg.inv(S) @ y)
                except np.linalg.LinAlgError:
                    d2 = 1e9
                # apply gating
                if d2 <= gate_maha:
                    C[i, j] = d2
                else:
                    C[i, j] = 1e9

        if not use_hungarian:
            # fallback: greedy (kept for debugging)
            assigned_tracks = set()
            assigned_meas = set()
            flat = [(C[i, j], i, j) for i in range(T) for j in range(M)]
            flat.sort(key=lambda x: x[0])
            for d2, i, j in flat:
                if d2 > gate_maha:
                    break
                if i in assigned_tracks or j in assigned_meas:
                    continue
                assigned_tracks.add(i)
                assigned_meas.add(j)
                self.tracks[i].update(Z[j], H, R)
            for i, tr in enumerate(self.tracks):
                if i not in assigned_tracks:
                    tr.misses += 1
            if spawn_unmatched:
                for j, z in enumerate(Z):
                    if j not in assigned_meas:
                        self.spawn(z)
            self.prune(max_misses)
            return

        # --- D56-B CHANGE ---
        # Global assignment with "no match" options:
        #
        # We build a (T+M) x (T+M) square cost matrix:
        #   [ C          track_unmatched (T x T diag) ]
        #   [ meas_unmatched (M x M diag)      0      ]
        #
        # Interpreting result:
        # - track i assigned to measurement j (j < M) => update
        # - track i assigned to its dummy col (j >= M) => miss++
        # - measurement j assigned to its dummy row => spawn new track
        #
        n = T + M
        BIG = 1e9

        A = np.full((n, n), BIG, dtype=np.float64)

        # top-left: C
        A[:T, :M] = C

        # top-right: track no-match diagonal
        for i in range(T):
            A[i, M + i] = float(track_no_match_cost)

        # bottom-left: measurement no-match diagonal
        for j in range(M):
            A[T + j, j] = float(meas_no_match_cost)

        # bottom-right: zeros (allows dummy-dummy)
        A[T:, M:] = 0.0

        assign = hungarian_min_cost(A)

        # Decode assignments
        track_assigned_meas = [-1] * T
        meas_is_matched = [False] * M

        for i in range(T):
            j = assign[i]
            if j < M and A[i, j] < BIG / 2:
                track_assigned_meas[i] = j
                meas_is_matched[j] = True
            else:
                track_assigned_meas[i] = -1

        # Apply updates/misses
        for i, tr in enumerate(self.tracks):
            j = track_assigned_meas[i]
            if j >= 0:
                tr.update(Z[j], H, R)
            else:
                tr.misses += 1

        # Spawn for unmatched measurements
        if spawn_unmatched:
            for j in range(M):
                if not meas_is_matched[j]:
                    self.spawn(Z[j])

        self.prune(max_misses)


# =========================
# ASCII render (30x20)
# - truth objects: 1,2
# - tracks: a/b or A/B
# - confirmed but very uncertain => force lowercase (shows uncertainty)
# =========================

def render_ascii(world_size, robot_xy, robot_theta, objects, obstacles, mot: MOT,
                 grid_w=30, grid_h=20,
                 unc_hi=6.0):
    def to_cell(xy):
        gx = int(clamp((xy[0] / world_size) * (grid_w - 1), 0, grid_w - 1))
        gy = int(clamp((xy[1] / world_size) * (grid_h - 1), 0, grid_h - 1))
        return gx, gy

    canvas = [["." for _ in range(grid_w)] for _ in range(grid_h)]

    # obstacle
    for (xmin, ymin, xmax, ymax) in obstacles:
        x0, y0 = to_cell((xmin, ymin))
        x1, y1 = to_cell((xmax, ymax))
        for yy in range(min(y0, y1), max(y0, y1) + 1):
            for xx in range(min(x0, x1), max(x0, x1) + 1):
                canvas[grid_h - 1 - yy][xx] = "#"

    # truth objects
    for o in objects:
        x, y = to_cell(o.xy)
        canvas[grid_h - 1 - y][x] = str(o.oid % 10)

    # tracks
    for tr in mot.tracks:
        x, y = to_cell(tr.mu[:2])
        idx = (tr.tid - 1) % 26
        letter = chr(ord("A") + idx)
        if not tr.confirmed:
            letter = letter.lower()

        pos_unc = float(np.trace(tr.Sigma[:2, :2]))
        if pos_unc > unc_hi:
            letter = letter.lower()

        canvas[grid_h - 1 - y][x] = letter

    # robot
    rx, ry = to_cell(robot_xy)
    canvas[grid_h - 1 - ry][rx] = "R"

    # heading marker
    hx = robot_xy[0] + 0.8 * math.cos(robot_theta)
    hy = robot_xy[1] + 0.8 * math.sin(robot_theta)
    hx, hy = to_cell((hx, hy))
    if 0 <= hx < grid_w and 0 <= hy < grid_h:
        if canvas[grid_h - 1 - hy][hx] == ".":
            canvas[grid_h - 1 - hy][hx] = "^"

    return "\n".join("".join(row) for row in canvas)


# =========================
# Main: D56-B run
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--print_every", type=int, default=2)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--world_size", type=float, default=30.0)

    # vision
    ap.add_argument("--fov_range", type=float, default=40.0)
    ap.add_argument("--fov_half_angle", type=float, default=120.0)
    ap.add_argument("--meas_noise_std", type=float, default=0.6)

    # objects
    ap.add_argument("--num_objects", type=int, default=2)
    ap.add_argument("--accel_std", type=float, default=0.0)   # deterministic default
    ap.add_argument("--friction", type=float, default=1.0)

    # tracker model
    ap.add_argument("--process_pos_std", type=float, default=0.10)
    ap.add_argument("--process_vel_std", type=float, default=0.30)
    ap.add_argument("--gate_maha", type=float, default=9.0)
    ap.add_argument("--max_misses", type=int, default=30)

    # --- D56-B CHANGE ---
    ap.add_argument("--track_no_match_cost", type=float, default=12.0,
                    help="Cost for leaving a track unmatched (higher => force matches)")
    ap.add_argument("--meas_no_match_cost", type=float, default=12.0,
                    help="Cost for leaving a measurement unmatched (higher => fewer spawns)")
    ap.add_argument("--use_hungarian", type=int, default=1)

    # rendering
    ap.add_argument("--grid_w", type=int, default=30)
    ap.add_argument("--grid_h", type=int, default=20)

    # obstacle
    ap.add_argument("--obstacle", type=str, default="14,17,24,22")

    # scenario controls
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--force_initial_spawn", type=int, default=1)

    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    oxmin, oymin, oxmax, oymax = [float(x) for x in args.obstacle.split(",")]
    obstacles = [(oxmin, oymin, oxmax, oymax)]

    # Robot fixed for comparability
    robot_xy = np.array([4.0, 4.0], dtype=np.float32)
    robot_theta = 0.70

    # Objects: same deterministic occlusion story as D56-A
    objects: List[Obj] = []
    if args.deterministic:
        objects.append(Obj(
            oid=1,
            xy=np.array([20.0, 24.0], dtype=np.float32),
            v=np.array([-0.2, 0.0], dtype=np.float32),
        ))
        objects.append(Obj(
            oid=2,
            xy=np.array([10.0, 19.5], dtype=np.float32),
            v=np.array([1.2, 0.0], dtype=np.float32),
        ))
    else:
        for i in range(args.num_objects):
            xy = np.array([np.random.uniform(6, args.world_size - 2),
                           np.random.uniform(6, args.world_size - 2)], dtype=np.float32)
            v = np.random.randn(2).astype(np.float32) * 1.0
            objects.append(Obj(oid=i + 1, xy=xy, v=v))

    # Kalman model
    dt = args.dt
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], dtype=np.float32)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=np.float32)

    Q = np.diag([args.process_pos_std**2, args.process_pos_std**2,
                 args.process_vel_std**2, args.process_vel_std**2]).astype(np.float32)
    R = np.diag([args.meas_noise_std**2, args.meas_noise_std**2]).astype(np.float32)

    mot = MOT()

    # D56-A behavior retained: spawn at start so tracks exist immediately
    if args.force_initial_spawn:
        for o in objects:
            z0 = o.xy + np.random.randn(2).astype(np.float32) * args.meas_noise_std
            mot.spawn(z0)

    for t in range(args.steps):
        step_world(objects, dt, args.world_size, args.accel_std, args.friction)

        # measurements
        Z: List[np.ndarray] = []
        vis = 0
        for o in objects:
            if visible(tuple(robot_xy), robot_theta, tuple(o.xy),
                       obstacles, args.fov_range, args.fov_half_angle):
                vis += 1
                z = o.xy + np.random.randn(2).astype(np.float32) * args.meas_noise_std
                Z.append(z)

        mot.step(
            Z=Z,
            F=F, Q=Q,
            H=H, R=R,
            gate_maha=args.gate_maha,
            max_misses=args.max_misses,
            spawn_unmatched=True,
            use_hungarian=bool(args.use_hungarian),
            track_no_match_cost=args.track_no_match_cost,
            meas_no_match_cost=args.meas_no_match_cost,
        )

        if (t % args.print_every == 0) or (t == args.steps - 1):
            confirmed = sum(1 for tr in mot.tracks if tr.confirmed)
            print(f"[D56-B][t={t:03d}] vis_meas={vis}  tracks={len(mot.tracks)}  confirmed={confirmed}")
            print(render_ascii(args.world_size, robot_xy, robot_theta,
                               objects, obstacles, mot,
                               grid_w=args.grid_w, grid_h=args.grid_h,
                               unc_hi=6.0))
            print()

            for tr in mot.tracks:
                pos_unc = float(np.trace(tr.Sigma[:2, :2]))
                idx = (tr.tid - 1) % 26
                name = chr(ord("A") + idx)
                if not tr.confirmed:
                    name = name.lower()
                mode = "UPDATE" if tr.misses == 0 and vis > 0 else "COAST"
                unc_level = "LOW" if pos_unc < 3.0 else ("MID" if pos_unc < 6.0 else "HIGH")
                print(f"  track {name} (tid={tr.tid:02d}) conf={int(tr.confirmed)} "
                      f"mode={mode:6s} miss={tr.misses:02d} "
                      f"mu=({tr.mu[0]:5.2f},{tr.mu[1]:5.2f}) pos_unc={pos_unc:6.2f} unc={unc_level}")
            print("-" * 60)


if __name__ == "__main__":
    main()
