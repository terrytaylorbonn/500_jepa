#54_d56A_multi_object_belief_observable.py
import argparse
import math
import random
from dataclasses import dataclass, field
from typing import List

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
            o.v = friction * o.v  # constant-ish velocity if friction=1.0

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

    def step(self, Z: List[np.ndarray], F, Q, H, R, gate_maha: float,
             max_misses: int, spawn_unmatched: bool = True):
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

        # cost matrix: Mahalanobis distance
        cost = np.full((T, M), 1e9, dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            for j, z in enumerate(Z):
                y, S = tr.innovation(z, H, R)
                try:
                    d2 = float(y.T @ np.linalg.inv(S) @ y)
                except np.linalg.LinAlgError:
                    d2 = 1e9
                cost[i, j] = d2

        # greedy assignment (minimal)
        assigned_tracks = set()
        assigned_meas = set()
        flat = [(cost[i, j], i, j) for i in range(T) for j in range(M)]
        flat.sort(key=lambda x: x[0])

        for d2, i, j in flat:
            if d2 > gate_maha:
                break
            if i in assigned_tracks or j in assigned_meas:
                continue
            assigned_tracks.add(i)
            assigned_meas.add(j)
            self.tracks[i].update(Z[j], H, R)

        # misses
        for i, tr in enumerate(self.tracks):
            if i not in assigned_tracks:
                tr.misses += 1

        # spawn for unmatched measurements
        if spawn_unmatched:
            for j, z in enumerate(Z):
                if j not in assigned_meas:
                    self.spawn(z)

        self.prune(max_misses)


# =========================
# ASCII render (30x20)
# - truth objects: 1,2
# - tracks: a/b or A/B
#   confirmed but very uncertain => force lowercase (so uncertainty is visible)
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

        # if position uncertainty is high, force lowercase to show "uncertain"
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
# Main: D56-A "observable" run
# =========================

def main():
    ap = argparse.ArgumentParser()

    # keep it small/readable
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
    ap.add_argument("--accel_std", type=float, default=0.0)   # deterministic by default
    ap.add_argument("--friction", type=float, default=1.0)

    # tracker
    ap.add_argument("--process_pos_std", type=float, default=0.10)
    ap.add_argument("--process_vel_std", type=float, default=0.30)
    ap.add_argument("--gate_maha", type=float, default=9.0)
    ap.add_argument("--max_misses", type=int, default=30)

    # rendering
    ap.add_argument("--grid_w", type=int, default=30)
    ap.add_argument("--grid_h", type=int, default=20)

    # obstacle
    ap.add_argument("--obstacle", type=str, default="14,17,24,22")

    # D56-A behavior controls
    ap.add_argument("--deterministic", type=int, default=1)   # 1 = fixed scenario
    ap.add_argument("--force_initial_spawn", type=int, default=1)

    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    oxmin, oymin, oxmax, oymax = [float(x) for x in args.obstacle.split(",")]
    obstacles = [(oxmin, oymin, oxmax, oymax)]

    # Robot: keep it fixed in D56-A so visibility changes come from object motion/occlusion
    if args.deterministic:
        robot_xy = np.array([4.0, 4.0], dtype=np.float32)
        robot_theta = 0.70
    else:
        robot_xy = np.array([4.0, 4.0], dtype=np.float32)
        robot_theta = 0.60

    # Objects: deterministic occlusion story
    objects: List[Obj] = []
    if args.deterministic:
        # Object 1 stays visible (upper-right-ish), slow drift
        objects.append(Obj(
            oid=1,
            xy=np.array([20.0, 24.0], dtype=np.float32),
            v=np.array([ -0.2,  0.0], dtype=np.float32),
        ))
        # Object 2 passes behind the obstacle from left->right (occlusion event)
        objects.append(Obj(
            oid=2,
            xy=np.array([10.0, 19.5], dtype=np.float32),
            v=np.array([  1.2,  0.0], dtype=np.float32),
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

    # D56-A: force initial spawn so you always see tracks immediately
    if args.force_initial_spawn:
        for o in objects:
            z0 = o.xy + np.random.randn(2).astype(np.float32) * args.meas_noise_std
            mot.spawn(z0)

    for t in range(args.steps):
        # (optional) small robot wobble if not deterministic
        if not args.deterministic:
            robot_theta = wrap_angle(robot_theta + 0.02)

        # step world
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

        # tracker update
        prev_misses = [tr.misses for tr in mot.tracks]
        mot.step(Z, F, Q, H, R,
                 gate_maha=args.gate_maha,
                 max_misses=args.max_misses,
                 spawn_unmatched=True)

        # print
        if (t % args.print_every == 0) or (t == args.steps - 1):
            confirmed = sum(1 for tr in mot.tracks if tr.confirmed)
            print(f"[D56-A][t={t:03d}] vis_meas={vis}  tracks={len(mot.tracks)}  confirmed={confirmed}")
            print(render_ascii(args.world_size, robot_xy, robot_theta,
                               objects, obstacles, mot,
                               grid_w=args.grid_w, grid_h=args.grid_h,
                               unc_hi=6.0))
            print()

            # per-track debug: mode + uncertainty
            for tr in mot.tracks:
                pos_unc = float(np.trace(tr.Sigma[:2, :2]))
                idx = (tr.tid - 1) % 26
                name = chr(ord("A") + idx)
                if not tr.confirmed:
                    name = name.lower()

                # Was this track updated this step? (misses reset to 0 on update)
                mode = "UPDATE" if tr.misses == 0 and vis > 0 else "COAST"
                unc_level = "LOW" if pos_unc < 3.0 else ("MID" if pos_unc < 6.0 else "HIGH")

                print(f"  track {name} (tid={tr.tid:02d}) conf={int(tr.confirmed)} "
                      f"mode={mode:6s} miss={tr.misses:02d} "
                      f"mu=({tr.mu[0]:5.2f},{tr.mu[1]:5.2f}) pos_unc={pos_unc:6.2f} unc={unc_level}")
            print("-" * 60)

if __name__ == "__main__":
    main()


