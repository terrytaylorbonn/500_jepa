#58_d57_micro_identity_stability.py

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


# =========================
# World (micro)
# =========================

@dataclass
class Obj:
    oid: int
    xy: np.ndarray  # (2,)
    v: np.ndarray   # (2,)
    feat: float     # --- D57 CHANGE --- "appearance feature" per object (e.g., -1 vs +1)


def step_world(objects: List[Obj], dt: float, world_size: float):
    for o in objects:
        o.xy = o.xy + o.v * dt
        # bounce in [0, world_size]
        for k in range(2):
            if o.xy[k] < 0:
                o.xy[k] = 0
                o.v[k] *= -1.0
            if o.xy[k] > world_size:
                o.xy[k] = world_size
                o.v[k] *= -1.0


# =========================
# Measurement (pos + feature)
# =========================

@dataclass
class Meas:
    z_xy: np.ndarray  # (2,)
    z_f: float        # --- D57 CHANGE --- measured appearance feature


# =========================
# Kalman Track (pos/vel) + feature belief
# =========================

@dataclass
class Track:
    tid: int
    mu: np.ndarray        # (4,) [px,py,vx,vy]
    Sigma: np.ndarray     # (4,4)
    # --- D57 CHANGE --- appearance feature state (simple scalar Gaussian)
    f_mu: float
    f_var: float

    age: int = 0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False

    def predict(self, F, Q, f_process_var: float):
        self.mu = F @ self.mu
        self.Sigma = F @ self.Sigma @ F.T + Q
        # --- D57 CHANGE ---
        self.f_var = self.f_var + f_process_var
        self.age += 1

    def innovation_pos(self, z_xy, H, R):
        zhat = H @ self.mu
        S = H @ self.Sigma @ H.T + R
        y = z_xy - zhat
        return y, S

    def update(self, meas: Meas, H, R, f_meas_var: float):
        # pos update (KF)
        y, S = self.innovation_pos(meas.z_xy, H, R)
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ y
        I = np.eye(4, dtype=np.float32)
        self.Sigma = (I - K @ H) @ self.Sigma

        # --- D57 CHANGE --- feature update (scalar KF)
        # prior: f ~ N(f_mu, f_var), meas: z_f ~ N(f, f_meas_var)
        # posterior:
        #   Kf = f_var / (f_var + f_meas_var)
        #   f_mu <- f_mu + Kf*(z_f - f_mu)
        #   f_var <- (1-Kf)*f_var
        Kf = self.f_var / (self.f_var + f_meas_var)
        self.f_mu = self.f_mu + Kf * (meas.z_f - self.f_mu)
        self.f_var = (1.0 - Kf) * self.f_var

        self.hits += 1
        self.misses = 0
        if self.hits >= 3:
            self.confirmed = True


# =========================
# Hungarian (min-cost assignment)
# =========================

def hungarian_min_cost(cost: np.ndarray) -> List[int]:
    n = cost.shape[0]
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int32)
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

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

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

    def spawn(self, meas: Meas, init_vel_std=0.6, init_pos_std=0.8,
              init_f_var=1.0):
        mu = np.array([meas.z_xy[0], meas.z_xy[1],
                       np.random.randn() * init_vel_std,
                       np.random.randn() * init_vel_std], dtype=np.float32)
        Sigma = np.diag([
            init_pos_std**2, init_pos_std**2,
            (2.0 * init_vel_std)**2, (2.0 * init_vel_std)**2
        ]).astype(np.float32)

        # --- D57 CHANGE --- initialize feature belief from measurement
        t = Track(
            tid=self.next_id,
            mu=mu, Sigma=Sigma,
            f_mu=float(meas.z_f),
            f_var=float(init_f_var),
        )
        self.next_id += 1
        self.tracks.append(t)

    def prune(self, max_misses):
        self.tracks = [t for t in self.tracks if t.misses <= max_misses]

    def step(self,
             meas_list: List[Meas],
             F, Q, H, R,
             gate_maha: float,
             max_misses: int,
             track_no_match_cost: float,
             meas_no_match_cost: float,
             # --- D57 CHANGE ---
             use_feature_cost: bool,
             feat_weight: float,
             f_meas_var: float,
             f_process_var: float):

        for tr in self.tracks:
            tr.predict(F, Q, f_process_var=f_process_var)

        T = len(self.tracks)
        M = len(meas_list)

        if T == 0:
            for m in meas_list:
                self.spawn(m)
            return

        if M == 0:
            for tr in self.tracks:
                tr.misses += 1
            self.prune(max_misses)
            return

        # cost matrix C (T x M): position maha + optional feature term
        BIG = 1e9
        C = np.full((T, M), BIG, dtype=np.float64)

        for i, tr in enumerate(self.tracks):
            for j, m in enumerate(meas_list):
                y, S = tr.innovation_pos(m.z_xy, H, R)
                try:
                    d2_pos = float(y.T @ np.linalg.inv(S) @ y)
                except np.linalg.LinAlgError:
                    d2_pos = BIG

                if d2_pos > gate_maha:
                    continue

                d2 = d2_pos

                # --- D57 CHANGE --- identity stability via appearance feature
                if use_feature_cost:
                    # treat feature as 1D gaussian: (z_f - f_mu)^2 / (f_var + f_meas_var)
                    denom = (tr.f_var + f_meas_var)
                    d2_feat = ((m.z_f - tr.f_mu) ** 2) / max(1e-6, denom)
                    d2 = d2_pos + feat_weight * d2_feat

                C[i, j] = d2

        # augmented assignment with no-match rows/cols
        n = T + M
        A = np.full((n, n), BIG, dtype=np.float64)
        A[:T, :M] = C

        # track i can choose "no measurement"
        for i in range(T):
            A[i, M + i] = float(track_no_match_cost)

        # measurement j can choose "spawn"
        for j in range(M):
            A[T + j, j] = float(meas_no_match_cost)

        A[T:, M:] = 0.0

        assign = hungarian_min_cost(A)

        track_assigned = [-1] * T
        meas_matched = [False] * M

        for i in range(T):
            j = assign[i]
            if j < M and A[i, j] < BIG / 2:
                track_assigned[i] = j
                meas_matched[j] = True

        # apply
        for i, tr in enumerate(self.tracks):
            j = track_assigned[i]
            if j >= 0:
                tr.update(meas_list[j], H, R, f_meas_var=f_meas_var)
            else:
                tr.misses += 1

        # spawn unmatched meas
        for j in range(M):
            if not meas_matched[j]:
                self.spawn(meas_list[j])

        self.prune(max_misses)


# =========================
# Visualization (compact, minimal)
#   '+' halo = UPDATE
#   ',' halo = COAST
# =========================

def render_ascii(world_size: float,
                 objects: List[Obj],
                 mot: MOT,
                 grid_w: int,
                 grid_h: int,
                 unc_low: float,
                 unc_mid: float) -> str:
    def to_cell(xy):
        gx = int(clamp((xy[0] / world_size) * (grid_w - 1), 0, grid_w - 1))
        gy = int(clamp((xy[1] / world_size) * (grid_h - 1), 0, grid_h - 1))
        return gx, gy

    canvas = [["." for _ in range(grid_w)] for _ in range(grid_h)]

    def set_if_empty(cx, cy, ch):
        if 0 <= cx < grid_w and 0 <= cy < grid_h:
            rr = grid_h - 1 - cy
            if canvas[rr][cx] == ".":
                canvas[rr][cx] = ch

    def set_force(cx, cy, ch):
        if 0 <= cx < grid_w and 0 <= cy < grid_h:
            canvas[grid_h - 1 - cy][cx] = ch

    # truth objects (always drawn)
    for o in objects:
        x, y = to_cell(o.xy)
        set_force(x, y, str(o.oid % 10))

    # halos (draw first)
    for tr in mot.tracks:
        tx, ty = to_cell(tr.mu[:2])
        pos_unc = float(np.trace(tr.Sigma[:2, :2]))
        if pos_unc < unc_low:
            unc_level = "LOW"
        elif pos_unc < unc_mid:
            unc_level = "MID"
        else:
            unc_level = "HIGH"

        mode = "UPDATE" if tr.misses == 0 else "COAST"
        halo_char = "+" if mode == "UPDATE" else ","

        if unc_level == "LOW":
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif unc_level == "MID":
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        for dx, dy in offsets:
            set_if_empty(tx + dx, ty + dy, halo_char)

    # track centers
    for tr in mot.tracks:
        tx, ty = to_cell(tr.mu[:2])
        idx = (tr.tid - 1) % 26
        letter = chr(ord("A") + idx)
        if not tr.confirmed:
            letter = letter.lower()

        pos_unc = float(np.trace(tr.Sigma[:2, :2]))
        if pos_unc >= unc_mid:
            letter = letter.lower()

        set_force(tx, ty, letter)

    return "\n".join("".join(row) for row in canvas)


# =========================
# Micro scenario: two-object crossing
# =========================

def make_crossing_micro(world_size: float) -> List[Obj]:
    # Pre-positioned to "get interesting" quickly.
    # They cross near the center within ~10-20 steps.
    return [
        Obj(oid=1,
            xy=np.array([world_size * 0.35, world_size * 0.55], dtype=np.float32),
            v=np.array([+0.90, -0.10], dtype=np.float32),
            feat=-1.0),   # --- D57 CHANGE ---
        Obj(oid=2,
            xy=np.array([world_size * 0.65, world_size * 0.45], dtype=np.float32),
            v=np.array([-0.90, +0.10], dtype=np.float32),
            feat=+1.0),   # --- D57 CHANGE ---
    ]


def main():
    ap = argparse.ArgumentParser()

    # --- MICRO DEMO defaults ---
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--world_size", type=float, default=16.0)

    # compact output: we only print a window around closest approach
    ap.add_argument("--window_before", type=int, default=5)
    ap.add_argument("--window_after", type=int, default=7)

    # measurement noise
    ap.add_argument("--meas_noise_std", type=float, default=0.55)

    # --- D57 CHANGE --- feature noise + weight
    ap.add_argument("--use_feature_cost", type=int, default=1)        # 1 = D57 on, 0 = D56-like
    ap.add_argument("--feat_noise_std", type=float, default=0.45)
    ap.add_argument("--feat_weight", type=float, default=3.0)
    ap.add_argument("--f_process_var", type=float, default=0.02)      # drift in feature belief

    # tracker process / gate
    ap.add_argument("--process_pos_std", type=float, default=0.08)
    ap.add_argument("--process_vel_std", type=float, default=0.25)
    ap.add_argument("--gate_maha", type=float, default=9.0)
    ap.add_argument("--max_misses", type=int, default=50)

    # association knobs
    ap.add_argument("--track_no_match_cost", type=float, default=12.0)
    ap.add_argument("--meas_no_match_cost", type=float, default=12.0)

    # render
    ap.add_argument("--grid_w", type=int, default=22)
    ap.add_argument("--grid_h", type=int, default=12)
    ap.add_argument("--unc_low", type=float, default=2.2)
    ap.add_argument("--unc_mid", type=float, default=5.0)

    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # objects + tracker
    objects = make_crossing_micro(args.world_size)
    mot = MOT()

    # KF model
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

    # --- D57 CHANGE --- feature measurement variance
    f_meas_var = float(args.feat_noise_std ** 2)

    # force initial spawn (from first noisy measurement)
    for o in objects:
        zxy = o.xy + np.random.randn(2).astype(np.float32) * args.meas_noise_std
        zf = float(o.feat + np.random.randn() * args.feat_noise_std)
        mot.spawn(Meas(zxy, zf))

    # collect snapshots and detect closest approach (truth distance)
    snapshots = []
    dists = []

    for t in range(args.steps):
        # advance world
        step_world(objects, args.dt, args.world_size)

        # measurements: always 2 in this micro scenario (no FOV/occlusion)
        meas_list: List[Meas] = []
        for o in objects:
            zxy = o.xy + np.random.randn(2).astype(np.float32) * args.meas_noise_std
            zf = float(o.feat + np.random.randn() * args.feat_noise_std)
            meas_list.append(Meas(zxy, zf))

        mot.step(
            meas_list=meas_list,
            F=F, Q=Q, H=H, R=R,
            gate_maha=args.gate_maha,
            max_misses=args.max_misses,
            track_no_match_cost=args.track_no_match_cost,
            meas_no_match_cost=args.meas_no_match_cost,
            use_feature_cost=bool(args.use_feature_cost),
            feat_weight=args.feat_weight,
            f_meas_var=f_meas_var,
            f_process_var=float(args.f_process_var),
        )

        # distance between truth objects
        d = float(np.linalg.norm(objects[0].xy - objects[1].xy))
        dists.append(d)

        # store minimal snapshot
        confirmed = sum(1 for tr in mot.tracks if tr.confirmed)
        grid = render_ascii(
            args.world_size, objects, mot,
            grid_w=args.grid_w, grid_h=args.grid_h,
            unc_low=args.unc_low, unc_mid=args.unc_mid
        )
        # compact track line: include feature mean too (for intuition)
        tr_lines = []
        for tr in mot.tracks:
            idx = (tr.tid - 1) % 26
            name = chr(ord("A") + idx)
            if not tr.confirmed:
                name = name.lower()
            pos_unc = float(np.trace(tr.Sigma[:2, :2]))
            unc_level = "LOW" if pos_unc < args.unc_low else ("MID" if pos_unc < args.unc_mid else "HIGH")
            mode = "UPDATE" if tr.misses == 0 else "COAST"
            tr_lines.append(
                f"{name}: {mode} miss={tr.misses:02d} "
                f"mu=({tr.mu[0]:4.1f},{tr.mu[1]:4.1f}) "
                f"unc={unc_level:4s} "
                f"f={tr.f_mu:+.2f}"
            )

        snapshots.append((t, d, len(mot.tracks), confirmed, grid, tr_lines))

    # print only the window around closest approach
    t_star = int(np.argmin(np.array(dists)))
    lo = max(0, t_star - args.window_before)
    hi = min(args.steps - 1, t_star + args.window_after)

    print("D57 micro demo: identity stability during crossing")
    print("Legend: 1,2 truth | A,B tracks | '+' update halo | ',' coast halo | lowercase = very uncertain/unconfirmed")
    print(f"use_feature_cost={args.use_feature_cost} feat_weight={args.feat_weight}  (set --use_feature_cost 0 to see D56-like swapping)")
    print(f"Closest approach at t={t_star} (printing t={lo}..{hi})")
    print("-" * 60)

    for (t, d, ntrk, conf, grid, tr_lines) in snapshots[lo:hi + 1]:
        print(f"[D57][t={t:02d}] d(1,2)={d:4.2f}  tracks={ntrk}  confirmed={conf}")
        print(grid)
        for s in tr_lines:
            print("  " + s)
        print("-" * 60)


if __name__ == "__main__":
    main()
