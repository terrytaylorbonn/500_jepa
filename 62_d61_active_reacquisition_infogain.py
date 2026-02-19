#!/usr/bin/env python3
# D61: Active re-acquisition via expected-visibility (info-gain) planning
# - Keeps the same belief (mu, sigma) tracker as D60d
# - When vision is lost: chooses actions by simulating short rollouts and scoring
#   expected visibility probability under belief (Monte Carlo samples).
#
# Run:
#   python 62_d61_active_reacquisition_infogain.py --steps 240 --out_gif d61.gif
#
# Notes:
# - This is intentionally small + readable.
# - You should see fewer “endless spiral” minutes and more purposeful turns toward likely re-sighting angles.

import math
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:
    import imageio


# -----------------------------
# Geometry helpers
# -----------------------------
def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def seg_intersect(p1, p2, q1, q2) -> bool:
    # segment intersection (robust enough for our simple sim)
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def on_segment(a, b, c):
        # c on segment ab
        return (min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and
                min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9)

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    # general
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return True

    # collinear cases
    if abs(o1) < 1e-9 and on_segment(p1, p2, q1): return True
    if abs(o2) < 1e-9 and on_segment(p1, p2, q2): return True
    if abs(o3) < 1e-9 and on_segment(q1, q2, p1): return True
    if abs(o4) < 1e-9 and on_segment(q1, q2, p2): return True
    return False

def line_intersects_rect(p, q, rect) -> bool:
    x1, y1, x2, y2 = rect  # minx,miny,maxx,maxy
    # If either endpoint is inside rect, treat as intersecting (occluded)
    if (x1 <= p[0] <= x2 and y1 <= p[1] <= y2) or (x1 <= q[0] <= x2 and y1 <= q[1] <= y2):
        return True
    corners = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    edges = [(corners[0], corners[1]),
             (corners[1], corners[2]),
             (corners[2], corners[3]),
             (corners[3], corners[0])]
    for a,b in edges:
        if seg_intersect(p, q, a, b):
            return True
    return False

def in_fov(robot_xy, robot_th, obj_xy, fov_range, fov_half_angle_rad) -> bool:
    dx = obj_xy[0] - robot_xy[0]
    dy = obj_xy[1] - robot_xy[1]
    r = math.hypot(dx, dy)
    if r > fov_range:
        return False
    ang = math.atan2(dy, dx)
    dth = wrap_pi(ang - robot_th)
    return abs(dth) <= fov_half_angle_rad


# -----------------------------
# World + sensor
# -----------------------------
class World:
    def __init__(self, W=40.0, H=40.0, obstacle=(14, 17, 24, 22), seed=0):
        self.W = W
        self.H = H
        self.obstacle = obstacle
        self.rng = np.random.default_rng(seed)

        # robot state
        self.rx = 8.0
        self.ry = 8.0
        self.rth = 0.2

        # # object state (random walk)
        # self.ox = 31.0
        # self.oy = 20.0
        # self.ovx = -0.25
        # self.ovy = 0.15
        
        ######## object state (random walk) -- start visible, then drift toward obstacle
        self.ox = 20.0
        self.oy = 14.0
        self.ovx = 0.30
        self.ovy = 0.25

    def step(self, v, w, dt=0.25):
        # robot unicycle
        self.rth = wrap_pi(self.rth + w * dt)
        self.rx += v * math.cos(self.rth) * dt
        self.ry += v * math.sin(self.rth) * dt
        self.rx = float(np.clip(self.rx, 1.0, self.W - 1.0))
        self.ry = float(np.clip(self.ry, 1.0, self.H - 1.0))

        # object mild random walk with boundary bounce
        ax = self.rng.normal(0.0, 0.08)
        ay = self.rng.normal(0.0, 0.08)
        self.ovx = 0.92 * self.ovx + ax
        self.ovy = 0.92 * self.ovy + ay
        self.ox += self.ovx
        self.oy += self.ovy

        if self.ox < 1.0 or self.ox > self.W - 1.0:
            self.ovx *= -1
            self.ox = float(np.clip(self.ox, 1.0, self.W - 1.0))
        if self.oy < 1.0 or self.oy > self.H - 1.0:
            self.ovy *= -1
            self.oy = float(np.clip(self.oy, 1.0, self.H - 1.0))

    def sense(self, fov_range=15.0, fov_half_angle_deg=35.0, meas_noise=0.6):
        fov_half = math.radians(fov_half_angle_deg)
        rxy = (self.rx, self.ry)
        oxy = (self.ox, self.oy)

        visible = in_fov(rxy, self.rth, oxy, fov_range, fov_half) and (not line_intersects_rect(rxy, oxy, self.obstacle))
        if not visible:
            return None, 0

        x = np.array([self.ox, self.oy], dtype=np.float32)
        x += self.rng.normal(0.0, meas_noise, size=(2,)).astype(np.float32)
        return x, 1


# -----------------------------
# Belief (simple isotropic Gaussian)
# -----------------------------
class Belief:
    def __init__(self, mu=(8.0, 8.0), sigma=3.0):
        self.has_lock = False
        self.mu = np.array(mu, dtype=np.float32)
        self.sigma = float(sigma)

        ###################### CHANGE ######################
        # Count consecutive "blind" frames; used to decide when to unlock.
        self.lost_count = 0
        ###################### CHANGE ######################

    def predict(self, q_inflate=0.18, sigma_max=12.0):
        # when blind, uncertainty inflates
        self.sigma = float(min(sigma_max, math.sqrt(self.sigma * self.sigma + q_inflate * q_inflate)))

    def update(self, z, meas_noise=0.6, gate_distance_good=1.0, gate_distance_bad=10.0):
        # gated Kalman-ish update
        if z is None:
            ###################### CHANGE ######################
            # If we've been blind for a couple frames, allow next seen measurement
            # to "lock-on" once (bootstrap path).
            self.lost_count += 1
            if self.lost_count >= 2:   # tweak: 1=immediate, 2=avoid single-frame dropout
                self.has_lock = False
            ###################### CHANGE ######################
            return 0.0, 0.0, 0.0  # d, gate, K

        ###################### CHANGE ######################
        # Visible again: reset blind counter.
        self.lost_count = 0
        ###################### CHANGE ######################

        z = np.array(z, dtype=np.float32)
        d = float(np.linalg.norm(z - self.mu))

        ####### Bootstrap: first time we see something after being blind, lock-on once.
        if not self.has_lock:
            self.mu = z.copy()
            self.sigma = float(max(0.8, meas_noise * 1.5))
            self.has_lock = True
            return d, 1.0, 1.0

        # distance-based gate (0..1)
        if d <= gate_distance_good:
            gate = 1.0
        elif d >= gate_distance_bad:
            gate = 0.0
        else:
            gate = 1.0 - (d - gate_distance_good) / (gate_distance_bad - gate_distance_good)

        # Kalman gain (scalar isotropic)
        s2 = self.sigma * self.sigma
        r2 = meas_noise * meas_noise
        K = s2 / (s2 + r2 + 1e-9)

        alpha = gate * K
        self.mu = (1.0 - alpha) * self.mu + alpha * z

        # shrink sigma when we accepted measurement (gate>0)
        self.sigma = float(math.sqrt(max(0.05, (1.0 - gate * K) * s2)))

        return d, gate, K# class Belief:
#     def __init__(self, mu=(8.0, 8.0), sigma=3.0):
#         self.has_lock = False ##################################
#         self.mu = np.array(mu, dtype=np.float32)
#         self.sigma = float(sigma)

#     def predict(self, q_inflate=0.18, sigma_max=12.0):
#         # when blind, uncertainty inflates
#         self.sigma = float(min(sigma_max, math.sqrt(self.sigma * self.sigma + q_inflate * q_inflate)))

#     def update(self, z, meas_noise=0.6, gate_distance_good=1.0, gate_distance_bad=10.0):
#         # gated Kalman-ish update
#         if z is None:
#             return 0.0, 0.0, 0.0  # d, gate, K

#         z = np.array(z, dtype=np.float32)
#         d = float(np.linalg.norm(z - self.mu))

#         ####### Bootstrap: first time we see something after being blind, lock-on once.
#         if not self.has_lock:
#             self.mu = z.copy()
#             self.sigma = float(max(0.8, meas_noise * 1.5))
#             self.has_lock = True
#             return d, 1.0, 1.0

#         # distance-based gate (0..1)
#         if d <= gate_distance_good:
#             gate = 1.0
#         elif d >= gate_distance_bad:
#             gate = 0.0
#         else:
#             gate = 1.0 - (d - gate_distance_good) / (gate_distance_bad - gate_distance_good)

#         # Kalman gain (scalar isotropic)
#         s2 = self.sigma * self.sigma
#         r2 = meas_noise * meas_noise
#         K = s2 / (s2 + r2 + 1e-9)

#         alpha = gate * K
#         self.mu = (1.0 - alpha) * self.mu + alpha * z

#         # shrink sigma when we accepted measurement (gate>0)
#         self.sigma = float(math.sqrt(max(0.05, (1.0 - gate * K) * s2)))

#         return d, gate, K


# -----------------------------
# D61: Info-gain action selection when lost
# -----------------------------
def expected_visibility_score(world: World, belief: Belief, v, w,
                              horizon=6, dt=0.25,
                              n_samples=64,
                              fov_range=15.0, fov_half_angle_deg=35.0,
                              obstacle=(14, 17, 24, 22),
                              step_cost=0.02, turn_cost=0.01, drift_cost=0.002):
    """
    Simulate robot rollout under (v,w) repeated horizon steps, but object is not simulated.
    Instead we sample object positions from belief N(mu, sigma^2 I) and check if visible
    from any pose in the rollout. Score ~ P(visible sometime) - small costs.
    """
    fov_half = math.radians(fov_half_angle_deg)

    # rollout robot poses (copy)
    rx, ry, rth = world.rx, world.ry, world.rth
    poses = []
    for _ in range(horizon):
        rth = wrap_pi(rth + w * dt)
        rx += v * math.cos(rth) * dt
        ry += v * math.sin(rth) * dt
        rx = float(np.clip(rx, 1.0, world.W - 1.0))
        ry = float(np.clip(ry, 1.0, world.H - 1.0))
        poses.append((rx, ry, rth))

    # Monte Carlo object samples from belief
    mu = belief.mu.astype(np.float32)
    sig = float(belief.sigma)
    samples = np.random.normal(loc=mu, scale=sig, size=(n_samples, 2)).astype(np.float32)
    samples[:, 0] = np.clip(samples[:, 0], 1.0, world.W - 1.0)
    samples[:, 1] = np.clip(samples[:, 1], 1.0, world.H - 1.0)

    # count samples that become visible at any step
    hit = 0
    for sxy in samples:
        sx, sy = float(sxy[0]), float(sxy[1])
        visible_any = False
        for (px, py, pth) in poses:
            if in_fov((px, py), pth, (sx, sy), fov_range, fov_half):
                if not line_intersects_rect((px, py), (sx, sy), obstacle):
                    visible_any = True
                    break
        hit += 1 if visible_any else 0

    p_vis = hit / float(n_samples)

    # costs to prevent crazy spinning forever
    cost = step_cost * horizon + turn_cost * abs(w) * horizon + drift_cost * abs(v) * horizon
    ##########################################
    if abs(v) < 1e-6:  
        cost += 0.15
        ####################################3
    return p_vis - cost


def choose_action_D61(world: World, belief: Belief, lost_count: int,
                      v_set=(0.0, 0.7, 1.2),
                      w_set=(-1.2, -0.7, -0.3, 0.0, 0.3, 0.7, 1.2),
                      horizon=6, n_samples=64,
                      fov_range=15.0, fov_half_angle_deg=35.0):
    # If not very lost, do a mild “toward belief” action.
    # If very lost, rely more on info-gain scoring.
    # We’ll blend a small “go toward mu” preference.
    dx = float(belief.mu[0] - world.rx)
    dy = float(belief.mu[1] - world.ry)
    ang_to_mu = math.atan2(dy, dx)
    dth = wrap_pi(ang_to_mu - world.rth)

    best = (-1e9, 0.0, 0.0)  # score, v, w
    for v in v_set:
        for w in w_set:
            s = expected_visibility_score(
                world, belief, v, w,
                horizon=horizon,
                n_samples=n_samples,
                fov_range=fov_range,
                fov_half_angle_deg=fov_half_angle_deg,
                obstacle=world.obstacle,
            )

            # mild preference to turn toward belief (stronger when not too lost)
            toward = max(0.0, 1.0 - abs(dth) / math.pi)
            s += (0.06 if lost_count < 40 else 0.02) * toward

            ###################### CHANGE ######################
            # discourage v=0 earlier + stronger (encourage movement to change viewpoint)
            if v == 0.0:
                s -= (0.10 + 0.002 * lost_count)
            ###################### CHANGE ######################

            # # if super lost, discourage v=0 (encourage movement to change viewpoint)
            # if lost_count > 80 and v == 0.0:
            #     s -= 0.05

            if s > best[0]:
                best = (s, v, w)

    return float(best[1]), float(best[2]), float(best[0])


# -----------------------------
# Plotting
# -----------------------------
def draw_frame(world: World, belief: Belief, z, vis, t, lost, mode, info_score=None,
               fov_range=15.0, fov_half_angle_deg=35.0):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, world.W)
    ax.set_ylim(0, world.H)
    ax.set_aspect("equal")
    ax.set_title(f"D61 t={t:03d}  vis={vis}  lost={lost:03d}  mode={mode}" + ("" if info_score is None else f"  score={info_score:+.3f}"))

    # obstacle
    x1, y1, x2, y2 = world.obstacle
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2))

    # true object
    ax.scatter([world.ox], [world.oy], marker="x", s=80)

    # robot
    ax.scatter([world.rx], [world.ry], s=60)
    # heading
    hx = world.rx + 1.8 * math.cos(world.rth)
    hy = world.ry + 1.8 * math.sin(world.rth)
    ax.plot([world.rx, hx], [world.ry, hy], linewidth=2)

    # FOV wedge (approx lines)
    half = math.radians(fov_half_angle_deg)
    for sign in (-1, +1):
        ang = world.rth + sign * half
        ax.plot([world.rx, world.rx + fov_range * math.cos(ang)],
                [world.ry, world.ry + fov_range * math.sin(ang)], linestyle="--", linewidth=1)

    # belief ellipse (1-sigma circle)
    circ = plt.Circle((float(belief.mu[0]), float(belief.mu[1])),
                      radius=float(belief.sigma), fill=False, linewidth=2)
    ax.add_patch(circ)
    ax.scatter([belief.mu[0]], [belief.mu[1]], s=40)

    # measurement
    if z is not None:
        ax.scatter([z[0]], [z[1]], marker="+", s=80)
        ax.plot([world.rx, z[0]], [world.ry, z[1]], linewidth=1, alpha=0.6)

    # legend-ish text
    txt = f"robot=({world.rx:4.1f},{world.ry:4.1f}) th={world.rth:+.2f}\n" \
          f"true =({world.ox:4.1f},{world.oy:4.1f})\n" \
          f"mu   =({belief.mu[0]:4.1f},{belief.mu[1]:4.1f})  s={belief.sigma:4.2f}"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=9, va="bottom")

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img


# -----------------------------
# Main loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_gif", type=str, default="d61.gif")

    # sensor
    ap.add_argument("--fov_range", type=float, default=15.0)
    ap.add_argument("--fov_half_angle_deg", type=float, default=35.0)
    ap.add_argument("--meas_noise", type=float, default=0.6)

    # belief gating
    ap.add_argument("--gate_good", type=float, default=1.0)
    ap.add_argument("--gate_bad", type=float, default=10.0)
    # ap.add_argument("--q_inflate", type=float, default=0.18)
    ap.add_argument("--q_inflate", type=float, default=0.60)

    # D61 planner
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--mc_samples", type=int, default=64)

    args = ap.parse_args()

    world = World(seed=args.seed)
    belief = Belief(mu=(world.rx, world.ry), sigma=3.0)

    frames = []

    lost = 0
    mode = "TRACK"

    # simple “track” controller gains
    kv = 0.35
    kw = 1.20
    v_max = 1.4
    w_max = 1.6

    print("[D61] Starting active re-acquisition (info-gain)")
    print(f"  steps      = {args.steps}")
    print(f"  fov_range  = {args.fov_range}")
    print(f"  fov_half   = {args.fov_half_angle_deg} deg")
    print(f"  horizon    = {args.horizon}")
    print(f"  mc_samples = {args.mc_samples}")

    for t in range(args.steps):
        # sense
        z, vis = world.sense(
            fov_range=args.fov_range,
            fov_half_angle_deg=args.fov_half_angle_deg,
            meas_noise=args.meas_noise
        )

        if vis == 0:
            lost += 1
            belief.predict(q_inflate=args.q_inflate)
        ##############################################3
        # else:
        #     lost = 0

        ###################### CHANGE ######################
        # Dampen overly-optimistic p_vis when belief is very uncertain.
        # Prevents the planner from thinking "everything is probably visible somewhere".
        #p_vis *= 1.0 / (1.0 + 0.15 * sig)
        ###################### CHANGE ######################
    
        # update belief
        d, gate, K = belief.update(
            z,
            meas_noise=args.meas_noise,
            gate_distance_good=args.gate_good,
            gate_distance_bad=args.gate_bad
        )

        # choose action
        info_score = None
        if vis == 1:
            mode = "TRACK"
            dx = float(belief.mu[0] - world.rx)
            dy = float(belief.mu[1] - world.ry)
            ang = math.atan2(dy, dx)
            dth = wrap_pi(ang - world.rth)
            dist = math.hypot(dx, dy)

            v = float(np.clip(kv * dist, 0.0, v_max))
            w = float(np.clip(kw * dth, -w_max, w_max))
        else:
            mode = "INFOGAIN"
            v, w, info_score = choose_action_D61(
                world, belief, lost_count=lost,
                horizon=args.horizon,
                n_samples=args.mc_samples,
                fov_range=args.fov_range,
                fov_half_angle_deg=args.fov_half_angle_deg
            )
            # clip
            v = float(np.clip(v, 0.0, v_max))
            w = float(np.clip(w, -w_max, w_max))

        # apply
        world.step(v, w)

        # logging (keep in your style)
        if z is None:
            zstr = "None"
        else:
            zstr = f"({float(z[0]):4.1f},{float(z[1]):4.1f})"

        if (t % 1) == 0:
            if info_score is None:
                print(f"t={t:03d}  x={zstr:>12} vis={vis}  lost={lost:03d}  "
                      f"B=({belief.mu[0]:4.1f},{belief.mu[1]:4.1f}) s={belief.sigma:4.2f}  "
                      f"d={d:5.2f} gate={gate:4.2f} K={K:4.2f}  mode={mode:6}  a=({v:+.2f},{w:+.2f})")
            else:
                print(f"t={t:03d}  x={zstr:>12} vis={vis}  lost={lost:03d}  "
                      f"B=({belief.mu[0]:4.1f},{belief.mu[1]:4.1f}) s={belief.sigma:4.2f}  "
                      f"d={d:5.2f} gate={gate:4.2f} K={K:4.2f}  mode={mode:6}  a=({v:+.2f},{w:+.2f})  score={info_score:+.3f}")

        # render
        frame = draw_frame(
            world, belief, z, vis, t, lost, mode,
            info_score=info_score,
            fov_range=args.fov_range,
            fov_half_angle_deg=args.fov_half_angle_deg
        )
        frames.append(frame)

    imageio.mimsave(args.out_gif, frames, duration=0.08)
    print(f"[D61] Saved: {args.out_gif}")


if __name__ == "__main__":
    main()