# 64_d63_multi_object_search.py
import math, random
import numpy as np

DT = 0.25

def wrap_pi(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def in_fov(rx, ry, rth, tx, ty, rng=25.0, half_deg=70.0):
    dx, dy = tx - rx, ty - ry
    d = math.hypot(dx, dy)
    if d > rng:
        return False
    ang = math.atan2(dy, dx)
    dth = wrap_pi(ang - rth)
    return abs(dth) <= math.radians(half_deg)

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def main(
    steps=160,
    fov_range=25.0,
    fov_half_deg=70.0,
    seed=0,
    log_every=2,
):
    random.seed(seed)
    np.random.seed(seed)

    # -------------------------
    # Robot state
    # -------------------------
    rx, ry, rth = 5.0, 5.0, 0.0

    # -------------------------
    # True target motion (ground truth)
    # -------------------------
    tx, ty = 20.0, 14.0
    tvx, tvy = 0.2, 0.0

    # -------------------------
    # Occluder: vertical wall segment blocks line of sight
    # -------------------------
    wall_x = 14.0
    wall_y0, wall_y1 = 10.0, 20.0

    def occluded(rx, ry, tx, ty):
        # Does segment (robot->target) cross x=wall_x within [wall_y0, wall_y1]?
        if (rx - wall_x) * (tx - wall_x) >= 0:
            return False
        t = (wall_x - rx) / (tx - rx + 1e-9)
        y_cross = ry + t * (ty - ry)
        return (wall_y0 <= y_cross <= wall_y1)

    # -------------------------
    # Belief state (D64 core)
    # -------------------------
    Bx, By = tx, ty      # belief mean (start correct)
    s = 0.6              # belief uncertainty (std-ish)

    # Belief update knobs (minimal + stable)
    meas_noise = 0.15    # measurement noise when VIS=1
    alpha = 0.55         # how strongly we trust measurement when visible
    s_shrink = 0.65      # multiply s when we see (uncertainty shrinks)
    s_grow = 1.08        # multiply s when lost (uncertainty grows)
    s_min, s_max = 0.15, 12.0

    lost = 0

    for t in range(steps):
        # --- true target motion ---
        tx += tvx * DT
        ty += tvy * DT

        vis = in_fov(rx, ry, rth, tx, ty, fov_range, fov_half_deg) and (not occluded(rx, ry, tx, ty))

        if vis:
            # ================= TRACK (robot control) =================
            lost = 0
            ang = math.atan2(ty - ry, tx - rx)
            dth = wrap_pi(ang - rth)
            w = clamp(1.5 * dth, -1.0, 1.0)
            v = 1.2
            mode = "TRACK"

            # ================= Belief update (measurement) =================
            # No fancy filter yet: just noisy measurement + exponential update.
            zx = tx + np.random.randn() * meas_noise
            zy = ty + np.random.randn() * meas_noise

            Bx = (1 - alpha) * Bx + alpha * zx
            By = (1 - alpha) * By + alpha * zy
            s = clamp(s * s_shrink, s_min, s_max)

        else:
            # ================= LOST / REACQUIRE (robot control) =================
            lost += 1
            v = 0.9
            w = 0.5 if (lost // 20) % 2 == 0 else -0.5
            mode = "SWEEP"

            # ================= Belief update (prediction only) =================
            # Minimal motion model: "target keeps its last known velocity".
            # Here we just use the true tvx/tvy to keep the demo simple.
            # Later (D65+) we'll remove that cheat and estimate it.
            Bx += tvx * DT
            By += tvy * DT
            s = clamp(s * s_grow, s_min, s_max)

        # --- robot motion ---
        rth = wrap_pi(rth + w * DT)
        rx += v * math.cos(rth) * DT
        ry += v * math.sin(rth) * DT

        # --- compact output ---
        if t % log_every == 0:
            if vis:
                print(
                    f"t={t:03d} VIS  mode={mode}  "
                    f"robot=({rx:5.2f},{ry:5.2f})  "
                    f"x=({tx:5.2f},{ty:5.2f})  "
                    f"B=({Bx:5.2f},{By:5.2f})  s={s:4.2f}  lost={lost:03d}"
                )
            else:
                print(
                    f"t={t:03d} LOST mode={mode}  "
                    f"robot=({rx:5.2f},{ry:5.2f})  "
                    f"x=(  None,  None)  "
                    f"B=({Bx:5.2f},{By:5.2f})  s={s:4.2f}  lost={lost:03d}"
                )

if __name__ == "__main__":
    main()