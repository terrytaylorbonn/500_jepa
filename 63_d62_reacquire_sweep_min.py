#63_d62_reacquire_sweep_min.py
# 63_d62_reacquire_sweep_min.py
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
    if d > rng: return False
    ang = math.atan2(dy, dx)
    dth = wrap_pi(ang - rth)
    return abs(dth) <= math.radians(half_deg)

def main(steps=160, fov_range=25.0, fov_half_deg=70.0, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    # robot state
    rx, ry, rth = 5.0, 5.0, 0.0

    # target: constant velocity (drifts behind an "occluder line")
    tx, ty = 20.0, 14.0
    tvx, tvy = 0.2, 0.0

    # simple occluder: vertical wall segment blocking line of sight
    wall_x = 14.0
    wall_y0, wall_y1 = 10.0, 20.0

    def occluded(rx, ry, tx, ty):
        # check if ray from robot to target crosses the wall segment
        if (rx - wall_x) * (tx - wall_x) >= 0:
            return False
        t = (wall_x - rx) / (tx - rx + 1e-9)
        y_cross = ry + t * (ty - ry)
        return (wall_y0 <= y_cross <= wall_y1)

    lost = 0
    has_lock = True

    for t in range(steps):
        # --- target motion ---
        tx += tvx * DT
        ty += tvy * DT

        vis = in_fov(rx, ry, rth, tx, ty, fov_range, fov_half_deg) and (not occluded(rx, ry, tx, ty))

        if vis:
            # ================= TRACK =================
            lost = 0
            has_lock = True
            ang = math.atan2(ty - ry, tx - rx)
            dth = wrap_pi(ang - rth)
            w = max(-1.0, min(1.0, 1.5 * dth))
            v = 1.2
            mode = "TRACK"
        else:
            # ================= LOST / REACQUIRE =================
            lost += 1
            has_lock = False

            # core concept: move + sweep camera
            v = 0.9

            # alternate sweep direction every ~20 steps
            if (lost // 20) % 2 == 0:
                w = 0.5
            else:
                w = -0.5

            mode = "SWEEP"

        # --- robot motion ---
        rth = wrap_pi(rth + w * DT)
        rx += v * math.cos(rth) * DT
        ry += v * math.sin(rth) * DT

        # --- short output ---
        if t % 2 == 0:
            if vis:
                print(f"t={t:03d}  VIS  mode={mode}  robot=({rx:5.2f},{ry:5.2f})  target=({tx:5.2f},{ty:5.2f})")
            else:
                print(f"t={t:03d}  LOST={lost:03d}  mode={mode}  robot=({rx:5.2f},{ry:5.2f})")

if __name__ == "__main__":
    main()