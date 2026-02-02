
#!/usr/bin/env python3
"""
42_D49b2A_online_full_supervision.py

[D49b2A] Online residual world model (full supervision)

Extends D49b1:
- same 3-object 2D world
- classical vs NN residual belief
- NN learns online during rollout
- plots:
    (1) xy trajectories (true / classical / NN)
    (2) per-object belief error over time
"""

import math, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

DT = 0.1
STEPS = 400
LR = 1e-3

def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi

# -----------------------------------
# world init
# -----------------------------------
def init_world():
    robot = {'x':0., 'y':0., 'th':0.}
    objs = []
    objs.append({'p':np.array([2.,0.],np.float32),
                 'v':np.array([0.01,0.],np.float32)})
    objs.append({'p':np.array([3.,1.],np.float32),
                 'v':np.array([0.01,-0.01],np.float32)})
    objs.append({'p':np.array([3.5,-1.],np.float32),
                 'v':np.array([0.01,0.],np.float32)})
    return robot, objs

# -----------------------------------
# simple robot control
# -----------------------------------
def control(t):
    time = t*DT
    v = 0.2
    if 5<time<8:    w=math.radians(20)
    elif 8<=time<11:w=math.radians(-20)
    else:           w=0.
    return v,w

# -----------------------------------
# residual MLP
# -----------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4,64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,2)
    def forward(self,x):
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        return self.l3(x)

# -----------------------------------
# main
# -----------------------------------
def main():
    r_cl, o_cl = init_world()
    r_nn, o_nn = init_world()

    model = MLP()
    opt = optim.Adam(model.parameters(), lr=LR)
    lossfn = nn.MSELoss()

    true_traj  = [[] for _ in o_cl]
    cl_traj    = [[] for _ in o_cl]
    nn_traj    = [[] for _ in o_cl]
    err_cl     = [[] for _ in o_cl]
    err_nn     = [[] for _ in o_cl]

    for t in range(STEPS):
        # ----- classical -----
        v,w = control(t)
        r_cl['x'] += v*math.cos(r_cl['th'])*DT
        r_cl['y'] += v*math.sin(r_cl['th'])*DT
        r_cl['th'] = wrap(r_cl['th'] + w*DT)
        for o in o_cl:
            o['p'] = o['p'] + o['v']*DT

        # ----- NN world -----
        v,w = control(t)
        r_nn['x'] += v*math.cos(r_nn['th'])*DT
        r_nn['y'] += v*math.sin(r_nn['th'])*DT
        r_nn['th'] = wrap(r_nn['th'] + w*DT)

        for i,(oc,on) in enumerate(zip(o_cl,o_nn)):
            # belief prediction
            B = on['p']
            Bpred = B + on['v']*DT

            # full supervision target from classical world
            Btrue_next = oc['p']  # classical already stepped

            # NN residual
            x = np.array([Bpred[0], Bpred[1], v*DT, w*DT], np.float32)[None,:]
            xt = torch.from_numpy(x)
            pred = model(xt)
            d = pred.detach().numpy()[0]
            on['p'] = Bpred + d

            # online update
            y = (Btrue_next - Bpred).astype(np.float32)[None,:]
            yt = torch.from_numpy(y)
            loss = lossfn(model(xt), yt)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # record
        for i,(oc,on) in enumerate(zip(o_cl,o_nn)):
            true_traj[i].append(oc['p'].copy())
            cl_traj[i].append(oc['p'].copy())
            nn_traj[i].append(on['p'].copy())
            err_cl[i].append(np.linalg.norm(oc['p'] - oc['p']))   # classical = baseline (0)
            err_nn[i].append(np.linalg.norm(oc['p'] - on['p']))

    true_traj = [np.stack(x) for x in true_traj]
    cl_traj   = [np.stack(x) for x in cl_traj]
    nn_traj   = [np.stack(x) for x in nn_traj]
    err_cl    = [np.stack(x) for x in err_cl]
    err_nn    = [np.stack(x) for x in err_nn]


    # v5 ----- plot trajectories -----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax0, ax1 = ax

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # choose which object to visualize
    obj_idx = 0

    pts_true = true_traj[obj_idx]
    pts_cl   = cl_traj[obj_idx]
    pts_nn   = nn_traj[obj_idx]

    # main lines (make them a bit transparent so markers stand out)
    ax0.plot(pts_true[:, 0], pts_true[:, 1],
             linestyle='-', color='tab:blue', alpha=0.5, label="true")
    ax0.plot(pts_cl[:, 0], pts_cl[:, 1],
             linestyle='--', color='tab:green', alpha=0.6, label="classical")
    ax0.plot(pts_nn[:, 0], pts_nn[:, 1],
             linestyle=':', color='tab:orange', alpha=0.9, label="nn")

    # ============================
    #   BIG START / END MARKERS
    # ============================

    # TRUE: start = green square, end = green circle
    ax0.scatter(pts_true[0, 0],  pts_true[0, 1],
                marker='s', s=80,
                facecolors='none', edgecolors='green',
                linewidths=2, zorder=10, label="true start")
    ax0.scatter(pts_true[-1, 0], pts_true[-1, 1],
                marker='o', s=80,
                facecolors='green', edgecolors='black',
                linewidths=1, zorder=10, label="true end")

    # CLASSICAL: start/end in blue
    ax0.scatter(pts_cl[0, 0],  pts_cl[0, 1],
                marker='s', s=70,
                facecolors='none', edgecolors='blue',
                linewidths=2, zorder=10, label="classical start")
    ax0.scatter(pts_cl[-1, 0], pts_cl[-1, 1],
                marker='o', s=70,
                facecolors='blue', edgecolors='black',
                linewidths=1, zorder=10, label="classical end")

    # NN: start/end in red so they pop
    ax0.scatter(pts_nn[0, 0],  pts_nn[0, 1],
                marker='s', s=90,
                facecolors='none', edgecolors='red',
                linewidths=2, zorder=11, label="nn start")
    ax0.scatter(pts_nn[-1, 0], pts_nn[-1, 1],
                marker='o', s=90,
                facecolors='red', edgecolors='black',
                linewidths=1, zorder=11, label="nn end")

    # v5 --- add time labels along NN trajectory ---
    step = 5
    for t in range(0, len(pts_nn), step):
        x, y = pts_nn[t]
        ax0.text(x, y, f"{t}", fontsize=7, color='red', zorder=12)

    ax0.set_title("Trajectories (obj0)")
    ax0.set_aspect('equal')
    ax0.grid(True)
    ax0.legend(fontsize=7, loc="best")

    # ----- plot errors -----
    for i, c in enumerate(colors):
        ax1.plot(err_nn[i], color=c, label=f"obj{i} nn error")
    ax1.set_title("Belief error over time")
    ax1.set_xlabel("t")
    ax1.set_ylabel("||true - B||")
    ax1.grid(True)
    ax1.legend(fontsize=7)

    fig.suptitle("[D49b2A] Online residual dynamics (full supervision)")
    plt.tight_layout()
    plt.show()


    # V2 # ----- plot trajectories -----
    # fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # ax0, ax1 = ax

    # # --- choose a single object to visualize ---
    # obj_idx = 0  # 0, 1, or 2

    # pts_true = true_traj[obj_idx]   # shape (T, 2)
    # pts_cl   = cl_traj[obj_idx]
    # pts_nn   = nn_traj[obj_idx]

    # # main lines
    # ax0.plot(pts_true[:, 0], pts_true[:, 1],
    #          linestyle='-',  color='tab:blue', label=f"obj{obj_idx} true")
    # ax0.plot(pts_cl[:, 0], pts_cl[:, 1],
    #          linestyle='--', color='tab:blue', alpha=0.6, label=f"obj{obj_idx} classical")
    # ax0.plot(pts_nn[:, 0], pts_nn[:, 1],
    #          linestyle=':',  color='tab:blue', alpha=0.9, label=f"obj{obj_idx} nn")

    # # arrows along the *true* trajectory to show direction
    # step = 10  # sample every 10th point for arrows
    # P = pts_true[::step]                  # positions
    # V = np.diff(pts_true[::step], axis=0) # segment vectors

    # ax0.quiver(
    #     P[:-1, 0], P[:-1, 1],   # arrow bases
    #     V[:, 0],  V[:, 1],      # arrow directions
    #     angles='xy', scale_units='xy', scale=1.0,
    #     width=0.005, color='k'
    # )

    # ax0.set_title("Trajectories (single object)")
    # ax0.set_aspect('equal')
    # ax0.grid(True)
    # ax0.legend(fontsize=7)

    # # ----- plot errors -----
    # # If you still want all three error curves:
    # colors = ['tab:blue', 'tab:orange', 'tab:green']
    # for i, c in enumerate(colors):
    #     ax1.plot(err_nn[i], color=c, label=f"obj{i} nn error")

    # ax1.set_title("Belief error over time")
    # ax1.set_xlabel("t")
    # ax1.set_ylabel("||true - B||")
    # ax1.grid(True)
    # ax1.legend(fontsize=7)

    # fig.suptitle("[D49b2A] Online residual dynamics (full supervision)")
    # plt.tight_layout()
    # plt.show()


    # V1 # ----- plot trajectories -----
    # fig,ax = plt.subplots(1,2,figsize=(10,4))
    # ax0,ax1=ax

    # colors=['tab:blue','tab:orange','tab:green']

    # i = 0   # choose object index here

    # ax0.plot(true_traj[i][:,0], true_traj[i][:,1],
    #         linestyle='-', color='tab:blue', label=f"obj{i} true")

    # ax0.plot(cl_traj[i][:,0], cl_traj[i][:,1],
    #         linestyle='--', color='tab:blue', alpha=0.6, label=f"obj{i} classical")

    # ax0.plot(nn_traj[i][:,0], nn_traj[i][:,1],
    #         linestyle=':', color='tab:blue', alpha=0.9, label=f"obj{i} nn")

    # # for i,c in enumerate(colors):
    # #     ax0.plot(true_traj[i][:,0], true_traj[i][:,1],
    # #             linestyle='-', color=c, label=f"obj{i} true")
    # #     ax0.plot(cl_traj[i][:,0], cl_traj[i][:,1],
    # #             linestyle='--', color=c, alpha=0.6, label=f"obj{i} classical")

    # #     ax0.plot(nn_traj[i][:,0], nn_traj[i][:,1],
    # #             linestyle=':', color=c, alpha=0.9, label=f"obj{i} nn")

    # pts = true_traj[i]
    # ax0.quiver(
    #     pts[0:-1:5,0], pts[0:-1:5,1],     # sample points
    #     pts[1::5,0]-pts[0:-1:5,0],        # dx
    #     pts[1::5,1]-pts[0:-1:5,1],        # dy
    #     angles='xy', scale_units='xy', scale=1.0,
    #     width=0.005, color='k'
    # )

    # ax0.set_title("Trajectories")
    # ax0.set_aspect('equal'); ax0.grid(True); ax0.legend(fontsize=7)

    # # ----- plot errors -----
    # for i,c in enumerate(colors):
    #     ax1.plot(err_nn[i], color=c, label=f"obj{i} nn error")
    # ax1.set_title("Belief error over time")
    # ax1.set_xlabel("t"); ax1.set_ylabel("||true - B||")
    # ax1.grid(True); ax1.legend(fontsize=7)

    # fig.suptitle("[D49b2A] Online residual dynamics (full supervision)")
    # plt.tight_layout(); plt.show()


if __name__=="__main__":
    main()