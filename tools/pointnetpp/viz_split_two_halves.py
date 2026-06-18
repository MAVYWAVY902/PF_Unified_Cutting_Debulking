"""2-panel video: cut ONE block down the middle -> does warp split it into TWO independent
halves? GRAVITY-driven dramatic version: the two halves fall AND spread apart (V-shape).
left = split OFF (welded -> falls as one rigid block); right = split ON (two halves diverge).
Vertical mid-band removed, free block, outward nudge + gravity (down = z axis), no floor.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.warp_cluster_split import split_clusters_inplace

cfg = MESH_CONFIGS[0]; device = "cuda:0"; top = cfg["top_axis"]
CUTAX = 0; GRAVAX = 2                     # cut along x
T_EL = 2; STEPS = 160; SPLIT_EVERY = 5; CAP_EVERY = 2; NUDGE = 0.6
DAMP = 0.98                              # dissipative: halves slide apart then SETTLE (no under-damped recoil)


def run(do_split):
    solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), rings=2, fix_threshold_fraction=0.0)
    N = solver.N; solver.damping = DAMP; solver.num_iterations = 2
    pos = mi["vertices"].copy(); bb0, bb1 = pos.min(0), pos.max(0); sp = bb1 - bb0
    fr = (pos[:, CUTAX] - bb0[CUTAX]) / sp[CUTAX]
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    mean_el = float(np.linalg.norm(pos[ea] - pos[eb], axis=1).mean())
    Tf = T_EL * mean_el / sp[CUTAX]
    removed = (fr >= 0.5 - Tf / 2) & (fr < 0.5 + Tf / 2)
    leftm = fr < (0.5 - Tf / 2); rightm = fr >= (0.5 + Tf / 2)
    solver.gravity = wp.vec3(0., 0., 0.); solver.floor_y = float(pos[:, 2].min() - 100.0)  # gravity OFF: pure slide-apart
    sw = torch.tensor(np.full(N, 0.5, np.float32), device=device)
    twt = torch.tensor(np.full(N, 0.30, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); far = pos.max(0) + 100.
    solver.reset_to_rest(pos)
    st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rest_state.items()}
    st["alpha"][removed] = 0.0; st["particle_damage"][removed] = 10.0
    vel = np.zeros((N, 3), np.float32); vel[leftm, CUTAX] = -NUDGE; vel[rightm, CUTAX] = +NUDGE
    st["vel"] = vel
    caps = []
    with torch.no_grad():
        for step in range(STEPS):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, 0.0, sw, twt)
            if do_split and step % SPLIT_EVERY == 0:
                split_clusters_inplace(solver, drop_small=True)
            st = solver_save_state(solver)
            st["alpha"][removed] = 0.0
            if step % CAP_EVERY == 0:
                a = solver.p["alpha"].numpy()
                caps.append((pw.cpu().numpy().copy(), a > 0.5))
    return caps, (bb0, bb1, leftm, rightm)


capsOFF, geo = run(False)
capsON, _ = run(True)
bb0, bb1, leftm, rightm = geo
nF = min(len(capsOFF), len(capsON))

fig = plt.figure(figsize=(11, 6))
axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]


def draw(i):
    for ax, caps, lab in zip(axes, [capsOFF, capsON], ["split OFF (welded block)", "split ON (two halves)"]):
        ax.clear()
        pf, alive = caps[i]
        for mask, col in [(leftm & alive, "crimson"), (rightm & alive, "darkorange")]:
            P = pf[mask]
            # screen: x = cut/spread axis, y = remaining lateral (top), z(up) = gravity axis
            ax.scatter(P[:, CUTAX], P[:, top], P[:, GRAVAX], s=2, c=col, alpha=0.7, linewidths=0)
        ax.set_title(lab, fontsize=12)
        ax.view_init(elev=16, azim=-72); ax.set_box_aspect([2.2, 1, 1])
        span = bb1[CUTAX] - bb0[CUTAX]
        ax.set_xlim(bb0[CUTAX] - 0.9 * span, bb1[CUTAX] + 0.9 * span)   # room to slide apart
        ax.set_ylim(bb0[top], bb1[top])
        ax.set_zlim(bb0[GRAVAX], bb1[GRAVAX])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle(f"Cut block down the middle: do the two halves slide apart as independent pieces?  frame {i}/{nF}", fontsize=11)


anim = FuncAnimation(fig, draw, frames=nF, interval=70)
out = REPO / "tools/pointnetpp/viz_out/split_two_halves.gif"
anim.save(str(out), writer=PillowWriter(fps=14))
print(f"saved {out}  ({nF} frames)", flush=True)
