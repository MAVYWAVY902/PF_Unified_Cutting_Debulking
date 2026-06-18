"""2-panel video, gravity along the cube's NATURAL up/down (top_axis = y = axis 1).
Cut block down the middle -> two halves fall DOWN (-y) and spread apart (outward nudge),
land on a manual y-floor (friction pins the separation). left = split OFF (welded, one
block falls); right = split ON (two independent halves). Prints fall + gap diagnostics."""
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

cfg = MESH_CONFIGS[0]; device = "cuda:0"; top = cfg["top_axis"]   # = 1 (y), the natural up
CUTAX = 0; YAX = top                          # cut along x; gravity along NATURAL up/down (y)
DEPTHAX = [a for a in range(3) if a not in (CUTAX, YAX)][0]
T_EL = 2; STEPS = 150; SPLIT_EVERY = 5; CAP_EVERY = 2; NUDGE = 0.4; DAMP = 0.98
FALL_ROOM = 0.18                              # manual y-floor this far below the block bottom


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
    g = [0., 0., 0.]; g[YAX] = -9.81
    solver.gravity = wp.vec3(*g); solver.floor_y = float(pos[:, 2].min() - 100.0)  # disable built-in (z) floor
    floor_lvl = float(bb0[YAX] - FALL_ROOM)
    sw = torch.tensor(np.full(N, 0.5, np.float32), device=device)
    twt = torch.tensor(np.full(N, 0.30, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); far = pos.max(0) + 100.
    solver.reset_to_rest(pos)
    st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rest_state.items()}
    st["alpha"][removed] = 0.0; st["particle_damage"][removed] = 10.0
    vel = np.zeros((N, 3), np.float32); vel[leftm, CUTAX] = -NUDGE; vel[rightm, CUTAX] = +NUDGE
    st["vel"] = vel
    y0 = pos[leftm | rightm, YAX].mean()
    caps = []
    with torch.no_grad():
        for step in range(STEPS):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, 0.0, sw, twt)
            if do_split and step % SPLIT_EVERY == 0:
                split_clusters_inplace(solver, drop_small=True)
            st = solver_save_state(solver)
            st["alpha"][removed] = 0.0
            # manual y-floor: clamp, stop downward vel, friction pins x-separation
            p = st["pos"]; v = st["vel"]
            below = p[:, YAX] < floor_lvl
            if below.any():
                p[below, YAX] = floor_lvl
                v[below, YAX] = 0.0
                v[below, CUTAX] *= 0.5
            st["pos"] = p; st["vel"] = v
            if step % CAP_EVERY == 0:
                a = solver.p["alpha"].numpy() > 0.5
                caps.append((pw.cpu().numpy().copy(), a))
            if step % 30 == 0:
                a = solver.p["alpha"].numpy() > 0.5
                fall = (y0 - pw.cpu().numpy()[(leftm | rightm) & a, YAX].mean()) * 1000
                gp = pw.cpu().numpy()
                gap = (gp[rightm & a, CUTAX].mean() - gp[leftm & a, CUTAX].mean()) * 1000
                print(f"  [{'ON ' if do_split else 'OFF'}] step {step:3d}: fall(-y)={fall:6.1f}mm  L-R gap={gap:6.1f}mm", flush=True)
    return caps, (bb0, bb1, leftm, rightm, floor_lvl)


print("=== gravity along natural y (top_axis); manual y-floor ===", flush=True)
capsOFF, geo = run(False)
capsON, _ = run(True)
bb0, bb1, leftm, rightm, floor_lvl = geo
nF = min(len(capsOFF), len(capsON))

fig = plt.figure(figsize=(11, 6))
axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]


def draw(i):
    for ax, caps, lab in zip(axes, [capsOFF, capsON], ["split OFF (one block)", "split ON (two halves)"]):
        ax.clear()
        pf, alive = caps[i]
        for mask, col in [(leftm & alive, "crimson"), (rightm & alive, "darkorange")]:
            P = pf[mask]
            ax.scatter(P[:, CUTAX], P[:, DEPTHAX], P[:, YAX], s=2, c=col, alpha=0.7, linewidths=0)  # up = y (natural)
        ax.set_title(lab, fontsize=12)
        ax.view_init(elev=12, azim=-72); ax.set_box_aspect([1.9, 1, 1.5])
        span = bb1[CUTAX] - bb0[CUTAX]
        ax.set_xlim(bb0[CUTAX] - 0.7 * span, bb1[CUTAX] + 0.7 * span)
        ax.set_ylim(bb0[DEPTHAX], bb1[DEPTHAX])
        ax.set_zlim(floor_lvl - 0.03, bb1[YAX])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle(f"Cut + gravity (natural down): two independent halves fall & spread  frame {i}/{nF}", fontsize=11)


anim = FuncAnimation(fig, draw, frames=nF, interval=70)
out = REPO / "tools/pointnetpp/viz_out/split_two_halves_gravity.gif"
anim.save(str(out), writer=PillowWriter(fps=14))
print(f"saved {out}  ({nF} frames)", flush=True)
