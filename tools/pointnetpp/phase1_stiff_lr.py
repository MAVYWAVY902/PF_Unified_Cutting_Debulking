"""Phase 1 (stiff variant): L/R heterogeneous STIFFNESS visual contrast.

Stiff directly modulates cluster shape-matching pullback → position signal
strong. Vs tau which only changes alpha (saturated at s=100).

Each mesh, two configs:
   A: stiff_left=0.2 (soft), stiff_right=0.8 (tough)
   B: stiff_left=0.8, stiff_right=0.2 (mirror)
Same uniform tau=0.30.
Same knife trajectory.

Verify visible cavity + position asymmetry before Phase 2.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402


STIFF_HIGH = 0.8
STIFF_LOW = 0.2
TAU_UNIFORM = 0.30


def build_lr_stiff(pos_rest, stiff_left, stiff_right):
    x_mid = 0.5 * (pos_rest[:, 0].min() + pos_rest[:, 0].max())
    left_mask = pos_rest[:, 0] < x_mid
    stiff = np.where(left_mask, stiff_left, stiff_right).astype(np.float32)
    return stiff, left_mask, x_mid


def forward_one_config(cfg, stiff_truth, traj, device):
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.95
    solver.num_iterations = 2
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)
    pos_rest = mesh_info["vertices"]
    solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))
    tau_uniform = np.full(N, TAU_UNIFORM, dtype=np.float32)
    theta = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    solver.reset_to_rest(pos_rest)
    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta, stiff_truth, traj,
        cfg["knife_radius_abs"], device, tau_truth_np=tau_uniform)
    return obs_pos, obs_alpha


def render_2panel_3d(pos_A, alpha_A, label_A,
                       pos_B, alpha_B, label_B,
                       knife_pos, bbox_min, bbox_max, x_mid,
                       t, T, mesh_name, surface_mask, out_path):
    fig = plt.figure(figsize=(14, 6.5))
    azim = 30.0 + (t / max(T - 1, 1)) * 60.0
    elev = 22.0
    size_box = (bbox_max - bbox_min).max()
    cx = 0.5 * (bbox_min + bbox_max)
    for j, (pos, alpha, label) in enumerate([(pos_A, alpha_A, label_A),
                                                 (pos_B, alpha_B, label_B)]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        sm = surface_mask
        a = alpha[sm]; p = pos[sm]
        alive = a > 0.5; dead = ~alive
        ax.scatter(p[alive, 0], p[alive, 1], p[alive, 2],
                    c="lightgray", s=8, edgecolors="none", alpha=0.25, depthshade=True)
        ax.scatter(p[dead, 0], p[dead, 1], p[dead, 2],
                    c="red", s=35, edgecolors="darkred", linewidth=0.4,
                    alpha=0.95, depthshade=True)
        yv = np.linspace(bbox_min[1], bbox_max[1], 6)
        zv = np.linspace(bbox_min[2], bbox_max[2], 6)
        Y, Z = np.meshgrid(yv, zv)
        X = np.full_like(Y, x_mid)
        ax.plot_wireframe(X, Y, Z, color="blue", linewidth=0.4, alpha=0.3)
        ax.scatter([knife_pos[0]], [knife_pos[1]], [knife_pos[2]], marker="X",
                    s=200, color="black", edgecolor="white", linewidth=1.5, zorder=10)
        ax.set_xlim(cx[0] - 0.6 * size_box, cx[0] + 0.6 * size_box)
        ax.set_ylim(cx[1] - 0.6 * size_box, cx[1] + 0.6 * size_box)
        ax.set_zlim(cx[2] - 0.6 * size_box, cx[2] + 0.6 * size_box)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        cav = float(dead.mean())
        ax.set_title(f"{label}  ·  surface cav={cav*100:.1f}%", fontsize=10)
    fig.suptitle(f"{mesh_name}  ·  frame {t+1}/{T}  ·  Left/Right heterogeneous STIFFNESS",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=85, bbox_inches="tight")
    plt.close(fig)


def run_one_mesh(cfg, device, T=30):
    print(f"\n{'='*78}\n{cfg['name']}  Phase 1 (STIFF): L/R contrast viz\n{'='*78}", flush=True)
    solver, _, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    pos_rest = mesh_info["vertices"]
    N = solver.N
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    top_axis = cfg["top_axis"]

    stiff_A, lr_mask, x_mid = build_lr_stiff(pos_rest, STIFF_LOW, STIFF_HIGH)
    stiff_B, _, _ = build_lr_stiff(pos_rest, STIFF_HIGH, STIFF_LOW)
    print(f"  N={N}  x_mid={x_mid:.4f}  left={int(lr_mask.sum())} right={N - int(lr_mask.sum())}", flush=True)

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold

    spec = dict(cfg["traj_specs"]["A"])
    spec["sweep_range"] = (0.10, 0.90)
    traj = make_traj_from_spec(spec, bbox_min, bbox_max, T)

    print(f"  Config A: stiff_left={STIFF_LOW} (soft), stiff_right={STIFF_HIGH} (tough)", flush=True)
    pos_A, alpha_A = forward_one_config(cfg, stiff_A, traj, device)
    print(f"  Config B: stiff_left={STIFF_HIGH} (tough), stiff_right={STIFF_LOW} (soft)", flush=True)
    pos_B, alpha_B = forward_one_config(cfg, stiff_B, traj, device)

    final_dead_A = (alpha_A[-1] < 0.5)
    final_dead_B = (alpha_B[-1] < 0.5)
    a_left_A = int((final_dead_A & surface_mask & lr_mask).sum())
    a_right_A = int((final_dead_A & surface_mask & ~lr_mask).sum())
    asym_A = (a_left_A - a_right_A) / max(a_left_A + a_right_A, 1)
    a_left_B = int((final_dead_B & surface_mask & lr_mask).sum())
    a_right_B = int((final_dead_B & surface_mask & ~lr_mask).sum())
    asym_B = (a_left_B - a_right_B) / max(a_left_B + a_right_B, 1)
    print(f"  Cavity asymmetry A={asym_A:+.3f} (expect +, soft-left more cavity)  B={asym_B:+.3f} (expect -)")
    p_diff = float(np.linalg.norm(pos_A[-1][surface_mask] - pos_B[-1][surface_mask], axis=1).mean() * 1000)
    a_diff = float((alpha_A[-1][surface_mask] - alpha_B[-1][surface_mask]).__abs__().mean())
    print(f"  Surface obs diff (A vs B) at final: pos {p_diff:.3f} mm  alpha {a_diff:.4f}", flush=True)

    out_dir = REPO / "tools" / "pointnetpp" / f"phase1_stiff_{cfg['name']}"
    out_dir.mkdir(exist_ok=True)
    for t in range(T):
        render_2panel_3d(
            pos_A[t], alpha_A[t], f"Config A: stiff_L={STIFF_LOW}, stiff_R={STIFF_HIGH}",
            pos_B[t], alpha_B[t], f"Config B: stiff_L={STIFF_HIGH}, stiff_R={STIFF_LOW}",
            traj[t], bbox_min, bbox_max, x_mid,
            t, T, cfg["name"], surface_mask, out_dir / f"frame_{t:02d}.png")
    frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T)]
    gif_path = REPO / "tools" / "pointnetpp" / f"phase1_stiff_{cfg['name']}.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=200, loop=0)
    print(f"  saved {gif_path}", flush=True)
    return dict(asym_A=asym_A, asym_B=asym_B, p_diff_mm=p_diff, a_diff=a_diff)


def main():
    device = "cuda:0"
    t0 = time.time()
    results = {}
    USED = ["M1_cube", "M2_tumor"]
    for cfg in MESH_CONFIGS:
        if cfg["name"] not in USED: continue
        try:
            r = run_one_mesh(cfg, device, T=30)
            results[cfg["name"]] = r
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}"); traceback.print_exc()
    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    print(f"{'mesh':<12} {'asym_A':>10} {'asym_B':>10} {'pos_diff_mm':>12} {'alpha_diff':>11}")
    for m, r in results.items():
        print(f"{m:<12} {r['asym_A']:>+10.3f} {r['asym_B']:>+10.3f} {r['p_diff_mm']:>12.3f} {r['a_diff']:>11.4f}")
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
