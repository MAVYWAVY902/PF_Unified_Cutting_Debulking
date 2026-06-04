"""Phase 1: Left/right heterogeneous tau visual contrast.

Setup:
   - Each mesh split along X axis (left half vs right half)
   - Config A: tau_left=0.15, tau_right=0.55 (left soft, right tough)
   - Config B: tau_left=0.55, tau_right=0.15 (mirror)
   - Same stiff = 0.5 everywhere (decouple stiff from tau contrast)
   - Knife sweeps along X top surface, partially embedded (center at mesh top)
   - Same trajectory both configs

Render 2-panel 3D perspective per frame. Check:
   - Do A and B produce visibly different cavity shape?
   - Is cavity asymmetry consistent with left/right tau pattern?

If yes → proceed to Phase 2 (calibration). If too similar → increase contrast.
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


# Left/right tau parameters — large contrast for visibility
TAU_HIGH = 0.55   # tough side
TAU_LOW = 0.15    # soft side
STIFF_UNIFORM = 0.5

CONFIGS = [
    ("A_soft_left", TAU_LOW, TAU_HIGH),    # tau_left=0.15, tau_right=0.55
    ("B_soft_right", TAU_HIGH, TAU_LOW),   # tau_left=0.55, tau_right=0.15
]


def build_lr_truth(pos_rest, tau_left, tau_right):
    """Build left/right asymmetric tau field. Split along X axis at midpoint."""
    N = pos_rest.shape[0]
    x_mid = 0.5 * (pos_rest[:, 0].min() + pos_rest[:, 0].max())
    left_mask = pos_rest[:, 0] < x_mid
    tau_truth = np.where(left_mask, tau_left, tau_right).astype(np.float32)
    return tau_truth, left_mask


def forward_one_config(cfg, tau_truth, traj, device):
    """Run forward sim with given tau field. Returns positions + alpha per frame."""
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.95
    solver.num_iterations = 2
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)
    pos_rest = mesh_info["vertices"]
    solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))

    stiff_uniform = np.full(N, STIFF_UNIFORM, dtype=np.float32)
    theta = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    solver.reset_to_rest(pos_rest)
    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta, stiff_uniform, traj,
        cfg["knife_radius_abs"], device, tau_truth_np=tau_truth)
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
        # X-midplane wireframe (left/right boundary)
        yv = np.linspace(bbox_min[1], bbox_max[1], 8)
        zv = np.linspace(bbox_min[2], bbox_max[2], 8)
        Y, Z = np.meshgrid(yv, zv)
        X = np.full_like(Y, x_mid)
        ax.plot_wireframe(X, Y, Z, color="blue", linewidth=0.4, alpha=0.3)
        # Knife
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
    fig.suptitle(f"{mesh_name}  ·  frame {t+1}/{T}  ·  Left/Right heterogeneous τ  "
                  f"(blue plane = left/right boundary)",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=85, bbox_inches="tight")
    plt.close(fig)


def run_one_mesh(cfg, device, T=30):
    print(f"\n{'='*78}\n{cfg['name']}  Phase 1: L/R tau contrast viz\n{'='*78}", flush=True)
    solver, _, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    pos_rest = mesh_info["vertices"]
    N = solver.N
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    top_axis = cfg["top_axis"]

    # Build left/right truth for A and B
    tau_A, lr_mask = build_lr_truth(pos_rest, TAU_LOW, TAU_HIGH)
    tau_B, _ = build_lr_truth(pos_rest, TAU_HIGH, TAU_LOW)
    x_mid = 0.5 * (pos_rest[:, 0].min() + pos_rest[:, 0].max())
    print(f"  N={N}  bbox={bbox_size}  x_mid={x_mid:.4f}  left={int(lr_mask.sum())} right={N - int(lr_mask.sum())}", flush=True)

    # Surface mask (top 10% along top_axis)
    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold

    # Knife trajectory — sweep across X at top of mesh, partially embedded.
    # Standard spec "A" sweeps narrow range; extend to 10%-90% to traverse both halves.
    spec = dict(cfg["traj_specs"]["A"])
    spec["sweep_range"] = (0.10, 0.90)
    traj = make_traj_from_spec(spec, bbox_min, bbox_max, T)

    print(f"  Config A: tau_left={TAU_LOW}, tau_right={TAU_HIGH} (soft left)", flush=True)
    pos_A, alpha_A = forward_one_config(cfg, tau_A, traj, device)
    print(f"  Config B: tau_left={TAU_HIGH}, tau_right={TAU_LOW} (soft right)", flush=True)
    pos_B, alpha_B = forward_one_config(cfg, tau_B, traj, device)

    # Cavity statistics
    final_dead_A = (alpha_A[-1] < 0.5)
    final_dead_B = (alpha_B[-1] < 0.5)
    print(f"\n  Final cavity stats (frame {T}):")
    print(f"    A surface_dead={int((final_dead_A & surface_mask).sum())}, "
           f"  left surface_dead={int((final_dead_A & surface_mask & lr_mask).sum())},  "
           f"right surface_dead={int((final_dead_A & surface_mask & ~lr_mask).sum())}")
    print(f"    B surface_dead={int((final_dead_B & surface_mask).sum())}, "
           f"  left surface_dead={int((final_dead_B & surface_mask & lr_mask).sum())},  "
           f"right surface_dead={int((final_dead_B & surface_mask & ~lr_mask).sum())}")
    # Asymmetry index
    a_left_A = int((final_dead_A & surface_mask & lr_mask).sum())
    a_right_A = int((final_dead_A & surface_mask & ~lr_mask).sum())
    asym_A = (a_left_A - a_right_A) / max(a_left_A + a_right_A, 1)
    a_left_B = int((final_dead_B & surface_mask & lr_mask).sum())
    a_right_B = int((final_dead_B & surface_mask & ~lr_mask).sum())
    asym_B = (a_left_B - a_right_B) / max(a_left_B + a_right_B, 1)
    print(f"    A asymmetry index (L-R)/(L+R) = {asym_A:+.3f}  (positive = more cavity on soft-left)")
    print(f"    B asymmetry index             = {asym_B:+.3f}  (positive = more cavity on soft-left, expect NEGATIVE since B has soft-right)")

    # Surface-only obs difference between A and B at last frame
    a_diff = float((alpha_A[-1][surface_mask] - alpha_B[-1][surface_mask]).__abs__().mean())
    p_diff = float(np.linalg.norm(pos_A[-1][surface_mask] - pos_B[-1][surface_mask], axis=1).mean() * 1000)
    print(f"    Surface obs difference (A vs B) at final frame:")
    print(f"      pos mean L2 = {p_diff:.3f} mm    alpha mean abs = {a_diff:.4f}")

    # Render frames
    out_dir = REPO / "tools" / "pointnetpp" / f"phase1_lr_{cfg['name']}"
    out_dir.mkdir(exist_ok=True)
    for t in range(T):
        render_2panel_3d(
            pos_A[t], alpha_A[t], f"Config A: τ_L={TAU_LOW}, τ_R={TAU_HIGH}",
            pos_B[t], alpha_B[t], f"Config B: τ_L={TAU_HIGH}, τ_R={TAU_LOW}",
            traj[t], bbox_min, bbox_max, x_mid,
            t, T, cfg["name"], surface_mask, out_dir / f"frame_{t:02d}.png")
    frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T)]
    gif_path = REPO / "tools" / "pointnetpp" / f"phase1_lr_{cfg['name']}.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=200, loop=0)
    print(f"  saved {gif_path}", flush=True)
    return dict(asym_A=asym_A, asym_B=asym_B, p_diff_mm=p_diff, a_diff=a_diff)


def main():
    device = "cuda:0"
    t0 = time.time()
    results = {}
    for cfg in MESH_CONFIGS:
        try:
            r = run_one_mesh(cfg, device, T=30)
            results[cfg["name"]] = r
        except Exception as e:
            import traceback
            print(f"  {cfg['name']} FAILED: {e}"); traceback.print_exc()
    print(f"\n{'='*78}\nSUMMARY — visibility of L/R contrast\n{'='*78}")
    print(f"{'mesh':<12} {'asym_A':>10} {'asym_B':>10} {'pos_diff_mm':>12} {'alpha_diff':>11}")
    for m, r in results.items():
        print(f"{m:<12} {r['asym_A']:>+10.3f} {r['asym_B']:>+10.3f} {r['p_diff_mm']:>12.3f} {r['a_diff']:>11.4f}")
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
