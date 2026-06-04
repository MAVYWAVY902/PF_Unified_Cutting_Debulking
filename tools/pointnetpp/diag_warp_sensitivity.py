"""Warp forward sim sensitivity test — does it respect spatial stiff pattern?

Three forward sims with SAME knife trajectory but DIFFERENT stiff fields:
   A. Uniform soft   (stiff = 0.3 everywhere)
   B. Uniform stiff  (stiff = 1.5 everywhere)
   C. Truth pattern  (stiff = 0.3 outside, 1.5 inside inclusion sphere)
                      ★ 5x stiff contrast at inclusion

If A vs B vs C produce VISUALLY DIFFERENT deformation:
   → Warp forward sim respects stiff heterogeneity properly
   → Hier vs Liang's tiny 0.08 stiff diff truly is too small to matter
   → Conclusion holds: hier not contribution under our setup

If A vs B vs C look the same:
   → Warp forward sim ignores stiff or has sensitivity bug
   → Need to investigate solver implementation
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
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff,
)
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, predict_on_traj  # noqa: E402
from tools.pointnetpp.viz_animation_long import (  # noqa: E402
    make_extended_traj_x_sweep, render_2view_frame,
)


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T_extended = 30
    NUM_ITER = 16  # high sensitivity regime

    print(f"\n=== Warp sensitivity sanity test ===")
    print(f"  Three stiff fields with 5x contrast, num_iter={NUM_ITER}")
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = NUM_ITER
    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = cfg["incl_radius_abs"]

    # Build inclusion mask
    _, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=1.5, stiff_out=0.3,
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    print(f"  N={N}, inclusion {int(inclusion_mask.sum())}")

    # Three test stiff fields
    stiff_A = np.full(N, 0.3, dtype=np.float32)         # uniform soft
    stiff_B = np.full(N, 1.5, dtype=np.float32)         # uniform stiff
    stiff_C = np.where(inclusion_mask, 1.5, 0.3).astype(np.float32)  # truth pattern (5x contrast)

    # Build extended trajectory
    traj_long = make_extended_traj_x_sweep(bbox_min, bbox_max, T_extended,
                                             y_frac=1.0, z_frac=0.5,
                                             x_start_frac=0.10, x_end_frac=0.90)

    # Forward each
    ctx = dict(
        solver=solver, rest_state=rest_state,
        knife_radius=knife_radius, pos_rest=pos_rest,
        theta_3d_np=np.array([3.0, 100.0, 0.3], dtype=np.float32),
        device=device,
    )

    print(f"\n[Forward A: uniform 0.3]", flush=True)
    pos_A, alpha_A = predict_on_traj(stiff_A, tau_truth, traj_long, ctx)
    cav_A = [float((a < 0.5).mean()) for a in alpha_A]
    displ_A_final = np.linalg.norm(pos_A[-1] - pos_rest, axis=1)
    print(f"  cavity final {cav_A[-1]*100:.2f}%, mean displ {displ_A_final.mean():.3e}, max {displ_A_final.max():.3e}")

    print(f"\n[Forward B: uniform 1.5]", flush=True)
    pos_B, alpha_B = predict_on_traj(stiff_B, tau_truth, traj_long, ctx)
    cav_B = [float((a < 0.5).mean()) for a in alpha_B]
    displ_B_final = np.linalg.norm(pos_B[-1] - pos_rest, axis=1)
    print(f"  cavity final {cav_B[-1]*100:.2f}%, mean displ {displ_B_final.mean():.3e}, max {displ_B_final.max():.3e}")

    print(f"\n[Forward C: inclusion pattern 5x contrast]", flush=True)
    pos_C, alpha_C = predict_on_traj(stiff_C, tau_truth, traj_long, ctx)
    cav_C = [float((a < 0.5).mean()) for a in alpha_C]
    displ_C_final = np.linalg.norm(pos_C[-1] - pos_rest, axis=1)
    print(f"  cavity final {cav_C[-1]*100:.2f}%, mean displ {displ_C_final.mean():.3e}, max {displ_C_final.max():.3e}")

    # Pairwise position differences
    print(f"\n[Pairwise pos differences at final frame]")
    diff_AB = np.linalg.norm(pos_A[-1] - pos_B[-1], axis=1)
    diff_AC = np.linalg.norm(pos_A[-1] - pos_C[-1], axis=1)
    diff_BC = np.linalg.norm(pos_B[-1] - pos_C[-1], axis=1)
    print(f"  A (uniform 0.3) vs B (uniform 1.5):  mean {diff_AB.mean():.3e}, max {diff_AB.max():.3e}")
    print(f"  A (uniform 0.3) vs C (inclusion):    mean {diff_AC.mean():.3e}, max {diff_AC.max():.3e}")
    print(f"  B (uniform 1.5) vs C (inclusion):    mean {diff_BC.mean():.3e}, max {diff_BC.max():.3e}")

    # Visual perception threshold reminder
    print(f"\n  Visual perception threshold: ~100 microns = 1e-4 m")
    print(f"  A vs B mean diff / threshold: {diff_AB.mean() / 1e-4:.1f}× (visually {'visible' if diff_AB.mean() > 1e-4 else 'subtle'})")
    print(f"  A vs C mean diff / threshold: {diff_AC.mean() / 1e-4:.1f}× (visually {'visible' if diff_AC.mean() > 1e-4 else 'subtle'})")

    # === Render key frames ===
    out_dir = REPO / "tools" / "pointnetpp" / "viz_warp_sensitivity"
    out_dir.mkdir(exist_ok=True)
    print(f"\n[Render frames]")
    for t in range(T_extended):
        out_path = out_dir / f"frame_{t:02d}.png"
        render_2view_frame(
            [pos_A[t], pos_B[t], pos_C[t]],
            [alpha_A[t], alpha_B[t], alpha_C[t]],
            ["A: uniform 0.3 (soft)", "B: uniform 1.5 (stiff)", "C: inclusion 0.3/1.5 (5x)"],
            knife_pos=traj_long[t],
            bbox_min=bbox_min, bbox_max=bbox_max,
            incl_center=incl_center, incl_radius=incl_radius,
            frame_t=t, T_total=T_extended,
            cavity_per_frame=[cav_A, cav_B, cav_C],
            out_path=out_path,
        )
        if t % 5 == 0:
            print(f"  rendered frame {t}", flush=True)

    frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T_extended)]
    gif_path = REPO / "tools" / "pointnetpp" / "viz_warp_sensitivity.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=250, loop=0)
    print(f"  saved {gif_path}")

    # Cavity progression
    print(f"\n[Cavity progression]")
    print(f"  Frame    A (soft)   B (stiff)  C (incl)   B-A         C-A")
    for t in [0, 5, 10, 15, 20, 25, T_extended - 1]:
        print(f"  {t:5d}   {cav_A[t]*100:7.3f}%   {cav_B[t]*100:7.3f}%   {cav_C[t]*100:7.3f}%   "
              f"{(cav_B[t]-cav_A[t])*100:+8.3f}%   {(cav_C[t]-cav_A[t])*100:+8.3f}%")


if __name__ == "__main__":
    main()
