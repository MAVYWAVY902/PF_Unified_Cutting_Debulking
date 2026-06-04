"""Hier vs Liang with LARGE inclusion (~15% of mesh) — testing if hier
visibility scales with inclusion size.

Previous tests used 0.79% inclusion → hier impact diluted across 21745 particles.
With ~15% inclusion (~3000 particles), hier's stiff differences should be
amplified ~20x in aggregate metrics.

Setup:
   M1_cube, num_iter=16 (high sensitivity)
   Inclusion radius = 0.10 m (~3-4x larger than original 0.04)
   T_calib=10, T_predict=30
   Same calibration pipeline

If hier vs Liang visible now → hier IS contribution for substantial inclusions
If still indistinguishable → hier truly not visible at any inclusion scale
"""

import sys
import time
import math
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
from PIL import Image  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff,
)
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, predict_on_traj  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402
from tools.pointnetpp.viz_animation_long import (  # noqa: E402
    make_extended_traj_x_sweep, render_2view_frame,
)


def main():
    cfg = MESH_CONFIGS[0]
    device = "cuda:0"
    T_calib = 10
    T_extended = 30
    N_epochs = 60
    lambda_E = 1e-5
    stiff_in_truth = 0.8
    NUM_ITER = 16
    INCL_RADIUS_LARGE = 0.10  # ★ 2.5x larger than original 0.04

    print(f"\n=== Large inclusion test (radius={INCL_RADIUS_LARGE}, num_iter={NUM_ITER}) ===")
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = NUM_ITER
    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=stiff_in_truth, stiff_out=0.3,
        center_override=tuple(incl_center), radius_override=INCL_RADIUS_LARGE)
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)
    top_axis = cfg["top_axis"]
    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    n_incl = int(inclusion_mask.sum())
    print(f"  N={N}  inclusion {n_incl}/{N} ({n_incl/N*100:.2f}%)  surface {int(surface_mask.sum())}/{N}")

    theta_truth = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    from tools.pointnetpp.multi_mesh_test import make_traj_from_spec
    traj_calib = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T_calib)
    solver.reset_to_rest(pos_rest)
    print(f"\n[Truth obs T_calib={T_calib} with inclusion ~{n_incl/N*100:.0f}%]", flush=True)
    obs_calib_pos, obs_calib_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, traj_calib, knife_radius, device,
        tau_truth_np=tau_truth)
    cav_truth_calib = float((obs_calib_alpha[-1] < 0.5).mean())
    print(f"  Truth calib cavity: {cav_truth_calib*100:.2f}%")

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_calib_pos]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_calib_alpha]
    surface_idx = torch.tensor(np.where(surface_mask)[0], dtype=torch.long, device=device)
    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64), dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)
    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    H = build_hierarchy(x_rest, M1=512, M2=64, k_neighbors=3)
    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=traj_calib, knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth, theta_3d_np=theta_truth,
        stiff_init_np=stiff_init, stiff_truth=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask,
        surface_idx=surface_idx, surface_mask=surface_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H,
    )

    print(f"\n[Calibrate Liang]", flush=True)
    t0 = time.time()
    stiff_L, tau_L, _ = calibrate("Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                     lambda_smooth=0.0, lambda_E=lambda_E)
    print(f"  done {time.time()-t0:.0f}s  Liang stiff_in_int = {stiff_L[inclusion_mask & ~surface_mask].mean():.3f}")

    print(f"\n[Calibrate Hier+Liang]", flush=True)
    t0 = time.time()
    stiff_H, tau_H, _ = calibrate("Hier+Liang_inner", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                     lambda_smooth=0.0, lambda_E=lambda_E)
    print(f"  done {time.time()-t0:.0f}s  Hier stiff_in_int = {stiff_H[inclusion_mask & ~surface_mask].mean():.3f}")

    print(f"\n[Stiff field comparison]")
    diff_stiff = stiff_H - stiff_L
    print(f"  Per-particle Hier - Liang stiff diff:")
    print(f"    mean abs = {np.abs(diff_stiff).mean():.4f}")
    print(f"    Inside incl: Liang {stiff_L[inclusion_mask].mean():.3f}  Hier {stiff_H[inclusion_mask].mean():.3f}  Δ {(stiff_H-stiff_L)[inclusion_mask].mean():+.3f}")
    print(f"    Outside:     Liang {stiff_L[~inclusion_mask].mean():.3f}  Hier {stiff_H[~inclusion_mask].mean():.3f}  Δ {(stiff_H-stiff_L)[~inclusion_mask].mean():+.3f}")

    # Extended trajectory
    traj_long = make_extended_traj_x_sweep(bbox_min, bbox_max, T_extended,
                                             y_frac=1.0, z_frac=0.5,
                                             x_start_frac=0.10, x_end_frac=0.90)

    print(f"\n[Forward T={T_extended}]", flush=True)
    solver.reset_to_rest(pos_rest)
    truth_pos, truth_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth, stiff_truth, traj_long, knife_radius, device,
        tau_truth_np=tau_truth)
    pred_L_pos, pred_L_alpha = predict_on_traj(stiff_L, tau_L, traj_long, ctx)
    pred_H_pos, pred_H_alpha = predict_on_traj(stiff_H, tau_H, traj_long, ctx)
    cav_truth_long = [float((a < 0.5).mean()) for a in truth_alpha]
    cav_L = [float((a < 0.5).mean()) for a in pred_L_alpha]
    cav_H = [float((a < 0.5).mean()) for a in pred_H_alpha]
    print(f"  Final cavity: Truth={cav_truth_long[-1]*100:.2f}% Liang={cav_L[-1]*100:.2f}% Hier={cav_H[-1]*100:.2f}%")

    # Pred L vs H diff
    diff_LH = np.linalg.norm(pred_L_pos[-1] - pred_H_pos[-1], axis=1)
    print(f"\n[pred_L vs pred_H at frame 29]")
    print(f"  mean diff = {diff_LH.mean():.3e} m  ({diff_LH.mean()*1000:.3f} mm)")
    print(f"  max diff  = {diff_LH.max():.3e} m  ({diff_LH.max()*1000:.3f} mm)")
    print(f"  perception threshold ~1e-4 m (100 microns)")
    print(f"  mean / threshold ratio = {diff_LH.mean()/1e-4:.1f}× ({'visible' if diff_LH.mean() > 1e-4 else 'subtle'})")

    # Render
    out_dir = REPO / "tools" / "pointnetpp" / "viz_large_incl"
    out_dir.mkdir(exist_ok=True)
    print(f"\n[Render {T_extended} frames]")
    for t in range(T_extended):
        out_path = out_dir / f"frame_{t:02d}.png"
        render_2view_frame(
            [truth_pos[t], pred_L_pos[t], pred_H_pos[t]],
            [truth_alpha[t], pred_L_alpha[t], pred_H_alpha[t]],
            ["Truth", "Liang", "Hier+Liang"],
            knife_pos=traj_long[t],
            bbox_min=bbox_min, bbox_max=bbox_max,
            incl_center=incl_center, incl_radius=INCL_RADIUS_LARGE,
            frame_t=t, T_total=T_extended,
            cavity_per_frame=[cav_truth_long, cav_L, cav_H],
            out_path=out_path,
        )
        if t % 5 == 0:
            print(f"  rendered {t}", flush=True)

    frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T_extended)]
    gif_path = REPO / "tools" / "pointnetpp" / "viz_large_incl.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=250, loop=0)
    print(f"  saved {gif_path}")

    print(f"\n[Cavity progression]")
    print(f"  Frame    Truth     Liang     Hier      L-Truth     H-Truth     Hier-Liang")
    for t in [0, 5, 10, 15, 20, 25, T_extended - 1]:
        print(f"  {t:5d}   {cav_truth_long[t]*100:5.2f}%   {cav_L[t]*100:5.2f}%   "
              f"{cav_H[t]*100:5.2f}%   {(cav_L[t]-cav_truth_long[t])*100:+8.2f}%  "
              f"{(cav_H[t]-cav_truth_long[t])*100:+8.2f}%  "
              f"{(cav_H[t]-cav_L[t])*100:+8.2f}%")


if __name__ == "__main__":
    main()
