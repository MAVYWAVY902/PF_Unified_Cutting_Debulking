"""Phase 2: Calibration experiment.

Pipeline per mesh (cube + tumor):
   1. Generate fake-real obs using Config A (tau_left=0.15, tau_right=0.55)
   2. Initialize sim with uniform tau=0.30 (ignorant baseline)
   3. Calibrate Liang_inner + Hier+Liang_inner using surface obs
   4. Forward sim on HELD-OUT trajectory (different z lane than calib) with:
        - fake-real (truth Config A)
        - sim_init (uniform 0.30, NO calibration)
        - sim_Liang (Liang-calibrated)
        - sim_Hier (Hier-calibrated)
   5. Render 4-panel 3D perspective video
   6. Quantitative: sim-real surface overlap pre vs post

Validation:
   - sim_Liang and sim_Hier on T_held should be CLOSER to fake-real than sim_init
   - Quantitative drop in pos_MSE and rise in cav_IoU pre→post
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
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402
from tools.pointnetpp.eval_hier_phantom_realistic import (  # noqa: E402
    predict_on_traj, surface_overlap_metrics,
)


TAU_LEFT = 0.15   # fake-real (Config A)
TAU_RIGHT = 0.55
TAU_INIT = 0.30   # sim ignorant baseline
STIFF_UNIFORM = 0.5


def build_lr_tau(pos_rest, tau_left, tau_right):
    x_mid = 0.5 * (pos_rest[:, 0].min() + pos_rest[:, 0].max())
    left_mask = pos_rest[:, 0] < x_mid
    tau = np.where(left_mask, tau_left, tau_right).astype(np.float32)
    return tau, left_mask, x_mid


def render_4panel_3d(panels_data, knife_pos, bbox_min, bbox_max, x_mid,
                      t, T, mesh_name, surface_mask, out_path):
    """panels_data: list of (pos, alpha, label, metric_line) of length 4."""
    fig = plt.figure(figsize=(22, 6.5))
    azim = 30.0 + (t / max(T - 1, 1)) * 60.0
    elev = 22.0
    size_box = (bbox_max - bbox_min).max()
    cx = 0.5 * (bbox_min + bbox_max)
    for j, (pos, alpha, label, metric) in enumerate(panels_data):
        ax = fig.add_subplot(1, 4, j + 1, projection="3d")
        sm = surface_mask
        a = alpha[sm]; p = pos[sm]
        alive = a > 0.5; dead = ~alive
        ax.scatter(p[alive, 0], p[alive, 1], p[alive, 2],
                    c="lightgray", s=8, edgecolors="none", alpha=0.25, depthshade=True)
        ax.scatter(p[dead, 0], p[dead, 1], p[dead, 2],
                    c="red", s=30, edgecolors="darkred", linewidth=0.4,
                    alpha=0.95, depthshade=True)
        # Left/right boundary plane
        yv = np.linspace(bbox_min[1], bbox_max[1], 6)
        zv = np.linspace(bbox_min[2], bbox_max[2], 6)
        Y, Z = np.meshgrid(yv, zv)
        X = np.full_like(Y, x_mid)
        ax.plot_wireframe(X, Y, Z, color="blue", linewidth=0.4, alpha=0.3)
        # Knife
        ax.scatter([knife_pos[0]], [knife_pos[1]], [knife_pos[2]], marker="X",
                    s=180, color="black", edgecolor="white", linewidth=1.5, zorder=10)
        ax.set_xlim(cx[0] - 0.6 * size_box, cx[0] + 0.6 * size_box)
        ax.set_ylim(cx[1] - 0.6 * size_box, cx[1] + 0.6 * size_box)
        ax.set_zlim(cx[2] - 0.6 * size_box, cx[2] + 0.6 * size_box)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        cav = float(dead.mean())
        title = f"{label}\ncav={cav*100:.1f}%"
        if metric:
            title += f"\n{metric}"
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{mesh_name}  ·  frame {t+1}/{T}  ·  HELD-OUT trajectory  ·  "
                  f"Calibration: sim → fake-real",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=85, bbox_inches="tight")
    plt.close(fig)


def run_one_mesh(cfg, device, T_calib=15, T_held=30, N_epochs=30):
    print(f"\n{'='*80}\n{cfg['name']}  Phase 2: Calibration\n{'='*80}", flush=True)
    # v10 (2026-06-01): strong anchor (20% fix) + warm-up settle.
    # B — fix_threshold_fraction 0.05 → 0.20 (4× thicker anchor layer)
    solver, rest_state, mesh_info = setup_solver_with_rest(
        str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
    N = solver.N
    solver.damping = 0.95
    solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    # v9 (2026-06-01): per-mesh gravity aligned with top_axis.
    # cube/tumor top=Y → gravity=(0,-9.81,0), Y-down.
    # torus top=Z → gravity=(0,0,-9.81), Z-down.
    # fix_bottom uses Y axis (in day1_solver_smoke) — aligned with cube/tumor.
    if cfg["top_axis"] == 1:  # Y up
        solver.gravity = wp.vec3(0.0, -9.81, 0.0)
        # Floor in Z direction (existing floor_y is actually z-axis check)
        # Set far below mesh so no Z floor collision interfering with Y gravity
        solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    else:  # top_axis = 2, Z up (torus)
        solver.gravity = wp.vec3(0.0, 0.0, -9.81)
        solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))

    # v10 — A: WARM-UP SETTLE
    # Real phantom on table 已 经 在 gravity 下 equilibrated. Sim 从 rest pos
    # 出 发 立 即 free-fall artifacts. 这 里 跑 60 frames 没 knife 让 mesh settle.
    # Settled pos 作 为 新 "rest" 用 于 calibration.
    knife_radius = cfg["knife_radius_abs"]   # need this before warm-up
    n_warmup = 60
    print(f"  Warm-up settle ({n_warmup} frames, gravity only, knife far) ...", flush=True)
    solver.reset_to_rest(pos_rest)
    far_knife = pos_rest.max(0) + 10.0
    stiff_warm = np.full(N, STIFF_UNIFORM, dtype=np.float32)
    tau_warm = np.full(N, TAU_INIT, dtype=np.float32)
    theta_warm = np.array([3.0, 20.0, 0.3], dtype=np.float32)
    theta_warm_t = torch.tensor(theta_warm, dtype=torch.float32, device=device)
    stiff_warm_t = torch.tensor(stiff_warm, dtype=torch.float32, device=device)
    tau_warm_t = torch.tensor(tau_warm, dtype=torch.float32, device=device)
    state_warm = rest_state
    with torch.no_grad():
        for _ in range(n_warmup):
            alpha_w, pos_w, dv_w = WarpFullForwardOp.apply(
                theta_warm_t, solver, state_warm, far_knife, knife_radius,
                stiff_warm_t, tau_warm_t)
            state_warm = solver_save_state(solver)
    pos_settled = pos_w.cpu().numpy().copy()
    sag = float(np.linalg.norm(pos_settled - pos_rest, axis=1).max() * 1000)
    sag_mean = float(np.linalg.norm(pos_settled - pos_rest, axis=1).mean() * 1000)
    print(f"  Warm-up done: max sag={sag:.2f}mm  mean sag={sag_mean:.3f}mm", flush=True)
    # Replace rest pos + rest_state with settled
    pos_rest = pos_settled   # all subsequent calls use settled positions
    mesh_info["vertices"] = pos_settled
    # Pre-warmed state_warm has vel ≈ 0 (in equilibrium under gravity, damping killed velocity)
    rest_state = state_warm

    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    top_axis = cfg["top_axis"]

    # Truth (fake-real) tau: Config A
    tau_truth, lr_mask, x_mid = build_lr_tau(pos_rest, TAU_LEFT, TAU_RIGHT)
    stiff_uniform = np.full(N, STIFF_UNIFORM, dtype=np.float32)
    print(f"  N={N}  x_mid={x_mid:.4f}  left={int(lr_mask.sum())} right={N - int(lr_mask.sum())}", flush=True)

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]

    # v8 (2026-06-01): MULTI-TRAJECTORY calibration to escape local minima.
    # Calibrate on TWO trajectories: T_A (z=0.5 default lane) AND T_A2 (z=0.7).
    # Different cut lanes → different cavity patterns → tau identifiable globally.
    # Held-out still at z=0.25 (independent of both calib lanes).
    spec_calib = dict(cfg["traj_specs"]["A"])
    spec_calib["sweep_range"] = (0.10, 0.90)
    traj_calib = make_traj_from_spec(spec_calib, bbox_min, bbox_max, T_calib)

    # 2nd calibration trajectory: same sweep_range, but different non-top axis lane
    spec_calib2 = dict(cfg["traj_specs"]["A"])
    spec_calib2["sweep_range"] = (0.10, 0.90)
    fixed_calib2 = dict(spec_calib2["fixed"])
    other_axes_c = [i for i in range(3) if i != spec_calib2["sweep_axis"]]
    for ax_idx in other_axes_c:
        if ax_idx != top_axis:
            old_frac = fixed_calib2[ax_idx]
            fixed_calib2[ax_idx] = max(0.20, min(0.80, old_frac + 0.20))  # +0.2 lane shift
            break
    spec_calib2["fixed"] = fixed_calib2
    traj_calib2 = make_traj_from_spec(spec_calib2, bbox_min, bbox_max, T_calib)

    # Held-out trajectory: same X sweep but z=0.25 (different lane from both calibs)
    spec_held = dict(cfg["traj_specs"]["A"])
    spec_held["sweep_range"] = (0.10, 0.90)
    fixed_held = dict(spec_held["fixed"])
    # Shift z fixed value
    other_axes = [i for i in range(3) if i != spec_held["sweep_axis"]]
    # Choose the non-top axis as "lane" to shift
    for ax_idx in other_axes:
        if ax_idx != top_axis:
            old_frac = fixed_held[ax_idx]
            fixed_held[ax_idx] = max(0.20, min(0.80, old_frac - 0.25))   # shift by 0.25
            break
    spec_held["fixed"] = fixed_held
    traj_held = make_traj_from_spec(spec_held, bbox_min, bbox_max, T_held)

    # 2026-06-01 Option A: lower sharpness s=100→20 globally to make alpha
    # continuous and unlock gradient flow through ∂α/∂τ = -s·α(1-α).
    # At s=100 the sigmoid was binary, gradient ≈ 0 except in narrow cut moment.
    # At s=20 transition zone is 5× wider → 5× more frames carry signal.
    theta_truth_np = np.array([3.0, 20.0, 0.3], dtype=np.float32)

    # Generate fake-real on ALL trajectories (calib_1, calib_2, held)
    solver.reset_to_rest(pos_rest)
    obs_pos_calib, obs_alpha_calib = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_uniform, traj_calib,
        knife_radius, device, tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    obs_pos_calib2, obs_alpha_calib2 = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_uniform, traj_calib2,
        knife_radius, device, tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    obs_pos_held, obs_alpha_held = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_uniform, traj_held,
        knife_radius, device, tau_truth_np=tau_truth)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_calib]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha_calib]
    obs_pos_C_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_calib2]
    obs_alpha_C_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha_calib2]
    multi_traj = [(traj_calib, obs_pos_A_t, obs_alpha_A_t),
                  (traj_calib2, obs_pos_C_t, obs_alpha_C_t)]
    surface_idx = torch.tensor(surface_idx_np, dtype=torch.long, device=device)

    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edges_a_t = torch.tensor(edges_a_np, dtype=torch.long, device=device)
    edges_b_t = torch.tensor(edges_b_np, dtype=torch.long, device=device)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    M1 = min(512, max(32, N // 20))
    M2 = min(64, max(8, N // 200))
    H_hier = build_hierarchy(x_rest, M1=M1, M2=M2, k_neighbors=3)

    stiff_init = np.full(N, STIFF_UNIFORM, dtype=np.float32)
    tau_init = np.full(N, TAU_INIT, dtype=np.float32)

    # For sim_init forward, "inclusion_mask" + "stiff_truth" in ctx are only used by
    # calibrate's metrics path; predict_on_traj doesn't use them.
    inclusion_mask_dummy = lr_mask   # ANY mask works as placeholder

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=traj_calib, knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth_np, theta_3d_np=theta_truth_np,
        stiff_init_np=stiff_init, stiff_truth=stiff_uniform,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask_dummy,
        surface_idx=surface_idx, surface_mask=surface_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H_hier,
    )

    # sim_init forward on held-out
    print(f"  Forward sim_init (uniform τ=0.30) on held-out ...", flush=True)
    init_pos, init_alpha = predict_on_traj(stiff_init, tau_init, traj_held, ctx)

    # Calibrate Liang
    print(f"  Calibrate Liang_inner ({N_epochs} epochs) ...", flush=True)
    t0 = time.time()
    # v5 (2026-06-01): direct outer loss L_pos + lambda_area × L_cav_outer
    # Bypasses inner GD complexity; direct alpha gradient to tau.
    _, tau_L, _ = calibrate("L_pos_partial", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                lambda_smooth=0.0, lambda_E=1e-5, lambda_outer_cav=1.0,
                                lr_schedule=[(30, 0.2), (60, 0.2)],
                                return_best=True, multi_traj_obs=multi_traj)
    print(f"    done {time.time()-t0:.0f}s  "
           f"tau_L_left={tau_L[lr_mask].mean():.3f} (truth {TAU_LEFT})  "
           f"tau_L_right={tau_L[~lr_mask].mean():.3f} (truth {TAU_RIGHT})", flush=True)

    # Calibrate Hier
    print(f"  Calibrate Hier+Liang_inner ({N_epochs} epochs) ...", flush=True)
    t0 = time.time()
    _, tau_H, _ = calibrate("Hier+L_pos_partial", ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.5,
                                lambda_smooth=0.0, lambda_E=1e-5, lambda_outer_cav=1.0,
                                lr_schedule=[(30, 0.2), (60, 0.2)],
                                return_best=True, multi_traj_obs=multi_traj)
    print(f"    done {time.time()-t0:.0f}s  "
           f"tau_H_left={tau_H[lr_mask].mean():.3f} (truth {TAU_LEFT})  "
           f"tau_H_right={tau_H[~lr_mask].mean():.3f} (truth {TAU_RIGHT})", flush=True)

    # Forward Liang + Hier on held-out
    print(f"  Forward sim_Liang on held-out ...", flush=True)
    liang_pos, liang_alpha = predict_on_traj(stiff_init, tau_L, traj_held, ctx)
    print(f"  Forward sim_Hier on held-out ...", flush=True)
    hier_pos, hier_alpha = predict_on_traj(stiff_init, tau_H, traj_held, ctx)

    # Sim-real overlap metrics on held-out
    sr_init = surface_overlap_metrics(init_pos, obs_pos_held, init_alpha, obs_alpha_held, surface_idx_np)
    sr_L = surface_overlap_metrics(liang_pos, obs_pos_held, liang_alpha, obs_alpha_held, surface_idx_np)
    sr_H = surface_overlap_metrics(hier_pos, obs_pos_held, hier_alpha, obs_alpha_held, surface_idx_np)
    print(f"\n  Sim-real overlap on HELD-OUT trajectory:")
    print(f"    sim_init  : pos {sr_init['pos_mm']:.3f}mm  cav_IoU {sr_init['cav_iou']:.3f}  α_MSE {sr_init['alpha_mse']:.4f}")
    print(f"    sim_Liang : pos {sr_L['pos_mm']:.3f}mm  cav_IoU {sr_L['cav_iou']:.3f}  α_MSE {sr_L['alpha_mse']:.4f}")
    print(f"    sim_Hier  : pos {sr_H['pos_mm']:.3f}mm  cav_IoU {sr_H['cav_iou']:.3f}  α_MSE {sr_H['alpha_mse']:.4f}")
    print(f"    improvement (sim_init vs Liang): pos {(sr_init['pos_mm']-sr_L['pos_mm'])/sr_init['pos_mm']*100:+.1f}%  cav_IoU {(sr_L['cav_iou']-sr_init['cav_iou']):+.3f}")
    print(f"    improvement (sim_init vs Hier):  pos {(sr_init['pos_mm']-sr_H['pos_mm'])/sr_init['pos_mm']*100:+.1f}%  cav_IoU {(sr_H['cav_iou']-sr_init['cav_iou']):+.3f}")

    # Render 4-panel video
    print(f"  Render {T_held} frames ...", flush=True)
    out_dir = REPO / "tools" / "pointnetpp" / f"phase2_v10_calib_{cfg['name']}"
    out_dir.mkdir(exist_ok=True)
    for t in range(T_held):
        panels = [
            (obs_pos_held[t], obs_alpha_held[t], "fake-real (Config A)", None),
            (init_pos[t], init_alpha[t], "sim_init (τ=0.30 uniform)",
             f"pos={sr_init['pos_mm']:.2f}mm  IoU={sr_init['cav_iou']:.2f}"),
            (liang_pos[t], liang_alpha[t], "sim_Liang (calibrated)",
             f"pos={sr_L['pos_mm']:.2f}mm  IoU={sr_L['cav_iou']:.2f}"),
            (hier_pos[t], hier_alpha[t], "sim_Hier (calibrated)",
             f"pos={sr_H['pos_mm']:.2f}mm  IoU={sr_H['cav_iou']:.2f}"),
        ]
        render_4panel_3d(panels, traj_held[t], bbox_min, bbox_max, x_mid,
                          t, T_held, cfg["name"], surface_mask,
                          out_dir / f"frame_{t:02d}.png")
    frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T_held)]
    gif_path = REPO / "tools" / "pointnetpp" / f"phase2_v10_calib_{cfg['name']}.gif"
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                      duration=200, loop=0)
    print(f"  saved {gif_path}", flush=True)
    return dict(sr_init=sr_init, sr_L=sr_L, sr_H=sr_H,
                 tau_L=tau_L, tau_H=tau_H, tau_truth=tau_truth)


def main():
    device = "cuda:0"
    t0 = time.time()
    USED_MESHES = ["M1_cube", "M2_tumor"]   # skip torus per user instruction
    for cfg in MESH_CONFIGS:
        if cfg["name"] not in USED_MESHES:
            continue
        try:
            run_one_mesh(cfg, device, T_calib=15, T_held=30, N_epochs=100)
        except Exception as e:
            import traceback
            print(f"  {cfg['name']} FAILED: {e}"); traceback.print_exc()
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
