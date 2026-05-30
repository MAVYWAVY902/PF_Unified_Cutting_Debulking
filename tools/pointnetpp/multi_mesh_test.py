"""Multi-mesh robustness — does hierarchical's cross-trajectory advantage
hold beyond cube?

3 meshes:
   M1: cube_tet      N=21745  bbox=(0.3,0.3,0.3) m       — regular geometry
   M2: tumor_uniform N=25222  bbox=(0.031,0.018,0.024) m — organic shape, anatomical scale
   M3: torus         N=14886  bbox=(0.94,0.94,0.24) m    — non-convex topology with hole

Per-mesh setup is scaled to geometry: knife_radius, inclusion center & radius,
and trajectory range are all set as a fraction of the mesh bbox so that the
relative spatial scales match across meshes (e.g., inclusion ~20% of bbox extent).

For each mesh:
   - Build hierarchy on x_rest (FPS M1=512, M2=64)
   - Generate truth obs for 4 trajectories
   - Calibrate baseline + hierarchical on T_A only (15 epochs)
   - Predict each method on each trajectory
   - Report pos_err / alpha_err per (mesh, traj)

Aggregate: across 3 meshes × 3 held-out trajectories = 9 cells per metric.
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

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.day8_stable_inner_loop import gd_inner_loop_stable  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff, compute_metrics,
    inv_sigmoid_to_z, STIFF_LOW, STIFF_HIGH,
)
from tools.pointnetpp.hierarchy import build_hierarchy, reconstruct  # noqa: E402


# Per-mesh configurations — REDESIGNED 2026-05-29 after verify_cut_quality
# revealed previous settings caused EXCESSIVE cuts (tumor 50-69% cavity).
#
# Design principles:
#   1. knife_radius absolute, sized so half-sphere grazes top (not buried)
#   2. incl_radius small enough to leave room for surface scrape
#   3. Trajectories sweep in axes perpendicular to top_axis
#   4. T_C plunge limited to ≤ 10% of top_axis extent
#   5. Explicit top_axis per mesh (cube/tumor y up, torus z up)
MESH_CONFIGS = [
    dict(
        name="M1_cube",
        path="resources/cube/cube_tet.msh",
        top_axis=1,  # y up
        # bbox = (0.3, 0.3, 0.3) m; knife/incl as fraction of bbox dim
        knife_radius_abs=0.030,    # 10% of bbox dim
        incl_center_rel=(0.5, 0.80, 0.5),
        incl_radius_abs=0.040,     # 13% of bbox dim
        # Trajectory specs: sweep along one non-top axis; T_C is shallow plunge along top axis
        # (sweep_axis, sweep_range, fixed_positions, ...)
        traj_specs={
            "A": dict(sweep_axis=0, sweep_range=(0.42, 0.58), fixed={1: 1.00, 2: 0.50}),
            "B": dict(sweep_axis=2, sweep_range=(0.42, 0.58), fixed={0: 0.50, 1: 1.00}),
            "C": dict(sweep_axis=1, sweep_range=(1.00, 0.95), fixed={0: 0.50, 2: 0.50}),
            "D": dict(sweep_axis=0, sweep_range=(0.58, 0.74), fixed={1: 1.00, 2: 0.50}),
        },
        stiff_in=0.8, stiff_out=0.3,
        tau_in=0.20, tau_out=0.40,
    ),
    dict(
        name="M2_tumor",
        path="resources/tumor/tumor_uniform.msh",
        top_axis=1,  # y up
        # bbox = (0.031, 0.018, 0.024) m, smallest dim y=0.018
        knife_radius_abs=0.002,    # 11% of smallest y dim
        incl_center_rel=(0.5, 0.75, 0.5),
        incl_radius_abs=0.003,     # 17% of smallest y dim
        traj_specs={
            "A": dict(sweep_axis=0, sweep_range=(0.42, 0.58), fixed={1: 1.00, 2: 0.50}),
            "B": dict(sweep_axis=2, sweep_range=(0.42, 0.58), fixed={0: 0.50, 1: 1.00}),
            "C": dict(sweep_axis=1, sweep_range=(1.00, 0.95), fixed={0: 0.50, 2: 0.50}),
            "D": dict(sweep_axis=0, sweep_range=(0.58, 0.74), fixed={1: 1.00, 2: 0.50}),
        },
        stiff_in=0.8, stiff_out=0.3,
        tau_in=0.20, tau_out=0.40,
    ),
    dict(
        name="M3_torus",
        path="resources/torus/torus.msh",
        top_axis=2,  # z up (smallest dim 0.24)
        # bbox = (0.94, 0.94, 0.24) m; torus ring body offset from xy center
        # Sized to give ~2-3% cavity matching cube/tumor:
        #   first try 0.024m → 0.04% (way too small)
        #   second try 0.050m → 0.28% (still weak)
        #   third try 0.080m (33% of z-thickness) — target 2-3% cavity
        knife_radius_abs=0.080,
        # Inclusion placed at x=0.75 to land on ring body (avoid hole at xy center)
        incl_center_rel=(0.75, 0.50, 0.50),
        incl_radius_abs=0.080,
        traj_specs={
            # T_A: sweep along x at top of ring body
            "A": dict(sweep_axis=0, sweep_range=(0.68, 0.82), fixed={1: 0.50, 2: 1.00}),
            # T_B: sweep along y at same x — also on ring body (y=0.5 ± 0.1 stays on ring)
            "B": dict(sweep_axis=1, sweep_range=(0.42, 0.58), fixed={0: 0.75, 2: 1.00}),
            # T_C: plunge along z at one ring location
            "C": dict(sweep_axis=2, sweep_range=(1.00, 0.90), fixed={0: 0.75, 1: 0.50}),
            # T_D: x sweep offset to other side of ring body
            "D": dict(sweep_axis=0, sweep_range=(0.82, 0.96), fixed={1: 0.50, 2: 1.00}),
        },
        stiff_in=0.8, stiff_out=0.3,
        tau_in=0.20, tau_out=0.40,
    ),
]


def make_traj_from_spec(spec, bbox_min, bbox_max, T):
    """Materialize a trajectory from (sweep_axis, sweep_range, fixed) spec.

    spec keys:
       sweep_axis: 0/1/2 — which axis to sweep along
       sweep_range: (start_frac, end_frac) bbox-relative on sweep_axis
       fixed: dict {axis_idx: bbox_rel_frac} for the other 2 axes
    """
    sweep_axis = spec["sweep_axis"]
    s0, s1 = spec["sweep_range"]
    sweep_vals = np.linspace(bbox_min[sweep_axis] + s0 * (bbox_max[sweep_axis] - bbox_min[sweep_axis]),
                              bbox_min[sweep_axis] + s1 * (bbox_max[sweep_axis] - bbox_min[sweep_axis]),
                              T)
    fixed = spec["fixed"]
    traj = []
    for t in range(T):
        pos = np.zeros(3, dtype=np.float32)
        pos[sweep_axis] = sweep_vals[t]
        for ax, frac in fixed.items():
            pos[ax] = bbox_min[ax] + frac * (bbox_max[ax] - bbox_min[ax])
        traj.append(pos)
    return traj


def calibrate_method(mode, ctx, N_epochs=15, lr=0.03, lr_tau=0.01):
    """Calibrate baseline or hierarchical on T_A. Mirrors cross_trajectory_test.calibrate.
    Local copy to avoid coupling and to support per-mesh ctx properly."""
    device = ctx["device"]
    N = ctx["N"]
    stiff_init_np = ctx["stiff_init_np"]
    tau_init_np = ctx["tau_init_np"]
    tau_bound = ctx["tau_bound"]
    theta_3d_np = ctx["theta_3d_np"]
    knife_traj = ctx["knife_traj_A"]
    knife_radius = ctx["knife_radius"]
    rest_state = ctx["rest_state"]
    solver = ctx["solver"]
    obs_pos_t = ctx["obs_pos_A_t"]
    obs_alpha_t = ctx["obs_alpha_A_t"]
    edges_a_t = ctx["edges_a_t"]
    edges_b_t = ctx["edges_b_t"]
    edges_rest_t = ctx["edges_rest_t"]

    theta_torch = torch.tensor(theta_3d_np, dtype=torch.float32, device=device,
                                 requires_grad=False)
    z_L0_stiff = torch.tensor(inv_sigmoid_to_z(stiff_init_np),
                                dtype=torch.float32, device=device, requires_grad=True)
    z_L0_tau = torch.tensor(inv_sigmoid_to_z(tau_init_np, lo=tau_bound[0], hi=tau_bound[1]),
                              dtype=torch.float32, device=device, requires_grad=True)

    if mode == "baseline":
        params = [{"params": [z_L0_stiff], "lr": lr},
                   {"params": [z_L0_tau],   "lr": lr_tau}]
        use_hierarchy = False
        H = None
        z_L1_stiff = z_L2_stiff = z_L1_tau = z_L2_tau = None
    elif mode == "hierarchical":
        H = ctx["H"]
        M1, M2 = H.M1, H.M2
        z_L1_stiff = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=True)
        z_L2_stiff = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=True)
        z_L1_tau = torch.zeros(M1, dtype=torch.float32, device=device, requires_grad=True)
        z_L2_tau = torch.zeros(M2, dtype=torch.float32, device=device, requires_grad=True)
        sL1 = 1.0 / math.sqrt(N / M1)
        sL2 = 1.0 / math.sqrt(N / M2)
        params = [
            {"params": [z_L0_stiff], "lr": lr},
            {"params": [z_L1_stiff], "lr": lr * sL1},
            {"params": [z_L2_stiff], "lr": lr * sL2},
            {"params": [z_L0_tau],   "lr": lr_tau},
            {"params": [z_L1_tau],   "lr": lr_tau * sL1},
            {"params": [z_L2_tau],   "lr": lr_tau * sL2},
        ]
        use_hierarchy = True
    else:
        raise ValueError(mode)

    opt = torch.optim.Adam(params)
    T = len(knife_traj)
    best_L_gap = float("inf")
    best_stiff = stiff_init_np.copy()
    best_tau = tau_init_np.copy()

    for epoch in range(N_epochs):
        solver.reset_to_rest(ctx["pos_rest"])
        state = rest_state
        L_gaps = []
        for t in range(T):
            if use_hierarchy:
                z_flat_stiff = reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, H)
                z_flat_tau = reconstruct(z_L0_tau, z_L1_tau, z_L2_tau, H)
            else:
                z_flat_stiff = z_L0_stiff
                z_flat_tau = z_L0_tau
            stiff_pp_torch = STIFF_LOW + (STIFF_HIGH - STIFF_LOW) * torch.sigmoid(z_flat_stiff)
            tau_pp_torch = tau_bound[0] + (tau_bound[1] - tau_bound[0]) * torch.sigmoid(z_flat_tau)

            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state, knife_traj[t], knife_radius,
                stiff_pp_torch, tau_pp_torch,
            )
            tau_inner_val = float(tau_pp_torch.mean().item())
            delta_x, delta_dv, _ = gd_inner_loop_stable(
                pos_t, dv_t, obs_pos_t[t], obs_alpha_t[t],
                theta_torch, edges_a_t, edges_b_t, edges_rest_t,
                n_iter=30, lr_pos=0.5, lr_dv_factor=1.0,
                lambda_cav=1.0, lambda_E=0.0,
                s_inner=float(theta_torch[1].item()),
                tau_inner=tau_inner_val,
            )
            L_gap = (delta_x ** 2).sum() + (delta_dv ** 2).sum()
            opt.zero_grad()
            L_gap.backward()
            opt.step()
            L_gaps.append(float(L_gap.item()))

            state = solver_save_state(solver)
            dx_np = delta_x.detach().cpu().numpy()
            ddv_np = delta_dv.detach().cpu().numpy()
            state["pos"] = state["pos"] + dx_np
            state["particle_damage"] = np.maximum(state["particle_damage"] + ddv_np, 0.0)
            new_dv = state["particle_damage"]
            cur_s = float(theta_torch[1].item())
            cur_tau_arr = tau_pp_torch.detach().cpu().numpy()
            state["alpha"] = 1.0 / (1.0 + np.exp(np.clip(
                cur_s * (new_dv - cur_tau_arr), -50, 50)))

        mean_L = float(np.mean(L_gaps))
        with torch.no_grad():
            if use_hierarchy:
                z_flat_stiff_now = reconstruct(z_L0_stiff, z_L1_stiff, z_L2_stiff, H)
                z_flat_tau_now = reconstruct(z_L0_tau, z_L1_tau, z_L2_tau, H)
            else:
                z_flat_stiff_now = z_L0_stiff
                z_flat_tau_now = z_L0_tau
            stiff_now = (STIFF_LOW + (STIFF_HIGH - STIFF_LOW)
                            * torch.sigmoid(z_flat_stiff_now)).cpu().numpy()
            tau_now = (tau_bound[0] + (tau_bound[1] - tau_bound[0])
                          * torch.sigmoid(z_flat_tau_now)).cpu().numpy()

        if mean_L < best_L_gap - 1e-4:
            best_L_gap = mean_L
            best_stiff = stiff_now.copy()
            best_tau = tau_now.copy()
    return best_stiff, best_tau, best_L_gap


def predict_on_traj(stiff_pp, tau_pp, knife_traj, ctx):
    """Forward sim with FROZEN params, full reset between calls."""
    device = ctx["device"]
    theta_torch = torch.tensor(ctx["theta_3d_np"], dtype=torch.float32, device=device,
                                 requires_grad=False)
    stiff_torch = torch.tensor(stiff_pp, dtype=torch.float32, device=device, requires_grad=False)
    tau_torch = torch.tensor(tau_pp, dtype=torch.float32, device=device, requires_grad=False)
    solver = ctx["solver"]
    rest_state = ctx["rest_state"]
    knife_radius = ctx["knife_radius"]
    pos_rest = ctx["pos_rest"]
    solver.reset_to_rest(pos_rest)
    state_in = rest_state
    pred_pos, pred_alpha = [], []
    with torch.no_grad():
        for t, knife_pos in enumerate(knife_traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_torch, solver, state_in, knife_pos, knife_radius,
                stiff_torch, tau_torch,
            )
            pred_pos.append(pos_t.cpu().numpy().copy())
            pred_alpha.append(alpha_t.cpu().numpy().copy())
            state_in = solver_save_state(solver)
    return pred_pos, pred_alpha


def err_per_traj(pred_pos, truth_pos, pred_alpha, truth_alpha):
    T = len(pred_pos)
    pos_errs, alpha_errs = [], []
    for t in range(T):
        alive = truth_alpha[t] > 0.5
        d = np.linalg.norm(pred_pos[t] - truth_pos[t], axis=1)
        if alive.sum() > 0:
            pos_errs.append(float(d[alive].mean()))
        else:
            pos_errs.append(float(d.mean()))
        alpha_errs.append(float(np.abs(pred_alpha[t] - truth_alpha[t]).mean()))
    return float(np.mean(pos_errs)), float(np.mean(alpha_errs))


def run_one_mesh(cfg, device, T=10):
    print(f"\n{'='*72}")
    print(f"MESH: {cfg['name']}  ({cfg['path']})")
    print(f"{'='*72}")
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0)
    bbox_max = pos_rest.max(0)
    bbox_size = bbox_max - bbox_min
    bbox_diag = float(np.linalg.norm(bbox_size))
    print(f"  N={N}  bbox=({bbox_size[0]:.4f},{bbox_size[1]:.4f},{bbox_size[2]:.4f}) m  diag={bbox_diag:.4f} m")

    # Absolute params per mesh (set by hand to give realistic surface debulk)
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = cfg["incl_radius_abs"]
    top_axis = cfg["top_axis"]
    print(f"  top_axis={['x','y','z'][top_axis]}  knife_radius={knife_radius:.4f} ({knife_radius/bbox_size.min()*100:.1f}% of smallest bbox dim)")
    print(f"  incl_center=({incl_center[0]:.4f},{incl_center[1]:.4f},{incl_center[2]:.4f})  incl_radius={incl_radius:.4f}")

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center),
        radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    n_incl = int(inclusion_mask.sum())
    print(f"  inclusion: {n_incl}/{N} particles ({n_incl/N*100:.2f}%)")

    if n_incl == 0:
        print(f"  ⚠ skipping {cfg['name']}: no particles inside inclusion sphere")
        return None
    if n_incl == N:
        print(f"  ⚠ skipping {cfg['name']}: ALL particles inside inclusion sphere")
        return None

    # Build trajectories
    truth = {}
    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    for name, spec in cfg["traj_specs"].items():
        traj = make_traj_from_spec(spec, bbox_min, bbox_max, T)
        solver.reset_to_rest(pos_rest)
        obs_pos, obs_alpha = gen_obs_per_particle_stiff(
            solver, rest_state, theta_truth_np, stiff_truth, traj, knife_radius, device,
            tau_truth_np=tau_truth)
        cav = float((obs_alpha[-1] < 0.5).mean())
        truth[name] = dict(traj=traj, pos=obs_pos, alpha=obs_alpha)
        print(f"    T_{name}: knife start={traj[0]}, end={traj[-1]}  cavity={cav:.4f}")

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in truth["A"]["pos"]]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in truth["A"]["alpha"]]

    edges_a_t = torch.tensor(solver.e["a"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_b_t = torch.tensor(solver.e["b"].numpy().astype(np.int64),
                               dtype=torch.long, device=device)
    edges_a_np = solver.e["a"].numpy().astype(np.int64)
    edges_b_np = solver.e["b"].numpy().astype(np.int64)
    edge_rest_len = np.linalg.norm(pos_rest[edges_a_np] - pos_rest[edges_b_np], axis=1)
    edges_rest_t = torch.tensor(edge_rest_len, dtype=torch.float32, device=device)

    x_rest = torch.from_numpy(pos_rest).to(device, dtype=torch.float32)
    M1 = min(512, N // 20)
    M2 = min(64, N // 200)
    print(f"  Hierarchy: M1={M1}, M2={M2}")
    H = build_hierarchy(x_rest, M1=M1, M2=M2, k_neighbors=3)

    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=truth["A"]["traj"], knife_radius=knife_radius,
        stiff_init_np=stiff_init, stiff_truth_np=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        theta_3d_np=theta_truth_np, inclusion_mask=inclusion_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H, pos_rest=pos_rest,
    )

    print(f"  Calibrating baseline ...", flush=True)
    t0 = time.time()
    stiff_b, tau_b, Lgap_b = calibrate_method("baseline", ctx, N_epochs=15)
    t_b = time.time() - t0
    iou_b = compute_metrics(stiff_b, stiff_truth, inclusion_mask)["iou"]
    print(f"    baseline: {t_b:.0f}s  Lgap={Lgap_b:.3e}  IoU={iou_b:.3f}")

    print(f"  Calibrating hierarchical ...", flush=True)
    t0 = time.time()
    stiff_h, tau_h, Lgap_h = calibrate_method("hierarchical", ctx, N_epochs=15)
    t_h = time.time() - t0
    iou_h = compute_metrics(stiff_h, stiff_truth, inclusion_mask)["iou"]
    print(f"    hierarchical: {t_h:.0f}s  Lgap={Lgap_h:.3e}  IoU={iou_h:.3f}")

    results = {}
    for name in ["A", "B", "C", "D"]:
        traj = truth[name]["traj"]
        pb, ab = predict_on_traj(stiff_b, tau_b, traj, ctx)
        ph, ah = predict_on_traj(stiff_h, tau_h, traj, ctx)
        e_pos_b, e_a_b = err_per_traj(pb, truth[name]["pos"], ab, truth[name]["alpha"])
        e_pos_h, e_a_h = err_per_traj(ph, truth[name]["pos"], ah, truth[name]["alpha"])
        results[name] = dict(pos_b=e_pos_b, pos_h=e_pos_h, alpha_b=e_a_b, alpha_h=e_a_h)

    return dict(results=results, iou_b=iou_b, iou_h=iou_h,
                  Lgap_b=Lgap_b, Lgap_h=Lgap_h,
                  bbox_diag=bbox_diag, n_incl=n_incl, N=N)


def main():
    device = "cuda:0"
    T = 10
    all_results = {}
    t0_all = time.time()
    for cfg in MESH_CONFIGS:
        try:
            res = run_one_mesh(cfg, device, T=T)
            if res is not None:
                all_results[cfg["name"]] = res
        except Exception as e:
            print(f"\n✗ {cfg['name']} FAILED: {e}")
            import traceback; traceback.print_exc()
    print(f"\nTotal: {(time.time()-t0_all)/60:.1f} min")

    if not all_results:
        print("✗ No successful runs")
        return

    print(f"\n{'='*72}")
    print(f"MULTI-MESH SUMMARY — pos_err per (mesh, traj)")
    print(f"{'='*72}")
    print(f"\n  Per-cell pos_err (× 1e-5 m, scaled to mesh: × bbox_diag).")
    print(f"  Δ_rel = (hier - base) / base × 100%")
    print(f"  {'Mesh':10s}  {'bbox_diag':>10s}  {'T_A':>22s}  {'T_B':>22s}  {'T_C':>22s}  {'T_D':>22s}")
    for name, res in all_results.items():
        cells = []
        for tr in ["A", "B", "C", "D"]:
            b = res["results"][tr]["pos_b"]
            h = res["results"][tr]["pos_h"]
            rel = (h - b) / b * 100 if b > 1e-12 else 0
            mark = "★" if rel < -1.0 else ("✗" if rel > 1.0 else " ")
            # Normalize by bbox_diag for cross-mesh comparison
            b_norm = b / res["bbox_diag"] * 1e5
            h_norm = h / res["bbox_diag"] * 1e5
            cells.append(f"{b_norm:5.2f}/{h_norm:5.2f} {mark}{rel:+6.2f}%")
        print(f"  {name:10s}  {res['bbox_diag']:>10.4f}  " + "  ".join(cells))

    print(f"\n  ★ = hier beats base by >1%, ✗ = base beats hier by >1%")
    print(f"  Format: pos_err normalized by bbox_diag × 1e5 (dimensionless)")

    # IoU
    print(f"\n  IoU per mesh:")
    for name, res in all_results.items():
        di = res["iou_h"] - res["iou_b"]
        print(f"  {name:10s}  base={res['iou_b']:.3f}  hier={res['iou_h']:.3f}  Δ={di:+.3f}")

    # Aggregate held-out
    print(f"\n  Held-out wins (T_B + T_C + T_D × {len(all_results)} meshes = {3*len(all_results)} cells):")
    n_pos_wins = 0
    n_alpha_wins = 0
    n_cells = 0
    rel_pos_sum = 0.0
    rel_alpha_sum = 0.0
    for name, res in all_results.items():
        for tr in ["B", "C", "D"]:
            r = res["results"][tr]
            if r["pos_h"] < r["pos_b"]:
                n_pos_wins += 1
            if r["alpha_h"] < r["alpha_b"]:
                n_alpha_wins += 1
            rel_pos_sum += (r["pos_h"] - r["pos_b"]) / max(r["pos_b"], 1e-12) * 100
            rel_alpha_sum += (r["alpha_h"] - r["alpha_b"]) / max(r["alpha_b"], 1e-12) * 100
            n_cells += 1
    print(f"    Position: hier wins {n_pos_wins}/{n_cells}  (mean Δ_rel = {rel_pos_sum/n_cells:+.2f}%)")
    print(f"    Alpha:    hier wins {n_alpha_wins}/{n_cells}  (mean Δ_rel = {rel_alpha_sum/n_cells:+.2f}%)")

    print(f"\n{'='*72}")
    print(f"GEOMETRY-ROBUSTNESS VERDICT")
    print(f"{'='*72}")
    if n_pos_wins >= 0.75 * n_cells and n_alpha_wins >= 0.75 * n_cells:
        print(f"  ✓ ROBUST: hier wins ≥75% of held-out cells across {len(all_results)} meshes")
    elif n_pos_wins >= 0.5 * n_cells:
        print(f"  △ PARTIAL: hier wins majority but mesh-dependent")
    else:
        print(f"  ✗ NOT ROBUST: hier advantage is cube-specific")

    out_path = REPO / "tools" / "pointnetpp" / "multi_mesh_results.npz"
    np.savez(out_path, results=all_results)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
