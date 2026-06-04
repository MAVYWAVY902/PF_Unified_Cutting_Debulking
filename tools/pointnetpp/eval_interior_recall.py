"""Interior-only recall test — the REAL inference task.

D405 already directly sees surface inclusion (cavity holes in obs).
The valuable task: recover INTERIOR inclusion structure (hidden from camera)
from surface obs alone.

For each mesh × surface-inclusion config × method:
   1. Calibrate with partial obs (top 10% surface)
   2. Compute tau<0.3 prediction
   3. Stratify into 3 subsets:
        - SURFACE inclusion particles (directly observable region)
        - INTERIOR inclusion particles (truly hidden — the REAL task)
        - SURFACE healthy particles
        - INTERIOR healthy particles
   4. Report P/R/IoU per subset
   5. Show that:
        - Liang surface_R is high (gradient touches them)
        - Liang interior_R is ~0% (gradient doesn't reach them)
        - Hier interior_R is non-zero but with massive FP elsewhere

Honest assessment of what these methods can / cannot do.
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

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.stageA_inclusion_recovery import build_inclusion_truth, gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.hierarchy import build_hierarchy  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402
from tools.pointnetpp.diag_partial_obs import calibrate  # noqa: E402
from tools.pointnetpp.eval_surface_inclusion import SURFACE_INCL_CFG  # noqa: E402


def strat_PRI(pred_in, truth_in, subset_mask):
    """Compute Precision/Recall/IoU on a subset of particles."""
    p = pred_in & subset_mask
    t = truth_in & subset_mask
    tp = int((p & t).sum())
    fn = int((~p & t).sum())
    # FP within subset: predicted-in but NOT truth-in
    fp = int((p & ~t).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    union = (p | t).sum()
    IoU = 0.0 if union == 0 else tp / union
    return dict(tp=tp, fn=fn, fp=fp, P=P, R=R, IoU=IoU, n_truth=int(t.sum()), n_pred=int(p.sum()))


def run_one(cfg, device, T_calib=10, N_epochs=15):
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.95
    solver.num_iterations = 2
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)
    pos_rest = mesh_info["vertices"]
    solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))

    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    top_axis = cfg["top_axis"]
    incl_rel, incl_radius = SURFACE_INCL_CFG[cfg["name"]]
    incl_center = bbox_min + np.array(incl_rel) * bbox_size

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)

    surface_threshold = bbox_max[top_axis] - 0.10 * bbox_size[top_axis]
    surface_mask = pos_rest[:, top_axis] > surface_threshold
    surface_idx_np = np.where(surface_mask)[0]
    interior_mask = ~surface_mask

    # 4 stratified subsets
    surf_incl = surface_mask & inclusion_mask
    int_incl = interior_mask & inclusion_mask
    surf_healthy = surface_mask & ~inclusion_mask
    int_healthy = interior_mask & ~inclusion_mask
    print(f"  Stratification:")
    print(f"    SURFACE inclusion : {surf_incl.sum():>5}  ({surf_incl.sum()/inclusion_mask.sum()*100:.1f}% of truth)")
    print(f"    INTERIOR inclusion: {int_incl.sum():>5}  ({int_incl.sum()/inclusion_mask.sum()*100:.1f}% of truth)")
    print(f"    SURFACE healthy   : {surf_healthy.sum():>5}")
    print(f"    INTERIOR healthy  : {int_healthy.sum():>5}")
    print(f"    TOTAL N           : {N:>5}", flush=True)

    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    traj_calib = make_traj_from_spec(cfg["traj_specs"]["A"], bbox_min, bbox_max, T_calib)

    solver.reset_to_rest(pos_rest)
    obs_pos_calib, obs_alpha_calib = gen_obs_per_particle_stiff(
        solver, rest_state, theta_truth_np, stiff_truth, traj_calib, knife_radius, device,
        tau_truth_np=tau_truth)

    obs_pos_A_t = [torch.tensor(p, dtype=torch.float32, device=device) for p in obs_pos_calib]
    obs_alpha_A_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in obs_alpha_calib]
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

    stiff_init = np.full(N, 0.5, dtype=np.float32)
    tau_init = np.full(N, 0.30, dtype=np.float32)

    ctx = dict(
        solver=solver, rest_state=rest_state, N=N, device=device,
        knife_traj_A=traj_calib, knife_radius=knife_radius,
        pos_rest=pos_rest, theta_truth=theta_truth_np, theta_3d_np=theta_truth_np,
        stiff_init_np=stiff_init, stiff_truth=stiff_truth,
        tau_init_np=tau_init, tau_bound=(0.05, 0.95),
        inclusion_mask=inclusion_mask,
        surface_idx=surface_idx, surface_mask=surface_mask,
        obs_pos_A_t=obs_pos_A_t, obs_alpha_A_t=obs_alpha_A_t,
        edges_a_t=edges_a_t, edges_b_t=edges_b_t, edges_rest_t=edges_rest_t,
        H=H_hier,
    )

    results = {}
    for method in ["Liang_inner", "Hier+Liang_inner"]:
        t0 = time.time()
        _, tau, _ = calibrate(method, ctx, N_epochs=N_epochs, lr=0.03, lr_tau=0.01,
                                   lambda_smooth=0.0, lambda_E=1e-5)
        dt = time.time() - t0
        pred_in = tau < 0.3
        truth_in = tau_truth < 0.3

        # Stratified metrics
        all_mask = np.ones(N, dtype=bool)
        m_all = strat_PRI(pred_in, truth_in, all_mask)
        m_surf = strat_PRI(pred_in, truth_in, surface_mask)
        m_int = strat_PRI(pred_in, truth_in, interior_mask)

        # Movement diagnostics
        delta_from_init = tau - 0.30  # 大 negative 表 示 真 的 把 tau 拉 下 来 了
        moved = np.abs(delta_from_init) > 0.001  # 真 实 移 动 的 粒 子
        results[method] = dict(
            all=m_all, surf=m_surf, interior=m_int,
            tau_in_mean=float(tau[inclusion_mask].mean()),
            tau_out_mean=float(tau[~inclusion_mask].mean()),
            n_moved=int(moved.sum()),
            n_moved_inclusion=int((moved & inclusion_mask).sum()),
            n_moved_surface_incl=int((moved & surf_incl).sum()),
            n_moved_interior_incl=int((moved & int_incl).sum()),
            runtime=dt,
        )

    return dict(cfg=cfg, inclusion_mask=inclusion_mask, surface_mask=surface_mask,
                 results=results,
                 surf_incl=int(surf_incl.sum()), int_incl=int(int_incl.sum()))


def main():
    device = "cuda:0"
    t0_all = time.time()
    for cfg in MESH_CONFIGS:
        print(f"\n{'='*100}\n{cfg['name']}  SURFACE INCLUSION  Interior-only recall test\n{'='*100}", flush=True)
        try:
            data = run_one(cfg, device)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}"); traceback.print_exc(); continue
        for method in ["Liang_inner", "Hier+Liang_inner"]:
            r = data["results"][method]
            print(f"\n  [{method}]")
            print(f"    ALL particles    : P={r['all']['P']:.2f} R={r['all']['R']:.2f} IoU={r['all']['IoU']:.3f}  "
                   f"(TP={r['all']['tp']} FN={r['all']['fn']} FP={r['all']['fp']})")
            print(f"    SURFACE subset   : P={r['surf']['P']:.2f} R={r['surf']['R']:.2f} IoU={r['surf']['IoU']:.3f}  "
                   f"(TP={r['surf']['tp']} FN={r['surf']['fn']} FP={r['surf']['fp']})")
            print(f"    INTERIOR subset  : P={r['interior']['P']:.2f} R={r['interior']['R']:.2f} IoU={r['interior']['IoU']:.3f}  "
                   f"(TP={r['interior']['tp']} FN={r['interior']['fn']} FP={r['interior']['fp']})")
            print(f"    n_moved = {r['n_moved']}  of which inclusion={r['n_moved_inclusion']}  "
                   f"surf_incl={r['n_moved_surface_incl']}  int_incl={r['n_moved_interior_incl']}")
    print(f"\nTotal: {(time.time()-t0_all)/60:.1f} min")


if __name__ == "__main__":
    main()
