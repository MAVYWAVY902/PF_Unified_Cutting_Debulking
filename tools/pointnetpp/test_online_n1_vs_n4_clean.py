"""Clean N=1 (online single-pass) vs N=4 (offline replay), with repeats + stats.

3 patterns × {N=1, N=4} × 5 repeats (fresh obs each). Report mean±std±SEM.
Judge per pattern: is (N=4 - N=1) > 2×SEM_combined (replay helps) or indistinguishable
(online loses nothing)? Trust LR/DIAG (low var); SPOT wide bars.
Only axis (A) single-pass; (C) speed already ~50Hz; (B) correspondence gated.
"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.hierarchy import build_hierarchy
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.calibrate_online import calibrate_online
from tools.pointnetpp.surface_projection import column_grid_assignment
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD = 0.55; TAU_SOFT = 0.15; TAU_INIT = 0.30; STIFF_UNIFORM = 0.5
REP = 5


def main():
    cfg = MESH_CONFIGS[0]; device = "cuda:0"
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]),
                                                            fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    kr = cfg["knife_radius_abs"]; top_axis = cfg["top_axis"]
    solver.reset_to_rest(pos_rest); far = pos_rest.max(0) + 10.0
    sw = torch.tensor(np.full(N, STIFF_UNIFORM, np.float32), device=device)
    twt = torch.tensor(np.full(N, TAU_INIT, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); st = rest_state
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
    pos_rest = pw.cpu().numpy().copy(); rest_state = st
    bb0, bb1 = pos_rest.min(0), pos_rest.max(0); sp = bb1 - bb0
    lateral = [a for a in range(3) if a != top_axis]; sweep_axis, lat_axis = lateral[0], lateral[1]
    xn = (pos_rest[:, sweep_axis] - bb0[sweep_axis]) / sp[sweep_axis]
    zn = (pos_rest[:, lat_axis] - bb0[lat_axis]) / sp[lat_axis]
    PATTERNS = {
        "LR_HALVES":    xn < 0.5,
        "DIAG_STRIPES": (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0),
        "LATERAL_SPOT": ((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2,
    }
    surf = pos_rest[:, top_axis] > (bb1[top_axis] - 0.10 * sp[top_axis])
    surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=device)
    col_id, _ = column_grid_assignment(pos_rest, surf, top_axis, grid_n=8)
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    eat = torch.tensor(ea, dtype=torch.long, device=device); ebt = torch.tensor(eb, dtype=torch.long, device=device)
    erl = np.linalg.norm(pos_rest[ea] - pos_rest[eb], axis=1); ert = torch.tensor(erl, dtype=torch.float32, device=device)
    xr = torch.from_numpy(pos_rest).to(device, torch.float32)
    H = build_hierarchy(xr, M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=device, knife_radius=kr,
               pos_rest=pos_rest, theta_3d_np=np.array([3., 20., 0.3], np.float32),
               stiff_init_np=np.full(N, STIFF_UNIFORM, np.float32), tau_init_np=np.full(N, TAU_INIT, np.float32),
               tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt,
               edges_rest_t=ert, H=H, group_col_id=col_id)
    calA = make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.93, 10, 30)
    calB = make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.88, 10, 30)

    def gen_multi(hard):
        tt = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
        m = []
        for tj in (calA, calB):
            solver.reset_to_rest(pos_rest)
            op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                                np.full(N, STIFF_UNIFORM, np.float32), tj, kr, device, tau_truth_np=tt)
            m.append((tj, [torch.tensor(p, device=device) for p in op], [torch.tensor(a, device=device) for a in oa]))
        return m

    def calib(multi, hard, npass):
        _, tau, _ = calibrate_online("Group+L_pos_partial", ctx, multi, N_passes=npass, lr=0.03, lr_tau=0.5,
                                     lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                     return_best=True, verbose=False)
        return (tau[hard].mean() - tau[~hard].mean()) / (TAU_HARD - TAU_SOFT) * 100

    print(f"=== N=1 (online single-pass) vs N=4 (offline replay), ×{REP}, mean±std ===")
    for pname, hard in PATTERNS.items():
        v1 = np.array([calib(gen_multi(hard), hard, 1) for _ in range(REP)])
        v4 = np.array([calib(gen_multi(hard), hard, 4) for _ in range(REP)])
        sem = np.sqrt(v1.std()**2 / REP + v4.std()**2 / REP)
        diff = v4.mean() - v1.mean()
        verdict = "REPLAY HELPS" if abs(diff) > 2 * sem else "INDISTINGUISHABLE (online ok)"
        print(f"\n  [{pname}]", flush=True)
        print(f"    N=1: {v1.mean():6.1f}% ± {v1.std():4.1f}   vals={[f'{x:.0f}' for x in v1]}", flush=True)
        print(f"    N=4: {v4.mean():6.1f}% ± {v4.std():4.1f}   vals={[f'{x:.0f}' for x in v4]}", flush=True)
        print(f"    N4-N1 = {diff:+.1f}%   2*SEM = {2*sem:.1f}%   -> {verdict}", flush=True)
    print("\n(LR/DIAG low-var = trust; SPOT wide bars. Only axis-A single-pass; speed ~50Hz ok; correspondence gated.)")


if __name__ == "__main__":
    main()
