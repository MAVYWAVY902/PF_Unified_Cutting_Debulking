"""Two online-feasibility tests.

PART A (close the sparse gap WITHOUT replay): on LATERAL_SPOT (sparse, 17% hard),
does SLOWING the knife (more frames/line -> more gradient events per column in ONE
causal pass) recover as well as multi-pass replay? Key control = EQUAL Adam budget:
  N=1 @120 steps/line = 2400 Adam steps, single causal pass (spatially spread)
  vs N=4 @30 steps/line = 2400 Adam steps, 4 replays (repeated)
If single-pass-slow ~ multi-pass-fast -> online (no replay) suffices for sparse too.

PART B (real-time speed): wall-clock ms per online update (forward PBD + backward +
Adam + teacher-force) -> Hz. Compare to Liang's 0.3-1.1 Hz. (Perception NOT included.)
"""
import sys, time
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
    spot = ((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2
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
    tau_truth = np.where(spot, TAU_HARD, TAU_SOFT).astype(np.float32)

    def make_multi(steps):
        m = []
        for yf in (0.93, 0.88):
            tj = make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, yf, 10, steps)
            solver.reset_to_rest(pos_rest)
            op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                                np.full(N, STIFF_UNIFORM, np.float32), tj, kr, device, tau_truth_np=tau_truth)
            m.append((tj, [torch.tensor(p, device=device) for p in op], [torch.tensor(a, device=device) for a in oa]))
        return m

    def calib(multi, npass):
        _, tau, _ = calibrate_online("Group+L_pos_partial", ctx, multi, N_passes=npass, lr=0.03, lr_tau=0.5,
                                     lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                     return_best=True, verbose=False)
        return (tau[spot].mean() - tau[~spot].mean()) / (TAU_HARD - TAU_SOFT) * 100

    print("=== PART A: slow knife single-pass vs multi-pass replay (LATERAL_SPOT, sparse) ===")
    print(f"{'config':30s} | {'steps/line':>10} {'#frames(=Adam)':>14} | {'recovery':>9}")
    configs = [("N=1 fast (30/line)", 30, 1), ("N=1 slow (60/line)", 60, 1),
               ("N=1 slow (120/line)", 120, 1), ("N=4 fast (30/line) [replay ref]", 30, 4)]
    for label, steps, npass in configs:
        multi = make_multi(steps)
        nfr = npass * sum(len(t[0]) for t in multi)
        r = calib(multi, npass)
        print(f"{label:30s} | {steps:>10} {nfr:>14} | {r:8.0f}%", flush=True)
    print("  (key: N=1 @120 has SAME 2400 Adam steps as N=4 @30 — single-pass-slow vs replay)")

    print("\n=== PART B: real-time speed (ms per online update, forward+backward+Adam+teacher-force) ===")
    multi = make_multi(30)
    # warmup (compile kernels)
    calib(multi, 1)
    t0 = time.time(); calibrate_online("Group+L_pos_partial", ctx, multi, N_passes=1, lr=0.03, lr_tau=0.5,
                                       lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                       return_best=False, verbose=False)
    dt = time.time() - t0
    nfr = sum(len(t[0]) for t in multi)
    print(f"  {nfr} online updates in {dt:.2f}s  ->  {dt/nfr*1000:.1f} ms/update  =  {nfr/dt:.1f} Hz", flush=True)
    print(f"  (Liang PBD-RM-ON: 0.3-1.1 Hz incl perception; ours excludes perception)")


if __name__ == "__main__":
    main()
