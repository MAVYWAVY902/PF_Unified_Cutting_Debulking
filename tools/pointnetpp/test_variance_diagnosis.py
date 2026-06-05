"""Calibrate the ruler: where does the run-to-run variance come from?

A. Forward determinism: run the SAME forward (truth tau, same raster) twice -> obs RMS.
   If >0, Warp atomic_add is nondeterministic (the bottom noise source).
B. obs-side vs calibration-side variance (SPOT, N=4 ×5 each):
   - FIXED obs (generate once, reuse) -> variance from calibration's OWN forwards only.
   - FRESH obs (regenerate each time)  -> variance from obs-gen + calibration.
C. small vs large feature: GROUP N=4 ×5 on LR (large) vs SPOT (small) -> compare std.

Report mean±std + min..max so we can judge effect-vs-noise for prior claims.
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
REP = 5


def summ(name, vals):
    v = np.array(vals)
    print(f"  {name:28s}: mean={v.mean():6.1f}%  std={v.std():5.1f}%  "
          f"min..max={v.min():5.1f}..{v.max():5.1f}  vals={[f'{x:.0f}' for x in v]}", flush=True)
    return v


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
    LR = xn < 0.5
    SPOT = ((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2
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

    def calib(multi, hard):
        _, tau, _ = calibrate_online("Group+L_pos_partial", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5,
                                     lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                     return_best=True, verbose=False)
        return (tau[hard].mean() - tau[~hard].mean()) / (TAU_HARD - TAU_SOFT) * 100

    # ---- A: forward determinism ----
    print("=== A. forward determinism (same truth tau + raster, twice) ===")
    tt = np.where(SPOT, TAU_HARD, TAU_SOFT).astype(np.float32)
    o = []
    for _ in range(2):
        solver.reset_to_rest(pos_rest)
        op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                            np.full(N, STIFF_UNIFORM, np.float32), calA, kr, device, tau_truth_np=tt)
        o.append((np.stack(op), np.stack(oa)))
    pos_rms = float(np.sqrt(((o[0][0] - o[1][0]) ** 2).mean()))
    al_rms = float(np.sqrt(((o[0][1] - o[1][1]) ** 2).mean()))
    al_diff_n = int((np.abs(o[0][1] - o[1][1]) > 1e-6).sum())
    print(f"  obs pos RMS between 2 identical forwards = {pos_rms:.2e} m", flush=True)
    print(f"  obs alpha RMS = {al_rms:.2e}   (#alpha cells differing >1e-6: {al_diff_n})", flush=True)
    print(f"  => {'NONDETERMINISTIC (atomic_add)' if pos_rms > 1e-9 else 'deterministic'}", flush=True)

    # ---- B: fixed obs vs fresh obs (SPOT) ----
    print(f"\n=== B. obs-side vs calibration-side variance (SPOT, N=4 ×{REP}) ===")
    fixed = gen_multi(SPOT)
    summ("SPOT fixed-obs (calib-side)", [calib(fixed, SPOT) for _ in range(REP)])
    summ("SPOT fresh-obs (obs+calib)", [calib(gen_multi(SPOT), SPOT) for _ in range(REP)])

    # ---- C: large vs small feature ----
    print(f"\n=== C. large (LR) vs small (SPOT) feature std (fresh obs, N=4 ×{REP}) ===")
    summ("LR   fresh-obs", [calib(gen_multi(LR), LR) for _ in range(REP)])
    summ("SPOT fresh-obs", [calib(gen_multi(SPOT), SPOT) for _ in range(REP)])
    print("\n(judge: effect must be > 2-3x std to be real; compare to prior single-run claims)")


if __name__ == "__main__":
    main()
