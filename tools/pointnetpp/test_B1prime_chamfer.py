"""B1': does CORRESPONDENCE-FREE Chamfer state-correction recover like teacher forcing
(and stay stable, unlike plain free-run)?

3 state modes, GROUP N=4, LR+DIAG:
  teacher  = per-particle obs snap (uses correspondence) — reference
  free     = no correction (drifts, rise-then-collapse per B1_verify)
  chamfer  = snap sim surface -> NEAREST obs point (NO correspondence) = the real-D405 path
Report best-checkpoint recovery mean±std (×REP) + stability (best vs final, per-pass L_gap)
for chamfer. If chamfer ~ teacher and stable -> correspondence not needed for STATE (B de-risked).
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
REP = 3


def main():
    cfg = MESH_CONFIGS[0]; device = "cuda:0"
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    kr = cfg["knife_radius_abs"]; top_axis = cfg["top_axis"]
    solver.reset_to_rest(pos_rest); far = pos_rest.max(0) + 10.0
    sw = torch.tensor(np.full(N, STIFF_UNIFORM, np.float32), device=device)
    twt = torch.tensor(np.full(N, TAU_INIT, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); stt = rest_state
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, stt, far, kr, sw, twt); stt = solver_save_state(solver)
    pos_rest = pw.cpu().numpy().copy(); rest_state = stt
    bb0, bb1 = pos_rest.min(0), pos_rest.max(0); sp = bb1 - bb0
    lateral = [a for a in range(3) if a != top_axis]; sweep_axis, lat_axis = lateral[0], lateral[1]
    xn = (pos_rest[:, sweep_axis] - bb0[sweep_axis]) / sp[sweep_axis]
    zn = (pos_rest[:, lat_axis] - bb0[lat_axis]) / sp[lat_axis]
    PATTERNS = {"LR_HALVES": xn < 0.5,
                "DIAG_STRIPES": (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)}
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
        tt = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32); m = []
        for tj in (calA, calB):
            solver.reset_to_rest(pos_rest)
            op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                                np.full(N, STIFF_UNIFORM, np.float32), tj, kr, device, tau_truth_np=tt)
            m.append((tj, [torch.tensor(p, device=device) for p in op], [torch.tensor(a, device=device) for a in oa]))
        return m

    def contrast(tau, hard):
        return (tau[hard].mean() - tau[~hard].mean()) / (TAU_HARD - TAU_SOFT) * 100

    def run(multi, hard, mode, best=True):
        kw = dict(teacher_force=(mode == "teacher"), chamfer_state=(mode == "chamfer"))
        _, tau, hist = calibrate_online("Group+L_pos_partial", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5,
                                        lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                        return_best=best, **kw, verbose=False)
        return contrast(tau, hard), hist

    for pname, hard in PATTERNS.items():
        print(f"\n========== {pname} ==========", flush=True)
        for mode in ("teacher", "free", "chamfer"):
            vals = [run(gen_multi(hard), hard, mode)[0] for _ in range(REP)]
            v = np.array(vals)
            print(f"  {mode:8s}: best-recovery {v.mean():6.1f}% ± {v.std():4.1f}   vals={[f'{x:.0f}' for x in v]}", flush=True)
        # stability of chamfer: best vs final + per-pass L_gap
        multi = gen_multi(hard)
        cb, _ = run(multi, hard, "chamfer", best=True)
        cf, hist = run(multi, hard, "chamfer", best=False)
        lg = [np.mean([d["L_gap"] for d in hist if d["pass_idx"] == p]) for p in range(4)]
        print(f"  chamfer stability: best={cb:.0f}% final={cf:.0f}% (gap {cb-cf:+.0f})  "
              f"L_gap/pass={['%.2e'%x for x in lg]}  -> "
              f"{'STABLE (final~best, L_gap bounded)' if abs(cb-cf)<=15 else 'UNSTABLE (best masks collapse)'}", flush=True)
    print("\n(chamfer = correspondence-free; if ~teacher & stable -> B state-side de-risked.)")


if __name__ == "__main__":
    main()
