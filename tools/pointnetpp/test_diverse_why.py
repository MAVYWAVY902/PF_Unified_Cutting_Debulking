"""WHY is 'diverse' (orthogonal) multi-traj worse? Original test confounded DEPTH with
DIRECTION (multi-sim used depths 0.93+0.88; multi-div used 0.93+0.93). Isolate:

Single-traj alone (is the orthogonal cut intrinsically weaker?):
  calA      X@0.93 ;  calB_sim X@0.88 ;  calB_div93 Z@0.93 ;  calB_div88 Z@0.88
Multi (depth-controlled — only the 2nd cut's DIRECTION differs):
  multi-sim       = X@0.93 + X@0.88
  multi-div-CONF  = X@0.93 + Z@0.93   (original, depth-confounded)
  multi-div-FIX   = X@0.93 + Z@0.88   (depth-matched to multi-sim; only direction differs)
+ coverage stats per traj (removed surf particles, #columns touched).
DIAG only (low variance), GROUP N=4, ×5 mean±std.
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
    hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)        # DIAG
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

    print(f"cube extents: x={sp[sweep_axis]:.3f}  z={sp[lat_axis]:.3f}  (symmetric? {abs(sp[sweep_axis]-sp[lat_axis])<0.02})", flush=True)
    TR = {
        "X@0.93": make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.93, 10, 30),
        "X@0.88": make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.88, 10, 30),
        "Z@0.93": make_raster(bb0, bb1, top_axis, lat_axis, sweep_axis, 0.93, 10, 30),
        "Z@0.88": make_raster(bb0, bb1, top_axis, lat_axis, sweep_axis, 0.88, 10, 30),
    }

    def gen_one(tj):
        tt = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
        solver.reset_to_rest(pos_rest)
        op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                            np.full(N, STIFF_UNIFORM, np.float32), tj, kr, device, tau_truth_np=tt)
        return (tj, [torch.tensor(p, device=device) for p in op], [torch.tensor(a, device=device) for a in oa]), oa[-1]

    # coverage stats (once)
    si = surf_idx.cpu().numpy()
    print("coverage per traj (final removed surf particles, #of 64 columns touched):", flush=True)
    for name, tj in TR.items():
        _, fa = gen_one(tj)
        rmv = (fa[si] < 0.5)
        cols_touched = len(np.unique(col_id[si][rmv])) if rmv.any() else 0
        print(f"  {name}: removed_surf={int(rmv.sum())}/{len(si)}  cols_touched={cols_touched}/64", flush=True)

    def calib(traj_list):
        multi = [gen_one(tj)[0] for tj in traj_list]
        _, tau, _ = calibrate_online("Group+L_pos_partial", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5,
                                     lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                     return_best=True, verbose=False)
        return (tau[hard].mean() - tau[~hard].mean()) / (TAU_HARD - TAU_SOFT) * 100

    print("\n--- single-traj alone (N=4) ---", flush=True)
    singles = {}
    for name in ("X@0.93", "X@0.88", "Z@0.93", "Z@0.88"):
        v = np.array([calib([TR[name]]) for _ in range(REP)]); singles[name] = v
        print(f"  single {name}: {v.mean():6.1f}% ± {v.std():4.1f}", flush=True)

    print("\n--- multi (depth-controlled: only 2nd-cut DIRECTION differs) ---", flush=True)
    multis = {
        "multi-sim    (X@.93+X@.88)": ["X@0.93", "X@0.88"],
        "multi-div-CONF(X@.93+Z@.93)": ["X@0.93", "Z@0.93"],
        "multi-div-FIX (X@.93+Z@.88)": ["X@0.93", "Z@0.88"],
    }
    mres = {}
    for label, names in multis.items():
        v = np.array([calib([TR[n] for n in names]) for _ in range(REP)]); mres[label] = v
        print(f"  {label}: {v.mean():6.1f}% ± {v.std():4.1f}", flush=True)

    sim = mres["multi-sim    (X@.93+X@.88)"]; fix = mres["multi-div-FIX (X@.93+Z@.88)"]
    sem = np.sqrt(sim.std()**2/REP + fix.std()**2/REP)
    d = fix.mean() - sim.mean()
    print(f"\n  VERDICT (depth-controlled): multi-div-FIX − multi-sim = {d:+.1f}  (2SEM {2*sem:.1f}) -> "
          f"{'DIRECTION diversity still HURTS' if d < -2*sem else ('diversity helps' if d > 2*sem else 'SAME (the original gap was DEPTH, not direction)')}", flush=True)


if __name__ == "__main__":
    main()
