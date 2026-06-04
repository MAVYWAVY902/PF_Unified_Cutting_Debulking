"""Caveat #3 closure: does surface-projection column grouping beat per-particle on
LATERAL tau patterns OTHER than left/right halves?

Two new depth-uniform lateral truths:
   (1) DIAGONAL STRIPES — hard/soft bands along the (x+z) diagonal.
   (2) LATERAL SPOT     — a hard circular disk in the (x,z) plane, soft outside.

Both are surface-observable (depth-uniform) so column grouping SHOULD recover them
if the grouping benefit is general (not specific to the L/R boundary).

Trajectory = shallow RASTER over the whole top surface (sweep x, step z) so every
lateral column's surface gets cut/observed. Metrics:
   - contrast recovery = (tau[hard].mean - tau[soft].mean) / (0.55-0.15)
   - surface spatial corr = corr(tau_recovered[surf], tau_truth[surf])  (pattern fidelity)
Compare GROUP vs Hier vs per-particle (Liang), only parameterization varies.
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

TAU_HARD = 0.55; TAU_SOFT = 0.15; TAU_INIT = 0.30; STIFF_UNIFORM = 0.5


def make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, y_frac, n_lines, T_per_line):
    sp = bb1 - bb0
    y = bb0[top_axis] + y_frac * sp[top_axis]
    zs = bb0[lat_axis] + np.linspace(0.12, 0.88, n_lines) * sp[lat_axis]
    x_lo = bb0[sweep_axis] + 0.10 * sp[sweep_axis]
    x_hi = bb0[sweep_axis] + 0.90 * sp[sweep_axis]
    traj = []
    for li, z in enumerate(zs):
        xs = np.linspace(x_lo, x_hi, T_per_line)
        if li % 2 == 1:
            xs = xs[::-1]
        for x in xs:
            p = np.zeros(3, np.float32)
            p[top_axis] = y; p[sweep_axis] = x; p[lat_axis] = z
            traj.append(p.astype(np.float32))
    return traj


def main():
    cfg = MESH_CONFIGS[0]; device = "cuda:0"
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]),
                                                            fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    kr = cfg["knife_radius_abs"]; top_axis = cfg["top_axis"]

    # warm-up settle
    solver.reset_to_rest(pos_rest); far = pos_rest.max(0) + 10.0
    sw = torch.tensor(np.full(N, STIFF_UNIFORM, np.float32), device=device)
    twt = torch.tensor(np.full(N, TAU_INIT, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); st = rest_state
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
    pos_rest = pw.cpu().numpy().copy(); mesh_info["vertices"] = pos_rest; rest_state = st
    bb0, bb1 = pos_rest.min(0), pos_rest.max(0); sp = bb1 - bb0
    lateral = [a for a in range(3) if a != top_axis]
    sweep_axis, lat_axis = lateral[0], lateral[1]

    xn = (pos_rest[:, sweep_axis] - bb0[sweep_axis]) / sp[sweep_axis]
    zn = (pos_rest[:, lat_axis] - bb0[lat_axis]) / sp[lat_axis]

    # depth-uniform lateral truth patterns
    diag = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)            # diagonal stripes
    spot = ((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2             # lateral disk
    PATTERNS = {
        "DIAG_STRIPES": diag,
        "LATERAL_SPOT": spot,
    }

    surf = pos_rest[:, top_axis] > (bb1[top_axis] - 0.10 * sp[top_axis])
    surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=device)
    col_id, ginfo = column_grid_assignment(pos_rest, surf, top_axis, grid_n=8)
    print(f"N={N} surf={int(surf.sum())} GROUP K={ginfo['K']} "
          f"touch={ginfo['groups_touching_surface']}/{ginfo['K']}", flush=True)

    # two shallow rasters (slightly different depth) for multi-traj robustness
    trajA = make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.93, 10, 30)
    trajB = make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.88, 10, 30)
    print(f"raster T_A={len(trajA)} T_B={len(trajB)}", flush=True)

    # edges / hierarchy / ctx (shared)
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    eat = torch.tensor(ea, dtype=torch.long, device=device); ebt = torch.tensor(eb, dtype=torch.long, device=device)
    erl = np.linalg.norm(pos_rest[ea] - pos_rest[eb], axis=1)
    ert = torch.tensor(erl, dtype=torch.float32, device=device)
    xr = torch.from_numpy(pos_rest).to(device, torch.float32)
    H = build_hierarchy(xr, M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
    stiff_init = np.full(N, STIFF_UNIFORM, np.float32); tau_init = np.full(N, TAU_INIT, np.float32)
    stiff_uniform = np.full(N, STIFF_UNIFORM, np.float32)
    theta_truth = np.array([3., 20., 0.3], np.float32)
    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=device, knife_radius=kr,
               pos_rest=pos_rest, theta_3d_np=theta_truth, stiff_init_np=stiff_init, tau_init_np=tau_init,
               tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt,
               edges_rest_t=ert, H=H, group_col_id=col_id)

    surf_np = surf
    truth_surf_cache = {}
    for pname, hard in PATTERNS.items():
        tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
        truth_surf_cache[pname] = (hard, tau_truth)
        # gen obs for both rasters
        multi = []
        for traj in (trajA, trajB):
            solver.reset_to_rest(pos_rest)
            op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, stiff_uniform,
                                                traj, kr, device, tau_truth_np=tau_truth)
            multi.append((traj, [torch.tensor(p, device=device) for p in op],
                          [torch.tensor(a, device=device) for a in oa]))
        truth_surf_cache[pname] = (hard, tau_truth, multi)

    LR = 0.03; LR_TAU = 0.5; NP = 4; LC = 1.0
    def run(cond, label, multi, hard, tau_truth):
        t0 = time.time()
        _, tau, _ = calibrate_online(cond, ctx, multi, N_passes=NP, lr=LR, lr_tau=LR_TAU,
                                     lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=LC,
                                     return_best=True, log_every=2000, verbose=False)
        contrast = tau[hard].mean() - tau[~hard].mean()
        rec = contrast / (TAU_HARD - TAU_SOFT)
        # surface spatial fidelity
        corr = float(np.corrcoef(tau[surf_np], tau_truth[surf_np])[0, 1])
        print(f"    {label:11s}: hard={tau[hard].mean():.3f} soft={tau[~hard].mean():.3f} "
              f"recovery={rec*100:5.1f}%  surf-corr={corr:+.3f}  {time.time()-t0:.0f}s", flush=True)
        return rec, corr

    print("\n=== GROUP vs Hier vs per-particle on NON-L/R lateral patterns ===")
    print(f"(truth hard={TAU_HARD} soft={TAU_SOFT} init={TAU_INIT}; depth-uniform)")
    results = {}
    for pname in PATTERNS:
        hard, tau_truth, multi = truth_surf_cache[pname]
        print(f"\n  [{pname}]  hard frac = {100*hard.mean():.0f}%", flush=True)
        gG = run("Group+L_pos_partial", "GROUP", multi, hard, tau_truth)
        gH = run("Hier+L_pos_partial", "Hier", multi, hard, tau_truth)
        gL = run("L_pos_partial", "per-particle", multi, hard, tau_truth)
        results[pname] = (gG, gH, gL)

    print("\n================= SUMMARY =================")
    print(f"{'pattern':14s} | {'GROUP':>16s} | {'Hier':>16s} | {'per-particle':>16s}")
    for pname, (gG, gH, gL) in results.items():
        print(f"{pname:14s} | {gG[0]*100:5.1f}% c={gG[1]:+.2f} | "
              f"{gH[0]*100:5.1f}% c={gH[1]:+.2f} | {gL[0]*100:5.1f}% c={gL[1]:+.2f}", flush=True)
    print("(recovery = contrast vs truth 0.40; c = surface spatial correlation with truth)")
    print("GROUP should clearly beat per-particle on BOTH patterns -> grouping benefit is general.")


if __name__ == "__main__":
    main()
