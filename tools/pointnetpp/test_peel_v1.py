"""v1: peel-and-calibrate vs dynamic-surface vs static, on a DEPTH-varying tau (3 bands).
Tests the LOAD-BEARING claim:削一层把下一层当新鲜表面浅切, 能不能恢复埋藏层 tau.

Truth: 3 depth bands along cut axis, tau = [0.20 soft, 0.55 HARD(buried), 0.20 soft].
PEEL  : per band k -> shallow raster at band-k depth, surf = band-k top shell, GROUP
        calibrate (loss restricted to surf_k so cross-band knife contact is harmless),
        recovered_k = mean(tau over band k). 3 independent shallow calibrations.
DYNAMIC: one deep layered debulk, dynamic_surface=True, per-particle, per-band recovery.
STATIC : one deep debulk, static top-shell surf, per-particle, per-band recovery.
×5 mean±std. Focus = MIDDLE (buried) band. peel>>dynamic>>static => reduction holds.
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
from tools.pointnetpp.test_dynamic_surface_depth import make_layered_debulk

TAU_SOFT = 0.20; TAU_HARD = 0.55; TAU_INIT = 0.30; STIFF_UNIFORM = 0.5
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
    fr = (pos_rest[:, top_axis] - bb0[top_axis]) / sp[top_axis]   # depth fraction 0..1

    # 3 depth bands + truth tau
    BANDS = [("top",  0.78, 0.96, TAU_SOFT, 0.86),     # (name, lo, hi, tau, raster_frac)
             ("MID",  0.52, 0.70, TAU_HARD, 0.60),     # buried hard
             ("bot",  0.26, 0.44, TAU_SOFT, 0.34)]
    band_mask = [((fr >= lo) & (fr < hi)) for (_, lo, hi, _, _) in BANDS]
    tau_truth = np.full(N, TAU_SOFT, np.float32)
    for m, (_, _, _, tt, _) in zip(band_mask, BANDS):
        tau_truth[m] = tt
    print(f"N={N}  band sizes: " + ", ".join(f"{b[0]}={int(m.sum())}" for b, m in zip(BANDS, band_mask)), flush=True)

    surf_top = pos_rest[:, top_axis] > (bb1[top_axis] - 0.10 * sp[top_axis])
    surf_idx_top = torch.tensor(np.where(surf_top)[0], dtype=torch.long, device=device)
    col_id, _ = column_grid_assignment(pos_rest, surf_top, top_axis, grid_n=8)
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    eat = torch.tensor(ea, dtype=torch.long, device=device); ebt = torch.tensor(eb, dtype=torch.long, device=device)
    erl = np.linalg.norm(pos_rest[ea] - pos_rest[eb], axis=1); ert = torch.tensor(erl, dtype=torch.float32, device=device)
    xr = torch.from_numpy(pos_rest).to(device, torch.float32)
    H = build_hierarchy(xr, M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
    stiff_uniform = np.full(N, STIFF_UNIFORM, np.float32)
    theta = np.array([3., 20., 0.3], np.float32)

    def base_ctx(surf_idx, colid):
        return dict(solver=solver, rest_state=rest_state, N=N, device=device, knife_radius=kr,
                    pos_rest=pos_rest, theta_3d_np=theta, stiff_init_np=stiff_uniform,
                    tau_init_np=np.full(N, TAU_INIT, np.float32), tau_bound=(0.05, 0.95),
                    surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H,
                    group_col_id=colid, surf_top_axis=top_axis)

    def make_raster_at(rfrac, steps=30, nlines=10):
        ys = bb0[top_axis] + rfrac * sp[top_axis]
        zs = bb0[lat_axis] + np.linspace(0.12, 0.88, nlines) * sp[lat_axis]
        x_lo = bb0[sweep_axis] + 0.10 * sp[sweep_axis]; x_hi = bb0[sweep_axis] + 0.90 * sp[sweep_axis]
        traj = []
        for li, z in enumerate(zs):
            xs = np.linspace(x_lo, x_hi, steps)
            if li % 2 == 1: xs = xs[::-1]
            for x in xs:
                p = np.zeros(3, np.float32); p[top_axis] = ys; p[sweep_axis] = x; p[lat_axis] = z
                traj.append(p.astype(np.float32))
        return traj

    def gen_obs(traj):
        solver.reset_to_rest(pos_rest)
        op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta, stiff_uniform, traj, kr, device, tau_truth_np=tau_truth)
        return ([torch.tensor(p, device=device) for p in op], [torch.tensor(a, device=device) for a in oa], oa[-1])

    deep = make_layered_debulk(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.92, 0.30, 8, 60)

    def recov(tau, m):  # recovered fraction toward truth for band m
        return (tau[m].mean() - TAU_INIT) / (BANDS_tau(m) - TAU_INIT) * 100
    def BANDS_tau(m):
        # which band? return its truth tau
        for bm, (_, _, _, tt, _) in zip(band_mask, BANDS):
            if np.array_equal(bm, m): return tt
        return TAU_SOFT

    # ---- PEEL: per band, shallow raster at band depth, surf = band top shell ----
    def peel_band(k):
        name, lo, hi, tt, rfrac = BANDS[k]
        surf_k = (fr >= (hi - 0.10)) & (fr < (hi + 0.02))           # thin shell at band-k top
        si = torch.tensor(np.where(surf_k)[0], dtype=torch.long, device=device)
        ctx = base_ctx(si, col_id)
        traj = make_raster_at(rfrac)
        op, oa, fa = gen_obs(traj)
        # sanity: removal landing
        rm = fa < 0.5
        in_band = int((rm & band_mask[k]).sum()); out_band = int((rm & ~band_mask[k]).sum())
        _, tau, _ = calibrate_online("Group+L_pos_partial", ctx, [(traj, op, oa)], N_passes=4, lr=0.03,
                                     lr_tau=0.5, lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                     return_best=True, verbose=False)
        rk = (tau[band_mask[k]].mean() - TAU_INIT) / (tt - TAU_INIT) * 100
        return float(rk), in_band, out_band

    # ---- baselines: deep debulk, per-particle, static/dynamic ----
    def deep_recover(dynamic):
        ctx = base_ctx(surf_idx_top, col_id)
        op, oa, fa = gen_obs(deep)
        _, tau, _ = calibrate_online("L_pos_partial", ctx, [(deep, op, oa)], N_passes=4, lr=0.03, lr_tau=0.5,
                                     lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                     dynamic_surface=dynamic, return_best=True, verbose=False)
        return [float((tau[band_mask[k]].mean() - TAU_INIT) / (BANDS[k][3] - TAU_INIT) * 100) for k in range(3)]

    print("\n=== v1: depth-tau recovery (per band), ×{} mean±std ===".format(REP))
    # peel
    peel = {k: [] for k in range(3)}; sanity = {k: (0, 0) for k in range(3)}
    for r in range(REP):
        for k in range(3):
            rk, ib, ob = peel_band(k); peel[k].append(rk); sanity[k] = (ib, ob)
    # baselines
    stat = np.array([deep_recover(False) for _ in range(REP)])
    dyn = np.array([deep_recover(True) for _ in range(REP)])

    print(f"  {'band':6} | {'STATIC':>14} | {'DYNAMIC-surf':>14} | {'PEEL':>16} | sanity(in/out band)")
    for k, (name, lo, hi, tt, rf) in enumerate(BANDS):
        pv = np.array(peel[k])
        print(f"  {name:6} | {stat[:,k].mean():6.1f}±{stat[:,k].std():4.1f} | "
              f"{dyn[:,k].mean():6.1f}±{dyn[:,k].std():4.1f} | {pv.mean():6.1f}±{pv.std():4.1f}  | "
              f"{sanity[k][0]}/{sanity[k][1]}", flush=True)
    mk = 1  # MID
    print(f"\n  MIDDLE (buried hard τ=0.55): STATIC {stat[:,mk].mean():.0f}%  DYNAMIC {dyn[:,mk].mean():.0f}%  "
          f"PEEL {np.array(peel[mk]).mean():.0f}%  -> "
          f"{'REDUCTION HOLDS (peel recovers buried)' if np.array(peel[mk]).mean() > dyn[:,mk].mean()+15 else 'peel NOT better — reduction fragile'}", flush=True)


if __name__ == "__main__":
    main()
