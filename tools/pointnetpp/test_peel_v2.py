"""v2: does the peel reduction survive CUTTING DAMAGE to the layer below?

v1 calibrated each band PRISTINE (separate raster from rest). v2 adds the real effect:
cut layer k LEAKS damage into layer k+1 (sub-surface), so when layer k+1 becomes the
surface it is ALREADY pre-damaged -> its recovered tau is biased (appears softer).

Implementation: contiguous 3 bands. For each band k:
  PRISTINE (v1 ref): shallow raster from REST on band k -> calibrate band k (no pre-damage)
  PREDAMAGED (v2):   one CONTINUOUS deep debulk (descends through all bands, damage carried
                     forward; teacher-force resets pos NOT damage) + surface_idx = band-k top
                     -> band-k gradient fires only at its phase, but band k is PRE-DAMAGED by
                     the cut of bands above. Calibrate band k.
Compare MIDDLE (buried hard) recovery: v2/v1 = how much the reduction degrades ("打几折").
×3 mean±std.
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
    fr = (pos_rest[:, top_axis] - bb0[top_axis]) / sp[top_axis]

    # 3 CONTIGUOUS bands (so cutting one leaks damage into the next)
    BANDS = [("top", 0.66, 0.92, TAU_SOFT, 0.80),
             ("MID", 0.40, 0.66, TAU_HARD, 0.54),
             ("bot", 0.14, 0.40, TAU_SOFT, 0.28)]
    band_mask = [((fr >= lo) & (fr < hi)) for (_, lo, hi, _, _) in BANDS]
    tau_truth = np.full(N, TAU_SOFT, np.float32)
    for m, (_, _, _, tt, _) in zip(band_mask, BANDS):
        tau_truth[m] = tt
    print(f"N={N}  bands: " + ", ".join(f"{b[0]}={int(m.sum())}" for b, m in zip(BANDS, band_mask)), flush=True)

    surf_top = pos_rest[:, top_axis] > (bb1[top_axis] - 0.10 * sp[top_axis])
    col_id, _ = column_grid_assignment(pos_rest, surf_top, top_axis, grid_n=8)
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    eat = torch.tensor(ea, dtype=torch.long, device=device); ebt = torch.tensor(eb, dtype=torch.long, device=device)
    erl = np.linalg.norm(pos_rest[ea] - pos_rest[eb], axis=1); ert = torch.tensor(erl, dtype=torch.float32, device=device)
    xr = torch.from_numpy(pos_rest).to(device, torch.float32)
    H = build_hierarchy(xr, M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
    stiff_uniform = np.full(N, STIFF_UNIFORM, np.float32); theta = np.array([3., 20., 0.3], np.float32)

    def ctx_for(surf_idx):
        return dict(solver=solver, rest_state=rest_state, N=N, device=device, knife_radius=kr,
                    pos_rest=pos_rest, theta_3d_np=theta, stiff_init_np=stiff_uniform,
                    tau_init_np=np.full(N, TAU_INIT, np.float32), tau_bound=(0.05, 0.95),
                    surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H,
                    group_col_id=col_id, surf_top_axis=top_axis)

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

    deep = make_layered_debulk(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.90, 0.20, 8, 45)

    def gen(traj):
        solver.reset_to_rest(pos_rest)
        op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta, stiff_uniform, traj, kr, device, tau_truth_np=tau_truth)
        return ([torch.tensor(p, device=device) for p in op], [torch.tensor(a, device=device) for a in oa])

    def calib(traj, surf_idx, k):
        ctx = ctx_for(surf_idx)
        op, oa = gen(traj)
        _, tau, _ = calibrate_online("Group+L_pos_partial", ctx, [(traj, op, oa)], N_passes=4, lr=0.03,
                                     lr_tau=0.5, lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=1.0,
                                     return_best=True, verbose=False)
        tt = BANDS[k][3]
        return float((tau[band_mask[k]].mean() - TAU_INIT) / (tt - TAU_INIT) * 100)

    def surf_band(k):
        _, lo, hi, _, _ = BANDS[k]
        sk = (fr >= (hi - 0.10)) & (fr < (hi + 0.02))
        return torch.tensor(np.where(sk)[0], dtype=torch.long, device=device)

    print(f"\n=== v2: PRISTINE (v1-ref) vs PREDAMAGED (deep cut), ×{REP} ===")
    pri = {k: [] for k in range(3)}; pre = {k: [] for k in range(3)}
    for r in range(REP):
        for k in range(3):
            si = surf_band(k)
            pri[k].append(calib(make_raster_at(BANDS[k][4]), si, k))   # pristine: shallow raster from rest
            pre[k].append(calib(deep, si, k))                          # predamaged: deep continuous cut
    print(f"  {'band':6} | {'PRISTINE(v1)':>14} | {'PREDAMAGED(v2)':>15} | {'ratio v2/v1':>11}")
    for k, (name, lo, hi, tt, rf) in enumerate(BANDS):
        a = np.array(pri[k]); b = np.array(pre[k])
        ratio = b.mean() / a.mean() if abs(a.mean()) > 1 else float('nan')
        print(f"  {name:6} | {a.mean():6.1f}±{a.std():4.1f} | {b.mean():6.1f}±{b.std():4.1f}  | {ratio:6.2f}", flush=True)
    mk = 1
    a = np.array(pri[mk]).mean(); b = np.array(pre[mk]).mean()
    print(f"\n  MIDDLE (buried hard): PRISTINE {a:.0f}%  PREDAMAGED {b:.0f}%  -> degraded to {100*b/max(a,1e-6):.0f}% of ideal "
          f"({'robust (≥80%)' if b > 0.8*a else 'cutting-damage HURTS -> needs F/T anchor correction'})", flush=True)


if __name__ == "__main__":
    main()
