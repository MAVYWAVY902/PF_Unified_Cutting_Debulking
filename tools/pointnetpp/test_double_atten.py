"""Step 3B (decisive, numpy, no kernel edit): isolate the double-attenuation. On REAL
clusters near the cut (mixed alpha), compute the shape-match correction two ways and
compare for the DAMAGED particles. |corr_warp| < |corr_cpp| => warp weaker (softer) =>
double-attenuation confirmed.
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
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster


def svd_R(A):
    U, S, Vt = np.linalg.svd(A); R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


def main():
    cfg = MESH_CONFIGS[0]; device = "cuda:0"; ta = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
    solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), rings=2, fix_threshold_fraction=0.05)
    N = solver.N; solver.num_iterations = 6; solver.damping = 0.95
    pos_rest = mi["vertices"].copy(); bb0, bb1 = pos_rest.min(0), pos_rest.max(0); sp = bb1 - bb0
    lat = [a for a in range(3) if a != ta]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos_rest[:, 2].min() - 5.0)
    sw = torch.tensor(np.full(N, 0.85, np.float32), device=device)
    twt = torch.tensor(np.full(N, 0.45, np.float32), device=device)
    tht = torch.tensor(np.array([7., 20., 0.45], np.float32), device=device)
    traj = make_raster(bb0, bb1, ta, lat[0], lat[1], 0.85, 12, 20)
    solver.reset_to_rest(pos_rest); st = rest_state
    with torch.no_grad():
        for kp in traj:
            aw, pw, dv = WarpFullForwardOp.apply(tht, solver, st, np.asarray(kp, np.float32), kr, sw, twt)
            st = solver_save_state(solver)
    pos = pw.cpu().numpy().astype(np.float64); alpha = aw.cpu().numpy().astype(np.float64)
    dv_np = dv.cpu().numpy().astype(np.float64)
    starts = solver.c["starts"].numpy(); sizes = solver.c["sizes"].numpy()
    pids = solver.c["particle_ids"].numpy(); roff = solver.c["rest_offsets"].numpy().astype(np.float64)
    STIFF = 0.85

    cw_dam, cc_dam = [], []; n_mixed = 0
    for cid in range(solver.num_clusters):
        s0 = starts[cid]; n = sizes[cid]
        if n < 2: continue
        m = pids[s0:s0 + n]; a = alpha[m]
        dmg = (a > 0.05) & (a < 0.95)
        if not dmg.any(): continue
        n_mixed += 1
        p = pos[m]; q = roff[s0:s0 + n]
        w = a.copy()
        if w.sum() < 1e-6: continue
        cm = (w[:, None] * p).sum(0) / w.sum(); qcm = (w[:, None] * q).sum(0) / w.sum()
        A = (w[:, None] * (p - cm)).T @ (q - qcm)
        Rw = svd_R(A + 1e-6 * np.eye(3))
        corr_w = ((q - qcm) @ Rw.T + cm - p) * w[:, None]
        wb = (a > 0.5).astype(np.float64)
        if wb.sum() < 2: continue
        cmb = (wb[:, None] * p).sum(0) / wb.sum(); qcmb = (wb[:, None] * q).sum(0) / wb.sum()
        Ab = (wb[:, None] * (p - cmb)).T @ (q - qcmb)
        Rc = svd_R(Ab + 1e-6 * np.eye(3))
        corr_c = (q - qcmb) @ Rc.T + cmb - p
        soft = 1.0 - 0.95 * dv_np[m]
        cw_dam += list(np.linalg.norm(STIFF * soft[:, None] * corr_w, axis=1)[dmg])
        cc_dam += list(np.linalg.norm(STIFF * soft[:, None] * corr_c, axis=1)[dmg])
    cw = np.array(cw_dam); cc = np.array(cc_dam)
    print(f"\n=== Step3B: double-attenuation, {n_mixed} mixed-alpha clusters, {len(cw)} damaged particles ===")
    print(f"  mean |correction| warp(alpha-wt)={cw.mean()*1000:.4f}mm  cpp(binary)={cc.mean()*1000:.4f}mm")
    r = cw.mean() / cc.mean()
    print(f"  ratio warp/cpp = {r:.3f} -> "
          f"{'warp WEAKER (double-attenuation CONFIRMED)' if r < 0.85 else ('similar' if r < 1.15 else 'warp STRONGER')}", flush=True)


if __name__ == "__main__":
    main()
