"""Non-destructive PRESS verification (solver.enable_cutting=False):
 (A) STABILITY: does an embedded sphere indenter (sustained residual, no removal) stay
     bounded across increasing depth + num_iter=2? (predict: yes, residual reaches equilibrium)
 (B) k-OBSERVABILITY: do soft (k=0.2) vs stiff (k=0.8) give DISTINGUISHABLE surface
     deformation? (necessary for stiffness calibration)
 (C) confirm NO removal (α stays ~1, topology intact).
Knife sphere = position-controlled indenter, ramped down into the top, then held.
"""
import sys, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS

cfg = MESH_CONFIGS[0]; dev = "cuda:0"; THETA = np.array([3., 20., 0.3], np.float32)
PRESS_R = 0.08          # broad indenter (~5 edges) for a meaningful footprint
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2; solver.substeps = 1
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
top = cfg["top_axis"]
tht = torch.tensor(THETA, device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
# settle (knife far)
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, solver.knife_radius, swt(0.5), swt(0.3)); st = solver_save_state(solver)
prs = pw.cpu().numpy().copy(); REST = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
bb0, bb1 = prs.min(0), prs.max(0); bs = bb1 - bb0; center = 0.5 * (bb0 + bb1); diag = float(np.linalg.norm(bs))
lat = [a for a in range(3) if a != top]
clat = [0.5 * (bb0[a] + bb1[a]) for a in lat]
# surface particles directly under the indenter (top 12% + within PRESS_R laterally)
surf = prs[:, top] > (bb1[top] - 0.12 * bs[top])
near = ((prs[:, lat[0]] - clat[0]) ** 2 + (prs[:, lat[1]] - clat[1]) ** 2) < (PRESS_R * 1.2) ** 2
probe = surf & near
print(f"N={N} diag={diag:.3f}m  indenter_r={PRESS_R*1000:.0f}mm  probe particles under indenter: {int(probe.sum())}")


def press(stiff, depth_frac, T_ramp=80, T_hold=40):
    solver.enable_cutting = False; solver.knife_active = 1; solver.knife_radius = PRESS_R
    solver.num_iterations = 2; solver.muller_stiff_remap = False; solver.alpha_vel_scale = 0.0
    solver.reset_to_rest(prs); st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in REST.items()}
    sw, twt = swt(stiff), swt(0.30)
    z_start = bb1[top] + PRESS_R; z_end = bb1[top] - depth_frac * bs[top]
    maxpos = 0.0; nan = False
    with torch.no_grad():
        for s in range(T_ramp + T_hold):
            frac = min(1.0, s / T_ramp); zc = z_start + frac * (z_end - z_start)
            kp = np.zeros(3, np.float32); kp[top] = zc; kp[lat[0]] = clat[0]; kp[lat[1]] = clat[1]
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, kp, PRESS_R, sw, twt)
            st = solver_save_state(solver); p = st["pos"]
            if not np.isfinite(p).all(): nan = True; break
            maxpos = max(maxpos, float(np.abs(p - center).max()))
    a_min = float(solver.p["alpha"].numpy().min())
    defl = float((st["pos"][probe, top] - prs[probe, top]).mean() * 1000) if not nan else float("nan")  # mm (neg=down)
    return maxpos, defl, a_min, nan


print(f"\n{'depth':>6s} {'k=0.2(soft) defl':>17s} {'k=0.8(stiff) defl':>18s} {'|Δ|(obs)':>9s}  {'maxpos soft/stiff':>20s}  {'α_min':>7s}")
for depth in (0.10, 0.20, 0.30):
    mp_s, d_s, am_s, nan_s = press(0.2, depth)
    mp_h, d_h, am_h, nan_h = press(0.8, depth)
    stab = f"{mp_s:.2f}/{mp_h:.2f}m"
    flag_s = "NaN" if nan_s else ("BLEW" if mp_s > 3 * diag else "ok")
    flag_h = "NaN" if nan_h else ("BLEW" if mp_h > 3 * diag else "ok")
    obs = abs(d_s - d_h)
    print(f"{depth:6.2f} {d_s:14.2f}mm {d_h:15.2f}mm {obs:8.2f}mm  {stab:>20s}  {min(am_s,am_h):7.4f}  [{flag_s}/{flag_h}]", flush=True)
print("\nwant: stable (no BLEW/NaN), α_min≈1 (no removal), |Δ| >> 0 (soft vs stiff distinguishable = k observable)")
solver.enable_cutting = True; solver.knife_active = 0
