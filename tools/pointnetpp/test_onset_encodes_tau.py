"""Minimal prototype: does FRACTURE ONSET encode τ? Repeated shallow grazing raster (each rep
sweeps the whole surface; cumulative damage ramps over reps). Per surface particle, onset = first
frame α<0.5. Hypothesis: hard (τ=0.55) fractures LATER than soft (τ=0.15) → onset ∝ τ → τ readable
from the surface time-series (no depth, no saturation). Sweep grazing depth to find a regime where
onset SPREADS (not all-at-once, not none). DIAG truth, obs only (forward), corr-free-friendly.
"""
import sys, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD, TAU_SOFT, STIFF = 0.55, 0.15, 0.5
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2; solver.enable_cutting = True
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
tht = torch.tensor([3., 20., 0.3], device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, swt(STIFF), swt(0.3)); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
surf = pr[:, top] > (bb1[top] - 0.10 * bs[top])
xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
print(f"N={N} surf={int(surf.sum())}  onset-encodes-τ prototype (repeated grazing raster, DIAG truth)")


def onset_test(y_frac, R, n_lines=4, t_line=10):
    one = make_raster(bb0, bb1, top, sweep, la, y_frac, n_lines, t_line)
    traj = one * R                                # repeat R times: each rep sweeps whole surface
    solver.reset_to_rest(pr)
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                        np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
    A = np.stack([a for a in oa], 0)              # (T, N) alpha time series
    T = A.shape[0]
    fractured = (A < 0.5).any(0)                  # ever fractured?
    onset = np.where(fractured, (A < 0.5).argmax(0), T)   # first frame α<0.5 (T = never)
    sf = surf
    def reg(mask):
        m = mask & sf & fractured
        return (m.sum(), onset[mask & sf].astype(float))
    nh, oh = reg(hard); ns, os = reg(~hard)
    fr_h = 100 * (fractured & hard & sf).sum() / max((hard & sf).sum(), 1)
    fr_s = 100 * (fractured & ~hard & sf).sum() / max((~hard & sf).sum(), 1)
    # onset among FRACTURED surface particles
    oh_f = onset[hard & sf & fractured]; os_f = onset[~hard & sf & fractured]
    # correlation onset vs tau among fractured surface
    fm = sf & fractured
    corr = float(np.corrcoef(onset[fm], tau_truth[fm])[0, 1]) if fm.sum() > 10 else float("nan")
    return T, fr_h, fr_s, (oh_f.mean() if len(oh_f) else np.nan), (os_f.mean() if len(os_f) else np.nan), corr


print(f"\n{'y_frac':>7s} {'R':>3s} {'T':>5s} {'frac_hard':>9s} {'frac_soft':>9s} {'onset_hard':>10s} {'onset_soft':>10s} {'corr(onset,τ)':>13s}")
for yf in (0.90, 0.93, 0.96):
    for R in (12,):
        T, fh, fs, oh, os, c = onset_test(yf, R)
        print(f"{yf:7.2f} {R:3d} {T:5d} {fh:8.1f}% {fs:8.1f}% {oh:10.1f} {os:10.1f} {c:12.3f}", flush=True)
print("\nwant: onset_hard > onset_soft (hard fractures later) + corr(onset,τ)>0 + both partially fracture")
print("      = fracture ONSET encodes τ from surface time-series (the load-to-failure signal)")
