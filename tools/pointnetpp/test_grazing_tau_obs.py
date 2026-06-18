"""Grazing-cut τ observability sweep: is there a SHALLOW 'grazing' sweet spot that keeps
particles in the threshold band (α≈0.5, where ∂α/∂τ=s·α(1-α) is maximal) → strong τ signal +
stable, vs DEEP gross cut (α→0, signal lost + blow-up)?

Sweep cut depth y_frac (shallow→deep). Per depth, per frame measure:
  n_band = #particles 0.2<α<0.8 (τ-informative population)
  ‖g_τ‖  = ‖∂L_cav/∂τ‖ (single-frame backward, L_cav=‖α_sim−α_obs‖² on surface)
  max|pos-c| (blow-up marker)
Report per depth: Σ‖g_τ‖ (total τ signal), peak n_band, blow onset, peak max|pos|.
Uniform τ truth/init (magnitude probe). enable_cutting=True (debulk/cut mode).
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

cfg = MESH_CONFIGS[0]; dev = "cuda:0"; THETA = np.array([3., 20., 0.3], np.float32)
TAU_TRUTH, TAU_INIT, STIFF = 0.30, 0.40, 0.5
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2; solver.substeps = 1
solver.enable_cutting = True
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
tht = torch.tensor(THETA, device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, swt(STIFF), swt(TAU_INIT)); st = solver_save_state(solver)
prs = pw.cpu().numpy().copy(); REST = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
bb0, bb1 = prs.min(0), prs.max(0); bs = bb1 - bb0; center = 0.5 * (bb0 + bb1); diag = float(np.linalg.norm(bs))
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
surf = prs[:, top] > (bb1[top] - 0.12 * bs[top]); surf_t = torch.tensor(np.where(surf)[0], device=dev)
print(f"N={N} diag={diag:.3f}m  τ_truth={TAU_TRUTH} τ_init={TAU_INIT}  (deep raster default was y~0.45/0.93)")


def sweep_depth(y_frac):
    traj = make_raster(bb0, bb1, top, sweep, la, y_frac, 6, 20)
    # obs at truth
    solver.reset_to_rest(prs)
    op, oa = gen_obs_per_particle_stiff(solver, rest0 if False else {k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in REST.items()},
                                        np.array([3.,20.,0.3],np.float32), np.full(N,STIFF,np.float32), traj, kr, dev,
                                        tau_truth_np=np.full(N, TAU_TRUTH, np.float32))
    obs_a = [torch.tensor(a, device=dev) for a in oa]
    # roll perturbed, per-frame band + g_tau + maxpos
    st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in REST.items()}
    sumg = 0.0; peak_band = 0; peak_pos = 0.0; onset = None
    for t, kp in enumerate(traj):
        s_t = swt(STIFF).clone(); tau_t = swt(TAU_INIT).clone().requires_grad_(True)
        a_c, p_c, _ = WarpFullForwardOp.apply(tht, solver, {k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in st.items()},
                                              kp.astype(np.float32), kr, s_t, tau_t)
        L = ((a_c[surf_t] - obs_a[t][surf_t]) ** 2).sum()
        L.backward(); sumg += float(tau_t.grad.norm())
        st = solver_save_state(solver)
        a = solver.p["alpha"].numpy(); p = st["pos"]
        peak_band = max(peak_band, int(((a > 0.2) & (a < 0.8)).sum()))
        mp = float(np.abs(p - center).max()); peak_pos = max(peak_pos, mp)
        if onset is None and mp > 2 * diag: onset = t
    final_removed = float((solver.p["alpha"].numpy() < 0.5).mean()) * 100
    return sumg, peak_band, peak_pos, onset, len(traj), final_removed


print(f"\n{'y_frac':>7s} {'depth':>10s} {'Σ‖gτ‖':>9s} {'peak_band':>9s} {'removed%':>8s} {'onset':>7s} {'peak|pos|':>11s}")
for yf in (0.97, 0.93, 0.88, 0.80, 0.70, 0.55):
    sg, pb, pp, on, nf, rem = sweep_depth(yf)
    label = "grazing" if yf >= 0.9 else ("mid" if yf >= 0.75 else "deep/gross")
    ons = "none" if on is None else f"f{on}"
    pps = f"{pp:.2f}m" if pp < 1e4 else f"{pp:.1e}m"
    print(f"{yf:7.2f} {label:>10s} {sg:9.3f} {pb:9d} {rem:7.1f}% {ons:>7s} {pps:>11s}", flush=True)
print("\nwant: grazing (shallow) → high Σ‖gτ‖ + high peak_band + low removed% + stable (onset none);")
print("      deep → band collapses (α→0), τ-signal lost, blows up.")
