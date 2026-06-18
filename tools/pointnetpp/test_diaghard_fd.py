"""Distinguish DIAG hard under-fit root cause: TRUE observability limit vs LOSS-FORM (chamfer NN) limit.

DIAG τ field (hard=0.55 stripes, soft=0.15). With tau_soften ON, FD ∂pos/∂τ measured SEPARATELY on
hard-process-zone vs soft-process-zone particles (perturb that region's τ, measure its own Δpos).
  hard ∂pos/∂τ ≈ 0           → TRUE observability limit (hard too stiff, no pre-fracture signal) → F/T justified
  hard ∂pos/∂τ comparable    → signal EXISTS; chamfer NN-misassignment loses it → structure-aware loss, NOT F/T
Stable window: num_iter=1, shallow graze, fork one step from a saved state.
"""
import sys, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD, TAU_SOFT, STIFF, REACH_MM, R = 0.55, 0.15, 0.5, 4.0, 3
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 1; solver.enable_cutting = True
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
print(f"N={N} DIAG-hard FD: observability-limit vs chamfer-loss-limit  num_iter=1")

solver.reset_to_rest(pr0); far = pr0.max(0) + 10.
sw = torch.tensor(np.full(N, STIFF, np.float32), device=dev)
tht = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=dev); st = rest0
tw_settle = torch.tensor(np.full(N, 0.3, np.float32), device=dev)
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, tw_settle); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
tw = torch.tensor(tau_truth, device=dev)
y_center = bb1[top] - REACH_MM * 1e-3 + kr
base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
one = [(kp.copy().astype(np.float32)) for kp in base]
for kp in one: kp[top] = np.float32(y_center)
traj = one * R
thetat = torch.tensor(np.array([3., 20., 0.3], np.float32), device=dev)

solver.tau_soften = True
T = min(70, len(traj) - 1)
solver.reset_to_rest(pr); S = rest_state
with torch.no_grad():
    for t in range(T):
        _, _, _ = WarpFullForwardOp.apply(thetat, solver, S, traj[t], kr, sw, tw); S = solver_save_state(solver)

dmg = S["particle_damage"]; alive = S["alpha"] > 0.5
ratio = dmg / tau_truth
hard_pz = alive & hard & (ratio > 0.2) & (ratio < 0.9)
soft_pz = alive & (~hard) & (ratio > 0.2) & (ratio < 0.9)
print(f"frame {T}: hard process-zone={hard_pz.sum()}  soft process-zone={soft_pz.sum()}")
print(f"  hard d/τ alive range [{ratio[alive & hard].min():.2f},{ratio[alive & hard].max():.2f}]  "
      f"frac hard with d>0: {(dmg[hard] > 1e-6).mean():.2f}")


def fd_region(region_mask, delta=1e-3):
    """Perturb τ ONLY on region_mask particles, measure RMS ∂pos/∂τ on that region's process-zone."""
    knife = traj[T]
    with torch.no_grad():
        _, p0, _ = WarpFullForwardOp.apply(thetat, solver, S, knife, kr, sw, tw)
        tw1 = tw.clone(); tw1[torch.tensor(region_mask, device=dev)] += delta
        _, p1, _ = WarpFullForwardOp.apply(thetat, solver, S, knife, kr, sw, tw1)
    g = (p1 - p0).cpu().numpy() / delta
    return g


g_hard = fd_region(hard)
g_soft = fd_region(~hard)
rms_hard = np.sqrt((g_hard[hard_pz] ** 2).sum(1)).mean()
rms_soft = np.sqrt((g_soft[soft_pz] ** 2).sum(1)).mean()
print(f"\n∂pos/∂τ RMS (m per unit τ), tau_soften ON:")
print(f"  HARD region (perturb τ_hard, measure hard PZ): {rms_hard:.6e}")
print(f"  SOFT region (perturb τ_soft, measure soft PZ): {rms_soft:.6e}")
print(f"  hard/soft ratio: {rms_hard / (rms_soft + 1e-12):.3f}")
print("\nVERDICT:")
print("  hard RMS ≈ 0 (<<soft)  → TRUE observability limit → F/T justified")
print("  hard RMS comparable    → signal EXISTS; DIAG-hard under-fit is chamfer NN loss-form limit")
print("                            → structure-aware/obs-weighted position loss, NOT F/T")
