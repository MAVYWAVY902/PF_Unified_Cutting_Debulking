"""Settle two concerns:
  (1) GRAVITY sanity: settle cube under gravity, NO knife. Does it (a) reach equilibrium (not drift
      forever = gravity bug), (b) sag in -Y (toward the clamped bottom)? Report per-window max disp.
  (2) FLUID = stiffness, not gravity/num_iter: sweep STIFF (k). Higher k → smaller gravity sag (firmer)
      and check stability. If higher k = firmer + stable → fluidity is just low k. If higher k blows up
      → need XPBD for solid tissue.
All at num_iter=1 (eff-stiffness = k, cleanest).
"""
import sys, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS

cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 1; solver.enable_cutting = True
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
fixed = solver.p["fixed"].numpy(); free = fixed == 0
far = pr0.max(0) + 10.
print(f"N={N}  gravity=(0,-9.81,0)=-Y  top_axis={top}(Y)  fixed(bottom-Y)={fixed.sum()}")
print("knife parked far away (no cutting) for the settle test\n")


def settle(stiff_val, steps=120):
    solver.num_iterations = 1; solver.tau_soften = False
    solver.reset_to_rest(pr0)
    sw = torch.tensor(np.full(N, stiff_val, np.float32), device=dev)
    tw = torch.tensor(np.full(N, 0.3, np.float32), device=dev)
    tht = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=dev)
    st = rest0; prev = pr0.copy()
    log = []
    with torch.no_grad():
        for i in range(steps):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, tw)
            st = solver_save_state(solver)
            cur = pw.cpu().numpy()
            step_move = np.linalg.norm(cur - prev, axis=1)[free].max() * 1e3  # mm, this step
            prev = cur
            if (i + 1) % 30 == 0:
                log.append((i + 1, step_move))
    final = cur
    sag = final[free] - pr0[free]                      # displacement from initial
    sag_y = sag[:, 1].mean() * 1e3                      # mean Y displacement (mm); should be < 0 (-Y)
    sag_mag = np.linalg.norm(sag, axis=1).mean() * 1e3  # mean total displacement (mm) = firmness proxy
    nan = (not np.isfinite(final).all())
    return log, sag_y, sag_mag, nan


print("=== (1) GRAVITY sanity (STIFF=0.5) — per-30-step max single-step move (mm); should DECAY to ~0 ===")
log, sag_y, sag_mag, nan = settle(0.5)
for s, m in log:
    print(f"  step {s:3d}: max single-step move = {m:.4f} mm")
print(f"  → mean free-particle Y displacement = {sag_y:+.3f} mm  (negative = sagged -Y, correct)")
print(f"  → mean total displacement (sag magnitude) = {sag_mag:.3f} mm   NaN={nan}")
print(f"  VERDICT: if move decays to ~0 and Y-disp<0 → gravity correct & stable (not a bug)\n")

print("=== (2) FLUID = stiffness? sweep STIFF, settle under gravity, measure sag (firmness) ===")
print(f"  {'STIFF(k)':>9s}  {'sag_mag(mm)':>11s}  {'Y-disp(mm)':>11s}  {'final move(mm)':>14s}  stable?")
for k in (0.5, 0.8, 0.95):
    log, sag_y, sag_mag, nan = settle(k)
    fin_move = log[-1][1]
    print(f"  {k:>9.2f}  {sag_mag:>11.3f}  {sag_y:>+11.3f}  {fin_move:>14.4f}  {'NO(NaN)' if nan else 'yes'}")
print("\n  higher k → smaller sag = firmer (fluidity is low k, not gravity)")
print("  if higher k stays stable → calibrate at higher k for solid tissue; if blows up → XPBD needed")
