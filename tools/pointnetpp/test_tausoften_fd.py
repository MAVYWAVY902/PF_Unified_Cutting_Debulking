"""FD check for (b) τ-coupled cohesive softening (local_stiff = k·(1-0.95·clamp(d/τ,0,1))).

Goal: verify the POSITION channel is woken — ∂pos/∂τ on pre-fracture particles (0.2<d/τ<0.9)
should be ~0 with tau_soften OFF (current τ-blind position channel) and clearly >0 with it ON.
Also cross-check autodiff ∂L/∂τ vs FD (confirms the kernel's new d/τ branch is grad-correct).

Done in a STABLE window (num_iter=1, shallow light graze) so FD isn't corrupted by blow-up.
Forks at frame T from one saved state, takes ONE step with τ vs τ+δ → clean per-frame ∂pos/∂τ.
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

TAU0, STIFF, REACH_MM, R = 0.35, 0.5, 4.0, 3
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 1; solver.enable_cutting = True   # STABLE window
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
print(f"N={N} (b) τ-soften FD check  num_iter={solver.num_iterations}  reach={REACH_MM}mm")

# settle
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.
sw = torch.tensor(np.full(N, STIFF, np.float32), device=dev); tw = torch.tensor(np.full(N, TAU0, np.float32), device=dev)
tht = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=dev); st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, tw); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
y_center = bb1[top] - REACH_MM * 1e-3 + kr
base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
one = [(kp.copy().astype(np.float32)) for kp in base]
for kp in one: kp[top] = np.float32(y_center)
traj = one * R
theta = np.array([3., 20., 0.3], np.float32); thetat = torch.tensor(theta, device=dev)


def run_to(T, tau_soft):
    """Run rest→T-1, return saved state S and the knife pos at frame T-1's NEXT step."""
    solver.tau_soften = tau_soft
    solver.reset_to_rest(pr); st = rest_state
    with torch.no_grad():
        for t in range(T):
            _, _, _ = WarpFullForwardOp.apply(thetat, solver, st, traj[t], kr, sw, tw)
            st = solver_save_state(solver)
    return st


T = min(70, len(traj) - 1)
# pick the FD probe state at frame T (after T steps of cutting → process zone exists)
S = run_to(T, True)
dmg = S["particle_damage"]; alive = S["alpha"] > 0.5
ratio = dmg / TAU0
pf = alive & (ratio > 0.2) & (ratio < 0.9)        # pre-fracture process-zone particles
print(f"frame {T}: alive={alive.sum()}  process-zone(0.2<d/τ<0.9)={pf.sum()}  "
      f"d/τ range [{ratio[alive].min():.2f},{ratio[alive].max():.2f}]")


def fd_grad(tau_soft, delta=1e-3):
    solver.tau_soften = tau_soft
    knife = traj[T]
    with torch.no_grad():
        _, p0, _ = WarpFullForwardOp.apply(thetat, solver, S, knife, kr, sw, tw)
        tw1 = tw + delta
        _, p1, _ = WarpFullForwardOp.apply(thetat, solver, S, knife, kr, sw, tw1)
    g = (p1 - p0).cpu().numpy() / delta            # (N,3) ∂pos/∂τ
    return np.sqrt((g[pf] ** 2).sum(1)).mean()     # RMS |∂pos/∂τ| over process-zone particles (m per unit τ)


g_off = fd_grad(False)
g_on = fd_grad(True)
print(f"\n∂pos/∂τ RMS on process-zone particles (m per unit τ):")
print(f"  tau_soften OFF : {g_off:.6e}   ← current (position τ-blind)")
print(f"  tau_soften ON  : {g_on:.6e}   ← (b) d/τ softening")
print(f"  ratio ON/OFF   : {g_on / (g_off + 1e-12):.1f}×")

# autodiff vs FD on a scalar τ bump (confirms new branch is grad-correct)
solver.tau_soften = True
s = torch.zeros(1, device=dev, requires_grad=True)
tw_leaf = tw + s
_, p_ad, _ = WarpFullForwardOp.apply(thetat, solver, S, traj[T], kr, sw, tw_leaf)
pf_t = torch.tensor(pf, device=dev)
L = (p_ad[pf_t] ** 2).sum()
L.backward()
g_ad = float(s.grad.item())
# matching FD of the same scalar
with torch.no_grad():
    _, pa, _ = WarpFullForwardOp.apply(thetat, solver, S, traj[T], kr, sw, tw)
    _, pb, _ = WarpFullForwardOp.apply(thetat, solver, S, traj[T], kr, sw, tw + 1e-3)
    La = float((pa[pf_t] ** 2).sum()); Lb = float((pb[pf_t] ** 2).sum())
g_fd = (Lb - La) / 1e-3
print(f"\nautodiff vs FD on scalar τ bump (tau_soften ON):")
print(f"  autodiff dL/dτ = {g_ad:.6e}")
print(f"  FD       dL/dτ = {g_fd:.6e}")
print(f"  rel_err        = {abs(g_ad - g_fd) / (abs(g_fd) + 1e-12) * 100:.2f}%")
print("\nwant: ON >> OFF (channel woken) AND autodiff≈FD (new branch grad-correct)")
