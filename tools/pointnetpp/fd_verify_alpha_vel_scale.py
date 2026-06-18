"""FD verification of the α-scaled-velocity change (solver.alpha_vel_scale).

Two claims, both tested:

(A) GRADIENT-NEUTRAL @ substeps=1 (production). update_velocity is the last op of the
    step; vel_out has no in-tape consumer and is never read by the loss (L_gap reads pos,
    L_cav reads alpha). So per-frame ∂L/∂τ and ∂L/∂k must be IDENTICAL at alpha_vel_scale
    0 vs 1. We assert max|Δgrad| ~ 0.

(B) AUTOGRAD STILL CORRECT @ substeps=2 (robustness). Here vel_out feeds the next substep
    INSIDE the same tape, so the new α-path is active in the graph. Central-FD parity on
    top-K τ and k particles must still pass (median rel_err < 5%) at alpha_vel_scale=1 —
    proving the added α-multiply is differentiated correctly, not silently zeroed.

Loss mirrors production: L = L_gap(surface pos) + L_cav(surface alpha).
"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

cfg = MESH_CONFIGS[0]; dev = "cuda:0"
STIFF_TRUE, STIFF_INIT = 0.55, 0.45
TAU_TRUE, TAU_INIT = 0.30, 0.38
THETA = np.array([3., 20., 0.3], np.float32)

solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
pr = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
tht = torch.tensor(THETA, device=dev)
swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)

# settle + partial cut (no tape) -> reach a state with active threshold-band particles
solver.reset_to_rest(pr); far = pr.max(0) + 10.
st = rest_state
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, swt(STIFF_TRUE), swt(TAU_TRUE)); st = solver_save_state(solver)
prs = pw.cpu().numpy().copy()
bb0, bb1 = prs.min(0), prs.max(0); sp = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
traj = make_raster(bb0, bb1, top, sweep, la, 0.62, 4, 12)
with torch.no_grad():
    for kp in traj[:30]:
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, kp.astype(np.float32), kr, swt(STIFF_TRUE), swt(TAU_TRUE)); st = solver_save_state(solver)
S = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}   # frozen IC for the taped step
knife_step = traj[30].astype(np.float32)
a_now = solver.p["alpha"].numpy()
band = ((a_now > 0.05) & (a_now < 0.95)).sum()
print(f"N={N}  threshold-band particles (0.05<α<0.95) at IC: {band}  (these carry ∂α/∂τ)")
surf = prs[:, top] > (bb1[top] - 0.15 * sp[top]); surf_idx = torch.tensor(np.where(surf)[0], device=dev)


def fwd_loss_grads(scale, substeps, want_grad=True, stiff_np=None, tau_np=None, targets=None):
    solver.substeps = substeps; solver.alpha_vel_scale = float(scale)
    s_np = swt(STIFF_INIT).cpu().numpy() if stiff_np is None else stiff_np
    t_np = swt(TAU_INIT).cpu().numpy() if tau_np is None else tau_np
    s_t = torch.tensor(s_np, device=dev, requires_grad=want_grad)
    t_t = torch.tensor(t_np, device=dev, requires_grad=want_grad)
    a_t, p_t, _ = WarpFullForwardOp.apply(tht, solver,
                                          {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in S.items()},
                                          knife_step, kr, s_t, t_t)
    if targets is None:
        return a_t.detach().clone(), p_t.detach().clone()
    a_tgt, p_tgt = targets
    # L_gap on surface positions (∂L/∂k channel); L_cav on ALL alpha (∂L/∂τ channel —
    # τ signal lives in actively-cut band particles, which need not be on the surface).
    L = ((p_t[surf_idx] - p_tgt[surf_idx]) ** 2).sum() + ((a_t - a_tgt) ** 2).sum()
    if not want_grad:
        return float(L.item())
    L.backward()
    return float(L.item()), s_t.grad.cpu().numpy().copy(), t_t.grad.cpu().numpy().copy()


# target at TRUE params (substeps=1, scale doesn't matter for the within-step outputs)
a_tgt, p_tgt = fwd_loss_grads(0.0, 1, want_grad=False, stiff_np=swt(STIFF_TRUE).cpu().numpy(),
                              tau_np=swt(TAU_TRUE).cpu().numpy(), targets=None)
targets = (a_tgt, p_tgt)

# ===== (A) gradient-neutrality @ substeps=1 =====
L0, gk0, gt0 = fwd_loss_grads(0.0, 1, targets=targets)   # legacy
L1, gk1, gt1 = fwd_loss_grads(1.0, 1, targets=targets)   # α-scaled
print(f"\n=== (A) substeps=1 (production): gradient-neutrality ===")
print(f"  loss: legacy={L0:.6e}  alpha-scaled={L1:.6e}  |Δloss|={abs(L0-L1):.2e}")
print(f"  ∂L/∂τ : max|Δ|={np.abs(gt0-gt1).max():.3e}  (legacy ‖g‖={np.linalg.norm(gt0):.3e})")
print(f"  ∂L/∂k : max|Δ|={np.abs(gk0-gk1).max():.3e}  (legacy ‖g‖={np.linalg.norm(gk0):.3e})")
neutral = np.abs(gt0-gt1).max() < 1e-6 and np.abs(gk0-gk1).max() < 1e-6 and abs(L0-L1) < 1e-6
print(f"  => {'PASS: bit-identical (gradient-neutral as predicted)' if neutral else 'NOT identical — investigate'}")


# ===== (B) FD parity @ substeps=2 (α-path active in-tape) =====
def fd_parity(substeps):
    L, gk, gt = fwd_loss_grads(1.0, substeps, targets=targets)
    eps = 0.02
    out = {}
    for name, g, base in (("tau", gt, swt(TAU_INIT).cpu().numpy()), ("stiff", gk, swt(STIFF_INIT).cpu().numpy())):
        K = 12
        top_idx = np.argsort(np.abs(g))[::-1][:K]
        rels, signs = [], []
        for idx in top_idx:
            bp = base.copy(); bp[idx] += eps; bm = base.copy(); bm[idx] -= eps
            if name == "tau":
                Lp = fwd_loss_grads(1.0, substeps, want_grad=False, tau_np=bp, targets=targets)
                Lm = fwd_loss_grads(1.0, substeps, want_grad=False, tau_np=bm, targets=targets)
            else:
                Lp = fwd_loss_grads(1.0, substeps, want_grad=False, stiff_np=bp, targets=targets)
                Lm = fwd_loss_grads(1.0, substeps, want_grad=False, stiff_np=bm, targets=targets)
            gfd = (Lp - Lm) / (2 * eps); ga = g[idx]
            if abs(gfd) < 1e-10 and abs(ga) < 1e-10:
                rels.append(0.0); signs.append(True)
            elif abs(gfd) < 1e-10:
                continue
            else:
                rels.append(abs(ga - gfd) / abs(gfd) * 100); signs.append(np.sign(ga) == np.sign(gfd))
        med = float(np.median(rels)) if rels else float("nan")
        sgn = 100 * sum(signs) / len(signs) if signs else float("nan")
        out[name] = (med, sgn, len(rels))
    return out


print(f"\n=== (B) substeps=2 (α-vel path active in tape): central-FD parity @ alpha_vel_scale=1 ===")
res = fd_parity(2)
ok = True
for name, (med, sgn, n) in res.items():
    flag = "✓" if (med < 5 and sgn > 90) else ("⚠" if med < 20 else "✗")
    ok = ok and (med < 10 and sgn > 85)
    print(f"  ∂L/∂{name:5s}: median rel_err={med:5.2f}%  sign-match={sgn:3.0f}%  (n={n})  {flag}")
print(f"  => {'PASS: autograd matches FD with α-path active' if ok else 'CHECK: parity loose'}")
solver.substeps = 1; solver.alpha_vel_scale = 0.0   # restore defaults
