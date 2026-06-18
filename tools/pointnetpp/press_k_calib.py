"""Step 2: non-destructive PRESS stiffness (k) calibration — core forward + L_gap + FD parity.

press_forward_loss(): reusable building block (enable_cutting=False, knife sphere indenter).
FD parity: autograd ∂L_gap/∂k_pp vs central finite-difference on top-K particles (under press
excitation). Mirrors stageA_fd_parity but in PRESS mode. Pass = median rel_err<5%, sign>90%.
Probe is taken MID-RAMP (indenter still descending) so the surface is in active, k-dependent
motion (strong gradient).
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
PRESS_R = 0.08; K_SETUP = 0.5; K_TRUTH = 0.5; K_INIT = 0.35
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2; solver.substeps = 1
solver.enable_cutting = False                      # PRESS mode (no removal)
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
top = cfg["top_axis"]
tht = torch.tensor(THETA, device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
solver.knife_active = 1; solver.knife_radius = PRESS_R
# settle (knife far, cutting off)
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, PRESS_R, swt(K_SETUP), None); st = solver_save_state(solver)
prs = pw.cpu().numpy().copy(); REST = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
bb0, bb1 = prs.min(0), prs.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; clat = [0.5 * (bb0[a] + bb1[a]) for a in lat]
surf = prs[:, top] > (bb1[top] - 0.15 * bs[top]); surf_t = torch.tensor(np.where(surf)[0], device=dev)
DEPTH = 0.20; T_RAMP = 80; T_PROBE = 60                 # probe mid-ramp (descending)
z_start = bb1[top] + PRESS_R; z_end = bb1[top] - DEPTH * bs[top]

def knife_at(frame):
    frac = min(1.0, frame / T_RAMP); zc = z_start + frac * (z_end - z_start)
    kp = np.zeros(3, np.float32); kp[top] = zc; kp[lat[0]] = clat[0]; kp[lat[1]] = clat[1]; return kp

# ramp to T_PROBE-1 (no grad) -> pre-state S (indenter still descending)
solver.reset_to_rest(prs); st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in REST.items()}
with torch.no_grad():
    for f in range(T_PROBE):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, knife_at(f), PRESS_R, swt(K_SETUP), None); st = solver_save_state(solver)
S = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
kp_probe = knife_at(T_PROBE)
print(f"N={N} surf={int(surf.sum())}  PRESS mode (enable_cutting={solver.enable_cutting})  depth={DEPTH} probe@f{T_PROBE}")


def loss_at(k_np, targets=None, want_grad=False):
    s_t = torch.tensor(k_np, dtype=torch.float32, device=dev, requires_grad=want_grad)
    a, p, _ = WarpFullForwardOp.apply(tht, solver, {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in S.items()},
                                      kp_probe, PRESS_R, s_t, None)
    if targets is None:
        return p.detach().clone()
    L = ((p[surf_t] - targets[surf_t]) ** 2).sum()
    if not want_grad:
        return float(L.item())
    L.backward()
    return float(L.item()), s_t.grad.cpu().numpy().copy()

# obs at truth
p_truth = loss_at(np.full(N, K_TRUTH, np.float32))
assert np.isfinite(p_truth.cpu().numpy()).all(), "press forward not finite"
# autograd grad at init
L0, g = loss_at(np.full(N, K_INIT, np.float32), targets=p_truth, want_grad=True)
print(f"loss(k_init={K_INIT})={L0:.4e}  ‖∂L/∂k‖={np.linalg.norm(g):.4e}  nnz={int((g!=0).sum())}/{N}")

# FD parity on top-K by |g|
K = 12; eps = 0.03; base = np.full(N, K_INIT, np.float32)
top_idx = np.argsort(np.abs(g))[::-1][:K]
rels, signs = [], []
print(f"\n{'idx':>6s} {'g_autograd':>13s} {'g_FD':>13s} {'rel%':>8s} sign")
for idx in top_idx:
    bp = base.copy(); bp[int(idx)] += eps; bm = base.copy(); bm[int(idx)] -= eps
    Lp = loss_at(bp, targets=p_truth); Lm = loss_at(bm, targets=p_truth)
    gfd = (Lp - Lm) / (2 * eps); ga = g[int(idx)]
    rel = abs(ga - gfd) / max(abs(gfd), 1e-12) * 100
    ok = np.sign(ga) == np.sign(gfd)
    rels.append(rel); signs.append(ok)
    print(f"{int(idx):6d} {ga:+13.4e} {gfd:+13.4e} {rel:7.2f}% {'✓' if ok else '✗'}")
med = float(np.median(rels)); sgn = 100 * sum(signs) / len(signs)
print(f"\nmedian rel_err={med:.2f}%  sign-match={sgn:.0f}%")
print("PASS ✓ — press ∂L_gap/∂k autograd matches FD" if (med < 5 and sgn > 90)
      else ("PARTIAL ⚠" if med < 20 else "FAIL ✗"))
solver.enable_cutting = True; solver.knife_active = 0
