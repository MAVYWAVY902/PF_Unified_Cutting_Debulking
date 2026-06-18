"""Step 1: non-destructive PRESS k-calibration loop (per-frame Adam on k_pp) + smoke recovery.
PRESS mode (enable_cutting=False), knife sphere = position-controlled indenter ramp + hold.
Obs generated at k_truth; recover from k_init via per-frame Adam (state detached between frames,
like calibrate_online), return_best. Smoke = uniform k recovery (does loss drop, k → truth?).
"""
import sys, time, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS

cfg = MESH_CONFIGS[0]; dev = "cuda:0"; THETA = np.array([3., 20., 0.3], np.float32)
PRESS_R = 0.08; K_TRUTH = 0.50; K_INIT = 0.35
DEPTH = 0.20; T_RAMP = 60; T_HOLD = 20                      # short for smoke
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2; solver.substeps = 1
solver.enable_cutting = False; solver.knife_active = 1; solver.knife_radius = PRESS_R
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
top = cfg["top_axis"]
tht = torch.tensor(THETA, device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, PRESS_R, swt(K_TRUTH), None); st = solver_save_state(solver)
prs = pw.cpu().numpy().copy(); REST = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
bb0, bb1 = prs.min(0), prs.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; clat = [0.5 * (bb0[a] + bb1[a]) for a in lat]
surf = prs[:, top] > (bb1[top] - 0.15 * bs[top]); surf_t = torch.tensor(np.where(surf)[0], device=dev)
z_start = bb1[top] + PRESS_R; z_end = bb1[top] - DEPTH * bs[top]
def knife_at(f):
    frac = min(1.0, f / T_RAMP); zc = z_start + frac * (z_end - z_start)
    kp = np.zeros(3, np.float32); kp[top] = zc; kp[lat[0]] = clat[0]; kp[lat[1]] = clat[1]; return kp
traj = [knife_at(f) for f in range(T_RAMP + T_HOLD)]

def gen_obs(k_np):
    solver.reset_to_rest(prs); st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in REST.items()}
    obs = []
    with torch.no_grad():
        for kp in traj:
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, kp, PRESS_R, swt(0.0) + torch.tensor(k_np, device=dev), None)
            st = solver_save_state(solver); obs.append(pw.detach().clone())
    return obs

def calibrate(k_init_np, obs, N_passes=20, lr=0.03):
    """Full-trajectory backprop, ONE Adam step per pass: k is CONSTANT within a pass
    (consistent rollout, no within-pass state divergence / overshoot). Per-frame losses'
    gradients accumulate into the SAME k_t.grad (PyTorch accumulates), one opt.step()/pass."""
    k_t = torch.tensor(k_init_np, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([k_t], lr=lr)
    best = (float("inf"), k_init_np.copy()); hist = []
    for p in range(N_passes):
        solver.reset_to_rest(prs); st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in REST.items()}
        opt.zero_grad(); ploss = 0.0
        for f, kp in enumerate(traj):
            _, pos, _ = WarpFullForwardOp.apply(tht, solver, st, kp, PRESS_R, k_t, None)
            st = solver_save_state(solver)               # detached state carry
            L = ((pos[surf_t] - obs[f][surf_t]) ** 2).sum()
            L.backward()                                  # ACCUMULATE grad into k_t.grad
            ploss += float(L.item())
        opt.step()                                        # ONE step per pass
        with torch.no_grad(): k_t.clamp_(0.05, 2.0)
        km = k_t.detach().cpu().numpy()
        hist.append((p, ploss / len(traj), float(km.mean())))
        if ploss < best[0]: best = (ploss, km.copy())
    return best[1], hist

def calibrate_scalar(k0, obs, N_passes=20, lr=0.03):
    """SINGLE-SCALAR k (1 DOF, broadcast to all N). If this recovers 0.5 cleanly while the
    per-particle run overshoots, the bias is per-particle UNDER-DETERMINATION (→ grouping)."""
    k_s = torch.tensor([k0], dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([k_s], lr=lr)
    best = (float("inf"), k0); hist = []
    for p in range(N_passes):
        solver.reset_to_rest(prs); st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in REST.items()}
        opt.zero_grad(); ploss = 0.0
        for f, kp in enumerate(traj):
            _, pos, _ = WarpFullForwardOp.apply(tht, solver, st, kp, PRESS_R, k_s.expand(N), None)
            st = solver_save_state(solver)
            L = ((pos[surf_t] - obs[f][surf_t]) ** 2).sum()
            L.backward(); ploss += float(L.item())
        opt.step()
        with torch.no_grad(): k_s.clamp_(0.05, 2.0)
        hist.append((p, ploss / len(traj), float(k_s.item())))
        if ploss < best[0]: best = (ploss, float(k_s.item()))
    return best[1], hist


print(f"N={N} surf={int(surf.sum())}  PRESS k-recovery  k_truth={K_TRUTH} k_init={K_INIT}  ramp{T_RAMP}+hold{T_HOLD}")
obs = gen_obs(np.full(N, K_TRUTH, np.float32))
near = ((prs[:, lat[0]] - clat[0]) ** 2 + (prs[:, lat[1]] - clat[1]) ** 2) < (PRESS_R * 1.5) ** 2

t0 = time.time()
k_sc, hist_s = calibrate_scalar(K_INIT, obs, N_passes=20, lr=0.03)
print(f"\n--- SINGLE-SCALAR k (1 DOF) ---")
print(f"{'pass':>4s} {'mean_Lgap':>12s} {'k':>8s}")
for p, l, km in hist_s:
    if p % 2 == 0 or p == hist_s[-1][0]:
        print(f"{p:4d} {l:12.4e} {km:8.4f}")
print(f"recovered scalar k = {k_sc:.4f}  (truth {K_TRUTH})  [{time.time()-t0:.0f}s]")

t1 = time.time()
k_rec, hist = calibrate(np.full(N, K_INIT, np.float32), obs, N_passes=20, lr=0.03)
print(f"\n--- PER-PARTICLE k (N DOF, for comparison) ---")
print(f"recovered: ALL mean={k_rec.mean():.4f}  UNDER-INDENTER mean={k_rec[near & surf].mean():.4f}  (truth {K_TRUTH})  [{time.time()-t1:.0f}s]")
print(f"\nVERDICT: if scalar→~0.50 clean but per-particle under-indenter overshoots → "
      f"per-particle UNDER-DETERMINATION confirmed → grouping is the fix.")
solver.enable_cutting = True; solver.knife_active = 0
