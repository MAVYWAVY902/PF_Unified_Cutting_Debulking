"""RIGOROUS mechanism test for: 'late blow-up is INERT for τ but GARBAGE for stiff-via-cut'.

Measure, PER FRAME along the (perturbed) cutting trajectory, the two calibration channels
directly (single-frame backward, exactly what per-frame Adam uses):
  τ  channel:   L_cav = ||α_sim − α_obs||²(surf);   g_τ = ∂L_cav/∂τ_pp
  pos channel:  L_gap = ||x_sim − x_obs||²(surf);    g_k = ∂L_gap/∂stiff_pp
plus blow marker max|pos−c| and cut activity n_band (#particles 0.05<α<0.95).

Claim → after blow onset (~f105):
  τ:  α frozen → n_band→0, L_cav BOUNDED, ‖g_τ‖→~0   (INERT: can't pollute, signal already spent)
  k:  positions blow → L_gap and ‖g_k‖ EXPLODE by many orders  (GARBAGE: corrupts stiff-via-cut)
The ASYMMETRY (g_τ collapses while g_k explodes, same frames) is the proof; it's mechanistic /
low-variance (orders-of-magnitude), unlike recovery%.
"""
import sys, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state, solver_load_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD, TAU_SOFT, TAU_INIT = 0.55, 0.15, 0.30
STIFF_TRUTH, STIFF_INIT = 0.50, 0.40
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
kr = cfg["knife_radius_abs"]; top = cfg["top_axis"]
tht = torch.tensor([3., 20., 0.3], device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
# settle
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, swt(STIFF_TRUTH), swt(TAU_INIT)); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0; center = 0.5 * (bb0 + bb1); diag = float(np.linalg.norm(bs))
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
hard = (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0)
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
surf = pr[:, top] > (bb1[top] - 0.10 * bs[top]); surf_t = torch.tensor(np.where(surf)[0], device=dev)
traj = make_raster(bb0, bb1, top, sweep, la, 0.93, 10, 30)

# obs at TRUTH (also blows ~same frames)
solver.reset_to_rest(pr)
op, oa = gen_obs_per_particle_stiff(solver, rest, np.array([3., 20., 0.3], np.float32),
                                    np.full(N, STIFF_TRUTH, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
obs_pos = [torch.tensor(p, device=dev) for p in op]; obs_alpha = [torch.tensor(a, device=dev) for a in oa]
print(f"N={N} diag={diag:.3f}m  DIAG pattern, perturbed τ_init=0.30(truth 0.55/0.15) k_init=0.40(truth 0.50)")
print(f"\n{'frame':>5s} {'max|pos|':>10s} {'n_band':>7s}  {'L_cav':>9s} {'‖gτ‖':>10s}   {'L_gap':>10s} {'‖gk‖':>10s}")

st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rest.items()}
prev_alpha = solver.p["alpha"].numpy().copy()
SAMPLE = set(list(range(0, 30)) + list(range(90, 130)) + list(range(130, 300, 20)))
for t, kp in enumerate(traj):
    kpf = kp.astype(np.float32)
    # --- τ channel: L_cav backward -> g_tau ---
    s_t = swt(STIFF_INIT).clone().requires_grad_(True); tau_t = swt(TAU_INIT).clone().requires_grad_(True)
    a_c, p_c, _ = WarpFullForwardOp.apply(tht, solver, {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}, kpf, kr, s_t, tau_t)
    Lcav = ((a_c[surf_t] - obs_alpha[t][surf_t]) ** 2).sum()
    Lcav.backward()
    g_tau = float(tau_t.grad.norm()); Lcav_v = float(Lcav.item())
    st_next = solver_save_state(solver)   # advance state from this (truth-ish) step
    a_now = solver.p["alpha"].numpy()
    n_band = int(((a_now > 0.05) & (a_now < 0.95)).sum())
    maxpos = float(np.abs(st_next["pos"] - center).max())
    # --- pos channel: L_gap backward -> g_k (re-step from same input st) ---
    s_t2 = swt(STIFF_INIT).clone().requires_grad_(True); tau_t2 = swt(TAU_INIT).clone().requires_grad_(True)
    a_g, p_g, _ = WarpFullForwardOp.apply(tht, solver, {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}, kpf, kr, s_t2, tau_t2)
    Lgap = ((p_g[surf_t] - obs_pos[t][surf_t]) ** 2).sum()
    Lgap.backward()
    g_k = float(s_t2.grad.norm()); Lgap_v = float(Lgap.item())
    st = st_next
    if t in SAMPLE:
        def f(x): return f"{x:.2e}" if (x >= 1e4 or (0 < x < 1e-3)) else f"{x:8.3f}"
        print(f"{t:5d} {maxpos:10.3f} {n_band:7d}  {f(Lcav_v):>9s} {f(g_tau):>10s}   {f(Lgap_v):>10s} {f(g_k):>10s}", flush=True)
