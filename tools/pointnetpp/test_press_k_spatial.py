"""Spatial k recovery via MULTI-LOCATION press + column GROUPING.
Truth k field = lateral stiff/soft pattern (left half along sweep stiff, right soft; uniform
in depth). Scan the sphere indenter over several lateral positions (lateral coverage), accumulate
L_gap, optimize per-COLUMN-GROUP k (one DOF per lateral column, shared along unobservable depth —
reuses τ's column_grid_assignment). Recover: does the stiff region come out higher than soft?
PRESS mode (enable_cutting=False), full-trajectory backprop, one Adam step/pass.
"""
import sys, time, numpy as np, torch
from pathlib import Path
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.surface_projection import column_grid_assignment

cfg = MESH_CONFIGS[0]; dev = "cuda:0"; THETA = np.array([3., 20., 0.3], np.float32)
PRESS_R = 0.07; K_STIFF, K_SOFT, K_INIT = 0.80, 0.30, 0.45
DEPTH = 0.18; T_RAMP = 25; T_HOLD = 5; GRID_N = 6
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = 2; solver.substeps = 1
solver.enable_cutting = False; solver.knife_active = 1; solver.knife_radius = PRESS_R
pr0 = mi["vertices"]; solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
top = cfg["top_axis"]
tht = torch.tensor(THETA, device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
solver.reset_to_rest(pr0); far = pr0.max(0) + 10.; st = rest0
with torch.no_grad():
    for _ in range(60):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, PRESS_R, swt(0.5), None); st = solver_save_state(solver)
prs = pw.cpu().numpy().copy(); REST = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in st.items()}
bb0, bb1 = prs.min(0), prs.max(0); bs = bb1 - bb0
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
surf = prs[:, top] > (bb1[top] - 0.15 * bs[top]); surf_t = torch.tensor(np.where(surf)[0], device=dev)
# TRUTH: left half along sweep = stiff
mid_sweep = 0.5 * (bb0[sweep] + bb1[sweep])
stiff_region = prs[:, sweep] < mid_sweep
k_truth = np.where(stiff_region, K_STIFF, K_SOFT).astype(np.float32)
# column grouping (one k per lateral column, shared along depth)
col_id, _ = column_grid_assignment(prs, surf, top, grid_n=GRID_N)
col_t = torch.tensor(col_id, dtype=torch.long, device=dev); n_grp = int(col_id.max()) + 1
# MULTI-PRESS scan: 4 lateral positions along sweep (covers L & R), la at center
clat_la = 0.5 * (bb0[la] + bb1[la])
sweep_locs = bb0[sweep] + np.array([0.25, 0.42, 0.58, 0.75]) * bs[sweep]
z_start = bb1[top] + PRESS_R; z_end = bb1[top] - DEPTH * bs[top]
def traj_at(sx):
    out = []
    for f in range(T_RAMP + T_HOLD):
        frac = min(1.0, f / T_RAMP); zc = z_start + frac * (z_end - z_start)
        kp = np.zeros(3, np.float32); kp[top] = zc; kp[sweep] = sx; kp[la] = clat_la; out.append(kp)
    return out
presses = [traj_at(sx) for sx in sweep_locs]
print(f"N={N} surf={int(surf.sum())} groups={n_grp} presses={len(presses)} PRESS spatial k  stiff(L)={K_STIFF} soft(R)={K_SOFT}")

def gen_obs():
    obs = []
    for traj in presses:
        solver.reset_to_rest(prs); st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in REST.items()}
        seq = []
        with torch.no_grad():
            for kp in traj:
                _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, kp, PRESS_R, torch.tensor(k_truth, device=dev), None)
                st = solver_save_state(solver); seq.append(pw.detach().clone())
        obs.append(seq)
    return obs

def calibrate(obs, N_passes=14, lr=0.04):
    kg = torch.full((n_grp,), K_INIT, dtype=torch.float32, device=dev, requires_grad=True)
    opt = torch.optim.Adam([kg], lr=lr); best = (float("inf"), None); hist = []
    for p in range(N_passes):
        opt.zero_grad(); tot = 0.0
        for li, traj in enumerate(presses):
            solver.reset_to_rest(prs); st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in REST.items()}
            for f, kp in enumerate(traj):
                _, pos, _ = WarpFullForwardOp.apply(tht, solver, st, kp, PRESS_R, kg[col_t], None)
                st = solver_save_state(solver)
                L = ((pos[surf_t] - obs[li][f][surf_t]) ** 2).sum()
                L.backward(); tot += float(L.item())
        opt.step()
        with torch.no_grad(): kg.clamp_(0.05, 2.0)
        kpp = kg.detach()[col_t].cpu().numpy()
        cL = (kpp[stiff_region].mean() - kpp[~stiff_region].mean()) / (K_STIFF - K_SOFT) * 100
        hist.append((p, tot, kpp[stiff_region].mean(), kpp[~stiff_region].mean(), cL))
        if tot < best[0]: best = (tot, kg.detach()[col_t].cpu().numpy().copy())
    return best[1], hist

print("generating multi-press obs ...", flush=True)
obs = gen_obs(); t0 = time.time()
k_rec, hist = calibrate(obs)
print(f"\n{'pass':>4s} {'totL':>11s} {'k_stiff(L)':>10s} {'k_soft(R)':>9s} {'contrast%':>9s}")
for p, tl, ks, kso, cL in hist:
    if p % 2 == 0 or p == hist[-1][0]:
        print(f"{p:4d} {tl:11.3e} {ks:10.4f} {kso:9.4f} {cL:8.1f}%")
print(f"\nrecovered: stiff-region k={k_rec[stiff_region].mean():.4f} (truth {K_STIFF})  "
      f"soft-region k={k_rec[~stiff_region].mean():.4f} (truth {K_SOFT})  [{time.time()-t0:.0f}s]")
print("want: contrast% >> 0 (stiff region recovered higher than soft) = spatial k localized")
outd = REPO / "tools/pointnetpp/viz_out"; outd.mkdir(exist_ok=True)
np.savez(outd / "press_k_spatial.npz", pos=prs, surf=surf, k_truth=k_truth, k_rec=k_rec,
         sweep=sweep, la=la, top=top, sweep_locs=sweep_locs, K_STIFF=K_STIFF, K_SOFT=K_SOFT, press_r=PRESS_R)
print(f"saved {outd/'press_k_spatial.npz'}")
solver.enable_cutting = True; solver.knife_active = 0
