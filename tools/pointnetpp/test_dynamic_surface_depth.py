"""Dynamic-surface depth-observability A/B test (2026-06-03).

QUESTION (user's intuition): when surface particles are removed, the interior
beneath them is newly exposed and becomes the new visible surface. A depth camera
would SEE those newly-exposed particles. Does tracking this progressive exposure
(dynamic surface) let us recover the tau of a BURIED inclusion — i.e., does it lift
the depth null-space that a static shallow cut leaves?

SETUP:
   - Cube, per-particle tau. Background tau = init = 0.30 (no signal needed).
   - BURIED hard inclusion (tau=0.55) in the lower-middle, laterally centered, so a
     shallow cut never reaches it. Recovery is measured ONLY on the inclusion.
   - Trajectory = layered debulk: knife sweeps x at successively deeper y (removes
     a layer, exposing the next).

CONDITIONS (per-particle param, L_cav signal, teacher-forced):
   A  shallow  + static surface : inclusion never cut          -> reference (~0%)
   B_static    deep    + static surface : inclusion cut but NEVER observed
                                          (static top shell)    -> null-space baseline
   B_dyn       deep    + dynamic surface: inclusion exposed+cut+observed -> ??? (the test)

If B_dyn >> B_static -> progressive exposure lifts the depth null-space.
If B_dyn ~ B_static  -> static surface is enough; depth genuinely needs F/T.
Either way it's a real finding.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.hierarchy import build_hierarchy
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.calibrate_online import calibrate_online

TAU_BG = 0.30          # background = init -> only the inclusion carries signal
TAU_INCL = 0.55        # buried hard inclusion
TAU_INIT = 0.30
STIFF_UNIFORM = 0.5


def make_layered_debulk(bb0, bb1, top_axis, sweep_axis, lat_axis,
                        y_top_frac, y_bot_frac, n_layers, T_per_layer):
    """Knife sweeps `sweep_axis` back-and-forth at successively deeper `top_axis`
    levels (y_top_frac -> y_bot_frac). lat_axis held at center. Returns (T,3)."""
    sp = bb1 - bb0
    ys = bb0[top_axis] + np.linspace(y_top_frac, y_bot_frac, n_layers) * sp[top_axis]
    x_lo = bb0[sweep_axis] + 0.10 * sp[sweep_axis]
    x_hi = bb0[sweep_axis] + 0.90 * sp[sweep_axis]
    lat_c = bb0[lat_axis] + 0.50 * sp[lat_axis]
    traj = []
    for li, y in enumerate(ys):
        xs = np.linspace(x_lo, x_hi, T_per_layer)
        if li % 2 == 1:
            xs = xs[::-1]
        for x in xs:
            p = np.zeros(3, dtype=np.float32)
            p[top_axis] = y; p[sweep_axis] = x; p[lat_axis] = lat_c
            traj.append(p.astype(np.float32))
    return traj


def main():
    cfg = MESH_CONFIGS[0]; device = "cuda:0"
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]),
                                                            fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    kr = cfg["knife_radius_abs"]; top_axis = cfg["top_axis"]

    # warm-up settle
    solver.reset_to_rest(pos_rest); far = pos_rest.max(0) + 10.0
    sw = torch.tensor(np.full(N, STIFF_UNIFORM, np.float32), device=device)
    twt = torch.tensor(np.full(N, TAU_INIT, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); st = rest_state
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
    pos_rest = pw.cpu().numpy().copy(); mesh_info["vertices"] = pos_rest; rest_state = st
    bb0, bb1 = pos_rest.min(0), pos_rest.max(0); sp = bb1 - bb0

    lateral = [a for a in range(3) if a != top_axis]
    sweep_axis, lat_axis = lateral[0], lateral[1]

    # ---- buried hard inclusion: laterally centered, lower-middle depth band ----
    xc = bb0[sweep_axis] + 0.5 * sp[sweep_axis]
    zc = bb0[lat_axis] + 0.5 * sp[lat_axis]
    y_lo = bb0[top_axis] + 0.35 * sp[top_axis]   # inclusion bottom
    y_hi = bb0[top_axis] + 0.62 * sp[top_axis]   # inclusion top (buried below surface)
    incl = ((np.abs(pos_rest[:, sweep_axis] - xc) < 0.22 * sp[sweep_axis]) &
            (np.abs(pos_rest[:, lat_axis] - zc) < 0.22 * sp[lat_axis]) &
            (pos_rest[:, top_axis] > y_lo) & (pos_rest[:, top_axis] < y_hi))
    tau_truth = np.where(incl, TAU_INCL, TAU_BG).astype(np.float32)
    stiff_uniform = np.full(N, STIFF_UNIFORM, np.float32)
    print(f"N={N}  inclusion particles={int(incl.sum())} ({100*incl.mean():.1f}%)  "
          f"y-band=[{y_lo:.3f},{y_hi:.3f}] (depth frac 0.35-0.62)", flush=True)

    # static surface = top shell
    surf = pos_rest[:, top_axis] > (bb1[top_axis] - 0.10 * sp[top_axis])
    surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=device)
    print(f"static surface particles={int(surf.sum())} ({100*surf.mean():.1f}%)", flush=True)

    # ---- trajectories ----
    theta_truth = np.array([3., 20., 0.3], np.float32)
    # shallow: stays above the inclusion (0.92 -> 0.78)
    traj_shallow = make_layered_debulk(bb0, bb1, top_axis, sweep_axis, lat_axis,
                                       0.92, 0.78, n_layers=4, T_per_layer=70)
    # deep: plunges into the inclusion top (0.92 -> 0.50)
    traj_deep = make_layered_debulk(bb0, bb1, top_axis, sweep_axis, lat_axis,
                                    0.92, 0.50, n_layers=8, T_per_layer=60)
    print(f"traj_shallow T={len(traj_shallow)}  traj_deep T={len(traj_deep)}", flush=True)

    def gen(traj):
        solver.reset_to_rest(pos_rest)
        op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth,
                                            stiff_uniform, traj, kr, device,
                                            tau_truth_np=tau_truth)
        opt = [torch.tensor(p, device=device) for p in op]
        oat = [torch.tensor(a, device=device) for a in oa]
        return opt, oat

    print("Gen obs (shallow + deep) ...", flush=True)
    op_s, oa_s = gen(traj_shallow)
    op_d, oa_d = gen(traj_deep)

    # how much of the inclusion actually gets cut (alpha drops) under the deep traj?
    incl_t = torch.tensor(incl, device=device)
    final_alpha_deep = oa_d[-1]
    incl_cut = ((final_alpha_deep < 0.5) & incl_t).sum().item()
    print(f"deep traj: inclusion particles cut (final alpha<0.5) = {incl_cut}/{int(incl.sum())}", flush=True)

    # edges
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    eat = torch.tensor(ea, dtype=torch.long, device=device); ebt = torch.tensor(eb, dtype=torch.long, device=device)
    erl = np.linalg.norm(pos_rest[ea] - pos_rest[eb], axis=1)
    ert = torch.tensor(erl, dtype=torch.float32, device=device)
    xr = torch.from_numpy(pos_rest).to(device, torch.float32)
    H = build_hierarchy(xr, M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
    stiff_init = np.full(N, STIFF_UNIFORM, np.float32); tau_init = np.full(N, TAU_INIT, np.float32)

    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=device, knife_radius=kr,
               pos_rest=pos_rest, theta_3d_np=theta_truth, stiff_init_np=stiff_init, tau_init_np=tau_init,
               tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt,
               edges_rest_t=ert, H=H, surf_top_axis=top_axis)

    LR = 0.03; LR_TAU = 0.5; NP = 4; LC = 1.0

    def run(label, multi, dynamic):
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        _, tau, _ = calibrate_online("L_pos_partial", ctx, multi, N_passes=NP, lr=LR, lr_tau=LR_TAU,
                                     lr_decay_per_pass=0.6, lambda_hist=0.0, lambda_outer_cav=LC,
                                     dynamic_surface=dynamic, dyn_grid_n=8, dyn_band_frac=0.10,
                                     log_every=800, verbose=False)
        tau_incl = tau[incl].mean(); tau_bg = tau[~incl].mean()
        rec = (tau_incl - TAU_INIT) / (TAU_INCL - TAU_INIT)
        print(f"{label:32s}: tau_incl={tau_incl:.3f} tau_bg={tau_bg:.3f}  "
              f"recovery={rec*100:5.1f}%  mem={torch.cuda.max_memory_allocated()/1e6:.0f}MB  {time.time()-t0:.0f}s",
              flush=True)
        return rec

    print("\n=== A/B: depth observability via progressive exposure ===")
    print(f"(truth: inclusion tau={TAU_INCL}, init/bg={TAU_INIT}; recovery = (tau_incl-init)/(0.55-0.30))")
    rA  = run("A  shallow + static", [(traj_shallow, op_s, oa_s)], dynamic=False)
    rBs = run("B_static deep + static", [(traj_deep, op_d, oa_d)], dynamic=False)
    rBd = run("B_dyn    deep + DYNAMIC", [(traj_deep, op_d, oa_d)], dynamic=True)
    print("\nSUMMARY")
    print(f"  A  (shallow, static)  = {rA*100:5.1f}%   [reference: inclusion untouched]")
    print(f"  B_static (deep,static)= {rBs*100:5.1f}%   [null-space baseline: cut but unobserved]")
    print(f"  B_dyn (deep, dynamic) = {rBd*100:5.1f}%   [progressive exposure]")
    verdict = "LIFTS depth null-space" if rBd > rBs + 0.10 else "no lift (static enough)"
    print(f"  VERDICT: dynamic surface {verdict}  (B_dyn - B_static = {(rBd-rBs)*100:+.1f} pts)")


if __name__ == "__main__":
    main()
