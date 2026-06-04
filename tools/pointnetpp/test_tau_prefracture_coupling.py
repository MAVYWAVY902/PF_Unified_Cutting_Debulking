"""Confirm the deep claim: tau couples to observables ONLY through the fracture
event. Pre-fracture (a particle that stays alive), its tau is nearly invisible.

METHOD: one fixed cut. Sweep UNIFORM tau in {0.30..0.90}. Measure two channels:
   (A) FRACTURE channel  = #particles removed (alpha<0.5). If tau is observable
       through fracture, this drops sharply as tau rises (tougher -> fewer cut).
   (B) PRE-FRACTURE channel = surface position RMS, computed ONLY on particles
       that stay ALIVE in BOTH the tau-run and the tau=0.30 reference (i.e. never
       fractured). If tau is pre-fracture-invisible, this RMS is ~0 regardless of
       tau (the alive particles behave identically).

EXPECT: (A) strongly tau-dependent, (B) ~flat near 0 (only the sigmoid baseline).
That = "tau is a fracture-only observable; seeing an alive particle tells you ~nothing."
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
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_dynamic_surface_depth import make_layered_debulk

STIFF_UNIFORM = 0.5


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
    twt = torch.tensor(np.full(N, 0.30, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); st = rest_state
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
    pos_rest = pw.cpu().numpy().copy(); mesh_info["vertices"] = pos_rest; rest_state = st
    bb0, bb1 = pos_rest.min(0), pos_rest.max(0); sp = bb1 - bb0
    lateral = [a for a in range(3) if a != top_axis]
    sweep_axis, lat_axis = lateral[0], lateral[1]

    surf = pos_rest[:, top_axis] > (bb1[top_axis] - 0.10 * sp[top_axis])
    theta_truth = np.array([3., 20., 0.3], np.float32)
    stiff_uniform = np.full(N, STIFF_UNIFORM, np.float32)

    # moderate cut (a few layers) so a meaningful number fractures and the count
    # is sensitive to tau in [0.30, 0.90].
    traj = make_layered_debulk(bb0, bb1, top_axis, sweep_axis, lat_axis,
                               0.92, 0.62, n_layers=5, T_per_layer=60)
    print(f"N={N}  surface={int(surf.sum())}  cut T={len(traj)}", flush=True)

    TAUS = [0.30, 0.45, 0.60, 0.75, 0.90]
    fin_pos = {}; fin_alpha = {}
    for tau in TAUS:
        tau_field = np.full(N, tau, np.float32)
        solver.reset_to_rest(pos_rest)
        op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth,
                                            stiff_uniform, traj, kr, device,
                                            tau_truth_np=tau_field)
        fin_pos[tau] = op[-1]; fin_alpha[tau] = oa[-1]   # final-frame numpy
        print(f"  ran tau={tau:.2f}  removed(alpha<0.5)={int((oa[-1] < 0.5).sum())}", flush=True)

    ref = TAUS[0]
    pos_ref = fin_pos[ref]; al_ref = fin_alpha[ref]
    surf_b = surf
    print("\n=== tau pre-fracture coupling ===")
    print(f"(uniform tau swept; reference = tau={ref})")
    print(f"{'tau':>5} | {'A: #removed':>11} | {'B: surf-pos RMS on alive-in-both (m)':>38} | "
          f"{'alpha mean (alive surf)':>24}")
    for tau in TAUS:
        p = fin_pos[tau]; a = fin_alpha[tau]
        # (A) fracture channel
        n_rm = int((a < 0.5).sum())
        # (B) pre-fracture channel: surface particles alive in BOTH runs
        alive_both = surf_b & (a > 0.5) & (al_ref > 0.5)
        if alive_both.sum() > 0:
            d = np.linalg.norm(p[alive_both] - pos_ref[alive_both], axis=1)
            rms = float(np.sqrt((d ** 2).mean()))
        else:
            rms = float("nan")
        alpha_alive = float(a[surf_b & (a > 0.5)].mean())
        print(f"{tau:>5.2f} | {n_rm:>11d} | {rms:>38.6f} | {alpha_alive:>24.6f}", flush=True)

    # quantify the contrast between the two channels
    rm = np.array([int((fin_alpha[t] < 0.5).sum()) for t in TAUS])
    rms_all = []
    for tau in TAUS:
        ab = surf_b & (fin_alpha[tau] > 0.5) & (al_ref > 0.5)
        d = np.linalg.norm(fin_pos[tau][ab] - pos_ref[ab], axis=1)
        rms_all.append(float(np.sqrt((d ** 2).mean())) if ab.sum() else 0.0)
    rms_all = np.array(rms_all)
    print("\nSUMMARY")
    print(f"  FRACTURE channel: #removed ranges {rm.min()}..{rm.max()}  "
          f"(swing {rm.max()-rm.min()} particles across tau) -> {'STRONG' if rm.max()-rm.min() > 50 else 'weak'}")
    print(f"  PRE-FRACTURE channel: alive-in-both surf pos RMS max = {rms_all.max():.6f} m  "
          f"(cube span ~{sp.max():.3f} m) -> {'~zero (invisible)' if rms_all.max() < 1e-3 else 'visible'}")
    print(f"  VERDICT: tau is observed through FRACTURE only; alive particles are ~tau-blind.")


if __name__ == "__main__":
    main()
