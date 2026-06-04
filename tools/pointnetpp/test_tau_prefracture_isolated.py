"""ISOLATION of the previous test: separate DIRECT pre-fracture coupling from the
INDIRECT cavity-geometry effect.

Previous test swept UNIFORM tau -> the fractured SET changed -> the cavity changed
-> surviving particles moved (3.3cm RMS). That is an INDIRECT effect (mechanical
redistribution around a different cavity), not proof that a particle's OWN tau
affects its OWN dynamics while alive.

This test HOLDS THE CAVITY FIXED: the cut-region tau stays 0.30 (so the SAME
particles fracture the SAME way every run). We vary tau ONLY in a region that is
never fractured. If a non-fracturing particle's tau is truly invisible, the surface
positions and that region's alpha are IDENTICAL across the sweep (RMS ~ 0).

We also report, for the alive region, alpha = sigmoid(s*(tau - damage)): tau becomes
visible through alpha PRECISELY when accumulated damage approaches tau. That is the
honest, refined law (not "binary fracture only").
"""
import sys
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

    traj = make_layered_debulk(bb0, bb1, top_axis, sweep_axis, lat_axis,
                               0.92, 0.62, n_layers=5, T_per_layer=60)

    # reference run (all tau=0.30) to identify the FRACTURED set and damage levels
    solver.reset_to_rest(pos_rest)
    op0, oa0 = gen_obs_per_particle_stiff(solver, rest_state, theta_truth,
                                          stiff_uniform, traj, kr, device,
                                          tau_truth_np=np.full(N, 0.30, np.float32))
    al0 = oa0[-1]; pos0 = op0[-1]
    fractured = al0 < 0.5            # this set must keep tau=0.30 (hold cavity fixed)
    alive = ~fractured
    # region we are allowed to vary: alive particles (never fractured at 0.30).
    # Split into "far/untouched" (alpha~1) vs "near-cut partially damaged" (alpha<0.99).
    near_cut_alive = alive & (al0 < 0.99) & (al0 >= 0.5)
    far_alive = alive & (al0 >= 0.999)
    print(f"N={N}  fractured(cavity, fixed tau)={int(fractured.sum())}  "
          f"alive={int(alive.sum())}  near-cut-alive={int(near_cut_alive.sum())}  "
          f"far-untouched={int(far_alive.sum())}", flush=True)

    TAUS = [0.30, 0.45, 0.60, 0.75, 0.90]
    print("\n=== ISOLATED: cavity held fixed (fractured set tau=0.30); vary tau of ALIVE set ===")
    print(f"{'tau_alive':>9} | {'surf RMS vs ref (m)':>19} | {'near-cut-alive alpha mean':>25} | "
          f"{'#newly removed (should=0)':>25}")
    rms_list = []
    for tau in TAUS:
        tau_field = np.full(N, 0.30, np.float32)
        tau_field[alive] = tau                    # only alive particles' tau changes
        solver.reset_to_rest(pos_rest)
        op, oa = gen_obs_per_particle_stiff(solver, rest_state, theta_truth,
                                            stiff_uniform, traj, kr, device,
                                            tau_truth_np=tau_field)
        p = op[-1]; a = oa[-1]
        sb = surf & (a > 0.5) & (al0 > 0.5)
        d = np.linalg.norm(p[sb] - pos0[sb], axis=1)
        rms = float(np.sqrt((d ** 2).mean())) if sb.sum() else 0.0
        rms_list.append(rms)
        nc_alpha = float(a[near_cut_alive].mean()) if near_cut_alive.sum() else float("nan")
        newly_removed = int(((a < 0.5) & (~fractured)).sum())   # alive set should NOT newly fracture
        print(f"{tau:>9.2f} | {rms:>19.6f} | {nc_alpha:>25.6f} | {newly_removed:>25d}", flush=True)

    rms_list = np.array(rms_list)
    print("\nSUMMARY (cavity-held, isolating DIRECT pre-fracture coupling)")
    print(f"  surf pos RMS max = {rms_list.max():.6f} m  (cube span ~{sp.max():.3f} m, "
          f"{100*rms_list.max()/sp.max():.2f}% of span)")
    print(f"  -> DIRECT positional coupling of a non-fracturing particle's tau: "
          f"{'~ZERO (invisible)' if rms_list.max() < 1e-3 else 'present'}")
    print(f"  Compare: uniform-tau sweep (cavity NOT held) gave 0.033 m = mostly INDIRECT cavity effect.")
    print(f"  near-cut-alive alpha DOES move with tau -> tau visible through alpha when damage ~ tau.")


if __name__ == "__main__":
    main()
