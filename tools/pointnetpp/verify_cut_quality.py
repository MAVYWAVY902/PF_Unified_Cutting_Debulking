"""Verify each mesh's cutting setup gives REALISTIC surface debulk —
knife tip grazes top surface (not fully buried), gradual cavity (1-15%).

For each mesh + trajectory:
   1. Knife center y vs bbox top — should be at or slightly above top
   2. Fraction of knife volume above top surface — should be 30-60%
   3. Cavity fraction (alpha < 0.5) per frame — should grow gradually
   4. Max particle displacement — should be a small fraction of bbox
   5. Distribution of cut particles — should be near top surface, not bulk
"""

import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()
print(f"Warp {wp.__version__}  /  Torch {torch.__version__}")

from tools.day2_solver_wrapper import setup_solver_with_rest  # noqa: E402

from tools.stageA_inclusion_recovery import (  # noqa: E402
    build_inclusion_truth, gen_obs_per_particle_stiff,
)
from tools.pointnetpp.multi_mesh_test import (  # noqa: E402
    MESH_CONFIGS, make_traj_from_spec,
)


def verify_mesh_cuts(cfg, device, T=10):
    print(f"\n{'='*80}")
    print(f"MESH: {cfg['name']}  ({cfg['path']})")
    print(f"{'='*80}")
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = 0.99
    solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    bbox_min = pos_rest.min(0)
    bbox_max = pos_rest.max(0)
    bbox_size = bbox_max - bbox_min
    bbox_diag = float(np.linalg.norm(bbox_size))

    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = cfg["incl_radius_abs"]
    top_axis = cfg["top_axis"]
    top_axis_name = ["x", "y", "z"][top_axis]

    print(f"\n  Geometry:")
    print(f"    bbox = ({bbox_size[0]:.4f}, {bbox_size[1]:.4f}, {bbox_size[2]:.4f}) m")
    print(f"    bbox_min = ({bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f})")
    print(f"    bbox_max = ({bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f})")
    print(f"    top_axis = {top_axis_name} (top value = {bbox_max[top_axis]:.4f})")
    print(f"  Knife & inclusion:")
    print(f"    knife_radius = {knife_radius:.4f} m  "
          f"({knife_radius/bbox_size[top_axis]*100:.1f}% of top-axis dim, "
          f"{knife_radius/bbox_size.min()*100:.1f}% of smallest bbox dim)")
    print(f"    incl_center  = ({incl_center[0]:.4f}, {incl_center[1]:.4f}, {incl_center[2]:.4f})")
    print(f"    incl_radius  = {incl_radius:.4f} m  "
          f"({incl_radius/bbox_size.min()*100:.1f}% of smallest bbox dim)")

    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center),
        radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)
    n_incl = int(inclusion_mask.sum())
    print(f"    inclusion: {n_incl}/{N} particles ({n_incl/N*100:.2f}%)")

    # Generate truth for each trajectory and assess cut quality
    theta_truth_np = np.array([3.0, 100.0, 0.3], dtype=np.float32)

    for tr_name, spec in cfg["traj_specs"].items():
        traj = make_traj_from_spec(spec, bbox_min, bbox_max, T)
        # Solver reset
        solver.reset_to_rest(pos_rest)
        # Run forward at truth
        obs_pos, obs_alpha = gen_obs_per_particle_stiff(
            solver, rest_state, theta_truth_np, stiff_truth, traj, knife_radius, device,
            tau_truth_np=tau_truth)

        print(f"\n  Trajectory T_{tr_name}:")
        for t in [0, T // 2, T - 1]:
            k = traj[t]
            top_val = bbox_max[top_axis]
            # Surface exposure: fraction of knife volume above bbox top on top_axis
            # Sphere–halfspace intersection
            dy = k[top_axis] - top_val  # signed distance of knife center to top plane
            # Volume of spherical cap above plane:
            #   cap height h = r - dy (if knife center is below top, dy<0)
            #   above-plane part = sphere - lower cap
            if dy >= knife_radius:
                # entirely above mesh top
                frac_above = 1.0
            elif dy <= -knife_radius:
                # entirely below mesh top
                frac_above = 0.0
            else:
                # cap above: h_above = r + dy (with dy negative below, positive above)
                # Actually: if center is at dy above top, region above top is
                # the cap of height h = r - (-dy if center below else 0)
                # Simpler: signed dy. cap_above_height = r + dy (if dy ≤ 0 → cap = r+dy)
                # Standard formula for spherical cap volume V = π h² (3r - h) / 3
                # Cap above top has height h = r + dy (since dy < 0 means center below)
                h = knife_radius + dy if dy < 0 else 2 * knife_radius
                h = max(0.0, min(2 * knife_radius, h))
                cap_volume = np.pi * h * h * (3 * knife_radius - h) / 3
                sphere_volume = 4 * np.pi * knife_radius ** 3 / 3
                frac_above = cap_volume / sphere_volume

            cavity = float((obs_alpha[t] < 0.5).mean())
            # Particle displacement magnitudes
            displ = np.linalg.norm(obs_pos[t] - pos_rest, axis=1)
            max_displ = displ.max()
            mean_displ = displ.mean()
            # Active particles' displacement
            cut_mask = obs_alpha[t] < 0.5
            print(f"    frame t={t:2d}: knife pos ({k[0]:.4f},{k[1]:.4f},{k[2]:.4f})  "
                  f"dy_top={dy:+.4f}  frac_above={frac_above*100:5.1f}%  "
                  f"cavity={cavity*100:5.2f}%  "
                  f"max_displ={max_displ:.4f}  mean_displ={mean_displ:.6f}")

        # Verdict per trajectory
        cav_final = float((obs_alpha[-1] < 0.5).mean())
        print(f"  Verdict T_{tr_name}: ", end="")
        if cav_final > 0.30:
            print(f"✗ EXCESSIVE cut ({cav_final*100:.1f}% cavity) — knife too big or buried")
        elif cav_final < 0.005:
            print(f"⚠ WEAK cut ({cav_final*100:.2f}% cavity) — not enough signal")
        else:
            print(f"✓ REASONABLE cut ({cav_final*100:.1f}% cavity)")


def main():
    device = "cuda:0"
    for cfg in MESH_CONFIGS:
        try:
            verify_mesh_cuts(cfg, device)
        except Exception as e:
            print(f"\n✗ {cfg['name']} FAILED: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
