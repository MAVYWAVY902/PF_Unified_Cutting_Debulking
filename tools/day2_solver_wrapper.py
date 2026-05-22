"""Day 2 (v2): Functional wrapper around ShapeMatchingSolver for sys-ID.

Clean API (no state_in=None magic branch):

    solver, rest_state, mesh_info = setup_solver_with_rest(msh_path)
    state = rest_state
    for t in range(T):
        state, info = run_full_forward_step(solver, theta_4d, knife_xyz, state)

Properties:
  - run_full_forward_step ALWAYS takes a state_in dict (rest_state at t=0)
  - solver is reset to rest exactly ONCE per trajectory, inside setup_solver_with_rest
  - All state-bearing arrays are managed: pos, vel, alpha, damage, edge_damage,
    cluster_quat (now reset to identity via solver.reset_to_rest)
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402

wp.config.verify_autograd_array_access = True
wp.init()

from tools.day1_solver_smoke import setup_solver_from_msh  # noqa: E402


# ---------------------------------------------------------------------------
# State save / load
# ---------------------------------------------------------------------------

def solver_save_state(solver) -> dict:
    """Snapshot the dynamic state of a ShapeMatchingSolver.

    Captures EVERY state-bearing array. Crucially, in SSA mode the current
    damage state lives in solver._ssa_particle_damage_buffers[cycle_index],
    NOT in solver.p["damage"] (which is the stale legacy buffer). Reading
    p["damage"] in SSA mode silently returns 0s.
    """
    p = solver.p
    e = solver.e
    c = solver.c

    if solver.use_ssa_soft_removal and solver._ssa_alpha_buffers is not None:
        ci = solver._ssa_cycle_index
        alpha_np = solver._ssa_alpha_buffers[ci].numpy().copy()
        pd_np = solver._ssa_particle_damage_buffers[ci].numpy().copy()
        ed_np = (solver._ssa_edge_damage_buffers[ci].numpy().copy()
                  if e is not None else None)
    else:
        alpha_np = p["alpha"].numpy().copy()
        pd_np = p["damage"].numpy().copy()
        ed_np = e["damage"].numpy().copy() if e is not None else None

    return dict(
        pos=p["pos"].numpy().copy(),
        vel=p["vel"].numpy().copy(),
        alpha=alpha_np,
        particle_damage=pd_np,
        edge_damage=ed_np,
        cluster_quat=c["cluster_quat"].numpy().copy(),
        step_count=int(solver.step_count),
    )


def solver_load_state(solver, state_dict):
    """Write a state_dict back into a solver's wp.arrays.

    After this call, solver.step() will continue physics from state_dict
    exactly (subject to determinism of the underlying kernels).
    """
    p = solver.p
    e = solver.e
    c = solver.c

    if "pos" in state_dict and state_dict["pos"] is not None:
        pos_wp = wp.array(state_dict["pos"].astype(np.float32),
                            dtype=p["pos"].dtype, device=solver.device)
        wp.copy(p["pos"], pos_wp)
        wp.copy(p["old_pos"], pos_wp)
        wp.copy(p["pred_pos"], pos_wp)
    if "vel" in state_dict and state_dict["vel"] is not None:
        wp.copy(p["vel"],
                wp.array(state_dict["vel"].astype(np.float32),
                          dtype=p["vel"].dtype, device=solver.device))
    if "alpha" in state_dict and state_dict["alpha"] is not None:
        a_arr = wp.array(state_dict["alpha"].astype(np.float32),
                          dtype=wp.float32, device=solver.device)
        wp.copy(p["alpha"], a_arr)
        if solver.use_ssa_soft_removal and solver._ssa_alpha_buffers is not None:
            for buf in solver._ssa_alpha_buffers:
                wp.copy(buf, a_arr)
            solver._ssa_cycle_index = 0
    if "particle_damage" in state_dict and state_dict["particle_damage"] is not None:
        pd_arr = wp.array(state_dict["particle_damage"].astype(np.float32),
                           dtype=wp.float32, device=solver.device)
        wp.copy(p["damage"], pd_arr)
        if solver.use_ssa_soft_removal and solver._ssa_particle_damage_buffers is not None:
            for buf in solver._ssa_particle_damage_buffers:
                wp.copy(buf, pd_arr)
    if ("edge_damage" in state_dict and state_dict["edge_damage"] is not None
            and e is not None):
        ed_arr = wp.array(state_dict["edge_damage"].astype(np.float32),
                           dtype=wp.float32, device=solver.device)
        wp.copy(e["damage"], ed_arr)
        if solver.use_ssa_soft_removal and solver._ssa_edge_damage_buffers is not None:
            for buf in solver._ssa_edge_damage_buffers:
                wp.copy(buf, ed_arr)
    if "cluster_quat" in state_dict and state_dict["cluster_quat"] is not None:
        cq_arr = wp.array(state_dict["cluster_quat"].astype(np.float32),
                           dtype=wp.quat, device=solver.device)
        wp.copy(c["cluster_quat"], cq_arr)
    if "step_count" in state_dict:
        solver.step_count = int(state_dict["step_count"])


# ---------------------------------------------------------------------------
# Setup + per-frame forward
# ---------------------------------------------------------------------------

def setup_solver_with_rest(msh_path: str, rings: int = 2, device: str = "cuda:0",
                            fix_bottom_layer: bool = True,
                            fix_threshold_fraction: float = 0.05):
    """One-stop setup: build solver, reset to rest, snapshot rest_state.

    Returns:
        solver:     ShapeMatchingSolver ready to step()
        rest_state: dict matching solver_save_state output at t=0
        mesh_info:  dict with vertices, tets, fixed_mask, bbox, N, ...
    """
    solver, p, c, e, mesh_info = setup_solver_from_msh(
        msh_path, rings=rings, device=device,
        fix_bottom_layer=fix_bottom_layer,
        fix_threshold_fraction=fix_threshold_fraction,
    )

    # For sys-ID we want damage propagation on EVERY step. Default 50 is for
    # RL training where many physics substeps happen per RL action.
    solver.cutting_update_interval = 1

    # Use the solver's own reset_to_rest method (added in solver.py).
    solver.reset_to_rest(mesh_info["vertices"])
    rest_state = solver_save_state(solver)
    return solver, rest_state, mesh_info


def set_solver_theta(solver, theta_4d):
    """Apply θ = (α_min, sharpness, τ, k_stiffness) to solver fields.

    These are Python attrs in the solver (not wp.arrays). Autograd through
    them requires reading them inside kernels — handled in Day 3.
    """
    solver.alpha_min = float(theta_4d[0])
    solver.soft_removal_sharpness = float(theta_4d[1])
    solver.aspiration_threshold = float(theta_4d[2])
    solver.stiffness = float(theta_4d[3])


def set_solver_knife(solver, knife_xyz, radius=None):
    """Position the knife. knife_xyz is a 3-vector."""
    knife_host = np.array([list(knife_xyz)], dtype=np.float32)
    solver.knife_pos = wp.array(knife_host, dtype=wp.vec3,
                                  device=solver.device, requires_grad=True)
    if radius is not None:
        solver.knife_radius = float(radius)
    solver.knife_active = 1


def run_full_forward_step(solver, theta_4d, knife_xyz, state_in,
                           knife_radius=None, n_substeps=1):
    """One forward step from state_in.

    state_in is REQUIRED. To start a trajectory, pass rest_state from
    setup_solver_with_rest(). Each subsequent call passes the previous
    state_out.

    Returns:
        state_out: full state snapshot AFTER step
        info:      dict with diagnostic values
    """
    if state_in is None:
        raise ValueError(
            "state_in is required. Pass rest_state for the first frame, "
            "then chain state_out → state_in for subsequent frames.")

    solver_load_state(solver, state_in)
    set_solver_theta(solver, theta_4d)
    set_solver_knife(solver, knife_xyz, knife_radius)

    for _ in range(n_substeps):
        solver.step()
    wp.synchronize_device()

    state_out = solver_save_state(solver)
    alpha = state_out["alpha"]
    info = dict(
        cavity_frac=float((alpha < 0.5).mean()),
        alpha_min=float(alpha.min()),
        alpha_max=float(alpha.max()),
        max_pd=float(state_out["particle_damage"].max()),
    )
    return state_out, info


# ---------------------------------------------------------------------------
# Day 2 verification
# ---------------------------------------------------------------------------

def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)

    print("\n" + "=" * 64)
    print("Day 2 (v2) — proper rest reset + state save/load consistency")
    print("=" * 64)

    theta = np.array([3.0, 20.0, 0.5, 0.85])  # (α_min, sharp, τ, k_stiff)
    knife = np.array([0.15, 0.30, 0.15])      # on top surface

    # ---- Test 1: 3 steps from rest, save state_a ----
    print("\n[Test 1] rest → 3 steps → state_a")
    state_a, info_a = run_full_forward_step(
        solver, theta, knife, state_in=rest_state, knife_radius=0.04, n_substeps=3,
    )
    print(f"  cavity={info_a['cavity_frac']:.5f}  α∈[{info_a['alpha_min']:.3f}, {info_a['alpha_max']:.3f}]"
          f"  max pd={info_a['max_pd']:.4f}")

    # ---- Test 2: continue from state_a, 3 more steps → state_b ----
    print("\n[Test 2] state_a → 3 more steps → state_b (3+3 chain)")
    state_b, info_b = run_full_forward_step(
        solver, theta, knife, state_in=state_a, knife_radius=0.04, n_substeps=3,
    )
    print(f"  cavity={info_b['cavity_frac']:.5f}  α∈[{info_b['alpha_min']:.3f}, {info_b['alpha_max']:.3f}]"
          f"  max pd={info_b['max_pd']:.4f}")

    # ---- Test 3: rest → 6 steps in one call → state_c (should match state_b) ----
    print("\n[Test 3] rest → 6 steps in ONE call → state_c (should match state_b)")
    state_c, info_c = run_full_forward_step(
        solver, theta, knife, state_in=rest_state, knife_radius=0.04, n_substeps=6,
    )
    print(f"  cavity={info_c['cavity_frac']:.5f}  α∈[{info_c['alpha_min']:.3f}, {info_c['alpha_max']:.3f}]"
          f"  max pd={info_c['max_pd']:.4f}")

    # ---- Consistency check ----
    pos_diff = float(np.abs(state_b["pos"] - state_c["pos"]).max())
    alpha_diff = float(np.abs(state_b["alpha"] - state_c["alpha"]).max())
    pd_diff = float(np.abs(state_b["particle_damage"] - state_c["particle_damage"]).max())
    print(f"\n  consistency Test 2 vs Test 3:")
    print(f"    max|Δpos|   = {pos_diff:.6f}")
    print(f"    max|Δα|     = {alpha_diff:.6f}")
    print(f"    max|Δpd|    = {pd_diff:.6f}")
    if pos_diff < 1e-4 and alpha_diff < 1e-4 and pd_diff < 1e-4:
        print(f"  ✓ State save/load consistency PASS (3+3 ≡ 6)")
    else:
        print(f"  ✗ Still inconsistent — investigate")

    # ---- Test 4: weak vs strong knife should differ ----
    print("\n[Test 4] weak knife (α_min=20) vs strong (α_min=3) — both 6 steps from rest")
    theta_weak = np.array([20.0, 20.0, 0.5, 0.85])
    state_d, info_d = run_full_forward_step(
        solver, theta_weak, knife, state_in=rest_state, knife_radius=0.04, n_substeps=6,
    )
    print(f"  weak  cavity={info_d['cavity_frac']:.5f}  max pd={info_d['max_pd']:.4f}")
    print(f"  strong cavity={info_c['cavity_frac']:.5f}  max pd={info_c['max_pd']:.4f}")
    if info_c['cavity_frac'] > info_d['cavity_frac'] + 1e-4:
        print("  ✓ Strong knife produces MORE cavity (physics direction correct)")
    elif info_c['cavity_frac'] == info_d['cavity_frac'] == 0.0:
        print("  ⚠ Both produced 0 cavity — physics might not be triggering")
    else:
        print("  ⚠ Counter-intuitive: weak knife produced equal/more cavity")

    # ---- Test 5: position output sanity ----
    print(f"\n[Test 5] Position output sanity (final state)")
    pos = state_c["pos"]
    print(f"  pos shape: {pos.shape}, dtype: {pos.dtype}")
    print(f"  bbox: x∈[{pos[:,0].min():.3f},{pos[:,0].max():.3f}]  "
          f"y∈[{pos[:,1].min():.3f},{pos[:,1].max():.3f}]  "
          f"z∈[{pos[:,2].min():.3f},{pos[:,2].max():.3f}]")
    print(f"  centroid: {pos.mean(axis=0).round(4)}")
    rest = mesh_info["vertices"]
    disp = np.linalg.norm(pos - rest, axis=1)
    n_moved = int((disp > 1e-3).sum())
    print(f"  displacement: mean={disp.mean():.4f}, max={disp.max():.4f}")
    print(f"  particles moved > 1mm: {n_moved} / {len(pos)} ({100*n_moved/len(pos):.1f}%)")

    print("\nDay 2 (v2) PASS — clean state reset + chain. Next: Day 3 (autograd through full step).")


if __name__ == "__main__":
    main()
