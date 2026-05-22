"""Systematic debug of save/load consistency.

3 diagnostic tests progressively narrow where state goes missing:
  A: save → load → save (no step between). Should be idempotent.
  B: 1-step from rest WITHOUT save/load vs WITH save/load. Should be identical.
  C: deep state inspection — list all values, find which array differs.
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

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, solver_save_state, solver_load_state,
    run_full_forward_step,
)


def compare_states(s1, s2, label=""):
    """Compare two state dicts. Return summary of max diffs per key."""
    diffs = {}
    keys = set(s1.keys()) | set(s2.keys())
    for k in keys:
        v1 = s1.get(k); v2 = s2.get(k)
        if v1 is None and v2 is None:
            diffs[k] = "both None"; continue
        if v1 is None or v2 is None:
            diffs[k] = f"one None ({v1 is None}, {v2 is None})"; continue
        if isinstance(v1, int) or isinstance(v2, int):
            diffs[k] = f"scalar: {v1} vs {v2} (diff={abs(int(v1)-int(v2))})"; continue
        a1 = np.asarray(v1); a2 = np.asarray(v2)
        if a1.shape != a2.shape:
            diffs[k] = f"shape: {a1.shape} vs {a2.shape}"; continue
        m_abs = float(np.abs(a1 - a2).max())
        m_rel = float(np.abs(a1 - a2).max() / max(np.abs(a1).max(), 1e-12))
        n_diff = int((np.abs(a1 - a2) > 1e-8).sum())
        diffs[k] = f"max|Δ|={m_abs:.2e}  max_rel={m_rel:.2e}  n_diff>1e-8={n_diff}/{a1.size}"
    print(f"\n=== compare: {label} ===")
    for k in sorted(diffs):
        print(f"  {k:>20s}: {diffs[k]}")
    return diffs


def main():
    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)

    theta = np.array([3.0, 20.0, 0.5, 0.85])
    knife = np.array([0.15, 0.30, 0.15])

    # ============================================================
    # Test A: save → load → save idempotency (NO step in between)
    # ============================================================
    print("\n" + "=" * 64)
    print("[A] Idempotency: save → load → save (no step)")
    print("=" * 64)

    state_orig = solver_save_state(solver)
    solver_load_state(solver, state_orig)
    state_reloaded = solver_save_state(solver)
    compare_states(state_orig, state_reloaded, "save → load → save (zero step)")

    # ============================================================
    # Test B: 1 step rest+direct vs rest+save+load+1step
    # ============================================================
    print("\n" + "=" * 64)
    print("[B] 1-step diff: direct vs save-load-then-step")
    print("=" * 64)

    # B1: solver fresh from rest_state, 1 step
    solver_load_state(solver, rest_state)
    state_b1, _ = run_full_forward_step(
        solver, theta, knife, state_in=rest_state, knife_radius=0.04, n_substeps=1,
    )

    # B2: solver fresh from rest_state, save → load → 1 step
    solver_load_state(solver, rest_state)
    intermediate = solver_save_state(solver)
    solver_load_state(solver, intermediate)
    state_b2, _ = run_full_forward_step(
        solver, theta, knife, state_in=intermediate, knife_radius=0.04, n_substeps=1,
    )

    compare_states(state_b1, state_b2, "1-step direct vs 1-step after save+load")

    # ============================================================
    # Test C: 3-step direct vs save-load-3-step
    # ============================================================
    print("\n" + "=" * 64)
    print("[C] 3-step diff: direct vs save-load-3-step")
    print("=" * 64)

    # C1: rest → 3 steps
    state_c1, _ = run_full_forward_step(
        solver, theta, knife, state_in=rest_state, knife_radius=0.04, n_substeps=3,
    )
    # C2: rest → save → load → 3 steps
    fresh = solver_save_state(solver)
    # Already at end of C1's state, need to reset
    solver_load_state(solver, rest_state)
    fresh_rest = solver_save_state(solver)
    state_c2, _ = run_full_forward_step(
        solver, theta, knife, state_in=fresh_rest, knife_radius=0.04, n_substeps=3,
    )

    compare_states(state_c1, state_c2, "3-step direct vs 3-step after save+load(rest)")

    # ============================================================
    # Test D: 3-step vs 1+1+1 step (manual chain)
    # ============================================================
    print("\n" + "=" * 64)
    print("[D] 3-step continuous vs 1+1+1 saved chain")
    print("=" * 64)

    # D1: rest → 3 steps continuous
    state_d1, _ = run_full_forward_step(
        solver, theta, knife, state_in=rest_state, knife_radius=0.04, n_substeps=3,
    )
    # D2: rest → 1 step → save → load → 1 step → save → load → 1 step
    s_t = rest_state
    for i in range(3):
        s_t, _ = run_full_forward_step(
            solver, theta, knife, state_in=s_t, knife_radius=0.04, n_substeps=1,
        )
    state_d2 = s_t

    compare_states(state_d1, state_d2, "3-step continuous vs 1+1+1 chained")

    print("\nDONE — see which Test reveals the first nonzero diff.")


if __name__ == "__main__":
    main()
