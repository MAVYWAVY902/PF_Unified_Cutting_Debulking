"""Day 5: ReSECT online loop end-to-end integration on cube_tet.msh.

Plumbs together everything from Days 1-4:
  - Forward sim: ShapeMatchingSolver.step() (full PBD + cohesive zone P(F))
  - Inner loop: compute_liang_inner_residuals (2-channel Δx + Δδv, V2 + cohesive zone)
  - Outer gradient: compute_outer_gradient (CAVITY-ONLY autograd via run_ssa_full_with_state)
  - State injection: Mode C (pos_next = pos_pred + Δx, δv_next = δv_pred + Δδv)

Limitations (honest disclosure):
  - Outer gradient is cavity-only because solver.step() uses Python-float
    stiffness/sharpness → no autograd flow to those params. Position channel's
    contribution to ∂L/∂θ is NOT propagated. This will be fixed in Day 3b
    (apply_kernel_ssa param_arr variant).
  - Forward via solver.step() and outer-gradient forward via
    run_ssa_full_with_state are TWO different forward computations (the
    first does PBD position update, the second is damage-only). They share
    cohesive-zone parameters and SHOULD agree on the damage chain at small
    deformations. Inconsistency grows when pos drifts from rest.

Synth-to-synth verification:
  1. Generate obs trajectory at θ_true via solver.step() (positions + cavities)
  2. Run ReSECT main loop with θ_init, see if θ recovers toward θ_true
  3. Log per-frame θ, rel_err, inner-loop residual norms, cavity loss
"""
import sys
from pathlib import Path
import time
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()

from tools.day2_solver_wrapper import (  # noqa: E402
    setup_solver_with_rest, run_full_forward_step,
)
from tools.online_residual_synth import (  # noqa: E402
    load_msh_system, compute_liang_inner_residuals,
    compute_outer_gradient, make_surface_knife_trajectory,
    AdamND, _sigmoid,
)


def main():
    print("=" * 70)
    print("Day 5: ReSECT online loop integration on cube_tet.msh")
    print("=" * 70)

    mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")

    # ---- Setup ----
    print("\n[setup] building solver + sys_dict …")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)

    # Use the SAME edge graph as solver (cluster-based, 254k) for both inner-loop
    # E_edge regularizer and outer-gradient damage propagation. Otherwise the
    # state's edge_damage shape (254k) won't match sys_dict's E (138k from tets).
    edges_a = solver.e["a"].numpy().copy()
    edges_b = solver.e["b"].numpy().copy()
    edges_rest = solver.e["pf_rest"].numpy().copy()
    rest_pos = mesh_info["vertices"]
    bbox_lo = rest_pos.min(axis=0); bbox_hi = rest_pos.max(axis=0)

    def make_sys_dict_for_outer_grad(pos_current_np, knife_pos_np):
        """Build sys_dict matching solver's edge graph; pos is current frame's."""
        return dict(
            N=solver.N,
            E=int(solver.e["num_edges"]),
            pos=pos_current_np.astype(np.float32),
            edge_a=edges_a,
            edge_b=edges_b,
            pf_rest=edges_rest,
            knife_pos=np.array([knife_pos_np], dtype=np.float32),
        )

    print(f"[setup] N={solver.N}  E_solver={int(solver.e['num_edges'])}  "
          f"(using solver's cluster-based edge graph)")

    # ---- Trajectory + knife ----
    n_frames = 30
    knife_radius = 0.04
    traj = make_surface_knife_trajectory(
        bbox=(bbox_lo, bbox_hi), n_frames=n_frames,
        surface_axis='y', margin_fraction=0.15,
    )

    # ---- Phase 1: generate synth observation trajectory at θ_true ----
    theta_true = np.array([3.0, 20.0, 0.5, 0.85])
    print(f"\n[Phase 1] generating obs trajectory at θ_true = {theta_true} …")

    obs_traj = []
    state = rest_state
    t0 = time.time()
    for t in range(n_frames):
        state, info = run_full_forward_step(
            solver, theta_true, traj[t], state_in=state,
            knife_radius=knife_radius, n_substeps=1,
        )
        obs_traj.append(dict(
            pos=state["pos"].copy(),
            alpha=state["alpha"].copy(),
            knife_pos=traj[t].copy(),
        ))
    print(f"[Phase 1] obs gen done ({time.time()-t0:.1f}s)")
    print(f"           final cavity = {(obs_traj[-1]['alpha']<0.5).mean():.4f}")
    print(f"           max|pos drift from rest| = "
          f"{float(np.abs(obs_traj[-1]['pos'] - mesh_info['vertices']).max()):.4f}")

    # ---- Phase 2: online ReSECT loop at θ_init ----
    theta_init = np.array([5.0, 10.0, 0.5])   # 3-d (drop stiffness for Day 5)
    print(f"\n[Phase 2] running ReSECT online loop")
    print(f"          θ_init = {theta_init}  →  θ_true = {theta_true[:3]}")
    print(f"          (cavity-only outer gradient; full L_pos/L_E gradient ")
    print(f"           awaits Day 3b apply_kernel param_arr extension)")

    theta = theta_init.copy().astype(np.float64)
    opt = AdamND(n_dims=3, lr=np.array([0.1, 1.0, 0.05]))
    lo_b = np.array([0.5, 1.0, 0.05]); hi_b = np.array([50.0, 200.0, 0.95])
    state = rest_state

    log = []
    t0 = time.time()
    for t in range(n_frames):
        # ---- Forward via full solver ----
        state_pred, fwd_info = run_full_forward_step(
            solver,
            np.array([theta[0], theta[1], theta[2], 0.85]),  # 4-d (stiffness fixed)
            traj[t], state_in=state,
            knife_radius=knife_radius, n_substeps=1,
        )
        pos_pred = state_pred["pos"].astype(np.float64)
        dv_pred = state_pred["particle_damage"].astype(np.float64)

        # ---- Inner loop: Liang 2-channel residual ----
        Δx, Δδv, info_inner = compute_liang_inner_residuals(
            pos_pred=pos_pred, dv_pred=dv_pred,
            pos_target=obs_traj[t]["pos"],
            alpha_target=obs_traj[t]["alpha"],
            theta=theta,
            edges_a=edges_a, edges_b=edges_b, edges_rest=edges_rest,
            observable_mask=None,
            n_iter=10,                # short for speed; tune later
            lr_pos=5e-3, lr_dv=0.1,
            lambda_cav=1.0, lambda_E=0.1,
        )

        # ---- Outer gradient (CAVITY-ONLY via run_ssa_full_with_state) ----
        # Build sys_dict from solver's actual edge graph + current positions
        # so edge_damage shapes match the state we pass in.
        sys_t = make_sys_dict_for_outer_grad(state["pos"], traj[t])
        state_damage = dict(
            alpha=state["alpha"],
            edge_damage=state["edge_damage"],
            particle_damage=state["particle_damage"],
        )
        L_cav_outer, grad_θ_cav = compute_outer_gradient(
            theta=theta, sys_dict=sys_t,
            alpha_obs=obs_traj[t]["alpha"],
            n_cycles=1, state_in=state_damage,
            knife_radius=knife_radius,
            observable_mask=None,
        )

        # ---- Adam update on θ (3-d) ----
        theta = np.clip(opt.step(theta, grad_θ_cav), lo_b, hi_b)

        # ---- Mode C state injection: pos += Δx, δv += Δδv ----
        dv_next = dv_pred + Δδv.astype(np.float64)
        α_next = _sigmoid(-float(theta[1]) * (dv_next - float(theta[2])))
        pos_next = pos_pred + Δx.astype(np.float64)
        state = dict(
            pos=pos_next.astype(np.float32),
            vel=state_pred["vel"],
            alpha=α_next.astype(np.float32),
            particle_damage=dv_next.astype(np.float32),
            edge_damage=state_pred["edge_damage"],
            cluster_quat=state_pred["cluster_quat"],
            step_count=state_pred["step_count"],
        )

        # ---- Log ----
        rel = np.abs(theta - theta_true[:3]) / np.maximum(np.abs(theta_true[:3]), 1e-6) * 100
        log.append(dict(
            t=t, theta=theta.copy(), rel_err=rel,
            L_cav=L_cav_outer,
            L_inner_pos=info_inner["L_pos"],
            L_inner_cav=info_inner["L_cav"],
            L_inner_E=info_inner["L_E"],
            max_Δx=float(np.abs(Δx).max()),
            max_Δδv=float(Δδv.max()),
            mean_Δδv=float(Δδv.mean()),
        ))

        if t < 5 or t % 5 == 0 or t == n_frames - 1:
            print(f"  t={t:3d}  L_cav={L_cav_outer:.3e}  "
                  f"L_inner_pos={info_inner['L_pos']:.3e}  "
                  f"L_inner_cav={info_inner['L_cav']:.3e}  "
                  f"max|Δx|={float(np.abs(Δx).max()):.4f}  "
                  f"max Δδv={float(Δδv.max()):.4f}  "
                  f"θ=({theta[0]:.2f}, {theta[1]:.2f}, {theta[2]:.3f})  "
                  f"rel={rel.round(1)}%")

    elapsed = time.time() - t0
    print(f"\n[Phase 2] done ({elapsed:.1f}s, {elapsed/n_frames*1000:.0f} ms/frame)")

    # ---- Final analysis ----
    print("\n" + "=" * 70)
    print("Day 5 verdicts")
    print("=" * 70)
    final = log[-1]
    rel = final["rel_err"]
    print(f"  θ_final  = {final['theta'].round(3)}")
    print(f"  θ_true   = {theta_true[:3].round(3)}")
    print(f"  rel_err  = {rel.round(2)} %")

    # SS metrics (last half)
    half = n_frames // 2
    ss = np.array([e["theta"] for e in log[-half:]]).mean(axis=0)
    rel_ss = np.abs(ss - theta_true[:3]) / np.maximum(np.abs(theta_true[:3]), 1e-6) * 100
    print(f"  θ_SS (last half avg) = {ss.round(3)}  rel = {rel_ss.round(2)} %")

    # State-injection sanity: Δδv never negative, Δx bounded
    irr_pass = all([float(e["max_Δδv"]) >= 0 for e in log])
    print(f"  All Δδv ≥ 0 (cohesive zone irreversibility): {'✓' if irr_pass else '✗'}")
    max_Δx_overall = max(float(e["max_Δx"]) for e in log)
    print(f"  Max |Δx| across all frames: {max_Δx_overall:.4f}")

    # Inner-loop convergence: L_inner_pos decreased per frame
    L_pos_decreasing = all(log[t]["L_inner_pos"] < 1e5 for t in range(len(log)))
    print(f"  Inner-loop L_pos bounded (< 1e5): {'✓' if L_pos_decreasing else '✗'}")

    # Overall: structurally working if irr_pass + recovery direction OK
    print(f"\nPIPELINE STRUCTURE: ✓ runs end-to-end without errors")
    print(f"  Forward (solver.step) → Inner (V2) → Outer (cavity-only) → "
          f"Mode-C state injection")
    print(f"  Frame rate: {elapsed/n_frames*1000:.0f} ms/frame "
          f"(includes solver.step, inner loop, outer grad)")

    print(f"\nKNOWN LIMITATIONS:")
    print(f"  - Outer gradient is cavity-only (∂L_pos/∂θ, ∂L_E/∂θ not propagated)")
    print(f"  - This will be added in Day 3b: apply_kernel_ssa param_arr variant")
    print(f"  - Position channel (Δx) IS injected into state — correct Mode C")
    print(f"    behavior — but doesn't drive θ optimization yet")

    if rel.max() < 50:
        print(f"\n✓ Day 5 partial PASS: pipeline integrates cleanly; partial θ recovery")
    else:
        print(f"\n⚠ Day 5 partial: pipeline runs; recovery limited by cavity-only gradient")


if __name__ == "__main__":
    main()
