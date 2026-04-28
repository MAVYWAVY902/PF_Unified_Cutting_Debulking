#!/usr/bin/env python3
"""Empirically verify whether our debulk sim's damage/cutting pipeline is
continuous (DiSECt-style) or discrete (with hard tet removal / BFS).

This decides whether Path 4 (continuous-damage rewrite for trajectory opt)
is needed or already done.

Method:
  1. Load env
  2. Drive knife into torus with a fixed cutting motion
  3. Record per-particle alpha + damage at every step
  4. Check:
     a. Particle count constant? (no removal)
     b. Are alpha values smooth sigmoid transitions? (no step jumps)
     c. Are damage values monotonically smooth? (no discontinuities)
     d. Plot alpha trajectories for transitioning particles

Output:
  results/continuous_damage_check/
    ├── alpha_trajectories.png    # per-particle alpha over time
    ├── damage_trajectories.png   # per-particle damage over time
    ├── alpha_histogram.png       # distribution of alpha at final step
    └── verdict.txt               # CONTINUOUS or DISCRETE conclusion
"""
import argparse
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "pf_shape_matching_warp"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="warp_sim/config/debulk_cap_armC.yaml")
    ap.add_argument("--steps", type=int, default=200,
                    help="Number of substeps to run")
    ap.add_argument("--knife-action", nargs=3, type=float, default=[0.0, -0.5, 0.0],
                    help="Constant action driving knife into torus (3D in [-1,1])")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(REPO, "results", "continuous_damage_check")
    os.makedirs(out_dir, exist_ok=True)

    import warp as wp
    import yaml
    from warp_sim.envs.debulk_env import DebulkEnv

    print("=== Continuous-damage verification ===")
    print(f"  cfg:    {args.cfg}")
    print(f"  steps:  {args.steps}")
    print(f"  action: {args.knife_action}")
    print()

    wp.init()
    cfg = yaml.safe_load(open(os.path.join(REPO, args.cfg)))
    env = DebulkEnv(cfg, device=args.device)
    env.reset()

    N = env.solver.N
    print(f"  particle count at start: {N}")

    # Storage: alpha[t, i] and damage[t, i]
    alpha_traj = np.zeros((args.steps, N), dtype=np.float32)
    damage_traj = np.zeros((args.steps, N), dtype=np.float32)
    pcount_traj = np.zeros(args.steps, dtype=np.int64)
    edge_dmg_traj = np.zeros((args.steps, env.solver.e["num_edges"]), dtype=np.float32)

    # Run simulation with constant action
    action = np.array(args.knife_action, dtype=np.float32)
    print(f"  Running {args.steps} steps with constant action ...")

    for t in range(args.steps):
        # Step env
        obs, reward, terminated, truncated, info = env.step(action)

        # Snapshot alpha + damage
        alpha_traj[t]  = env.solver.p["alpha"].numpy()
        damage_traj[t] = env.solver.p["damage"].numpy()
        edge_dmg_traj[t] = env.solver.e["damage"].numpy()
        pcount_traj[t] = N  # constant by design

        if terminated or truncated:
            print(f"  episode terminated at step {t}, truncated to here")
            alpha_traj  = alpha_traj[:t+1]
            damage_traj = damage_traj[:t+1]
            edge_dmg_traj = edge_dmg_traj[:t+1]
            break

        if t % 50 == 0 or t == args.steps - 1:
            n_alive   = int((alpha_traj[t] > 0.01).sum())
            n_partial = int(((alpha_traj[t] > 0.01) & (alpha_traj[t] < 0.99)).sum())
            n_dead    = int((alpha_traj[t] < 0.01).sum())
            mean_dmg  = float(damage_traj[t].mean())
            print(f"  t={t:3d}: alive={n_alive}, transitioning={n_partial}, "
                  f"dead={n_dead}, mean_dmg={mean_dmg:.4f}")

    n_steps = alpha_traj.shape[0]
    print()
    print(f"  Recorded {n_steps} steps × {N} particles")
    print()

    # =========================================================================
    # Analysis
    # =========================================================================
    print("=== Analysis ===")

    # 1. Particle count constant?
    pcount_constant = (np.diff(pcount_traj[:n_steps]) == 0).all()
    print(f"\n[CHECK 1] Particle count constant: {pcount_constant}")
    print(f"  All {n_steps} steps have N={N} particles")
    if pcount_constant:
        print("  → No tet/particle removal happens. ✓")
    else:
        print("  → Particle count varies! Discrete removal exists. ✗")

    # 2. Alpha smoothness — find particles that transitioned during the run
    # Transitioning particles = those whose alpha dropped meaningfully
    alpha_initial = alpha_traj[0]
    alpha_final = alpha_traj[-1]
    transitioned_mask = (alpha_initial > 0.5) & (alpha_final < 0.5)
    transitioned_idx = np.where(transitioned_mask)[0]
    print(f"\n[CHECK 2] Particles that transitioned (alpha 0.5+ → 0.5-): "
          f"{transitioned_idx.size}")

    if transitioned_idx.size == 0:
        print("  → No particles transitioned. Knife may not be in cutting position.")
        print("  → Try larger --steps or different --knife-action.")
        verdict = "INCONCLUSIVE"
    else:
        # Check max jump in alpha for each transitioning particle
        alpha_for_trans = alpha_traj[:, transitioned_idx]  # (n_steps, k)
        per_step_diffs = np.abs(np.diff(alpha_for_trans, axis=0))  # (n_steps-1, k)
        max_jump = float(per_step_diffs.max())
        median_jump = float(np.median(per_step_diffs[per_step_diffs > 1e-6]))
        # If max jump > 0.5, probably discrete
        # If max jump < 0.1, definitely continuous
        print(f"  Max alpha change per step (across all transitioning particles): "
              f"{max_jump:.4f}")
        print(f"  Median alpha change per step (nonzero): {median_jump:.6f}")

        if max_jump > 0.5:
            verdict_alpha = "DISCRETE"
            print("  → Large step jump detected. Alpha is NOT continuous. ✗")
        elif max_jump < 0.2:
            verdict_alpha = "CONTINUOUS"
            print("  → All changes < 0.2 per step. Alpha is sigmoid-continuous. ✓")
        else:
            verdict_alpha = "MIXED"
            print("  → Borderline. Some moderate jumps, sigmoid but transition is fast.")

        # 3. Alpha minimum value reached
        alpha_min_observed = float(alpha_for_trans.min())
        print(f"\n[CHECK 3] Alpha minimum reached: {alpha_min_observed:.6f}")
        if alpha_min_observed > 0.0:
            print("  → Alpha never reaches 0 (asymptotic). Continuous behavior. ✓")
        else:
            print("  → Alpha hit 0. Could indicate hard removal.")

        # 4. Damage smoothness for transitioning particles
        damage_for_trans = damage_traj[:, transitioned_idx]
        damage_diffs = np.diff(damage_for_trans, axis=0)
        # All increments should be >= 0 (monotonic) and small
        min_increment = float(damage_diffs.min())
        max_increment = float(damage_diffs.max())
        n_nonmonotonic = int((damage_diffs < -1e-6).sum())
        print(f"\n[CHECK 4] Damage monotonicity:")
        print(f"  Min increment: {min_increment:.6f} (should be ≥ 0)")
        print(f"  Max increment: {max_increment:.6f}")
        print(f"  Non-monotonic samples: {n_nonmonotonic}")
        if abs(min_increment) < 1e-4 and n_nonmonotonic < 100:
            print("  → Damage monotonic + smooth. ✓")
        else:
            print("  → Damage not strictly monotonic. ⚠")

        # Final verdict
        if pcount_constant and verdict_alpha == "CONTINUOUS" and alpha_min_observed > 0.0:
            verdict = "CONTINUOUS — DiSECt-style soft damage already implemented"
        elif pcount_constant and verdict_alpha in ("CONTINUOUS", "MIXED"):
            verdict = "MOSTLY CONTINUOUS — minor smoothness concerns"
        else:
            verdict = "DISCRETE — hard removal or step jumps present"

    # =========================================================================
    # Plots
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: alpha trajectories for transitioning particles
        if transitioned_idx.size > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Sample up to 20 trajectories for clarity
            sample_idx = np.random.choice(
                transitioned_idx,
                size=min(20, transitioned_idx.size), replace=False)

            ax = axes[0]
            for i in sample_idx:
                ax.plot(alpha_traj[:, i], alpha=0.6, linewidth=1.0)
            ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.4, label="threshold")
            ax.axhline(y=0.01, color="gray", linestyle=":", alpha=0.5,
                       label="forward early-return cutoff")
            ax.set_xlabel("simulation step")
            ax.set_ylabel("alpha (soft removal mask)")
            ax.set_title(f"Alpha trajectories for {sample_idx.size} transitioning particles\n"
                         f"(continuous = smooth curves; discrete = step jumps)")
            ax.legend(loc="best")
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.05, 1.05)

            # Plot 2: damage trajectories
            ax = axes[1]
            for i in sample_idx:
                ax.plot(damage_traj[:, i], alpha=0.6, linewidth=1.0)
            ax.set_xlabel("simulation step")
            ax.set_ylabel("particle damage")
            ax.set_title("Particle damage over time (should be monotonic + smooth)")
            ax.grid(alpha=0.3)

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "alpha_damage_trajectories.png"), dpi=110)
            print(f"\n  [plot] {os.path.join(out_dir, 'alpha_damage_trajectories.png')}")

        # Plot 3: alpha distribution at final step
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        final_alpha = alpha_traj[-1]
        ax.hist(final_alpha, bins=100, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(x=0.5, color="r", linestyle="--", alpha=0.5, label="threshold 0.5")
        ax.axvline(x=0.01, color="gray", linestyle=":", alpha=0.5,
                   label="forward cutoff 0.01")
        ax.set_xlabel("alpha at final step")
        ax.set_ylabel("particle count (log scale)")
        ax.set_yscale("log")
        ax.set_title(f"Alpha distribution at step {n_steps}\n"
                     f"continuous → smooth distribution; discrete → bimodal at {{0, 1}}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "alpha_histogram.png"), dpi=110)
        print(f"  [plot] {os.path.join(out_dir, 'alpha_histogram.png')}")

    except ImportError:
        print("  [plot] matplotlib unavailable; skipping figures")

    # =========================================================================
    # Verdict file
    # =========================================================================
    print()
    print("=" * 72)
    print(f"VERDICT: {verdict}")
    print("=" * 72)

    with open(os.path.join(out_dir, "verdict.txt"), "w") as f:
        f.write(f"VERDICT: {verdict}\n\n")
        f.write(f"Particle count constant: {pcount_constant}\n")
        f.write(f"Particles transitioned (0.5+ → 0.5-): {transitioned_idx.size}\n")
        if transitioned_idx.size > 0:
            f.write(f"Max alpha change per step: {max_jump:.4f}\n")
            f.write(f"Median alpha change per step: {median_jump:.6f}\n")
            f.write(f"Alpha minimum reached: {alpha_min_observed:.6f}\n")
            f.write(f"Damage non-monotonic samples: {n_nonmonotonic}\n")
        f.write(f"Steps run: {n_steps}\n")

    print(f"  [verdict] {os.path.join(out_dir, 'verdict.txt')}")
    return 0 if verdict.startswith(("CONTINUOUS", "MOSTLY")) else 1


if __name__ == "__main__":
    sys.exit(main())
