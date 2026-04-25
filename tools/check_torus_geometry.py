#!/usr/bin/env python3
"""Verify Z-up torus geometry after env settle.

Catches the class of bug that hit us 2026-04-25: a coordinate-axis mismatch
between physics (gravity = -Z) and mesh setup (mesh shift + fixed mask used Y
axis). Symptom = settled torus is asymmetric / vertically stretched / centroid
off-center.

This tool:
  1. Loads the env exactly as training does (uses the same yaml).
  2. Checks settled position centroid + bbox against expected values for a
     properly-flat Z-up torus on the floor.
  3. Reports pass/fail per check with concrete numbers and tolerances.
  4. Optionally saves a diagnostic plot (--plot) showing the actual settled
     geometry — top-down (XY), side (XZ), front (YZ), and 3D — so you can
     eyeball whether the ring is symmetric.

Usage:
  python tools/check_torus_geometry.py
  python tools/check_torus_geometry.py --cfg warp_sim/config/debulk_cap_armA.yaml --plot
  python tools/check_torus_geometry.py --major-r 0.35 --minor-r 0.12

Exit codes:
  0  all checks passed
  1  one or more checks failed
  2  setup error (bad cfg path, env init failed)

Why this lives in tools/ and not as a unit test: it needs the full Warp
runtime + GPU + 200-step settle pass. ~30 seconds, not test-suite-friendly.
"""
import argparse
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "pf_shape_matching_warp"))


def check(name, observed, expected, tol, *, unit="", lower_better_negative=False):
    """One pass/fail line with formatted numbers."""
    delta = abs(observed - expected)
    ok = delta <= tol
    status = "PASS" if ok else "FAIL"
    arrow = "✓" if ok else "✗"
    print(f"  [{status}] {arrow} {name:30s}  observed={observed:.4f}{unit}  "
          f"expected={expected:.4f}{unit}  |Δ|={delta:.4f}  tol=±{tol:.4f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="warp_sim/config/debulk_cap_armC.yaml",
                    help="Env yaml (default: armC capsule task)")
    ap.add_argument("--major-r", type=float, default=0.35,
                    help="Expected torus major R (center-to-tube center). Default 0.35.")
    ap.add_argument("--minor-r", type=float, default=0.12,
                    help="Expected torus minor r (tube radius). Default 0.12.")
    ap.add_argument("--plot", action="store_true",
                    help="Save diagnostic 4-panel plot to results/geometry_check/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=None,
                    help="Plot output dir (default: results/geometry_check/)")
    args = ap.parse_args()

    # Lazy imports so --help works without warp/torch installed
    import warp as wp
    import yaml
    from warp_sim.envs.debulk_env import DebulkEnv

    cfg_path = os.path.join(REPO, args.cfg) if not os.path.isabs(args.cfg) else args.cfg
    if not os.path.exists(cfg_path):
        print(f"[error] cfg not found: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    print(f"=== Torus Geometry Check ===")
    print(f"  cfg        : {cfg_path}")
    print(f"  expected R : {args.major_r}  (major radius)")
    print(f"  expected r : {args.minor_r}  (minor radius)")
    print()

    wp.init()
    cfg = yaml.safe_load(open(cfg_path))
    print(f"  Loading env (will run {cfg.get('settle_steps', 200)}-step settle)...")
    print()
    env = DebulkEnv(cfg, device=args.device)

    pos = env.solver.p["pos"].numpy()
    fixed = env._fixed_np
    tumor_mask = env._tumor_mask
    boundary = env._boundary_mask_np == 1
    interior = env._interior_mask_np == 1

    cx, cy, cz = pos.mean(axis=0)
    bbox_lo = pos.min(axis=0)
    bbox_hi = pos.max(axis=0)
    bbox = bbox_hi - bbox_lo

    # Tumor-only bbox (excludes any fixtures / nerve etc)
    tumor_pos = pos[tumor_mask]
    t_lo = tumor_pos.min(axis=0)
    t_hi = tumor_pos.max(axis=0)
    t_bbox = t_hi - t_lo
    t_cx, t_cy, t_cz = tumor_pos.mean(axis=0)

    print(f"=== Settled state ===")
    print(f"  All particles  : N={len(pos):>5d}, fixed={int(fixed.sum())}, "
          f"tumor={int(tumor_mask.sum())}, "
          f"capsule={int(boundary.sum())}, interior={int(interior.sum())}")
    print(f"  All centroid   : ({cx:.4f}, {cy:.4f}, {cz:.4f})")
    print(f"  All bbox       : low={bbox_lo}  high={bbox_hi}  size={bbox}")
    print(f"  Tumor centroid : ({t_cx:.4f}, {t_cy:.4f}, {t_cz:.4f})")
    print(f"  Tumor bbox size: {t_bbox}")
    print()

    R, r = args.major_r, args.minor_r
    # Expected for properly flat Z-up torus, after mesh shifted to z=0.001 + settle:
    #   X centroid ≈ 0.5 (symmetric in mesh frame around X)
    #   Y centroid ≈ 0.5 (symmetric in mesh frame around Y)
    #   Z centroid ≈ minor_r ≈ r (centerline of tube above floor)
    #   X bbox    ≈ 2(R+r) (full ring outer extent)
    #   Y bbox    ≈ 2(R+r) (same — ring is in XY plane)
    #   Z bbox    ≈ 2r (tube thickness)
    print(f"=== Pass/Fail Checks (Z-up flat torus on floor) ===")

    results = []
    # Centroid checks — torus generated centered at (0.5, 0.5, ?).
    # X/Y centroid should remain near 0.5 after settle (no horizontal motion).
    results.append(check("centroid X", cx, 0.5, 0.02))
    results.append(check("centroid Y", cy, 0.5, 0.02))
    # Z centroid: mesh was shifted so min_z = 0.001; for flat torus, centroid_z
    # should be at minor_r ≈ r above min_z, i.e. ~r + 0.001.
    results.append(check("centroid Z (flat-torus rest)", cz, r + 0.001, 0.03,
                          unit=""))
    # bbox sizes — these tell us if the ring is symmetric in XY plane and not
    # vertically stretched.
    results.append(check("bbox X (≈2·(R+r))", t_bbox[0], 2 * (R + r), 0.04))
    results.append(check("bbox Y (≈2·(R+r))", t_bbox[1], 2 * (R + r), 0.04))
    # Z thickness: a healthy flat torus is 2·r thick.
    results.append(check("bbox Z (≈2·r — tube thickness)", t_bbox[2], 2 * r,
                          0.04))
    # X/Y symmetry check: ring centroid distance from mesh center should be ~0
    # for a perfectly symmetric ring. Y-axis mesh-shift bug breaks this.
    results.append(check("X-Y aspect ratio", t_bbox[0] / max(1e-6, t_bbox[1]),
                          1.0, 0.10))
    # Z height vs X width — for a flat torus, height should be much less than width.
    # Tolerance accounts for ~10-15% gravity-driven settling compression of the
    # soft tetrahedral mesh; tight enough to flag axis-swap bugs (which produce
    # >50% deviation), loose enough to not flag normal stiffness=0.6 settling.
    aspect_zx = t_bbox[2] / max(1e-6, t_bbox[0])
    expected_zx = (2 * r) / (2 * (R + r))  # = r / (R+r)
    results.append(check("Z/X aspect (flat ratio)", aspect_zx, expected_zx,
                          0.05))

    print()
    n_pass = sum(results)
    n_total = len(results)
    print(f"=== Summary: {n_pass}/{n_total} checks passed ===")

    if n_pass < n_total:
        print()
        print("⚠️  Geometry is NOT a properly-flat Z-up torus. Likely causes:")
        print("    1. Mesh shift uses wrong axis  (debulk_env.py vertices[:, ?])")
        print("    2. Fixed-mask threshold uses wrong axis (debulk_env.py)")
        print("    3. Solver gravity not aligned with floor axis (solver.py)")
        print("    4. Floor-collision kernel checks wrong p[?] (kernels.py)")
        print("    5. Mesh file (.msh) generated in wrong axis convention")
        print()
        print("    Run grep over the codebase for occurrences of [:, 1] / p[1]")
        print("    that should be [:, 2] / p[2] under Z-up convention.")

    if args.plot:
        save_diagnostic_plot(env, pos, tumor_mask, boundary, interior,
                              R, r, args.out_dir)

    sys.exit(0 if n_pass == n_total else 1)


def save_diagnostic_plot(env, pos, tumor_mask, boundary, interior, R, r, out_dir):
    """4-panel plot — same layout as visualize_torus_setup.py for visual continuity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or os.path.join(REPO, "results", "geometry_check")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Top-down XY
    ax = axes[0]
    ax.scatter(pos[interior, 0], pos[interior, 1], c="royalblue", s=1.5, alpha=0.30,
               label=f"interior ({int(interior.sum())})")
    ax.scatter(pos[boundary, 0], pos[boundary, 1], c="crimson", s=2, alpha=0.60,
               label=f"capsule ({int(boundary.sum())})")
    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Top-down (XY) — should look like a doughnut")
    ax.grid(alpha=0.2); ax.legend(fontsize=7)

    # 2. Side XZ
    ax = axes[1]
    ax.scatter(pos[interior, 0], pos[interior, 2], c="royalblue", s=1.5, alpha=0.30)
    ax.scatter(pos[boundary, 0], pos[boundary, 2], c="crimson", s=2, alpha=0.60)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.5)
    ax.set_xlabel("X"); ax.set_ylabel("Z (gravity ↓)")
    ax.set_title("Side (XZ) — flat band, thick=2·r")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4, label="floor (Z=0)")
    ax.axhline(y=2 * r, color="green", linestyle=":", alpha=0.5, label=f"expected top (2r={2*r:.2f})")
    ax.grid(alpha=0.2); ax.legend(fontsize=7)

    # 3. Front YZ
    ax = axes[2]
    ax.scatter(pos[interior, 1], pos[interior, 2], c="royalblue", s=1.5, alpha=0.30)
    ax.scatter(pos[boundary, 1], pos[boundary, 2], c="crimson", s=2, alpha=0.60)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.5)
    ax.set_xlabel("Y"); ax.set_ylabel("Z")
    ax.set_title("Front (YZ) — should mirror XZ")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.4)
    ax.axhline(y=2 * r, color="green", linestyle=":", alpha=0.5)
    ax.grid(alpha=0.2)

    # 4. 3D
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    sub = np.random.choice(np.where(interior)[0],
                            size=min(2000, int(interior.sum())), replace=False)
    ax.scatter(pos[sub, 0], pos[sub, 1], pos[sub, 2],
                c="royalblue", s=2, alpha=0.20, depthshade=False)
    ax.scatter(pos[boundary, 0], pos[boundary, 1], pos[boundary, 2],
                c="crimson", s=3, alpha=0.55, depthshade=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 0.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("3D — should be a flat doughnut on Z=0")
    ax.view_init(elev=22, azim=-55)

    fig.suptitle(f"Settled geometry — N={len(pos)}, "
                 f"centroid=({pos.mean(0)[0]:.3f}, {pos.mean(0)[1]:.3f}, {pos.mean(0)[2]:.3f})",
                 fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "geometry_check.png")
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\n[plot] saved diagnostic: {out_path}")


if __name__ == "__main__":
    main()
