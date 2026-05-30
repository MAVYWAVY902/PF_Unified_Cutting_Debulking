"""Phase 0.2 — Verify FPS on actual cube particle x_rest.

Pull x_rest from the same setup_solver_with_rest pipeline used by Stage A
inclusion recovery, run FPS to L1 (64) and L2 (16) centers, and check
geometric uniformity vs random sampling. Also save a viz NPZ for inspection.
"""

import sys
import time
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
from tools.pointnetpp.fps import fps  # noqa: E402


def main():
    mesh_path = str(REPO / "resources" / "cube" / "cube_tet.msh")
    print(f"\n=== Setup solver from {mesh_path} ===")
    solver, rest_state, mesh_info = setup_solver_with_rest(mesh_path)
    vertices_np = mesh_info["vertices"]            # [N, 3] numpy
    N = vertices_np.shape[0]
    print(f"  N particles: {N}")
    print(f"  bbox min: {vertices_np.min(0).round(4).tolist()}")
    print(f"  bbox max: {vertices_np.max(0).round(4).tolist()}")

    x_rest = torch.from_numpy(vertices_np).to("cuda", dtype=torch.float32)

    # --- L1 FPS (mid-scale, ~64) -----------------------------------------
    M1 = 64
    torch.cuda.synchronize(); t0 = time.time()
    idx_L1 = fps(x_rest, num_samples=M1)
    torch.cuda.synchronize(); dt_L1 = time.time() - t0
    centers_L1 = x_rest[idx_L1]

    # --- L2 FPS (coarse-scale, ~16) --------------------------------------
    M2 = 16
    torch.cuda.synchronize(); t0 = time.time()
    idx_L2 = fps(x_rest, num_samples=M2)
    torch.cuda.synchronize(); dt_L2 = time.time() - t0
    centers_L2 = x_rest[idx_L2]

    print(f"\n=== FPS results ===")
    print(f"  L1 {M1} centers: shape {tuple(centers_L1.shape)}  time {dt_L1*1e3:.1f} ms")
    print(f"  L2 {M2} centers: shape {tuple(centers_L2.shape)}  time {dt_L2*1e3:.1f} ms")
    print(f"  Unique L1: {len(set(idx_L1.cpu().tolist()))} / {M1}")
    print(f"  Unique L2: {len(set(idx_L2.cpu().tolist()))} / {M2}")

    # --- Uniformity check: FPS vs random ---------------------------------
    def min_pairwise(pts):
        pd = torch.cdist(pts, pts)
        pd.fill_diagonal_(float("inf"))
        return pd.min().item()

    def random_min_pairwise(n_samples, n_trials=5):
        vals = []
        for _ in range(n_trials):
            rand_idx = torch.randperm(N, device="cuda")[:n_samples]
            vals.append(min_pairwise(x_rest[rand_idx]))
        return float(np.mean(vals))

    fps_L1 = min_pairwise(centers_L1)
    rnd_L1 = random_min_pairwise(M1)
    fps_L2 = min_pairwise(centers_L2)
    rnd_L2 = random_min_pairwise(M2)
    bbox_diag = float(np.linalg.norm(vertices_np.max(0) - vertices_np.min(0)))

    print(f"\n=== Geometric uniformity ===")
    print(f"  bbox diag:               {bbox_diag:.4f} m")
    print(f"  L1 min pair (FPS)    : {fps_L1:.4f} m  ({fps_L1/bbox_diag*100:.1f}% diag)")
    print(f"  L1 min pair (random) : {rnd_L1:.4f} m  ({rnd_L1/bbox_diag*100:.1f}% diag)")
    print(f"  L1 uniformity ratio  : {fps_L1/rnd_L1:.2f}×")
    print(f"  L2 min pair (FPS)    : {fps_L2:.4f} m  ({fps_L2/bbox_diag*100:.1f}% diag)")
    print(f"  L2 min pair (random) : {rnd_L2:.4f} m  ({rnd_L2/bbox_diag*100:.1f}% diag)")
    print(f"  L2 uniformity ratio  : {fps_L2/rnd_L2:.2f}×")

    # --- Coverage check: each particle's distance to nearest L1 center --
    d_to_L1 = torch.cdist(x_rest, centers_L1).min(dim=1).values  # [N]
    d_to_L2 = torch.cdist(x_rest, centers_L2).min(dim=1).values
    print(f"\n=== Coverage (particle → nearest center distance) ===")
    print(f"  L1 mean: {d_to_L1.mean().item():.4f}, max: {d_to_L1.max().item():.4f} m")
    print(f"  L2 mean: {d_to_L2.mean().item():.4f}, max: {d_to_L2.max().item():.4f} m")

    # --- Save NPZ for offline viz ---------------------------------------
    out_path = REPO / "tools" / "pointnetpp" / "phase0_2_fps_cube.npz"
    np.savez(
        out_path,
        x_rest=vertices_np,
        idx_L1=idx_L1.cpu().numpy(),
        idx_L2=idx_L2.cpu().numpy(),
        centers_L1=centers_L1.cpu().numpy(),
        centers_L2=centers_L2.cpu().numpy(),
        d_to_L1=d_to_L1.cpu().numpy(),
        d_to_L2=d_to_L2.cpu().numpy(),
    )
    print(f"\nSaved {out_path}")

    # --- Gate criteria ---------------------------------------------------
    print(f"\n=== Phase 0.2 Gate ===")
    gate_pass = (
        fps_L1 / rnd_L1 > 1.5 and
        fps_L2 / rnd_L2 > 1.5 and
        len(set(idx_L1.cpu().tolist())) == M1 and
        len(set(idx_L2.cpu().tolist())) == M2
    )
    if gate_pass:
        print("  ✓ PASS — FPS uniformity > 1.5× random at both scales, all indices unique")
    else:
        print("  ✗ FAIL")
    return gate_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
