"""Step 1 isolation: NOISE FLOOR. Run the SAME warp cut twice (identical config) and
measure run-to-run position divergence (atomic_add nondeterminism amplified over 960
steps). Compare to the warp-vs-cpp FAR=0.96 / NEAR=6.74 mm. If noise floor ~= FAR,
then 'far-from-cut difference' is just nondeterminism, NOT a cluster-solve issue.
"""
import sys, struct, os
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp

CPP = Path("/tmp/cpp_nosc"); device = "cuda:0"
MSH = str(REPO / "resources/cube/cube_tet.msh")
ITERS = 6; STIFF = 0.85; DAMP = 0.95; KR = 0.025; ALPHA_MIN = 7.0; TAU = 0.45
SUBSTEPS = 8; SETTLE = 6


def read_meta():
    b = (CPP / "meta.bin").read_bytes(); N, nf, _ = struct.unpack("iii", b[:12]); return N, nf


def read_frame(i, N):
    b = (CPP / f"frame_{i:04d}.bin").read_bytes(); o = 0
    pos = np.frombuffer(b, np.float32, N * 3, o).reshape(N, 3); o += N * 12
    removed = np.frombuffer(b, np.int32, N, o).copy(); o += N * 4
    knife = np.frombuffer(b, np.float32, 3, o).copy()
    return pos, removed, knife


def warp_run(knife_traj, pos_rest, offm, nf):
    solver, rest_state, mi = setup_solver_with_rest(MSH, rings=2, fix_threshold_fraction=0.05)
    N = solver.N; solver.num_iterations = ITERS; solver.damping = DAMP
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = 0.0
    far = pos_rest.max(0) + 100.0
    sw = torch.tensor(np.full(N, STIFF, np.float32), device=device)
    twt = torch.tensor(np.full(N, TAU, np.float32), device=device)
    tht = torch.tensor(np.array([ALPHA_MIN, 20.0, TAU], np.float32), device=device)
    solver.reset_to_rest(pos_rest); st = rest_state
    with torch.no_grad():
        for frame in range(nf):
            kp = (knife_traj[frame] - offm).astype(np.float32) if frame >= SETTLE else far.astype(np.float32)
            for _ in range(SUBSTEPS):
                aw, pw, _ = WarpFullForwardOp.apply(tht, solver, st, kp, KR, sw, twt)
                st = solver_save_state(solver)
    return pw.cpu().numpy(), (aw.cpu().numpy() < 0.5)


def main():
    N, nf = read_meta()
    pos0_cpp = read_frame(0, N)[0]
    knife_traj = np.stack([read_frame(i, N)[2] for i in range(nf)])
    solver0, _, mi0 = setup_solver_with_rest(MSH, rings=2, fix_threshold_fraction=0.05)
    pos_rest = mi0["vertices"].copy()
    offm = (pos0_cpp - pos_rest).mean(0)

    print("running warp twice (identical config)...", flush=True)
    pw1, rem1 = warp_run(knife_traj, pos_rest, offm, nf)
    pw2, rem2 = warp_run(knife_traj, pos_rest, offm, nf)

    # stratify by distance to cut (warp's own removed set, union of both runs)
    rem = rem1 | rem2
    from scipy.spatial import cKDTree
    dcut = cKDTree(pos_rest[rem]).query(pos_rest)[0] if rem.any() else np.full(N, 1e9)
    el = 0.0157
    alive = ~rem1 & ~rem2
    near = alive & (dcut < 5 * el); farm = alive & (dcut > 15 * el)
    d = np.sqrt(((pw1 - pw2) ** 2).sum(1)) * 1000
    print(f"\n=== NOISE FLOOR (warp vs warp, same config, {nf}f x{SUBSTEPS}) ===")
    print(f"  run-to-run removed: {rem1.sum()} vs {rem2.sum()}  (diff {abs(int(rem1.sum())-int(rem2.sum()))})")
    print(f"  RMS(warp1 - warp2):")
    print(f"    FAR  (>15 edge) = {d[farm].mean():.3f} mm   (n={int(farm.sum())})")
    print(f"    NEAR (<5 edge)  = {d[near].mean():.3f} mm   (n={int(near.sum())})")
    print(f"    ALL alive       = {d[alive].mean():.3f} mm", flush=True)
    print(f"\n  compare to warp-vs-cpp:  FAR 0.96 / NEAR 6.74 mm")
    print(f"  => FAR: {'noise dominates (cluster-solve diff is SMALL/none)' if d[farm].mean() > 0.5 else 'real cluster-solve difference (noise small)'}", flush=True)


if __name__ == "__main__":
    main()
