"""Quantify warp vs cpp_sim consistency: replay the SAME knife cut (recorded by cpp_sim
record_trajectory) in warp, compare the resulting cavity (removed set).

cpp output: /tmp/cpp_cavity/{meta.bin, frame_NNNN.bin}
  frame: pos[N*3]f32 + removed[N]i32 + knife[3]f32 + damage[N]f32 + force[4]f32 + n_contact i32
Aligns indices (both load cube_tet.msh) via constant offset check, replays cpp knife
trajectory in warp with matched params, compares removed via IoU + counts + spatial.
"""
import sys, struct
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp

import os as _os
CPP = Path(_os.environ.get("CPP_DIR", "/tmp/cpp_cavity")); device = "cuda:0"
MSH = str(REPO / "resources/cube/cube_tet.msh")
# matched params from config/_test_a_uniform.yaml
import os
ITERS = 6; STIFF = 0.85; DAMP = 0.95; KR = 0.025; ALPHA_MIN = 7.0; TAU = 0.45
SUBSTEPS = int(os.environ.get("SUBSTEPS", "8")); SETTLE_FRAMES = 6


def read_meta():
    b = (CPP / "meta.bin").read_bytes()
    N, nf, pea = struct.unpack("iii", b[:12])
    return N, nf


def read_frame(i, N):
    b = (CPP / f"frame_{i:04d}.bin").read_bytes()
    o = 0
    pos = np.frombuffer(b, np.float32, N * 3, o).reshape(N, 3); o += N * 3 * 4
    removed = np.frombuffer(b, np.int32, N, o).copy(); o += N * 4
    knife = np.frombuffer(b, np.float32, 3, o).copy(); o += 12
    return pos, removed, knife


def main():
    N, nf = read_meta()
    print(f"cpp: N={N} frames={nf}", flush=True)
    pos0_cpp, rem0, _ = read_frame(0, N)
    knife_traj = np.stack([read_frame(i, N)[2] for i in range(nf)])
    rem_final_cpp = read_frame(nf - 1, N)[1].astype(bool)

    # warp solver on SAME mesh
    solver, rest_state, mi = setup_solver_with_rest(MSH, rings=2, fix_threshold_fraction=0.05)
    Nw = solver.N
    pos_rest = mi["vertices"].copy()
    print(f"warp: N={Nw}", flush=True)
    assert Nw == N, "particle count mismatch"

    # index alignment: cpp pos0 should equal warp rest + constant offset
    off = pos0_cpp - pos_rest
    off_std = off.std(0)
    print(f"offset cpp-warp: mean={off.mean(0)}  std={off_std}  (std small => index-aligned)", flush=True)
    aligned = bool((off_std < 1e-3).all())
    if not aligned:
        # fall back to NN match cpp->warp on rest geometry
        from scipy.spatial import cKDTree
        tree = cKDTree(pos_rest)
        _, idx = tree.query(pos0_cpp - off.mean(0))
        perm = idx  # cpp i -> warp perm[i]
        print(f"  NOT index-aligned; using NN match (max dup={np.bincount(perm).max()})", flush=True)
    else:
        perm = np.arange(N)
    offm = off.mean(0)

    # warp config to match cpp
    solver.num_iterations = ITERS; solver.damping = DAMP
    solver.gravity = wp.vec3(0.0, -9.81, 0.0)            # Y-up like cpp
    solver.floor_y = 0.0
    if hasattr(solver, "cutting_update_interval"):
        solver.cutting_update_interval = 4              # match cpp cutting_interval
    sw = torch.tensor(np.full(N, STIFF, np.float32), device=device)
    twt = torch.tensor(np.full(N, TAU, np.float32), device=device)
    tht = torch.tensor(np.array([ALPHA_MIN, 20.0, TAU], np.float32), device=device)  # (P(F) slope, removal sharpness, tau)
    far = pos_rest.max(0) + 100.0

    solver.reset_to_rest(pos_rest); st = rest_state
    with torch.no_grad():
        for frame in range(nf):
            kp = (knife_traj[frame] - offm).astype(np.float32) if frame >= SETTLE_FRAMES else far.astype(np.float32)
            for _ in range(SUBSTEPS):
                aw, pw, _ = WarpFullForwardOp.apply(tht, solver, st, kp, KR, sw, twt)  # (alpha, pos, dv)
                st = solver_save_state(solver)
            if frame % 20 == 0 or frame == nf - 1:
                print(f"  frame {frame}: warp removed={(aw.cpu().numpy()<0.5).sum()}  "
                      f"cpp removed={read_frame(frame, N)[1].sum()}", flush=True)
    alpha_w = aw.cpu().numpy()
    rem_warp = (alpha_w < 0.5)
    # map warp removed back to cpp index order
    rem_warp_cpp_order = rem_warp[perm]

    # compare
    cpp_set = rem_final_cpp; warp_set = rem_warp_cpp_order
    inter = (cpp_set & warp_set).sum(); union = (cpp_set | warp_set).sum()
    iou = inter / max(union, 1)
    only_cpp = (cpp_set & ~warp_set).sum(); only_warp = (~cpp_set & warp_set).sum()
    print(f"\n=== CAVITY COMPARISON (final frame) ===")
    print(f"  cpp removed   = {cpp_set.sum()}")
    print(f"  warp removed  = {warp_set.sum()}")
    print(f"  intersection  = {inter}   union = {union}")
    print(f"  IoU           = {iou:.3f}")
    print(f"  only-cpp (warp missed)  = {only_cpp}")
    print(f"  only-warp (warp extra)  = {only_warp}")
    # spatial: where do disagreements sit? (depth along cut, lateral)
    dis = cpp_set ^ warp_set
    if dis.any():
        dp = pos_rest[perm.argsort()][dis] if not aligned else pos_rest[dis]
        print(f"  disagreement bbox (warp frame): min={dp.min(0)} max={dp.max(0)}", flush=True)
    print(f"\n  => {'HIGH consistency' if iou > 0.7 else ('MODERATE' if iou > 0.4 else 'LOW consistency')} "
          f"(IoU {iou:.2f}); warp {'over' if only_warp > only_cpp else 'under'}-removes vs cpp", flush=True)

    # ---- CLUSTER-SOLVE EFFECT: position (displacement-from-rest) comparison ----
    pos_final_cpp = read_frame(nf - 1, N)[0]                 # cpp final pos (cpp frame)
    disp_cpp = pos_final_cpp - pos0_cpp                      # cpp displacement from its rest
    pw_np = pw.cpu().numpy()
    disp_warp_w = pw_np - pos_rest                           # warp displacement (warp frame)
    disp_warp = disp_warp_w[perm]                            # -> cpp index order
    alive_both = (~cpp_set) & (~warp_set)                    # particles alive in BOTH
    # distance of each particle to the cut (nearest removed particle, rest geom)
    from scipy.spatial import cKDTree
    rem_any = cpp_set | warp_set
    if rem_any.any():
        dtree = cKDTree(pos0_cpp[rem_any])
        dcut, _ = dtree.query(pos0_cpp)
    else:
        dcut = np.full(N, 1e9)
    el = 0.0157  # mean edge len
    near = alive_both & (dcut < 5 * el)
    far = alive_both & (dcut > 15 * el)
    def rms(m):
        d = disp_warp[m] - disp_cpp[m]
        return np.sqrt((d ** 2).sum(1)).mean() * 1000 if m.sum() else float('nan')
    cpp_mag = np.sqrt((disp_cpp[alive_both] ** 2).sum(1)).mean() * 1000
    def mag(arr, m):
        return np.sqrt((arr[m] ** 2).sum(1)).mean() * 1000 if m.sum() else float('nan')
    print(f"\n=== CLUSTER-SOLVE EFFECT (displacement warp vs cpp, mm) ===")
    print(f"  RMS(disp_warp - disp_cpp):  FAR={rms(far):.3f}  NEAR={rms(near):.3f}  ALL={rms(alive_both):.3f} mm")
    print(f"\n  --- Step3A directional: own |disp| magnitude (is warp SOFTER near cut?) ---")
    print(f"    FAR : warp |disp|={mag(disp_warp,far):.3f}  cpp |disp|={mag(disp_cpp,far):.3f} mm")
    print(f"    NEAR: warp |disp|={mag(disp_warp,near):.3f}  cpp |disp|={mag(disp_cpp,near):.3f} mm")
    wn, cn = mag(disp_warp, near), mag(disp_cpp, near)
    print(f"    => NEAR: warp/cpp = {wn/cn:.2f}x  -> "
          f"{'warp SOFTER (consistent with double-attenuation)' if wn > cn * 1.15 else ('warp STIFFER (opposite)' if wn < cn * 0.85 else 'similar magnitude')}", flush=True)


if __name__ == "__main__":
    main()
