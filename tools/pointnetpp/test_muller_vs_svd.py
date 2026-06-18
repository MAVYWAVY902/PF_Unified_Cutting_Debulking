"""Step 2: Muller (warp) vs SVD (cpp) rotation extraction on REAL A_pq matrices from a
deformed cube's clusters. Pure numpy, faithful to both. Match (~1e-4) => Muller NOT the
FAR cause; diverge => Muller IS a contributor.
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
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def muller_R(A, n_iter=8, q=None):
    if q is None: q = np.array([0., 0., 0., 1.])
    for _ in range(n_iter):
        R = quat_to_mat(q)
        num = np.cross(R[:, 0], A[:, 0]) + np.cross(R[:, 1], A[:, 1]) + np.cross(R[:, 2], A[:, 2])
        den = abs(R[:, 0] @ A[:, 0] + R[:, 1] @ A[:, 1] + R[:, 2] @ A[:, 2]) + 1e-9
        omega = num / den
        w = np.linalg.norm(omega)
        if w < 1e-9: break
        axis = omega / (w + 1e-20)
        s = np.sin(w / 2)
        dq = np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(w / 2)])
        q = qmul(dq, q); q = q / np.linalg.norm(q)
    return quat_to_mat(q)


def svd_R(A):
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R


def angle(R1, R2):
    c = (np.trace(R1.T @ R2) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def main():
    cfg = MESH_CONFIGS[0]; device = "cuda:0"
    solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), rings=2, fix_threshold_fraction=0.05)
    N = solver.N; solver.num_iterations = 6; solver.damping = 0.95
    pos_rest = mi["vertices"].copy()
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos_rest[:, 2].min() - 5.0)
    sw = torch.tensor(np.full(N, 0.85, np.float32), device=device)
    twt = torch.tensor(np.full(N, 0.45, np.float32), device=device)
    tht = torch.tensor(np.array([7., 20., 0.45], np.float32), device=device)
    far = pos_rest.max(0) + 100.0
    solver.reset_to_rest(pos_rest); st = rest_state
    with torch.no_grad():
        for _ in range(40):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, 0.0, sw, twt)
            st = solver_save_state(solver)
    pos = pw.cpu().numpy().astype(np.float64)
    starts = solver.c["starts"].numpy(); sizes = solver.c["sizes"].numpy()
    pids = solver.c["particle_ids"].numpy(); roff = solver.c["rest_offsets"].numpy().astype(np.float64)

    angs, corr = [], []
    for cid in range(0, solver.num_clusters, 7):
        s0 = starts[cid]; n = sizes[cid]
        if n < 2: continue
        m = pids[s0:s0 + n]
        p = pos[m]; q = roff[s0:s0 + n]
        pc = p - p.mean(0); qc = q - q.mean(0)
        A = pc.T @ qc
        Rs = svd_R(A); Rm = muller_R(A, 8)
        angs.append(angle(Rs, Rm))
        corr.append(np.sqrt(((qc @ Rm.T - qc @ Rs.T) ** 2).sum(1)).mean())
    a = np.array(angs); c = np.array(corr)
    print(f"\n=== Muller(warp,8) vs SVD(cpp) on {len(a)} real cluster A_pq ===")
    print(f"  rotation angle diff (deg): mean={a.mean():.2e} median={np.median(a):.2e} max={a.max():.2e}")
    print(f"  goal-pos diff:             mean={c.mean()*1000:.5f}mm  max={c.max()*1000:.5f}mm")
    print(f"  => {'Muller~=SVD: rotation NOT the FAR cause' if a.max()<1e-2 else 'Muller!=SVD: rotation IS a contributor'} "
          f"(max angle {a.max():.1e} deg, max goal {c.max()*1000:.4f}mm)", flush=True)


if __name__ == "__main__":
    main()
