"""Debug script: diagnose why Müller polar disagrees with scipy on near-identity input.
Key idea: construct A = R_gt * S where R_gt is KNOWN ground-truth rotation and S is
symmetric. Then the polar of A should be exactly R_gt. Compare Müller vs scipy vs R_gt."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pf_shape_matching_warp"))
import numpy as np
import warp as wp
wp.init()
from scipy.linalg import polar
from pf_shape_matching_warp.sim.kernels import extract_rotation_muller


@wp.kernel
def _batch(A: wp.array(dtype=wp.mat33),
           q_in: wp.array(dtype=wp.quat),
           n: int,
           R: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    q = extract_rotation_muller(A[i], q_in[i], n)
    R[i] = wp.quat_to_matrix(q)


def _rodrigues(axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*K @ K


def run_muller(A_np, n_iter, device="cuda:0"):
    A_wp = wp.array([A_np.astype(np.float32)], dtype=wp.mat33, device=device)
    q_init = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat, device=device)
    R_out = wp.zeros(1, dtype=wp.mat33, device=device)
    wp.launch(_batch, dim=1, inputs=[A_wp, q_init, n_iter, R_out])
    wp.synchronize()
    return R_out.numpy()[0]


def test_known_rotation(axis, angle, stretch_scale, label):
    R_gt = _rodrigues(np.array(axis, dtype=float), angle)
    # Symmetric positive-definite stretch
    S = np.eye(3) * stretch_scale
    A = R_gt @ S

    u_scipy, _ = polar(A.astype(np.float64))
    diff_scipy_vs_gt = np.abs(u_scipy - R_gt).max()

    print(f"\n--- {label} ---")
    print(f"  axis={axis}, angle={angle}, stretch={stretch_scale}")
    print(f"  det(A)={np.linalg.det(A):.6f}")
    print(f"  scipy U vs R_gt:   max |diff| = {diff_scipy_vs_gt:.3e}")

    for n_iter in [4, 8, 16, 32]:
        R_m = run_muller(A, n_iter)
        diff_vs_gt = np.abs(R_m - R_gt.astype(np.float32)).max()
        diff_vs_scipy = np.abs(R_m - u_scipy.astype(np.float32)).max()
        orth_err = np.linalg.norm(R_m.T @ R_m - np.eye(3))
        det_err = abs(np.linalg.det(R_m) - 1)
        print(f"  Müller({n_iter:2d}): vs R_gt={diff_vs_gt:.3e}  vs scipy={diff_vs_scipy:.3e}  "
              f"orth={orth_err:.2e}  |det-1|={det_err:.2e}")


print("=" * 70)
print("C.1 debug: Müller vs scipy vs ground-truth R on KNOWN inputs")
print("=" * 70)

# Trivial
test_known_rotation([0, 0, 1], 0.0, 1.0, "Identity (A = I)")
# Small rotation around y, identity stretch
test_known_rotation([0, 1, 0], 0.1, 1.0, "Small y-rotation (no stretch)")
# Moderate rotation around random axis, small stretch
test_known_rotation([0.577, 0.577, 0.577], 0.3, 1.05, "0.3 rad tri-axis, +5% stretch")
# Large rotation, identity stretch
test_known_rotation([1, 0, 0], 1.0, 1.0, "1.0 rad x-rotation, no stretch")
# Large rotation with large stretch
test_known_rotation([0, 1, 0], 2.0, 1.5, "2.0 rad y-rot, 50% stretch")
