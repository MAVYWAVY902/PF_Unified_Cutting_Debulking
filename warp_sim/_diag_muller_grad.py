"""Diagnose silent-zero in extract_rotation_muller.

Isolate: A (mat33, requires_grad) → Müller → R (mat33) → loss = sum(R).
Check if A.grad is populated after backward.

If A.grad = 0 even in this minimal setup, the Müller func itself is the problem
(Python for-loop variable re-binding not SSA-clean in Warp codegen).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "pf_shape_matching_warp"))
import numpy as np
import warp as wp
wp.init()
wp.config.verify_autograd_array_access = True
from pf_shape_matching_warp.sim.kernels import extract_rotation_muller


@wp.kernel
def _run_muller(
    A_in: wp.array(dtype=wp.mat33),
    q_init: wp.array(dtype=wp.quat),
    n_iter: int,
    R_out: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    q = extract_rotation_muller(A_in[i], q_init[i], n_iter)
    R_out[i] = wp.quat_to_matrix(q)


@wp.kernel
def _sum_mat(R: wp.array(dtype=wp.mat33), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    m = R[i]
    wp.atomic_add(out, 0,
                  m[0, 0] + m[0, 1] + m[0, 2] +
                  m[1, 0] + m[1, 1] + m[1, 2] +
                  m[2, 0] + m[2, 1] + m[2, 2])


def main():
    # A is a small rotation-ish matrix — clear forward answer
    A_np = np.array([[1.0, 0.1, 0.0],
                     [-0.1, 1.0, 0.0],
                     [0.0, 0.0, 1.05]], dtype=np.float32).reshape(1, 3, 3)

    for n_iter in [1, 2, 4, 8, 16]:
        A_leaf = wp.array(A_np, dtype=wp.mat33, device="cuda:0", requires_grad=True)
        q_init = wp.array([wp.quat(0.0, 0.0, 0.0, 1.0)], dtype=wp.quat,
                           device="cuda:0", requires_grad=False)
        R_out = wp.zeros(1, dtype=wp.mat33, device="cuda:0", requires_grad=True)
        loss_buf = wp.zeros(1, dtype=wp.float32, device="cuda:0", requires_grad=True)

        with wp.Tape() as tape:
            wp.launch(_run_muller, dim=1,
                      inputs=[A_leaf, q_init, n_iter, R_out])
            wp.launch(_sum_mat, dim=1, inputs=[R_out, loss_buf])

        tape.backward(loss=loss_buf)
        loss_val = float(loss_buf.numpy()[0])
        R = R_out.numpy()[0]
        grad = A_leaf.grad.numpy() if A_leaf.grad is not None else np.zeros_like(A_np)
        g_norm = float(np.linalg.norm(grad))
        print(f"n_iter={n_iter:2d}: loss={loss_val:.4f}, R[0,0]={R[0,0]:.4f}, |A.grad|={g_norm:.3e}")


if __name__ == "__main__":
    main()
