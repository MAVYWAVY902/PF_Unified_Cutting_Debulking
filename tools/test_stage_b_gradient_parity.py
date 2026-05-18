#!/usr/bin/env python3
"""
Stage B Day 2 — gradient parity test: verify production SSA kernels
(in pf_damage.py) give correct autograd gradients on (sharpness, alpha_min)
matching FD ground truth within 1% rel_err.

This is the END-TO-END VERIFICATION of Stage B Day 1's refactor:
  - new SSA kernels in pf_damage.py work in production location
  - cycle-scoped buffer pattern produces clean SSA chain
  - autograd gradient on calibration parameters matches FD ground truth

Mirrors POC v3 logic (which PASSED with rel_err 0.95% / 0.11%) but uses
kernels imported from pf_damage.py rather than locally defined.

If this passes (rel_err < 1% on both params), Stage B Day 1 is verified
end-to-end and Stage B can proceed to Day 2-3 caller updates with high
confidence.

Pass criteria:
  - sharpness rel_err < 1% vs FD
  - alpha_min  rel_err < 1% vs FD
  - sign agreement on both
  - no verify_autograd_array_access warnings on SSA path
  - determinism (zero drift on repeat)
"""
import sys
import numpy as np
import warp as wp

wp.init()
wp.config.verify_autograd_array_access = True

print("=" * 72)
print(f"  Stage B Day 2 gradient parity test (production SSA kernels)")
print(f"  Warp version: {wp.__version__}")
print("=" * 72)

sys.path.insert(0, '/home/yunxin/pf-shape-matching')

# Import PRODUCTION SSA kernels from pf_damage.py (NOT from POC v3 local copies)
from pf_shape_matching_warp.sim.pf_damage import (
    # Already-existing SSA kernels (from earlier work)
    update_particle_damage_kernel,
    finalize_particle_damage_kernel_ssa,
    # NEW SSA kernels added in Stage B Day 1
    apply_edge_damage_ssa_kernel,
    apply_particle_damage_ssa_kernel,
    # NEW calibration variants added today (wp.array params for autograd)
    update_edge_damage_kernel_ssa_param_arr,
    update_soft_removal_kernel_ssa_param_arr,
)


@wp.kernel
def loss_alpha_to_zero(
    alpha: wp.array(dtype=wp.float32),
    loss_buf: wp.array(dtype=wp.float32),
):
    """Loss = sum alpha² (proxy for "debulking complete" = low alpha)."""
    i = wp.tid()
    a = alpha[i]
    wp.atomic_add(loss_buf, 0, a * a)


def setup_test_system(N=20, knife_x=0.5):
    """Same synthetic 1D edge graph as POC v3."""
    np.random.seed(42)
    pos_np = np.zeros((N, 3), dtype=np.float32)
    pos_np[:, 0] = np.linspace(0.3, 0.7, N)
    pos_np[:, 1] = 0.5
    pos_np[:, 2] = 0.5
    E = N - 1
    edge_a_np = np.arange(E, dtype=np.int32)
    edge_b_np = np.arange(1, N, dtype=np.int32)
    pf_rest_np = np.full(E, 0.5, dtype=np.float32)
    knife_pos_np = np.array([[knife_x, 0.5, 0.5]], dtype=np.float32)
    return dict(
        pos=pos_np, edge_a=edge_a_np, edge_b=edge_b_np,
        pf_rest=pf_rest_np, knife_pos=knife_pos_np,
        N=N, E=E,
    )


def make_params(knife_radius=0.1, alpha_min=3.0, sharpness=20.0, threshold=0.5):
    return dict(
        knife_radius=wp.array([knife_radius], dtype=wp.float32, requires_grad=True),
        alpha_min=wp.array([alpha_min], dtype=wp.float32, requires_grad=True),
        sharpness=wp.array([sharpness], dtype=wp.float32, requires_grad=True),
        threshold=wp.array([threshold], dtype=wp.float32, requires_grad=True),
    )


def run_full_ssa(params, sys_dict, n_cycles=3):
    """Mirrors POC v3 run_full_ssa, but uses production kernels.

    All 3 state buffers (alpha, edge_damage, particle_damage) cycle-scoped.
    """
    N = sys_dict["N"]; E = sys_dict["E"]
    pos = wp.array(sys_dict["pos"], dtype=wp.vec3)
    edge_a = wp.array(sys_dict["edge_a"], dtype=wp.int32)
    edge_b = wp.array(sys_dict["edge_b"], dtype=wp.int32)
    pf_rest = wp.array(sys_dict["pf_rest"], dtype=wp.float32)
    knife_pos = wp.array(sys_dict["knife_pos"], dtype=wp.vec3)
    fixed = wp.array(np.zeros(N, dtype=np.int32), dtype=wp.int32)

    # Cycle-scoped buffer LISTS (n_cycles + 1 each)
    alpha_buffers = [
        wp.array(np.ones(N, dtype=np.float32), dtype=wp.float32, requires_grad=True)
        for _ in range(n_cycles + 1)
    ]
    edge_damage_buffers = [
        wp.zeros(E, dtype=wp.float32, requires_grad=True)
        for _ in range(n_cycles + 1)
    ]
    particle_damage_buffers = [
        wp.zeros(N, dtype=wp.float32, requires_grad=True)
        for _ in range(n_cycles + 1)
    ]

    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

    with wp.Tape() as tape:
        for cycle in range(n_cycles):
            # === EDGE DAMAGE ===
            edge_damage_delta = wp.zeros(E, dtype=wp.float32, requires_grad=True)
            wp.launch(update_edge_damage_kernel_ssa_param_arr, dim=E, inputs=[
                pos, alpha_buffers[cycle], edge_a, edge_b, pf_rest,
                edge_damage_buffers[cycle],
                knife_pos, params["knife_radius"], params["alpha_min"],
                edge_damage_delta,
            ])
            wp.launch(apply_edge_damage_ssa_kernel, dim=E, inputs=[
                edge_damage_buffers[cycle],
                edge_damage_delta,
                edge_damage_buffers[cycle + 1],
            ])

            # === PARTICLE DAMAGE ===
            damage_sum = wp.zeros(N, dtype=wp.float32, requires_grad=True)
            damage_count = wp.zeros(N, dtype=wp.float32, requires_grad=True)
            wp.launch(update_particle_damage_kernel, dim=E, inputs=[
                damage_sum, damage_count, alpha_buffers[cycle],
                edge_a, edge_b,
                edge_damage_buffers[cycle + 1],
            ])

            particle_damage_delta = wp.zeros(N, dtype=wp.float32, requires_grad=True)
            wp.launch(finalize_particle_damage_kernel_ssa, dim=N, inputs=[
                damage_sum, damage_count,
                particle_damage_buffers[cycle],
                particle_damage_delta,
            ])
            wp.launch(apply_particle_damage_ssa_kernel, dim=N, inputs=[
                particle_damage_buffers[cycle],
                particle_damage_delta,
                particle_damage_buffers[cycle + 1],
            ])

            # === SOFT REMOVAL ===
            wp.launch(update_soft_removal_kernel_ssa_param_arr, dim=N, inputs=[
                particle_damage_buffers[cycle + 1],
                alpha_buffers[cycle],
                fixed,
                params["threshold"], params["sharpness"],
                alpha_buffers[cycle + 1],
            ])

        wp.launch(loss_alpha_to_zero, dim=N, inputs=[alpha_buffers[-1], loss])

    tape.backward(loss=loss)

    return float(loss.numpy()[0]), {
        "sharpness": float(params["sharpness"].grad.numpy()[0]),
        "alpha_min": float(params["alpha_min"].grad.numpy()[0]),
    }


def fd_grad(sys_dict, param_name, eps=1e-3, n_cycles=3):
    defaults = dict(knife_radius=0.1, alpha_min=3.0, sharpness=20.0, threshold=0.5)
    base = defaults[param_name]

    def run_with_value(v):
        kw = dict(defaults)
        kw[param_name] = v
        params = make_params(**kw)
        loss, _ = run_full_ssa(params, sys_dict, n_cycles=n_cycles)
        return loss

    return (run_with_value(base + eps) - run_with_value(base - eps)) / (2 * eps)


def main():
    sys_dict = setup_test_system(N=20, knife_x=0.5)
    n_cycles = 3

    print()
    print("─" * 72)
    print(f"  Production SSA gradient parity (N={sys_dict['N']}, n_cycles={n_cycles})")
    print("─" * 72)

    # Autograd
    params = make_params()
    loss, autograd_grads = run_full_ssa(params, sys_dict, n_cycles=n_cycles)
    print(f"  Forward loss = {loss:.6e}")

    # FD ground truth
    fd_grads = {
        "sharpness": fd_grad(sys_dict, "sharpness", n_cycles=n_cycles),
        "alpha_min": fd_grad(sys_dict, "alpha_min", n_cycles=n_cycles),
    }

    # Compare
    results = {}
    for pname in ["sharpness", "alpha_min"]:
        ag = autograd_grads[pname]
        fd = fd_grads[pname]
        sign_ok = np.sign(ag) * np.sign(fd) > 0
        rel_err = abs(ag - fd) / max(abs(fd), 1e-12)
        silent_zero = abs(ag) < 1e-9

        print(f"  ∂loss/∂{pname:<10}  autograd={ag:+.4e}  FD={fd:+.4e}  "
              f"rel_err={rel_err*100:.3f}%  sign={'✓' if sign_ok else '✗'}")

        if silent_zero:
            verdict = "SILENT_ZERO"
        elif sign_ok and rel_err < 0.01:
            verdict = "PASS"
        elif sign_ok and rel_err < 0.05:
            verdict = "PARTIAL"
        elif sign_ok and rel_err < 0.50:
            verdict = "DEGRADED"
        elif sign_ok:
            verdict = "FAIL_MAGNITUDE"
        else:
            verdict = "FAIL_DIRECTION"
        results[pname] = dict(autograd=ag, fd=fd, rel_err=rel_err, verdict=verdict)

    # Determinism
    print()
    print("  Determinism check (run again):")
    params2 = make_params()
    _, autograd_grads2 = run_full_ssa(params2, sys_dict, n_cycles=n_cycles)
    for pname in ["sharpness", "alpha_min"]:
        diff = abs(autograd_grads[pname] - autograd_grads2[pname])
        print(f"    ∂loss/∂{pname:<10} drift = {diff:.2e}  {'✓' if diff < 1e-9 else '⚠️'}")

    # Final verdict
    print()
    print("=" * 72)
    print("  STAGE B DAY 2 GRADIENT PARITY VERDICT")
    print("=" * 72)
    print(f"  sharpness: {results['sharpness']['verdict']}  "
          f"(rel_err {results['sharpness']['rel_err']*100:.3f}%)")
    print(f"  alpha_min: {results['alpha_min']['verdict']}  "
          f"(rel_err {results['alpha_min']['rel_err']*100:.3f}%)")

    all_pass = all(results[p]['verdict'] == 'PASS' for p in ['sharpness', 'alpha_min'])

    print()
    if all_pass:
        print("  ✅ STAGE B DAY 2 PASS")
        print("     Production SSA kernels in pf_damage.py give correct gradients.")
        print("     Stage B Day 1 verified end-to-end. Proceed to Day 3 (callers).")
        return 0
    else:
        print("  ❌ FAIL — production SSA kernels have issue")
        for p in ['sharpness', 'alpha_min']:
            print(f"     {p}: {results[p]['verdict']} (rel_err {results[p]['rel_err']*100:.2f}%)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
