"""Test which Warp operations support automatic differentiation."""
import warp as wp
import numpy as np

wp.init()
device = "cuda:0"


# Test 1: SVD gradient
@wp.kernel
def test_svd_grad(A_flat: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    A = wp.mat33(A_flat[0], A_flat[1], A_flat[2],
                 A_flat[3], A_flat[4], A_flat[5],
                 A_flat[6], A_flat[7], A_flat[8])
    U, sigma, V = wp.svd3(A)
    R = U * wp.transpose(V)
    wp.atomic_add(out, 0, R[0, 0] + R[1, 1] + R[2, 2])


# Test 2: Sigmoid gradient
@wp.kernel
def test_sigmoid_grad(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    val = 1.0 / (1.0 + wp.exp(-10.0 * x[i]))
    wp.atomic_add(out, 0, val)


# Test 3: atomic_add vec3 gradient
@wp.kernel
def test_atomic_vec3(inp: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    v = inp[i]
    wp.atomic_add(out, 0, v[0] + v[1] + v[2])


# Test 4: Conditional write gradient
@wp.kernel
def test_conditional(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    val = x[i]
    if val > 0.5:
        wp.atomic_add(out, 0, val)


def run_test(name, kernel, inputs, dim=1):
    try:
        tape = wp.Tape()
        out = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        with tape:
            wp.launch(kernel, dim=dim, inputs=inputs + [out], device=device)
        tape.backward(loss=out)
        grad = inputs[0].grad
        if grad is not None:
            grad_np = grad.numpy()
            print(f"  {name}: PASS  out={out.numpy()[0]:.4f}  grad={grad_np}")
        else:
            print(f"  {name}: PARTIAL  out={out.numpy()[0]:.4f}  grad=None")
        tape.zero()
    except Exception as e:
        print(f"  {name}: FAIL  error={e}")


print("=" * 60)
print("Warp Autodiff Operation Tests")
print("=" * 60)

# Test 1: SVD
A = wp.array([1.0, 0.1, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 1.0],
             dtype=wp.float32, device=device, requires_grad=True)
run_test("SVD (wp.svd3)", test_svd_grad, [A], dim=1)

# Test 2: Sigmoid
x = wp.array([0.0, 0.5, -0.5], dtype=wp.float32, device=device, requires_grad=True)
run_test("Sigmoid (wp.exp)", test_sigmoid_grad, [x], dim=3)

# Test 3: atomic_add vec3
v = wp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
             dtype=wp.vec3, device=device, requires_grad=True)
run_test("atomic_add vec3", test_atomic_vec3, [v], dim=2)

# Test 4: Conditional
x4 = wp.array([0.3, 0.7, 0.9], dtype=wp.float32, device=device, requires_grad=True)
run_test("Conditional (if val>0.5)", test_conditional, [x4], dim=3)

print("\nDone.")
