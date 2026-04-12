"""Test: can gradients flow through the full simulation chain?
Simplified: knife_pos -> edge_damage -> particle_damage -> alpha -> loss
"""
import numpy as np
import warp as wp

from mesh.loader import (
    load_tet_mesh, build_vertex_neighbors,
    build_clusters_from_tets, pack_clusters_csr, build_edges,
)
from sim.particles import init_warp_arrays, init_cluster_arrays, init_edge_arrays
from sim.solver import ShapeMatchingSolver

wp.init()
device = "cuda:0"

# Load mesh
vertices, tets = load_tet_mesh("/home/yunxin/xpbd-tissue-sim/resource/cube/cube32.msh")
N = len(vertices)
vertices[:, 1] += -vertices[:, 1].min() + 0.001

vn = build_vertex_neighbors(tets, N)
cids, ccms, coffs = build_clusters_from_tets(vertices, vn, rings=2)
starts, sizes, pids, offsets, cms = pack_clusters_csr(cids, coffs, ccms)
ea, eb, epf = build_edges(vertices, cids, vn)

fixed = np.zeros(N, dtype=np.int32)
for i in range(N):
    if vertices[i, 1] < 0.05:
        fixed[i] = 1

# Init arrays WITH requires_grad on key arrays
pa = init_warp_arrays(vertices, fixed, device)
ca = init_cluster_arrays(starts, sizes, pids, offsets, cms, device)
ea_arr = init_edge_arrays(ea, eb, epf, device)

# Enable gradient tracking on simulation arrays
pa["pos"].requires_grad = True
pa["pred_pos"].requires_grad = True
pa["vel"].requires_grad = True
pa["old_pos"].requires_grad = True
pa["damage"].requires_grad = True
pa["alpha"].requires_grad = True
pa["deltas"].requires_grad = True
ea_arr["damage"].requires_grad = True

solver = ShapeMatchingSolver(pa, ca, ea_arr, device)
solver.cutting_update_interval = 1  # update every step for gradient flow

# Settle without knife (no tape needed)
print("Settling...")
solver.knife_active = 0
for _ in range(50):
    solver.step()
wp.synchronize()
print(f"  Settled centroid: {solver.get_centroid()}")

# Now test gradient flow with tape
print("\n" + "=" * 50)
print("Testing gradient flow through simulation")
print("=" * 50)

# Loss kernel
@wp.kernel
def sum_alpha_loss(alpha: wp.array(dtype=wp.float32), loss: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_add(loss, 0, alpha[i])

# Set knife
solver.set_knife_pos(0.5, 0.5, 0.5)
solver.knife_radius = 0.15
solver.knife_active = 1

loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

# Record 5 steps with tape
print("Recording 5 steps with wp.Tape()...")
tape = wp.Tape()
try:
    with tape:
        for step in range(5):
            solver.step()

        # Loss = sum of alpha (lower alpha = more removal = more debulking)
        wp.launch(sum_alpha_loss, dim=N, inputs=[pa["alpha"], loss], device=device)

    print(f"  Loss (sum alpha): {loss.numpy()[0]:.2f} (max possible: {N})")

    tape.backward(loss=loss)

    # Check which arrays have gradients
    for name, arr in [("pos", pa["pos"]), ("pred_pos", pa["pred_pos"]),
                      ("damage", pa["damage"]), ("alpha", pa["alpha"]),
                      ("edge_damage", ea_arr["damage"])]:
        if arr.grad is not None:
            g = arr.grad.numpy()
            nonzero = np.sum(np.abs(g.reshape(-1)) > 1e-10)
            print(f"  {name}.grad: EXISTS, nonzero={nonzero}, norm={np.linalg.norm(g.reshape(-1)):.6f}")
        else:
            print(f"  {name}.grad: None")

    print("\n  AUTODIFF STATUS: Gradient flow detected!" if pa["alpha"].grad is not None else "\n  AUTODIFF STATUS: Gradients NOT flowing through alpha")

except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")
