#!/usr/bin/env python3
"""Generate Warp Migration Plan PDF (Chinese)"""
from fpdf import FPDF

class PlanPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(100,100,100)
        self.cell(0, 8, 'P(F) Shape Matching -> NVIDIA Warp Migration Plan', align='R')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128,128,128)
        self.cell(0, 10, f'{self.page_no()} / {{nb}}', align='C')
    def stitle(self, t):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(20, 80, 40)
        self.cell(0, 10, t, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 80, 40)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)
    def stitle2(self, t):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(50,50,50)
        self.cell(0, 8, t, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
    def txt(self, t):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0,0,0)
        self.multi_cell(0, 5.5, t)
        self.ln(2)
    def bold_txt(self, label, t):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0,0,0)
        self.write(5.5, label)
        self.set_font('Helvetica', '', 10)
        self.write(5.5, t)
        self.ln(7)
    def code(self, c):
        self.set_font('Courier', '', 8)
        self.set_fill_color(240,243,240)
        self.set_text_color(30,30,30)
        x, w = self.l_margin, self.w - self.l_margin - self.r_margin
        lines = c.strip().split('\n')
        h = 4.0
        th = len(lines)*h + 6
        if self.get_y() + th > self.h - 20: self.add_page()
        y0 = self.get_y()
        self.rect(x, y0, w, th, 'F')
        self.set_xy(x+3, y0+3)
        for l in lines:
            self.cell(w-6, h, l, new_x="LMARGIN", new_y="NEXT")
            self.set_x(x+3)
        self.set_y(y0 + th + 3)
    def trow(self, cells, bold=False, fill=False, widths=None):
        self.set_font('Helvetica', 'B' if bold else '', 8.5)
        if fill: self.set_fill_color(225, 238, 225)
        if widths is None:
            widths = [int((self.w - self.l_margin - self.r_margin)/len(cells))] * len(cells)
        for c, w in zip(cells, widths):
            self.cell(w, 7, c, border=1, fill=fill)
        self.ln()
    def num_item(self, n, t):
        self.set_font('Helvetica', 'B', 10)
        self.cell(8, 5.5, f'{n}.')
        self.set_font('Helvetica', '', 10)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 8, 5.5, t)
        self.ln(1)
    def green_txt(self, t):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(20, 100, 40)
        self.multi_cell(0, 5.5, t)
        self.set_text_color(0,0,0)
        self.ln(1)

pdf = PlanPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 22)
pdf.set_text_color(20, 70, 40)
pdf.cell(0, 15, 'NVIDIA Warp Migration Plan', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 13)
pdf.set_text_color(80,80,80)
pdf.cell(0, 10, 'P(F) Shape Matching PBD -> Differentiable Simulation', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font('Helvetica', 'I', 10)
pdf.set_text_color(120,120,120)
pdf.cell(0, 8, 'C++/CUDA (forward-only) -> Python/Warp (auto-differentiable)', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)

# ============================================================
# Section 1: Environment
# ============================================================
pdf.stitle('1. Environment Verification')
pdf.txt('Warp 1.10.1 installed, CUDA 12.8, RTX 2070 (sm_75). Key APIs confirmed:')
w3 = [40, 25, 80]
pdf.trow(['Warp API', 'Status', 'CUDA Equivalent'], bold=True, fill=True, widths=w3)
pdf.trow(['wp.svd3', 'YES', 'svd3_cuda.h (custom header)'], widths=w3)
pdf.trow(['wp.atomic_add', 'YES', 'atomicAdd()'], widths=w3)
pdf.trow(['wp.HashGrid', 'YES', 'CUB sort + cell_start/end'], widths=w3)
pdf.trow(['wp.Tape', 'YES', 'N/A (no autodiff in CUDA)'], widths=w3)
pdf.trow(['wp.vec3 / wp.mat33', 'YES', 'float3 / manual 3x3'], widths=w3)
pdf.trow(['wp.kernel', 'YES', '__global__ void'], widths=w3)
pdf.ln(2)
pdf.green_txt('All blockers cleared. Full migration is feasible.')

# ============================================================
# Section 2: Migration Scope
# ============================================================
pdf.stitle('2. Migration Scope: 12 CUDA Kernels -> Warp Kernels')
w4 = [45, 18, 18, 64]
pdf.trow(['CUDA Kernel', 'Lines', 'Difficulty', 'Warp Notes'], bold=True, fill=True, widths=w4)
pdf.trow(['predict_kernel', '15', 'Easy', 'vel += g*dt, pred = pos + vel*dt'], widths=w4)
pdf.trow(['collision_kernel', '25', 'Easy', 'floor clamp + sphere SDF'], widths=w4)
pdf.trow(['clear_deltas_kernel', '5', 'Trivial', 'wp.zeros()'], widths=w4)
pdf.trow(['cluster_solve_kernel', '90', 'Hard', 'wp.svd3 + wp.atomic_add'], widths=w4)
pdf.trow(['apply_kernel', '10', 'Easy', 'pred += stiff * delta / count'], widths=w4)
pdf.trow(['update_velocity_kernel', '10', 'Easy', 'vel = (pred - old) / dt'], widths=w4)
pdf.trow(['update_edge_damage', '30', 'Medium', 'sigmoid P(F) + max damage'], widths=w4)
pdf.trow(['update_particle_damage', '15', 'Medium', 'wp.atomic_max (custom)'], widths=w4)
pdf.trow(['aspirate_kernel', '10', 'Easy', 'soft mask for differentiable'], widths=w4)
pdf.trow(['compute_cell_hash', '15', 'Easy', 'wp.HashGrid handles this'], widths=w4)
pdf.trow(['find_cell_bounds', '10', 'N/A', 'wp.HashGrid built-in'], widths=w4)
pdf.trow(['self_collision_kernel', '60', 'Medium', 'wp.hash_grid_query()'], widths=w4)
pdf.ln(2)
pdf.bold_txt('Core challenge: ', 'cluster_solve_kernel (warp shuffle reduction + SVD). Warp has no warp shuffle, need to restructure as per-particle kernel with shared memory or sequential accumulation per cluster.')

# ============================================================
# Section 3: Architecture
# ============================================================
pdf.stitle('3. Warp Architecture Design')
pdf.stitle2('3.1 Data Layout')
pdf.code(
"""# Particle data (Structure of Arrays)
pos       = wp.zeros(N, dtype=wp.vec3, device="cuda:0")
vel       = wp.zeros(N, dtype=wp.vec3, device="cuda:0")
pred_pos  = wp.zeros(N, dtype=wp.vec3, device="cuda:0")
old_pos   = wp.zeros(N, dtype=wp.vec3, device="cuda:0")
damage    = wp.zeros(N, dtype=wp.float32, device="cuda:0")
removed   = wp.zeros(N, dtype=wp.float32, device="cuda:0")  # float for differentiable
fixed     = wp.zeros(N, dtype=wp.int32, device="cuda:0")

# Cluster data (CSR format, same as CUDA)
cluster_starts  = wp.array(starts_np, dtype=wp.int32, device="cuda:0")
cluster_sizes   = wp.array(sizes_np, dtype=wp.int32, device="cuda:0")
particle_ids    = wp.array(ids_np, dtype=wp.int32, device="cuda:0")
rest_offsets    = wp.array(offsets_np, dtype=wp.vec3, device="cuda:0")

# Delta buffers
deltas       = wp.zeros(N, dtype=wp.vec3, device="cuda:0")
delta_counts = wp.zeros(N, dtype=wp.int32, device="cuda:0")""")

pdf.stitle2('3.2 Key Differentiability Change: Soft Removal')
pdf.code(
"""# CUDA (non-differentiable):
if damage[i] > 0.3:
    removed[i] = 1       # hard binary decision

# Warp (differentiable):
alpha[i] = wp.sigmoid(-k * (damage[i] - threshold))  # soft mask [0,1]
# alpha ~ 1 when healthy, alpha ~ 0 when damaged
# All subsequent computations use alpha[i] as weight:
#   pred_pos[i] contributes alpha[i] to cluster solve
#   mass_effective[i] = alpha[i] * mass[i]""")

pdf.stitle2('3.3 Cluster Solve Restructuring')
pdf.txt('CUDA uses 1 warp (32 threads) per cluster with __shfl_sync reduction. Warp has no warp shuffle. Two options:')
pdf.bold_txt('Option A (recommended): ', 'Per-cluster sequential loop in kernel. Each kernel instance handles one cluster, loops over its particles sequentially. Simple, correct, differentiable. Slower than warp shuffle but adequate for <30k clusters.')
pdf.bold_txt('Option B: ', 'Two-pass: Pass 1 = per-particle kernel computes partial sums (cm, A_pq contributions), atomicAdd to per-cluster accumulators. Pass 2 = per-cluster kernel does SVD + goal positions. More parallel but more complex.')
pdf.code(
"""# Option A: Per-cluster kernel (recommended for Warp)
@wp.kernel
def cluster_solve_kernel(
    pred_pos: wp.array(dtype=wp.vec3),
    alpha: wp.array(dtype=wp.float32),     # soft removal mask
    cluster_starts: wp.array(dtype=wp.int32),
    cluster_sizes: wp.array(dtype=wp.int32),
    particle_ids: wp.array(dtype=wp.int32),
    rest_offsets: wp.array(dtype=wp.vec3),
    deltas: wp.array(dtype=wp.vec3),
    delta_counts: wp.array(dtype=wp.int32)):

    cid = wp.tid()
    start = cluster_starts[cid]
    size = cluster_sizes[cid]

    # 1. Weighted centroid (sequential loop)
    cm = wp.vec3(0.0, 0.0, 0.0)
    total_w = float(0.0)
    for j in range(size):
        pid = particle_ids[start + j]
        w = alpha[pid]
        cm += w * pred_pos[pid]
        total_w += w
    if total_w < 1.5: return
    cm = cm / total_w

    # 2. Covariance matrix A_pq (sequential)
    # ... similar loop building 3x3 matrix ...

    # 3. SVD
    U, sigma, V = wp.svd3(A)
    R = U * wp.transpose(V)

    # 4. Goal positions -> atomic_add
    for j in range(size):
        pid = particle_ids[start + j]
        if alpha[pid] < 0.01: continue
        q = rest_offsets[start + j]
        goal = R * q + cm
        correction = goal - pred_pos[pid]
        wp.atomic_add(deltas, pid, correction)
        wp.atomic_add(delta_counts, pid, 1)""")

# ============================================================
# Section 4: Phase Plan
# ============================================================
pdf.stitle('4. 6-Phase Implementation Plan')

pdf.stitle2('Phase 1: Skeleton + Mesh Loading (Week 1)')
pdf.num_item(1, 'Create project: pf_shape_matching_warp/')
pdf.num_item(2, 'Use meshio to load .msh tet mesh -> numpy arrays')
pdf.num_item(3, 'Build cluster data from tet adjacency (reuse algorithm from tet_mesh.h)')
pdf.num_item(4, 'Initialize Warp arrays (pos, vel, clusters CSR, edges)')
pdf.num_item(5, 'Verify: print particle count, cluster count, edge count')
pdf.code(
"""import meshio, numpy as np, warp as wp

mesh = meshio.read("cube32.msh")
vertices = mesh.points          # (N, 3)
tets = mesh.cells_dict["tetra"] # (T, 4)

wp.init()
pos = wp.array(vertices, dtype=wp.vec3, device="cuda:0")""")
pdf.green_txt('Deliverable: Warp arrays initialized from .msh file, cluster CSR built.')

pdf.stitle2('Phase 2: Core Shape Matching (Week 2-3)')
pdf.num_item(1, 'Implement predict_kernel (trivial)')
pdf.num_item(2, 'Implement collision_kernel (floor + knife sphere SDF)')
pdf.num_item(3, 'Implement cluster_solve_kernel with wp.svd3 (core challenge)')
pdf.num_item(4, 'Implement apply_kernel, clear_deltas, update_velocity')
pdf.num_item(5, 'Verify: cube drops onto floor, deforms, settles. Compare with C++ output.')
pdf.code(
"""# Validation: compare Warp vs CUDA output
# Run both 100 steps on same input, compare final positions
max_diff = np.max(np.abs(warp_pos - cuda_pos))
assert max_diff < 1e-3, f"Mismatch: {max_diff}" """)
pdf.green_txt('Deliverable: Cube falls onto floor correctly. Numerically matches CUDA within 1e-3.')

pdf.stitle2('Phase 3: P(F) Damage + Soft Removal (Week 4)')
pdf.num_item(1, 'Implement P(F) edge damage kernel (sigmoid, knife surface distance)')
pdf.num_item(2, 'Implement particle damage aggregation (atomic max via atomic_add trick)')
pdf.num_item(3, 'Implement SOFT removal: alpha[i] = sigmoid(-k*(damage-threshold))')
pdf.num_item(4, 'Modify cluster_solve to weight by alpha[i]')
pdf.num_item(5, 'Verify: knife contact -> damage accumulates -> particles fade out')
pdf.green_txt('Deliverable: Debulking works with soft differentiable removal.')

pdf.stitle2('Phase 4: Autodiff Verification (Week 5)')
pdf.num_item(1, 'Wrap simulation loop in wp.Tape()')
pdf.num_item(2, 'Define loss: L = -V_removed + lambda * nerve_damage')
pdf.num_item(3, 'Call tape.backward(loss)')
pdf.num_item(4, 'Verify: dL/d(tool_pos) has correct sign (moving toward tumor -> gradient negative)')
pdf.num_item(5, 'Finite difference validation: compare analytic gradient vs FD')
pdf.code(
"""tape = wp.Tape()
with tape:
    for step in range(100):
        simulate_step(tool_pos, ...)
    loss = compute_loss(damage, alpha, nerve_mask)

tape.backward(loss)
grad_tool = tape.gradients[tool_pos]  # dL/d(tool_pos)

# Finite difference check
eps = 1e-4
tool_pos_plus = tool_pos + eps
loss_plus = run_forward(tool_pos_plus)
grad_fd = (loss_plus - loss) / eps
assert abs(grad_tool - grad_fd) / abs(grad_fd) < 0.05  # <5% error""")
pdf.green_txt('Deliverable: Analytic gradients match finite difference within 5%.')

pdf.stitle2('Phase 5: Trajectory Optimization (Week 6-7)')
pdf.num_item(1, 'Parameterize tool trajectory: tau(t) = spline with N control points')
pdf.num_item(2, 'Objective: max tumor_removed - lambda * nerve_damage')
pdf.num_item(3, 'Optimizer: Adam on control points, using Warp gradients')
pdf.num_item(4, 'Baseline comparisons: random, PPO, CEM')
pdf.num_item(5, 'Generate figures: optimized trajectory visualization, Pareto curve')
pdf.code(
"""control_points = wp.array(init_trajectory, dtype=wp.vec3, requires_grad=True)
optimizer = torch.optim.Adam([control_points.numpy()], lr=1e-3)

for epoch in range(200):
    tape = wp.Tape()
    with tape:
        trajectory = interpolate_spline(control_points, T)
        for t in range(T):
            set_tool_pos(trajectory[t])
            simulate_step(...)
        loss = -volume_removed + lambda_val * nerve_damage
    tape.backward(loss)
    optimizer.step()""")
pdf.green_txt('Deliverable: Optimized trajectory avoids nerve, removes max tumor. Paper-ready figures.')

pdf.stitle2('Phase 6: Visualization + Paper Experiments (Week 8)')
pdf.num_item(1, 'Polyscope or Open3D visualization of trajectory + damage field')
pdf.num_item(2, 'Ablation: P(F) vs threshold, differentiable vs non-differentiable')
pdf.num_item(3, 'Scalability: 5k, 20k, 100k particles timing')
pdf.num_item(4, 'Heterogeneous tissue: 3-layer nerve preservation demo')
pdf.num_item(5, 'Paper writing: CoRL / ICRA format')
pdf.green_txt('Deliverable: Complete paper draft with all figures and experiments.')

# ============================================================
# Section 5: File Structure
# ============================================================
pdf.stitle('5. Warp Project File Structure')
pdf.code(
"""pf_shape_matching_warp/
  |-- sim/
  |   |-- particles.py        # Particle data + wp.array initialization
  |   |-- clusters.py         # Cluster building from tet mesh + CSR packing
  |   |-- kernels.py          # All @wp.kernel definitions (predict, solve, apply...)
  |   |-- pf_damage.py        # P(F) edge damage + soft removal kernels
  |   |-- solver.py           # ShapeMatchingSolver class (step, init, loss)
  |
  |-- mesh/
  |   |-- loader.py           # meshio .msh loader -> numpy arrays
  |   |-- heterogeneous.py    # 3-layer tissue model
  |
  |-- optim/
  |   |-- trajectory.py       # Spline trajectory parameterization
  |   |-- optimizer.py        # Adam/L-BFGS on Warp gradients
  |   |-- baselines.py        # PPO, CEM, random baselines
  |
  |-- viz/
  |   |-- renderer.py         # Polyscope / Open3D visualization
  |
  |-- experiments/
  |   |-- exp_gradient_check.py   # Finite difference validation
  |   |-- exp_trajectory_opt.py   # Task 1: trajectory optimization
  |   |-- exp_pareto.py           # Task 3: volume vs nerve Pareto
  |   |-- exp_ablation.py         # P(F) vs threshold ablation
  |
  |-- main.py                 # Entry point""")

# ============================================================
# Section 6: CUDA -> Warp Mapping
# ============================================================
pdf.stitle('6. CUDA -> Warp API Mapping Reference')
w2 = [72, 72]
pdf.trow(['CUDA', 'Warp'], bold=True, fill=True, widths=w2)
pdf.trow(['__global__ void kernel(...)', '@wp.kernel def kernel(...)'], widths=w2)
pdf.trow(['kernel<<<blocks, threads>>>(...)', 'wp.launch(kernel, dim=N, ...)'], widths=w2)
pdf.trow(['float3', 'wp.vec3'], widths=w2)
pdf.trow(['float3* d_array', 'wp.array(dtype=wp.vec3)'], widths=w2)
pdf.trow(['cudaMalloc / cudaFree', 'wp.zeros() / automatic GC'], widths=w2)
pdf.trow(['cudaMemcpy H->D', 'wp.array(numpy_arr, device="cuda:0")'], widths=w2)
pdf.trow(['cudaMemcpy D->H', 'array.numpy()'], widths=w2)
pdf.trow(['atomicAdd(&x, val)', 'wp.atomic_add(array, idx, val)'], widths=w2)
pdf.trow(['__shfl_down_sync()', 'sequential loop (no warp shuffle)'], widths=w2)
pdf.trow(['svd3_cuda.h svd()', 'wp.svd3(A)'], widths=w2)
pdf.trow(['CUB radix sort + cell arrays', 'wp.HashGrid'], widths=w2)
pdf.trow(['N/A (no autodiff)', 'wp.Tape() + tape.backward()'], widths=w2)
pdf.trow(['cudaDeviceSynchronize()', 'wp.synchronize()'], widths=w2)

# ============================================================
# Section 7: Risks
# ============================================================
pdf.stitle('7. Risks and Mitigations')
w4r = [40, 18, 45, 42]
pdf.trow(['Risk', 'Impact', 'Mitigation', 'Timeline Impact'], bold=True, fill=True, widths=w4r)
pdf.trow(['cluster_solve perf', 'Medium', 'Sequential loop OK for <30k', '+0 (acceptable)'], widths=w4r)
pdf.trow(['wp.svd3 gradient', 'High', 'Test early in Phase 2', '+1 week if fails'], widths=w4r)
pdf.trow(['Tape memory for', 'Medium', 'Checkpoint every N steps', '+2 days'], widths=w4r)
pdf.trow([' long rollouts', '', '', ''], widths=w4r)
pdf.trow(['HashGrid differentiable?', 'Low', 'Self-collision not critical', '+0'], widths=w4r)
pdf.trow(['meshio .msh v4.1', 'Low', 'Already tested with meshio', '+0'], widths=w4r)

# ============================================================
# Section 8: Timeline
# ============================================================
pdf.stitle('8. Timeline Summary')
w_tl = [22, 60, 60]
pdf.trow(['Week', 'Task', 'Deliverable'], bold=True, fill=True, widths=w_tl)
pdf.trow(['1', 'Mesh loading + Warp array init', 'Particles + clusters in Warp'], widths=w_tl)
pdf.trow(['2-3', 'Core shape matching (SVD)', 'Cube falls correctly, matches CUDA'], widths=w_tl)
pdf.trow(['4', 'P(F) damage + soft removal', 'Debulking with differentiable mask'], widths=w_tl)
pdf.trow(['5', 'Autodiff verification', 'Gradients match FD within 5%'], widths=w_tl)
pdf.trow(['6-7', 'Trajectory optimization', 'Nerve-avoiding path, Pareto curves'], widths=w_tl)
pdf.trow(['8', 'Visualization + paper', 'Complete draft + figures'], widths=w_tl)
pdf.ln(3)
pdf.green_txt('Total: 8 weeks. Critical path: Week 2-3 (SVD kernel) -> Week 5 (gradient check).')

# ============================================================
# Section 9: Success Criteria
# ============================================================
pdf.stitle('9. Success Criteria')
pdf.num_item(1, 'Shape matching simulation matches CUDA output within 1e-3 (100 steps)')
pdf.num_item(2, 'P(F) damage accumulates correctly with soft sigmoid mask')
pdf.num_item(3, 'Analytic gradients (via wp.Tape) match finite difference within 5%')
pdf.num_item(4, 'Trajectory optimization converges: nerve_damage decreases, volume_removed increases')
pdf.num_item(5, 'Analytic gradient method outperforms PPO baseline in sample efficiency (100x fewer rollouts)')
pdf.num_item(6, '20k particles runs at >30 FPS forward, backward pass <100ms')

out = '/home/yunxin/pf-shape-matching/Warp_Migration_Plan.pdf'
pdf.output(out)
print(f'PDF generated: {out}')
