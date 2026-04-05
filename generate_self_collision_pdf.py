#!/usr/bin/env python3
"""Generate GPU Self-Collision Plan PDF"""
from fpdf import FPDF

class PlanPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'P(F) Shape Matching - GPU Self-Collision Plan', align='R')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(30, 60, 120)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def subsection_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def code_block(self, code):
        self.set_font('Courier', '', 8.5)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        w = self.w - self.l_margin - self.r_margin
        lines = code.strip().split('\n')
        h_per_line = 4.5
        total_h = len(lines) * h_per_line + 6
        if self.get_y() + total_h > self.h - 20:
            self.add_page()
        y_start = self.get_y()
        self.rect(x, y_start, w, total_h, 'F')
        self.set_xy(x + 3, y_start + 3)
        for line in lines:
            self.cell(w - 6, h_per_line, line, new_x="LMARGIN", new_y="NEXT")
            self.set_x(x + 3)
        self.set_y(y_start + total_h + 3)

    def bullet(self, text, indent=0):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        x = self.l_margin + indent
        self.set_x(x)
        self.cell(5, 5.5, chr(8226))
        self.multi_cell(self.w - self.r_margin - x - 5, 5.5, text)
        self.ln(1)

    def numbered(self, num, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(0, 0, 0)
        self.set_font('Helvetica', 'B', 10)
        self.cell(8, 5.5, f'{num}.')
        self.set_font('Helvetica', '', 10)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 8, 5.5, text)
        self.ln(1)

    def table_row(self, cells, bold=False, fill=False):
        self.set_font('Helvetica', 'B' if bold else '', 9)
        if fill:
            self.set_fill_color(230, 235, 245)
        col_widths = [45, 25, 25] if len(cells) == 3 else [55, 40, 40, 40]
        if len(cells) == 2:
            col_widths = [70, 105]
        for i, (cell, w) in enumerate(zip(cells, col_widths)):
            self.cell(w, 7, cell, border=1, fill=fill, align='C' if i > 0 and len(cells) > 2 else 'L')
        self.ln()


pdf = PlanPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font('Helvetica', 'B', 20)
pdf.set_text_color(20, 50, 100)
pdf.cell(0, 15, 'GPU Self-Collision', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, 'via Spatial Hash Grid', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font('Helvetica', 'I', 10)
pdf.set_text_color(120, 120, 120)
pdf.cell(0, 8, 'P(F) Shape Matching PBD  |  CUDA GPU Implementation Plan', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)

# 1. Context
pdf.section_title('1. Context')
pdf.body_text(
    'The GPU shape matching PBD simulation currently handles floor collision and knife (CUSA) collision, '
    'but has no self-collision detection between particles. After debulking removes large numbers of particles, '
    'the remaining structure\'s shape matching constraints weaken, leading to self-intersection artifacts. '
    'A GPU-based spatial hash grid with PBD position correction is needed to prevent particle overlap.'
)

# 2. Approach
pdf.section_title('2. Approach: Counting Sort Hash Grid + PBD Correction')

pdf.subsection_title('Core Idea')
pdf.numbered(1, 'Each Jacobi iteration: build spatial hash grid via counting sort')
pdf.numbered(2, 'Each particle queries 3x3x3 neighborhood: find overlapping pairs')
pdf.numbered(3, 'PBD-style correction: overlap -> atomicAdd into existing d_deltas/d_delta_counts, averaged together with shape matching in apply_kernel')
pdf.numbered(4, 'Floor + knife hard constraints applied last (unchanged)')

pdf.subsection_title('Modified step() Pipeline')
pdf.code_block(
"""for iter in [0, num_iterations):
    clear_deltas_kernel
    cluster_solve_kernel            // shape matching constraints
    if self_collision_enabled:
        build_hash_grid()           // 3 kernels + CUB radix sort
        self_collision_kernel       // query neighbors + atomicAdd corrections
    apply_kernel                    // merge shape matching + self-collision
    collision_kernel                // floor + knife hard constraints"""
)

pdf.body_text(
    'Key design: self-collision corrections accumulate into d_deltas (shared with shape matching), '
    'unified averaging in apply_kernel. Does not directly modify predicted_pos.'
)

# 3. Implementation
pdf.section_title('3. Implementation Steps')

# Step 1
pdf.subsection_title('Step 1: New GPU Data Structures')
pdf.body_text('Add to ShapeMatchingGPUSolver (shape_matching_gpu.h):')
pdf.code_block(
"""// Self-collision hash grid buffers
uint32_t* d_cell_hash = nullptr;
uint32_t* d_particle_index = nullptr;
uint32_t* d_cell_hash_sorted = nullptr;
uint32_t* d_particle_index_sorted = nullptr;
uint32_t* d_cell_start = nullptr;       // [TABLE_SIZE]
uint32_t* d_cell_end = nullptr;         // [TABLE_SIZE]
void* d_cub_temp = nullptr;
size_t cub_temp_bytes = 0;
static const int HASH_TABLE_SIZE = 32768;  // 2^15"""
)

pdf.body_text('Add to SimConfigGPU:')
pdf.code_block(
"""float self_collision_radius;      // particle collision radius, ~0.012
float self_collision_stiffness;   // PBD stiffness, 0.5-1.0
int self_collision_enabled;       // toggle on/off"""
)

# Step 2
pdf.subsection_title('Step 2: GPU Memory Allocation')
pdf.body_text(
    'Allocate hash grid buffers at the end of init() (N particles + TABLE_SIZE cells). '
    'Free in cleanup(). Query CUB temp storage size via '
    'cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes, ...).'
)

# Step 3
pdf.subsection_title('Step 3: Three New CUDA Kernels')

pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 7, 'Kernel 1: compute_cell_hash_kernel', new_x="LMARGIN", new_y="NEXT")
pdf.code_block(
"""// Per particle:
if (removed[i]) { hash = TABLE_SIZE; }  // sentinel -> sorted to end
else {
    ix = floor(pos.x / cell_size);
    iy = floor(pos.y / cell_size);
    iz = floor(pos.z / cell_size);
    hash = (ix*73856093 ^ iy*19349663 ^ iz*83492791) & (TABLE_SIZE-1);
}
cell_hash[i] = hash;
particle_index[i] = i;"""
)

pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 7, 'Kernel 2: find_cell_bounds_kernel', new_x="LMARGIN", new_y="NEXT")
pdf.code_block(
"""// Per sorted element i:
if (i == 0 || sorted_hash[i] != sorted_hash[i-1])
    cell_start[sorted_hash[i]] = i;
if (i == N-1 || sorted_hash[i] != sorted_hash[i+1])
    cell_end[sorted_hash[i]] = i + 1;"""
)

pdf.set_font('Helvetica', 'B', 10)
pdf.cell(0, 7, 'Kernel 3: self_collision_kernel', new_x="LMARGIN", new_y="NEXT")
pdf.code_block(
"""// Per particle i:
if (removed[i] || fixed[i]) return;
(ix, iy, iz) = cell coords of pred_pos[i]
for each of 27 neighbor cells (3x3x3 stencil):
    h = hash(cx, cy, cz) & (TABLE_SIZE-1)
    for k in cell_start[h] .. cell_end[h]:
        j = sorted_index[k]
        if (j <= i || removed[j]) continue;  // avoid double-count
        dist = length(pred_pos[i] - pred_pos[j])
        if (dist < 2*radius && dist > 1e-8):
            overlap = 2*radius - dist
            dir = (pred_pos[i] - pred_pos[j]) / dist
            corr = stiffness * 0.5 * overlap * dir
            atomicAdd(&deltas[i], +corr)
            atomicAdd(&deltas[j], -corr)
            atomicAdd(&delta_counts[i], 1)
            atomicAdd(&delta_counts[j], 1)"""
)

# Step 4
pdf.subsection_title('Step 4: build_hash_grid() Host Function')
pdf.code_block(
"""void ShapeMatchingGPUSolver::build_hash_grid() {
    int N = d_particles.N;
    int blocks = (N + 255) / 256;

    // 1. Compute per-particle cell hash
    compute_cell_hash_kernel<<<blocks, 256>>>(
        d_particles.predicted_pos, d_particles.removed,
        d_cell_hash, d_particle_index,
        config.self_collision_radius * 2.0f, N);

    // 2. Sort by hash (CUB radix sort, only 15 bits)
    cub::DeviceRadixSort::SortPairs(
        d_cub_temp, cub_temp_bytes,
        d_cell_hash, d_cell_hash_sorted,
        d_particle_index, d_particle_index_sorted,
        N, 0, 15);

    // 3. Find cell boundaries
    cudaMemset(d_cell_start, 0xFF, HASH_TABLE_SIZE * sizeof(uint32_t));
    cudaMemset(d_cell_end, 0xFF, HASH_TABLE_SIZE * sizeof(uint32_t));
    find_cell_bounds_kernel<<<blocks, 256>>>(
        d_cell_hash_sorted, d_cell_start, d_cell_end, N);
}"""
)

# Step 5
pdf.subsection_title('Step 5: Integration into step()')
pdf.body_text(
    'Insert build_hash_grid() + self_collision_kernel between cluster_solve_kernel and apply_kernel '
    'in the Jacobi iteration loop. Self-collision corrections flow through the same d_deltas/d_delta_counts '
    'path as shape matching, ensuring consistent Jacobi averaging.'
)

# Step 6
pdf.subsection_title('Step 6: Configuration Parameters')
pdf.code_block(
"""// In main_gpu_viewer.cpp:
gpu_solver.config.self_collision_radius = 0.012f;
gpu_solver.config.self_collision_stiffness = 0.8f;
gpu_solver.config.self_collision_enabled = 1;"""
)
pdf.body_text(
    'particle_radius = 0.012 gives collision distance 2r = 0.024, which is less than the minimum '
    'neighbor spacing ~0.03 in cube32.msh. This ensures self-collision does NOT trigger in the rest state, '
    'only activating when deformation brings particles closer than their natural spacing.'
)

# 4. Performance
pdf.section_title('4. Performance Budget')

pdf.table_row(['Operation', 'Per Iteration', '4 Iter/Step'], bold=True, fill=True)
pdf.table_row(['compute_hash', '0.02 ms', '0.08 ms'])
pdf.table_row(['CUB sort (20k)', '0.05 ms', '0.20 ms'])
pdf.table_row(['find_bounds', '0.03 ms', '0.12 ms'])
pdf.table_row(['self_collision', '0.10 ms', '0.40 ms'])
pdf.table_row(['Total', '~0.2 ms', '~0.8 ms'], bold=True, fill=True)
pdf.ln(4)

pdf.body_text(
    'Current step time is ~2ms. With self-collision: ~3ms. Still achieves >300 FPS headless, >100 FPS with viewer. '
    'Optimization: run self-collision only on iterations 0 and 2 (every other), halving cost to ~0.4ms.'
)

# 5. Files
pdf.section_title('5. Files to Modify')

pdf.table_row(['File', 'Changes'], bold=True, fill=True)
pdf.table_row(['solver/shape_matching_gpu.h', 'Hash grid buffers + config fields'])
pdf.table_row(['solver/shape_matching_gpu.cu', '3 new kernels + build_hash_grid() + step() + init()/cleanup()'])
pdf.table_row(['main_gpu_viewer.cpp', 'Set self_collision config parameters'])
pdf.table_row(['CMakeLists.txt', 'CUB include path (CUDA 11.8 ships CUB, usually not needed)'])

# 6. Verification
pdf.section_title('6. Verification')

pdf.numbered(1, 'Basic test: cube32 free fall (disable fixed particles), verify no self-intersection')
pdf.numbered(2, 'Debulking test: after removing large sections, verify remaining structure maintains no self-intersection')
pdf.numbered(3, 'Performance test: pf_sim_gpu headless, compare FPS with/without self-collision')
pdf.numbered(4, 'Correctness: in rest state, self-collision should NOT trigger (2*radius < min_spacing)')

# 7. Key Design Decisions
pdf.section_title('7. Key Design Decisions')

pdf.subsection_title('Why accumulate into d_deltas?')
pdf.body_text(
    'The existing Jacobi iteration averages multiple constraint contributions via delta_counts. '
    'Self-collision is just another constraint. Accumulating into the same buffer lets apply_kernel '
    'naturally blend shape matching (keep shape) with self-collision (prevent overlap). '
    'Direct modification of predicted_pos would fight with shape matching corrections.'
)

pdf.subsection_title('Why rebuild grid every iteration?')
pdf.body_text(
    'Positions change after each apply_kernel, so the grid from iteration 0 would be stale by iteration 3. '
    'Grid build is cheap (~0.1ms). As optimization, build only on iterations 0 and 2.'
)

pdf.subsection_title('Why counting sort (CUB) over atomic insertion?')
pdf.body_text(
    'Counting sort + CUB radix sort produces a perfectly compact, cache-friendly structure with zero hash collisions. '
    'CUB radix sort at 20k elements takes ~0.05ms. Atomic insertion requires collision resolution and has worse worst-case.'
)

pdf.subsection_title('Handling removed particles')
pdf.body_text(
    'compute_cell_hash_kernel assigns hash = TABLE_SIZE (sentinel) to removed particles. '
    'CUB sorts these to the end, so they never appear in any valid cell range.'
)

# Output
output_path = '/home/yunxin/pf-shape-matching/GPU_Self_Collision_Plan.pdf'
pdf.output(output_path)
print(f'PDF generated: {output_path}')
