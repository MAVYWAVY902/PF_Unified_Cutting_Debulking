#!/usr/bin/env python3
"""Generate Mesh-Based Self-Collision Plan PDF"""
from fpdf import FPDF

class PlanPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'P(F) Shape Matching - Mesh-Based Self-Collision Plan', align='R')
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
        self.set_font('Courier', '', 8)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        w = self.w - self.l_margin - self.r_margin
        lines = code.strip().split('\n')
        h_per_line = 4.2
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
        self.cell(5, 5.5, '-')
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
        if len(cells) == 3:
            col_widths = [50, 30, 30]
        elif len(cells) == 2:
            col_widths = [65, 110]
        else:
            col_widths = [45] * len(cells)
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
pdf.cell(0, 15, 'Mesh-Based Self-Collision', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, 'Vertex-Face Proximity on GPU', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font('Helvetica', 'I', 10)
pdf.set_text_color(120, 120, 120)
pdf.cell(0, 8, 'P(F) Shape Matching PBD  |  Replaces Particle-Based Approach', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)

# 1. Problem
pdf.section_title('1. Why Particle-Based Self-Collision Failed')
pdf.body_text(
    'The particle-based approach treats each vertex as a sphere of radius 0.012. '
    'Collision triggers when two sphere centers are within 2r = 0.024. However, surface triangles '
    'can interpenetrate even when all vertices are separated by more than 0.024. '
    'Two triangular faces can pass through each other while their vertices remain far apart.'
)

pdf.subsection_title('Particle vs Mesh Collision')
pdf.table_row(['Aspect', 'Particle-Based (old)'], bold=True, fill=True)
pdf.table_row(['Detects', 'Vertex-vertex overlap only'])
pdf.table_row(['Misses', 'Face-face interpenetration'])
pdf.table_row(['Suitable for', 'Granular/SPH, not continuous mesh'])
pdf.ln(2)
pdf.table_row(['Aspect', 'Mesh-Based (new)'], bold=True, fill=True)
pdf.table_row(['Detects', 'Vertex passing through face'])
pdf.table_row(['Handles', 'Surface self-intersection'])
pdf.table_row(['Suitable for', 'Deformable tet mesh (our case)'])

# 2. Approach
pdf.section_title('2. Approach: Vertex-Face Proximity + Spatial Hash Grid')

pdf.subsection_title('Core Algorithm')
pdf.numbered(1, 'Hash surface face centroids into spatial grid (reuse CUB radix sort pattern)')
pdf.numbered(2, 'For each surface vertex, query 3x3x3 neighboring cells to find candidate faces')
pdf.numbered(3, 'Skip topologically adjacent faces (vertex is part of face, or 1-ring neighbor)')
pdf.numbered(4, 'Compute point-triangle distance; if < thickness, apply PBD correction')
pdf.numbered(5, 'Corrections accumulated into d_deltas via atomicAdd (same as shape matching)')

pdf.subsection_title('Modified Jacobi Loop')
pdf.code_block(
"""for iter in [0, num_iterations):
    clear_deltas_kernel
    cluster_solve_kernel               // shape matching constraints
    if self_collision_enabled:
        build_face_hash_grid()         // hash face centroids (CUB sort)
        mesh_self_collision_kernel     // vertex vs non-adjacent faces
    apply_kernel                       // merge shape matching + collision
    collision_kernel                   // floor + knife hard constraints""")

# 3. GPU Data Structures
pdf.section_title('3. New GPU Data Structures')

pdf.subsection_title('Surface Mesh (shape_matching_gpu.h)')
pdf.code_block(
"""struct SurfaceMeshGPU {
    int* face_a;              // [max_faces] triangle vertex A
    int* face_b;              // [max_faces] triangle vertex B
    int* face_c;              // [max_faces] triangle vertex C
    int* vertex_is_surface;   // [N] 1 if surface vertex, 0 otherwise
    int num_faces;            // current face count
    int max_faces;            // allocated capacity
};""")

pdf.subsection_title('Vertex Adjacency CSR (for topological exclusion)')
pdf.code_block(
"""int* d_neighbor_starts;   // [N+1] CSR row pointers
int* d_neighbor_ids;      // [total_entries] sorted neighbor IDs
// Built once from tet_mesh.vertex_neighbors (topology fixed)""")

pdf.subsection_title('Face Hash Grid (reuses CUB sort pattern)')
pdf.code_block(
"""unsigned int* d_face_hash, d_face_index;         // unsorted
unsigned int* d_face_hash_sorted, d_face_idx_sorted; // sorted
unsigned int* d_face_cell_start, d_face_cell_end;    // [TABLE_SIZE]
void* d_face_cub_temp;""")

# 4. Key Kernels
pdf.section_title('4. CUDA Kernels')

pdf.subsection_title('4.1 compute_face_hash_kernel')
pdf.body_text('Per face: compute centroid, hash into grid cell.')
pdf.code_block(
"""__global__ void compute_face_hash_kernel(
    float3* positions, int* fa, int* fb, int* fc,
    unsigned int* hash, unsigned int* index,
    float cell_size, int table_size, int num_faces) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;
    float3 va=positions[fa[f]], vb=positions[fb[f]], vc=positions[fc[f]];
    float cx=(va.x+vb.x+vc.x)/3, cy=(va.y+vb.y+vc.y)/3, cz=(va.z+vb.z+vc.z)/3;
    int ix=floor(cx/cell_size), iy=floor(cy/cell_size), iz=floor(cz/cell_size);
    hash[f] = (ix*73856093 ^ iy*19349663 ^ iz*83492791) & (table_size-1);
    index[f] = f;
}""")

pdf.subsection_title('4.2 point_triangle_distance (device function)')
pdf.body_text(
    'Standard closest-point-on-triangle algorithm. Projects point onto triangle plane, '
    'computes barycentric coordinates. If projection is inside triangle, distance = point-to-plane. '
    'Otherwise, clamps to nearest edge or vertex. Returns distance + barycentric coords for '
    'correction distribution.'
)
pdf.code_block(
"""__device__ float point_triangle_distance(
    float3 p, float3 a, float3 b, float3 c,
    float3* closest, float* u, float* v, float* w) {
    // 1. Compute edges and project
    float3 ab=b-a, ac=c-a, ap=p-a;
    float d1=dot(ab,ap), d2=dot(ac,ap);
    float d3=dot(ab,ab), d4=dot(ab,ac), d5=dot(ac,ac);
    // 2. Barycentric coords via Cramer's rule
    // 3. Clamp to triangle edges if outside
    // 4. Return distance to closest point
    // (~40 lines, well-known algorithm)
}""")

pdf.subsection_title('4.3 is_neighbor (device function)')
pdf.body_text('Binary search in sorted CSR neighbor list for topological exclusion.')
pdf.code_block(
"""__device__ bool is_neighbor(int v, int u,
    int* starts, int* ids) {
    int lo = starts[v], hi = starts[v+1];
    while (lo < hi) {  // binary search
        int mid = (lo + hi) / 2;
        if (ids[mid] == u) return true;
        if (ids[mid] < u) lo = mid+1; else hi = mid;
    }
    return false;
}""")

pdf.subsection_title('4.4 mesh_self_collision_kernel')
pdf.body_text(
    'Per surface vertex: query 3x3x3 neighbor cells, iterate candidate faces, '
    'skip adjacent faces, compute point-triangle distance, apply PBD correction.'
)
pdf.code_block(
"""__global__ void mesh_self_collision_kernel(
    float3* pred_pos, int* removed, int* fixed, int* is_surface,
    int* face_a, int* face_b, int* face_c,
    int* nbr_starts, int* nbr_ids,
    uint* face_sorted_idx, uint* face_cell_start, uint* face_cell_end,
    float3* deltas, int* delta_counts,
    float thickness, float stiffness, float cell_size,
    int table_size, int N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N || removed[v] || !is_surface[v]) return;

    float3 p = pred_pos[v];
    // Query 3x3x3 neighbor cells
    for (dx,dy,dz in [-1,0,1]^3):
        h = hash(cell + offset)
        for each face f in cell:
            a,b,c = face vertices
            if (v==a || v==b || v==c) continue;  // skip own face
            if (is_neighbor(v,a) || is_neighbor(v,b) || is_neighbor(v,c))
                continue;  // skip 1-ring adjacent
            dist = point_triangle_distance(p, pos[a], pos[b], pos[c])
            if (dist < thickness):
                correction along face normal
                atomicAdd(deltas[v],  +0.25 * corr)  // vertex
                atomicAdd(deltas[a],  -0.75*u * corr) // face verts
                atomicAdd(deltas[b],  -0.75*v * corr)
                atomicAdd(deltas[c],  -0.75*w * corr)
}""")

# 5. Correction Distribution
pdf.section_title('5. PBD Correction Distribution')
pdf.body_text(
    'When vertex V is within thickness of face (A,B,C), 4 particles are involved. '
    'Correction is split: 1/4 to V (pushed away), 3/4 to face vertices (pushed back), '
    'weighted by barycentric coordinates of the closest point on the face.'
)
pdf.code_block(
"""// Vertex V: pushed along +normal
deltas[V] += 0.25 * (thickness - dist) * stiffness * normal

// Face vertices: pushed along -normal, weighted by barycentric
deltas[A] -= 0.75 * bary_u * (thickness - dist) * stiffness * normal
deltas[B] -= 0.75 * bary_v * (thickness - dist) * stiffness * normal
deltas[C] -= 0.75 * bary_w * (thickness - dist) * stiffness * normal""")

# 6. Surface Sync
pdf.section_title('6. Dynamic Surface Synchronization')
pdf.body_text(
    'Surface faces change when particles are removed (debulking). The CPU already calls '
    'rebuild_surface_faces() after removal. After this, call gpu_solver.upload_surface_mesh() '
    'to re-upload face_a/b/c and vertex_is_surface to GPU. Vertex neighbors (tet topology) '
    'never change, uploaded once at init.'
)
pdf.code_block(
"""// In main_gpu_viewer.cpp, after rebuild_surface_faces():
gpu_solver.upload_surface_mesh(
    tet_mesh.surface_faces,    // new face list
    N);                        // rebuild vertex_is_surface""")

# 7. Topological Exclusion
pdf.section_title('7. Topological Exclusion')
pdf.body_text(
    'A surface vertex must NOT collide with faces it is part of or topologically adjacent to. '
    'Without exclusion, every vertex would collide with its own faces, causing explosion.'
)
pdf.numbered(1, 'Skip if vertex V is a vertex of face (A,B,C): V==A || V==B || V==C')
pdf.numbered(2, 'Skip if V is a 1-ring mesh neighbor of any face vertex: '
    'is_neighbor(V,A) || is_neighbor(V,B) || is_neighbor(V,C)')
pdf.body_text(
    'The is_neighbor() check uses binary search in the sorted CSR adjacency list. '
    'vertex_neighbors is built from tet connectivity (each tet creates 6 edges), '
    'giving ~12-14 neighbors per vertex. Binary search cost: ~4 comparisons.'
)

# 8. Performance
pdf.section_title('8. Performance Budget')

pdf.table_row(['Operation', 'Per Iteration', '4 Iter/Step'], bold=True, fill=True)
pdf.table_row(['face hash + CUB sort', '0.07 ms', '0.28 ms'])
pdf.table_row(['mesh_collision (5k verts)', '0.3-0.5 ms', '1.2-2.0 ms'])
pdf.table_row(['Total', '~0.5 ms', '~1.5-2.0 ms'], bold=True, fill=True)
pdf.ln(3)
pdf.body_text(
    'Current step: ~2ms. With mesh collision: ~4ms. Viewer: ~80 FPS (acceptable). '
    'Optimization: run only on iterations 0 and 2, halving cost to ~1ms/step.'
)

# 9. Config
pdf.section_title('9. Configuration Parameters')
pdf.code_block(
"""gpu_solver.config.self_collision_thickness = 0.01f;  // collision thickness
gpu_solver.config.self_collision_stiffness = 0.5f;    // PBD stiffness
gpu_solver.config.self_collision_enabled = 1;          // toggle""")
pdf.body_text(
    'thickness = 0.01: less than minimum non-adjacent face distance (~0.06 for cube32), '
    'so rest state does not trigger collision. Only activates under deformation.'
)

# 10. Files
pdf.section_title('10. Files to Modify')
pdf.table_row(['File', 'Changes'], bold=True, fill=True)
pdf.table_row(['solver/shape_matching_gpu.h', 'SurfaceMeshGPU struct + neighbor CSR + methods'])
pdf.table_row(['solver/shape_matching_gpu.cu', '4 kernels + device funcs + hash grid + step()'])
pdf.table_row(['main_gpu_viewer.cpp', 'init_surface_mesh() + upload after rebuild'])
pdf.table_row(['core/tet_mesh.h', 'Optional: flatten_neighbors_csr() helper'])

# 11. Implementation Sequence
pdf.section_title('11. Implementation Sequence')
pdf.numbered(1, 'Add SurfaceMeshGPU struct and member variables to shape_matching_gpu.h')
pdf.numbered(2, 'Implement init_surface_mesh(): allocate GPU buffers, upload initial faces + neighbors CSR')
pdf.numbered(3, 'Implement upload_surface_mesh(): re-upload faces after removal')
pdf.numbered(4, 'Implement compute_face_hash_kernel + build_face_hash_grid() (copy CUB pattern)')
pdf.numbered(5, 'Implement point_triangle_distance device function (~40 lines)')
pdf.numbered(6, 'Implement is_neighbor device function (binary search, ~10 lines)')
pdf.numbered(7, 'Implement mesh_self_collision_kernel (~80 lines)')
pdf.numbered(8, 'Replace old particle self-collision call in step() Jacobi loop')
pdf.numbered(9, 'Hook up upload_surface_mesh() in main_gpu_viewer.cpp after rebuild_surface_faces()')
pdf.numbered(10, 'Add cleanup for new GPU allocations')

# 12. Verification
pdf.section_title('12. Verification')
pdf.numbered(1, 'Rest state: collision does NOT trigger (thickness < min non-adjacent face distance)')
pdf.numbered(2, 'Knife push: surface does not self-intersect when deformed')
pdf.numbered(3, 'Debulking: exposed interior surface does not intersect')
pdf.numbered(4, 'Performance: step time < 5ms with self-collision enabled')
pdf.numbered(5, 'Stability: no oscillation between shape matching and collision corrections')

# 13. Future
pdf.section_title('13. Future Enhancements')
pdf.bullet('Edge-edge collision: detect when two non-adjacent surface edges cross')
pdf.bullet('Continuous collision detection (CCD): prevent tunneling at high velocities')
pdf.bullet('2-ring topological exclusion: for very coarse meshes')

output_path = '/home/yunxin/pf-shape-matching/Mesh_Self_Collision_Plan.pdf'
pdf.output(output_path)
print(f'PDF generated: {output_path}')
