#pragma once

#include "cross_rec.h"
#include "tool_sheet.h"
#include <cuda_runtime.h>
#include <vector>

// GPU 端数据结构
struct ParticlesGPU {
    float3* pos;
    float3* vel;
    float3* predicted_pos;
    float3* pre_apply_pos; // contact_ccd: positions before each iteration's shape-matching apply
    float3* old_pos;       // 用于速度更新
    float* damage;         // per-particle MEAN edge damage (monotonic, only increases)
    float* damage_sum;     // [N] accumulator for mean aggregation — zeroed each cutting step
    float* damage_count;   // [N] incident-edge count (float for atomicAdd)
    int* removed;
    int* fixed;
    int N;
};

struct ClustersGPU {
    int* cluster_starts;   // CSR 行指针
    int* cluster_sizes;
    int* particle_ids;     // CSR 列索引
    float3* rest_offsets;
    float3* rest_cm;
    int num_clusters;
    int total_entries;     // sum of all cluster sizes
};

struct SimConfigGPU {
    float dt;
    float3 gravity;
    int num_iterations;
    float stiffness;
    float damping;
    float floor_y;
    float3 knife_pos;
    float knife_rot[9];  // 3x3 rotation matrix (row-major): local→world
    float knife_radius;
    int knife_active;
    // Self-collision
    float self_collision_radius;
    float self_collision_stiffness;
    int self_collision_enabled;
    // Rigid body friction
    float friction_s;
    float friction_k;
    // Tool type: 0 = sphere (default), 1 = mesh SDF
    int tool_type;
    // Cutting mode: 1 = enable aspiration (debulk, current behavior),
    //               0 = cut_only (edges break + clusters split, NO particle removal)
    // Default 1 → safe for headless tests (main_gpu.cpp / verify_gpu.cpp) that don't
    // explicitly set this field; gpu_aspirate is gated on knife_active anyway.
    int enable_aspiration = 1;
    // cluster_solve_kernel: damage-weighted shape matching strength α.
    // Particle i contributes weight w_i = max(0.05, 1 - α · damage_i) to cm/qcm/A_pq.
    // Müller 2005 effective mass instantiated with P(F): heavily-damaged particles
    // are 'soft mass', they still participate but don't dominate cluster shape.
    //   debulk   α=0.0 → w=1 always → behavior identical to original (no weighting)
    //   cut_only α=1.0 → P(F)-weighted shape matching, intact side dominates cluster
    // Default 0.0 → byte-identical to original kernel for any binary that doesn't
    // explicitly opt into cut_only (preserves RL training reproducibility).
    float damage_shape_weight = 0.0f;

    // Sharp-edge damage mode (replaces SDF damage when enabled).
    // Real physics: only the sharp edge of a blade cuts; the blade body merely
    // pushes (collision). SDF damage incorrectly treats the entire wedge volume
    // as cutter, causing horizontal slides to wrongly damage tissue.
    // When use_sharp_edge_damage=1, damage is computed from distance to the
    // sharp_edge_a → sharp_edge_b line segment (in tool LOCAL frame), not SDF.
    // Collision (tool_sdf_collision_kernel) remains full-SDF for non-penetration.
    int use_sharp_edge_damage = 0;

    // Step 2 tool switches (defaults = legacy behaviour; see solver/tool_query.cuh):
    //   sdf_normal_mode    0 legacy nearest-neighbour index-space normal | 1 trilinear world gradient
    //   sdf_pf_edge_radius 0 legacy P(F) sdist = phi | 1 sdist = phi - r_e (sphere intruder model)
    //   aspiration_reach   0 legacy contains test | >0 aspirate when phi <= reach (all tool types)
    int sdf_normal_mode = 0;
    int sdf_pf_edge_radius = 0;
    float aspiration_reach = 0.0f;
    // Step 3c: 1 = intact tissue edges also collide with the tool body (thin tools between particles)
    int edge_contact = 0;
    // R3: 1 = intact edges separated by the tool body (straddle) accumulate P(F) damage
    int straddle_fracture = 0;
    // A1: 1 = a particle inside a mesh tool is returned to the side it entered from (not the nearest face)
    int entry_side_contact = 0;
    // F1: 1 = mesh-tool contact repeats the projection until the particle is out (<= 8 pushes)
    int contact_iterate = 0;
    // F3: 1 = self-collision pairs with a mesh tool between them are skipped
    int selfcol_tool_block = 0;
    // 1 = CUT_CURVE cuts an edge when the sharp feature passes BETWEEN its particles (swept crossing);
    // 0 = legacy proximity (edge midpoint within r_e of the feature)
    int cut_curve_crossing = 0;
    // CCD tool contact: a particle whose path since a known-outside reference crossed the tool is put
    // back on the entry face (tool_query.cuh tool_project_out_ccd)
    int contact_ccd = 0;
    // R3 loads an edge only if the tool cuts its element (minimum ball) in two: no detour a -> c -> b
    // around the tool through K points of the ball sphere (straddle_detour, straddle_detour_k)
    int straddle_detour = 0;
    int straddle_detour_k = 32;
    // R3 rule: 0 = segment (the segment passes through the tool, outward normals opposed); 1 = sheet
    // (the tool's 2D intruders cut the edge's element in two, exact; tool_query.cuh tool_sheet_separates)
    int straddle_sheet = 0;
    // 1 = tissue-edge vs tool-edge CCD: a convex tool edge that sweeps across an intact tissue edge
    // pushes it (unless it is a sharp edge that leads: the cut channel decides those). Without it a
    // blunt part thinner than the particle spacing passes between particles (tool_query.cuh)
    int edge_sweep_contact = 0;
    // 1 (with crossing) = a crossing cuts only if the feature LEADS: the tool's motion relative to the
    // tissue lies in the feature's normal cone (cut_curve: leading)
    int cut_curve_leading = 0;
    float3 sharp_edge_a;   // endpoint A in tool local frame (only read if use_sharp_edge_damage=1)
    float3 sharp_edge_b;   // endpoint B

    // Max damage growth per substep — controls "deform first, then cut" timing.
    // Damage update formula: damage += min(target - damage, damage_growth_rate)
    //   1.0 → instant snap to target (original max-update behavior, byte-identical debulk)
    //   0.02 → ~25 substeps (≈50ms sim) to reach break threshold from full pressure
    //          → user sees 0.5s of elastic deformation before tissue yields
    // Lower = more elastic feel. Physical analogy: tissue tensile strength integral.
    float damage_growth_rate = 1.0f;
    // XPBD shape matching (solver: xpbd). Each cluster member is a zero-length link to its goal with
    // stiffness k per particle mass m (xpbd_stiffness = k/m, 1/s^2); the multipliers accumulate over
    // the iterations of a substep, so the converged deformation depends on k/m, not on dt or on
    // num_iterations. 0 = legacy PBD (stiffness applied per iteration).
    int solver_xpbd = 0;
    float xpbd_stiffness = 0.0f;
};

// SDF data (used for both rigid body collision and tool collision)
struct SDFData {
    float* d_sdf;           // [res_x * res_y * res_z] SDF grid on GPU
    int res_x, res_y, res_z;
    float3 origin;          // bounding box min corner (local frame)
    float3 cell_size;       // per-axis cell size
    int active;
    // contact_ccd: the tool's triangles in the same local frame (tri[4k..4k+2] = vertices, tri[4k+3] =
    // outward unit normal), for exact path tests; n_tri = 0 → SDF tracing only
    float3* d_tri = nullptr;
    int n_tri = 0;
    // edge_sweep_contact: convex tool edges (4 float3 per edge: P, Q, n1, n2) + sharp flags
    float3* d_cedge = nullptr;
    int* d_cedge_sharp = nullptr;
    int n_cedge = 0;
    // straddle_rule sheet: the tool's sheets and their cross-section segments (local frame)
    ToolSheetGPU* d_sheet = nullptr;
    float4* d_sheet_seg = nullptr;
    int n_sheet = 0;
};

// 边数据（cutting 用）
struct EdgesGPU {
    int* particle_a;      // [num_edges]
    int* particle_b;      // [num_edges]
    float* pf_rest;       // [num_edges]
    float* damage;        // [num_edges]
    int* broken;          // [num_edges]
    int num_edges;
    // Per-edge α field (optional, used when config.per_edge_alpha=true).
    // nullptr → fall back to scalar alpha_min in legacy kernel signature.
    float* alpha_min_per_edge = nullptr;  // [num_edges]
};

// GPU solver 接口（在 .cu 中实现）
class ShapeMatchingGPUSolver {
public:
    ParticlesGPU d_particles;
    ClustersGPU d_clusters;
    EdgesGPU d_edges;
    SDFData d_rigid_sdf = {};   // bone SDF (static)
    SDFData d_tool_sdf = {};    // tool SDF (local frame, moves with knife_pos)
    SimConfigGPU config;

    int step_count = 0;
    int cutting_update_interval = 200;
    float alpha_min = 3.0f;   // P(F) sigmoid steepness (was 10 — saturated damage on any contact)
    float aspiration_threshold = 0.3f;

    // Per-edge α toggle (mirrors SimConfigYAML.per_edge_alpha).
    // When true AND d_edges.alpha_min_per_edge != nullptr → dispatch uses per-edge α.
    // When false → legacy single-float alpha_min path (byte-identical).
    bool per_edge_alpha = false;

    // Damage model: 0 = "sharpness" (legacy, byte-identical), 1 = "toughness"
    // Toughness: damage_growth_rate scaled by 1/α per edge → low α = fast cut.
    int damage_model_mode = 0;

    // Per-particle τ (aspiration threshold) field.
    // When per_particle_tau=true AND d_tau_per_particle allocated, aspirate
    // uses per-particle threshold; else global aspiration_threshold.
    bool per_particle_tau = false;
    float* d_tau_per_particle = nullptr;   // [N] on GPU
    std::vector<float> tau_per_particle_cpu;  // CPU mirror for viewer

    void upload_per_particle_tau(const std::vector<float>& tau_per_particle);

    // Per-particle stiffness field — elasticity heterogeneity.
    // When per_particle_stiffness=true AND d_stiffness_per_particle allocated,
    // apply_kernel uses per-particle stiffness; else global config.stiffness.
    bool per_particle_stiffness = false;
    float* d_stiffness_per_particle = nullptr;   // [N] on GPU
    std::vector<float> stiffness_per_particle_cpu;  // CPU mirror for viewer

    void upload_per_particle_stiffness(const std::vector<float>& stiff_per_particle);

    // 初始化：分配 GPU 内存，上传数据
    void init(const std::vector<float3>& positions,
              const std::vector<float3>& velocities,
              const std::vector<int>& removed,
              const std::vector<int>& fixed,
              const std::vector<int>& cluster_starts,
              const std::vector<int>& cluster_sizes,
              const std::vector<int>& particle_ids,
              const std::vector<float3>& rest_offsets,
              const std::vector<float3>& rest_cm);

    // 初始化边数据（cutting 用）
    void init_edges(const std::vector<int>& edge_a,
                    const std::vector<int>& edge_b,
                    const std::vector<float>& edge_pf_rest);

    // 一个 timestep（含 cutting）
    void step();

    // 下载位置回 CPU（给渲染用）
    void download_positions(std::vector<float3>& positions);
    // Read-only diagnostic: predicted_pos as the cutting kernels saw it this step
    // (aspirate/damage run on predicted_pos after the Jacobi loop). No physics effect.
    void download_predicted_positions(std::vector<float3>& positions);
    void download_removed(std::vector<int>& removed);
    void download_damage(std::vector<float>& damage);

    // 下载边数据（BFS 分裂用）
    void download_edge_broken(std::vector<int>& broken);
    void download_edge_damage(std::vector<float>& damage);

    // 上传 removed（CPU 分裂后同步回 GPU）
    void upload_removed(const std::vector<int>& removed);

    // 重新上传簇数据（分裂后）
    void reupload_clusters(
        const std::vector<int>& cluster_starts,
        const std::vector<int>& cluster_sizes,
        const std::vector<int>& particle_ids,
        const std::vector<float3>& rest_offsets,
        const std::vector<float3>& rest_cm);

    // 重新上传边数据（剪枝后）
    void reupload_edges(
        const std::vector<int>& edge_a,
        const std::vector<int>& edge_b,
        const std::vector<float>& pf_rest,
        const std::vector<float>& damage,
        const std::vector<int>& broken);

    // Upload per-edge α field (must be called BEFORE step() if config.per_edge_alpha=true).
    // Length must equal d_edges.num_edges. Frees+reallocates if size changed.
    void upload_per_edge_alpha(const std::vector<float>& alpha_per_edge);

    // CPU copy of per-edge α (for viewer to look up per-particle averaged α).
    // Populated by upload_per_edge_alpha.
    std::vector<float> alpha_per_edge_cpu;

    // R1 — per-cluster local connectivity: shape matching is solved per sub-cluster (members still
    // connected through intact cutting edges INSIDE the cluster). Call after init() and
    // init_edges() with the same cluster CSR that was uploaded; the cluster CSR must not change
    // afterwards (cpu_bfs_split skips its global re-clustering when local_split is on).
    void setup_local_split(const std::vector<int>& cluster_starts, const std::vector<int>& cluster_sizes,
                           const std::vector<int>& particle_ids,
                           const std::vector<int>& edge_a, const std::vector<int>& edge_b);
    bool local_split = false;
    // diagnostic: sub-cluster label per cluster entry (empty when local_split is off)
    void download_labels(std::vector<int>& labels);
    void update_cluster_labels_public() {
        if (!local_split) return;
        int one = 1; cudaMemcpy(d_topo_dirty, &one, sizeof(int), cudaMemcpyHostToDevice);
        update_cluster_labels();
    }

    // Debug trace (diagnostics only): positions of chosen particles after each stage of step(),
    // with the tool pose of that substep. Enabled by set_trace(); off = no effect, no cost.
    struct TraceRec { int step; int stage; int iter; float3 kpos; float krot[9]; std::vector<float3> pos; };
    std::vector<int> trace_ids;          // empty = off
    bool trace_on = false;               // toggled per frame by the caller
    std::vector<TraceRec> trace_log;
    int* d_trace_ids = nullptr; float3* d_trace_buf = nullptr;
    void set_trace(const std::vector<int>& ids);
    void trace_point(int stage, int iter);   // stage: 0 predict, 1 first collision, 2 shape match,
                                             // 3 self collision, 4 tool collision, 5 bone, 6 end

    // A1: tool pose of the previous substep (set at the end of step())
    float3 knife_pos_prev = {0, 0, 0};
    float knife_rot_prev[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool knife_prev_valid = false;
    struct ToolDesc prev_tool_desc() const;
    int entry_side_active() const;
    int tool_contact_mode() const;

    // Sharp features of the CUT_CURVE channel (used when config.use_sharp_edge_damage = 1).
    // diagnostics: crossing events (tool_bench --cross-log)
    bool cross_log_on = false; int cross_log_cap = 1 << 16;
    CrossRec* d_cross_log = nullptr; int* d_cross_log_n = nullptr;
    std::vector<CrossRec> cross_log; std::vector<int> cross_log_step;
    CrossRec* d_strad_log = nullptr; int* d_strad_log_n = nullptr;          // R3 breaks (seg = -1, s = t, sdist = phi)
    std::vector<CrossRec> strad_log; std::vector<int> strad_log_step;
    // nrms (optional): two outward face normals per feature, the normal cone used by cut_curve: leading
    // contact_ccd: exact tool triangles (local frame, 4 float3 per triangle: v0 v1 v2 outward normal)
    void set_tool_triangles(const std::vector<float3>& tri4);
    void edge_sweep_contact(const float3* ref, const struct ToolDesc& ref_t);
    void set_tool_contact_edges(const std::vector<float3>& e4, const std::vector<int>& sharp);
    void set_tool_sheets(const std::vector<ToolSheetGPU>& sheets, const std::vector<float4>& seg);
    void set_cut_segments(const std::vector<float3>& segs, const std::vector<float3>& nrms = {}) {
        h_cut_segments = segs; h_cut_normals = nrms; cut_segments_dirty = true; }
    bool has_cut_normals() const { return !h_cut_normals.empty() && h_cut_normals.size() == h_cut_segments.size(); }
    const std::vector<float3>& cut_segments() const { return h_cut_segments; }

    // 初始化 SDF (通用: bone or tool)
    void init_sdf(SDFData& sdf, const std::vector<float>& sdf_data,
                  int res_x, int res_y, int res_z,
                  float3 origin, float3 cell_size, const char* label);

    // 清理
    void cleanup();

private:
    float3* d_deltas;
    int* d_delta_counts;
    // FleX-style constraint groups (Macklin 2014, Sec 4): self-collision has its own
    // delta buffer + its own apply pass, so it's NOT diluted by the ~60 shape-match
    // contributions per particle on dense tet meshes. cluster_solve → apply → clear_sc
    // → self_collision → apply_sc → tool/floor/bone collision.
    float3* d_deltas_sc = nullptr;
    int* d_delta_counts_sc = nullptr;
    // Step 3c edge contact: its own delta buffer (same constraint-group pattern)
    float3* d_deltas_ec = nullptr;
    int* d_delta_counts_ec = nullptr;
    // R1 sub-cluster data: internal edges per cluster (CSR) with member-local endpoints, labels
    int* d_cl_edge_start = nullptr;   // [C+1]
    int* d_cl_edge_eid = nullptr;     // [M] cutting-edge index
    int2* d_cl_edge_uv = nullptr;     // [M] local member indices of the two endpoints
    int* d_labels = nullptr;          // [total_entries]
    int* d_topo_dirty = nullptr;      // set by a new break / removal → labels recomputed next step
    int* d_cluster_multi = nullptr;   // [C] 1 = cluster split into >= 2 sub-clusters
    int max_cluster_size = 0;
    void update_cluster_labels();
    // XPBD (config.solver_xpbd): per cluster entry [total_entries], goal of this iteration (NaN = no
    // goal) and the accumulated multiplier of the substep (mass-normalised, metres)
    float3* d_xpbd_goal = nullptr;
    float3* d_xpbd_lambda = nullptr;
    int xpbd_entries = 0;
    void ensure_xpbd_buffers();

    // GPU cutting kernels
    void gpu_update_damage();
    // CUT_CURVE channel features (tool local frame, 2 float3 per segment; a point feature has
    // equal endpoints). Host copy set by set_cut_segments, uploaded on the next damage update.
    std::vector<float3> h_cut_segments, h_cut_normals;
    float3* d_cut_segments = nullptr;
    float3* d_cut_normals = nullptr;
    size_t d_cut_segments_cap = 0;
    bool cut_segments_dirty = false;
    // CUT_CURVE crossing mode: particle positions + tool pose at the previous damage evaluation
    float3* d_cut_prev_pos = nullptr;
    float3 cut_prev_pos_tool = {0, 0, 0};
    float cut_prev_rot_tool[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool cut_prev_valid = false;
    void sync_cut_segments();
    void gpu_aspirate();
    void gpu_update_particle_damage();

    // Self-collision hash grid
    static const int HASH_TABLE_SIZE = 32768;  // 2^15
    unsigned int* d_cell_hash = nullptr;
    unsigned int* d_particle_index = nullptr;
    unsigned int* d_cell_hash_sorted = nullptr;
    unsigned int* d_particle_index_sorted = nullptr;
    unsigned int* d_cell_start = nullptr;
    unsigned int* d_cell_end = nullptr;
    void* d_cub_temp = nullptr;
    size_t cub_temp_bytes = 0;

    void build_hash_grid();
};
