#pragma once

#include <cuda_runtime.h>
#include <vector>

// GPU 端数据结构
struct ParticlesGPU {
    float3* pos;
    float3* vel;
    float3* predicted_pos;
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
    float3 sharp_edge_a;   // endpoint A in tool local frame (only read if use_sharp_edge_damage=1)
    float3 sharp_edge_b;   // endpoint B

    // Max damage growth per substep — controls "deform first, then cut" timing.
    // Damage update formula: damage += min(target - damage, damage_growth_rate)
    //   1.0 → instant snap to target (original max-update behavior, byte-identical debulk)
    //   0.02 → ~25 substeps (≈50ms sim) to reach break threshold from full pressure
    //          → user sees 0.5s of elastic deformation before tissue yields
    // Lower = more elastic feel. Physical analogy: tissue tensile strength integral.
    float damage_growth_rate = 1.0f;
};

// SDF data (used for both rigid body collision and tool collision)
struct SDFData {
    float* d_sdf;           // [res_x * res_y * res_z] SDF grid on GPU
    int res_x, res_y, res_z;
    float3 origin;          // bounding box min corner (local frame)
    float3 cell_size;       // per-axis cell size
    int active;
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
    float* d_knife_rot = nullptr;  // [9] rotation matrix on GPU
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

    // GPU cutting kernels
    void gpu_update_damage();
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
