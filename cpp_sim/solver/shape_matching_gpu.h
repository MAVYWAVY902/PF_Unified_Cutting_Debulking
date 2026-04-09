#pragma once

#include <cuda_runtime.h>
#include <vector>

// GPU 端数据结构
struct ParticlesGPU {
    float3* pos;
    float3* vel;
    float3* predicted_pos;
    float3* old_pos;       // 用于速度更新
    float* damage;         // per-particle max edge damage
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
    float alpha_min = 10.0f;
    float aspiration_threshold = 0.3f;

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

    // 初始化 SDF (通用: bone or tool)
    void init_sdf(SDFData& sdf, const std::vector<float>& sdf_data,
                  int res_x, int res_y, int res_z,
                  float3 origin, float3 cell_size, const char* label);

    // 清理
    void cleanup();

private:
    float3* d_deltas;
    int* d_delta_counts;

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
