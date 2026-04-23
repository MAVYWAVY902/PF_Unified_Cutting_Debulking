#include "shape_matching_gpu.h"
#include "svd3_cuda.h"
#include <cub/cub.cuh>
#include <cstdio>

// ============== CUDA Kernels ==============

__global__ void predict_kernel(
    float3* pos, float3* vel, float3* predicted_pos, float3* old_pos,
    int* removed, int* fixed,
    float3 gravity, float dt, float damping, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i] || fixed[i]) return;

    old_pos[i] = pos[i];
    vel[i].x += gravity.x * dt;
    vel[i].y += gravity.y * dt;
    vel[i].z += gravity.z * dt;
    vel[i].x *= damping;
    vel[i].y *= damping;
    vel[i].z *= damping;
    predicted_pos[i].x = pos[i].x + vel[i].x * dt;
    predicted_pos[i].y = pos[i].y + vel[i].y * dt;
    predicted_pos[i].z = pos[i].z + vel[i].z * dt;
}

__global__ void collision_kernel(
    float3* pred_pos, float3* old_pos, int* removed, int* fixed,
    float floor_y, float3 knife_pos, float knife_radius, int knife_active,
    float mu_s, float mu_k, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i] || fixed[i]) return;

    // 地板 — y-clamp + Coulomb 摩擦（法向量恒为 +Y）
    // 和 sdf_collision_kernel 的 friction 段同构；normal=(0,1,0) 简化 dot/tangent 计算。
    // 没有摩擦时粒子会"滑冰"—— 垂直能量正确吸收（vy clamp 到 0），但水平速度只靠全局
    // damping=0.95 衰减，碎片能滑几 cm 才停，对手术 demo 不现实。
    if (pred_pos[i].y < floor_y) {
        float push = floor_y - pred_pos[i].y;  // 多少被推回地面上（>0）
        pred_pos[i].y = floor_y;

        if (mu_s > 0.0f || mu_k > 0.0f) {
            // 本 substep 水平位移（切向，法向 y 分量不算）
            float dpx = pred_pos[i].x - old_pos[i].x;
            float dpz = pred_pos[i].z - old_pos[i].z;
            float len_t = sqrtf(dpx*dpx + dpz*dpz);

            if (len_t > 1e-10f) {
                if (len_t < mu_s * push) {
                    // 静摩擦：切向位移 < μ_s × 法向接触力 → 完全粘住，本步抹掉水平移动
                    pred_pos[i].x -= dpx;
                    pred_pos[i].z -= dpz;
                } else {
                    // 动摩擦：按 μ_k × 法向推量 的比例削弱水平位移
                    float reduce = fminf(mu_k * push / len_t, 1.0f);
                    pred_pos[i].x -= dpx * reduce;
                    pred_pos[i].z -= dpz * reduce;
                }
            }
        }
    }

    // 刀具（球形 SDF）
    if (knife_active) {
        float dx = pred_pos[i].x - knife_pos.x;
        float dy = pred_pos[i].y - knife_pos.y;
        float dz = pred_pos[i].z - knife_pos.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dist < knife_radius && dist > 1e-10f) {
            float inv = knife_radius / dist;
            pred_pos[i].x = knife_pos.x + dx * inv;
            pred_pos[i].y = knife_pos.y + dy * inv;
            pred_pos[i].z = knife_pos.z + dz * inv;
        }
    }
}

__global__ void cluster_solve_kernel(
    float3* positions, int* removed,
    int* cluster_starts, int* cluster_sizes,
    int* particle_ids, float3* rest_offsets,
    float3* deltas, int* delta_counts,
    int num_clusters)
{
    int cid = blockIdx.x;
    if (cid >= num_clusters) return;
    int tid = threadIdx.x;

    int start = cluster_starts[cid];
    int size  = cluster_sizes[cid];
    if (size < 2) return;

    // 1. Warp reduction 求质心（跳过 removed 粒子）
    float cx = 0, cy = 0, cz = 0;
    float tw = 0;
    for (int j = tid; j < size; j += 32) {
        int pid = particle_ids[start + j];
        if (removed[pid]) continue;
        cx += positions[pid].x;
        cy += positions[pid].y;
        cz += positions[pid].z;
        tw += 1.0f;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        cx += __shfl_down_sync(0xFFFFFFFF, cx, offset);
        cy += __shfl_down_sync(0xFFFFFFFF, cy, offset);
        cz += __shfl_down_sync(0xFFFFFFFF, cz, offset);
        tw += __shfl_down_sync(0xFFFFFFFF, tw, offset);
    }
    float total_w = __shfl_sync(0xFFFFFFFF, tw, 0);
    if (total_w < 1.5f) return;  // 簇中活跃粒子不足 2 个，跳过
    cx = __shfl_sync(0xFFFFFFFF, cx, 0) / total_w;
    cy = __shfl_sync(0xFFFFFFFF, cy, 0) / total_w;
    cz = __shfl_sync(0xFFFFFFFF, cz, 0) / total_w;

    // 2a. 活跃粒子的 rest offset 均值（移除粒子后不再零均值）
    float qcx = 0, qcy = 0, qcz = 0;
    for (int j = tid; j < size; j += 32) {
        int pid = particle_ids[start + j];
        if (removed[pid]) continue;
        qcx += rest_offsets[start + j].x;
        qcy += rest_offsets[start + j].y;
        qcz += rest_offsets[start + j].z;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        qcx += __shfl_down_sync(0xFFFFFFFF, qcx, offset);
        qcy += __shfl_down_sync(0xFFFFFFFF, qcy, offset);
        qcz += __shfl_down_sync(0xFFFFFFFF, qcz, offset);
    }
    qcx = __shfl_sync(0xFFFFFFFF, qcx, 0) / total_w;
    qcy = __shfl_sync(0xFFFFFFFF, qcy, 0) / total_w;
    qcz = __shfl_sync(0xFFFFFFFF, qcz, 0) / total_w;

    // 2b. 协方差矩阵 A_pq (3x3)（用修正后的 rest offsets）
    float a11=0,a12=0,a13=0,a21=0,a22=0,a23=0,a31=0,a32=0,a33=0;
    for (int j = tid; j < size; j += 32) {
        int pid = particle_ids[start + j];
        if (removed[pid]) continue;
        float px = positions[pid].x - cx;
        float py = positions[pid].y - cy;
        float pz = positions[pid].z - cz;
        float qx = rest_offsets[start + j].x - qcx;
        float qy = rest_offsets[start + j].y - qcy;
        float qz = rest_offsets[start + j].z - qcz;
        a11 += px*qx; a12 += px*qy; a13 += px*qz;
        a21 += py*qx; a22 += py*qy; a23 += py*qz;
        a31 += pz*qx; a32 += pz*qy; a33 += pz*qz;
    }
    // Warp reduction for all 9 elements
    for (int offset = 16; offset > 0; offset >>= 1) {
        a11 += __shfl_down_sync(0xFFFFFFFF, a11, offset);
        a12 += __shfl_down_sync(0xFFFFFFFF, a12, offset);
        a13 += __shfl_down_sync(0xFFFFFFFF, a13, offset);
        a21 += __shfl_down_sync(0xFFFFFFFF, a21, offset);
        a22 += __shfl_down_sync(0xFFFFFFFF, a22, offset);
        a23 += __shfl_down_sync(0xFFFFFFFF, a23, offset);
        a31 += __shfl_down_sync(0xFFFFFFFF, a31, offset);
        a32 += __shfl_down_sync(0xFFFFFFFF, a32, offset);
        a33 += __shfl_down_sync(0xFFFFFFFF, a33, offset);
    }

    // 3. SVD → 旋转矩阵 R（只在 lane 0 计算）
    float u11,u12,u13,u21,u22,u23,u31,u32,u33;
    float s11,s12,s13,s21,s22,s23,s31,s32,s33;
    float v11,v12,v13,v21,v22,v23,v31,v32,v33;
    float r11,r12,r13,r21,r22,r23,r31,r32,r33;

    if (tid == 0) {
        // Broadcast A to lane 0 (already there from reduction)
        svd(a11,a12,a13,a21,a22,a23,a31,a32,a33,
            u11,u12,u13,u21,u22,u23,u31,u32,u33,
            s11,s12,s13,s21,s22,s23,s31,s32,s33,
            v11,v12,v13,v21,v22,v23,v31,v32,v33);

        // R = U * V^T
        multAB(u11,u12,u13,u21,u22,u23,u31,u32,u33,
               v11,v21,v31,v12,v22,v32,v13,v23,v33,
               r11,r12,r13,r21,r22,r23,r31,r32,r33);

        // 确保 det(R) > 0
        float det = r11*(r22*r33-r23*r32) - r12*(r21*r33-r23*r31) + r13*(r21*r32-r22*r31);
        if (det < 0) {
            u13 = -u13; u23 = -u23; u33 = -u33;
            multAB(u11,u12,u13,u21,u22,u23,u31,u32,u33,
                   v11,v21,v31,v12,v22,v32,v13,v23,v33,
                   r11,r12,r13,r21,r22,r23,r31,r32,r33);
        }
    }

    // Broadcast R to all threads
    r11 = __shfl_sync(0xFFFFFFFF, r11, 0);
    r12 = __shfl_sync(0xFFFFFFFF, r12, 0);
    r13 = __shfl_sync(0xFFFFFFFF, r13, 0);
    r21 = __shfl_sync(0xFFFFFFFF, r21, 0);
    r22 = __shfl_sync(0xFFFFFFFF, r22, 0);
    r23 = __shfl_sync(0xFFFFFFFF, r23, 0);
    r31 = __shfl_sync(0xFFFFFFFF, r31, 0);
    r32 = __shfl_sync(0xFFFFFFFF, r32, 0);
    r33 = __shfl_sync(0xFFFFFFFF, r33, 0);

    // 4. Goal positions → atomicAdd deltas（跳过 removed 粒子）
    for (int j = tid; j < size; j += 32) {
        int pid = particle_ids[start + j];
        if (removed[pid]) continue;
        float qx = rest_offsets[start + j].x - qcx;
        float qy = rest_offsets[start + j].y - qcy;
        float qz = rest_offsets[start + j].z - qcz;

        float gx = r11*qx + r12*qy + r13*qz + cx;
        float gy = r21*qx + r22*qy + r23*qz + cy;
        float gz = r31*qx + r32*qy + r33*qz + cz;

        atomicAdd(&deltas[pid].x, gx - positions[pid].x);
        atomicAdd(&deltas[pid].y, gy - positions[pid].y);
        atomicAdd(&deltas[pid].z, gz - positions[pid].z);
        atomicAdd(&delta_counts[pid], 1);
    }
}

__global__ void apply_kernel(
    float3* pred_pos, float3* deltas, int* delta_counts,
    float* damage, int* removed, int* fixed,
    float stiffness, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i] || fixed[i] || delta_counts[i] == 0) return;

    float local_stiff = stiffness * (1.0f - damage[i] * 0.95f);
    float inv_count = 1.0f / delta_counts[i];
    pred_pos[i].x += local_stiff * deltas[i].x * inv_count;
    pred_pos[i].y += local_stiff * deltas[i].y * inv_count;
    pred_pos[i].z += local_stiff * deltas[i].z * inv_count;
}

// Apply self-collision delta (FleX constraint groups, Macklin 2014, Sec 4).
// Differs from apply_kernel:
//   - Uses SEPARATE deltas_sc / delta_counts_sc (not shared with shape-match)
//     → average is over only the self-colliding pairs for this particle (typically 2-8),
//     not over the ~60 shape-match cluster memberships. No dilution.
//   - NO damage-based weakening. Self-collision is a kinematic non-penetration constraint;
//     a damaged particle should NOT be allowed to freely overlap with its neighbors.
//   - self_collision_kernel already bakes stiffness into the per-pair delta, so no
//     further stiffness multiplier here.
__global__ void apply_self_collision_kernel(
    float3* pred_pos, float3* deltas_sc, int* delta_counts_sc,
    int* removed, int* fixed, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i] || fixed[i] || delta_counts_sc[i] == 0) return;

    float inv_count = 1.0f / delta_counts_sc[i];
    pred_pos[i].x += deltas_sc[i].x * inv_count;
    pred_pos[i].y += deltas_sc[i].y * inv_count;
    pred_pos[i].z += deltas_sc[i].z * inv_count;
}

__global__ void clear_deltas_kernel(float3* deltas, int* counts, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    deltas[i] = make_float3(0, 0, 0);
    counts[i] = 0;
}

__global__ void update_velocity_kernel(
    float3* pos, float3* vel, float3* pred_pos, float3* old_pos,
    int* removed, int* fixed, float inv_dt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i] || fixed[i]) return;

    vel[i].x = (pred_pos[i].x - old_pos[i].x) * inv_dt;
    vel[i].y = (pred_pos[i].y - old_pos[i].y) * inv_dt;
    vel[i].z = (pred_pos[i].z - old_pos[i].z) * inv_dt;
    pos[i] = pred_pos[i];
}

// ============== Self-Collision Kernels ==============

__global__ void compute_cell_hash_kernel(
    float3* positions, int* removed,
    unsigned int* cell_hash, unsigned int* particle_index,
    float cell_size, int table_size, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (removed[i]) {
        cell_hash[i] = table_size;  // sentinel → sorted to end
    } else {
        int ix = (int)floorf(positions[i].x / cell_size);
        int iy = (int)floorf(positions[i].y / cell_size);
        int iz = (int)floorf(positions[i].z / cell_size);
        unsigned int h = ((unsigned int)(ix * 73856093) ^
                          (unsigned int)(iy * 19349663) ^
                          (unsigned int)(iz * 83492791)) & (table_size - 1);
        cell_hash[i] = h;
    }
    particle_index[i] = i;
}

__global__ void find_cell_bounds_kernel(
    unsigned int* sorted_hash, unsigned int* cell_start, unsigned int* cell_end, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    unsigned int h = sorted_hash[i];
    if (i == 0 || sorted_hash[i - 1] != h) {
        cell_start[h] = i;
    }
    if (i == N - 1 || sorted_hash[i + 1] != h) {
        cell_end[h] = i + 1;
    }
}

__global__ void self_collision_kernel(
    float3* pred_pos, int* removed, int* fixed,
    unsigned int* sorted_index, unsigned int* sorted_hash,
    unsigned int* cell_start, unsigned int* cell_end,
    float3* deltas, int* delta_counts,
    float radius, float stiffness, float cell_size,
    int table_size, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Get original particle index from sorted order
    unsigned int i = sorted_index[idx];
    if (removed[i] || fixed[i]) return;
    if (sorted_hash[idx] >= (unsigned int)table_size) return;  // sentinel

    float3 pi = pred_pos[i];
    int ix = (int)floorf(pi.x / cell_size);
    int iy = (int)floorf(pi.y / cell_size);
    int iz = (int)floorf(pi.z / cell_size);

    float collision_dist = 2.0f * radius;

    // 3x3x3 neighbor cells
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                unsigned int h = ((unsigned int)((ix + dx) * 73856093) ^
                                  (unsigned int)((iy + dy) * 19349663) ^
                                  (unsigned int)((iz + dz) * 83492791)) & (table_size - 1);

                unsigned int start = cell_start[h];
                if (start == 0xFFFFFFFF) continue;
                unsigned int end = cell_end[h];

                for (unsigned int k = start; k < end; k++) {
                    unsigned int j = sorted_index[k];
                    if (j <= i || removed[j]) continue;  // avoid double-count

                    float ex = pi.x - pred_pos[j].x;
                    float ey = pi.y - pred_pos[j].y;
                    float ez = pi.z - pred_pos[j].z;
                    float dist2 = ex * ex + ey * ey + ez * ez;

                    if (dist2 < collision_dist * collision_dist && dist2 > 1e-16f) {
                        float dist = sqrtf(dist2);
                        float overlap = collision_dist - dist;
                        float inv_dist = 1.0f / dist;
                        float cx = stiffness * 0.5f * overlap * ex * inv_dist;
                        float cy = stiffness * 0.5f * overlap * ey * inv_dist;
                        float cz = stiffness * 0.5f * overlap * ez * inv_dist;

                        atomicAdd(&deltas[i].x, cx);
                        atomicAdd(&deltas[i].y, cy);
                        atomicAdd(&deltas[i].z, cz);
                        atomicAdd(&delta_counts[i], 1);

                        atomicAdd(&deltas[j].x, -cx);
                        atomicAdd(&deltas[j].y, -cy);
                        atomicAdd(&deltas[j].z, -cz);
                        atomicAdd(&delta_counts[j], 1);
                    }
                }
            }
        }
    }
}

// ============== Cutting Kernels ==============

// 边 P(F) 计算 + damage 更新（每边一线程）
__global__ void update_edge_damage_kernel(
    float3* pred_pos, int* removed,
    int* edge_a, int* edge_b, float* pf_rest, float* damage, int* broken,
    float3 knife_pos, float knife_radius, float alpha_min, int num_edges)
{
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges || broken[eid]) return;

    int a = edge_a[eid], b = edge_b[eid];
    if (removed[a] || removed[b]) return;

    // 边中点
    float3 mid;
    mid.x = (pred_pos[a].x + pred_pos[b].x) * 0.5f;
    mid.y = (pred_pos[a].y + pred_pos[b].y) * 0.5f;
    mid.z = (pred_pos[a].z + pred_pos[b].z) * 0.5f;

    float dx = mid.x - knife_pos.x;
    float dy = mid.y - knife_pos.y;
    float dz = mid.z - knife_pos.z;
    float dist_to_knife_center = sqrtf(dx*dx + dy*dy + dz*dz);

    // 只更新刀具附近的边
    float search_r = knife_radius * 3.0f;
    if (dist_to_knife_center > search_r) return;

    // 边半径（circumsphere）
    float ex = pred_pos[a].x - pred_pos[b].x;
    float ey = pred_pos[a].y - pred_pos[b].y;
    float ez = pred_pos[a].z - pred_pos[b].z;
    float radius = sqrtf(ex*ex + ey*ey + ez*ez) * 0.5f;
    if (radius < 1e-10f) return;

    // 用刀具**表面**（非中心）作为闯入者
    // 刀具表面到边中点的距离 = dist_to_knife_center - knife_radius
    float intruder_dist = dist_to_knife_center - knife_radius;
    float sdist = intruder_dist - radius;
    float pf = 1.0f / (1.0f + expf(-alpha_min * sdist / radius));

    // Damage 更新（只增不减）
    float rest = pf_rest[eid];
    if (rest > 0.01f) {
        float target = fmaxf(0.0f, 1.0f - pf / rest);
        damage[eid] = fmaxf(damage[eid], target);
    }

    // 断裂检查
    if (damage[eid] > 0.5f) {
        broken[eid] = 1;
    }
}

// float atomicMax（CUDA 原生不支持，用 atomicCAS 模拟）
__device__ void atomicMaxFloat(float* addr, float val) {
    if (val <= 0.0f) return;
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) return;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

// 每粒子 damage 聚合（MEAN over incident edges，只增不减）
// Pass 1: 每条边把自己的 damage atomicAdd 到两个端点的 sum，同时 count +=1
// Pass 2 (下面的 finalize 核)：particle_damage = max(particle_damage, sum/count)
// 这是对原 atomicMax 聚合的修正 —— 原版"一条断边 = 两粒子满 damage"，过度敏感。
// 改 MEAN 后，内部粒子（~6 incident 边）需要多条边同时受损才能超过 aspiration 阈值。
// 与 Warp/Python 版 (pf_damage.py update_particle_damage_kernel) 语义对齐。
__global__ void accumulate_particle_damage_kernel(
    float* particle_damage_sum, float* particle_damage_count,
    int* removed,
    int* edge_a, int* edge_b, float* edge_damage, int num_edges)
{
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;
    int a = edge_a[eid], b = edge_b[eid];
    if (removed[a] || removed[b]) return;

    float val = edge_damage[eid];
    atomicAdd(&particle_damage_sum[a], val);
    atomicAdd(&particle_damage_sum[b], val);
    atomicAdd(&particle_damage_count[a], 1.0f);
    atomicAdd(&particle_damage_count[b], 1.0f);
}

// Pass 2: mean + monotonic update
__global__ void finalize_particle_damage_kernel(
    float* particle_damage,
    float* particle_damage_sum, float* particle_damage_count,
    int* removed, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i]) return;

    float cnt = particle_damage_count[i];
    if (cnt < 0.5f) return;  // 本次无 incident 边参与（邻粒子全被移除或远离刀具）

    float mean_damage = particle_damage_sum[i] / cnt;
    // 只增不减（与边 damage 的 monotonic 语义一致）
    if (mean_damage > particle_damage[i]) {
        particle_damage[i] = mean_damage;
    }
}

// CUSA 吸引：刀具内 + 高 damage → 标记 removed
__global__ void aspirate_kernel(
    float3* pred_pos, float* particle_damage, int* removed,
    float3 knife_pos, float knife_radius, float threshold, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i]) return;

    float dx = pred_pos[i].x - knife_pos.x;
    float dy = pred_pos[i].y - knife_pos.y;
    float dz = pred_pos[i].z - knife_pos.z;
    if (dx*dx + dy*dy + dz*dz > knife_radius * knife_radius) return;

    if (particle_damage[i] > threshold) {
        removed[i] = 1;
    }
}

// ============== Host Functions ==============

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
}

void ShapeMatchingGPUSolver::init(
    const std::vector<float3>& positions,
    const std::vector<float3>& velocities,
    const std::vector<int>& removed,
    const std::vector<int>& fixed,
    const std::vector<int>& cluster_starts,
    const std::vector<int>& cluster_sizes,
    const std::vector<int>& particle_ids,
    const std::vector<float3>& rest_offsets,
    const std::vector<float3>& rest_cm)
{
    int N = positions.size();
    int C = cluster_starts.size();
    int total = particle_ids.size();

    d_particles.N = N;
    d_clusters.num_clusters = C;
    d_clusters.total_entries = total;

    // 分配粒子数据
    CUDA_CHECK(cudaMalloc(&d_particles.pos, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_particles.vel, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_particles.predicted_pos, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_particles.old_pos, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_particles.damage, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_particles.damage_sum,   N * sizeof(float)));  // MEAN-aggregation accumulator
    CUDA_CHECK(cudaMalloc(&d_particles.damage_count, N * sizeof(float)));  // incident-edge count
    CUDA_CHECK(cudaMalloc(&d_particles.removed, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_particles.fixed, N * sizeof(int)));

    // 上传粒子数据
    CUDA_CHECK(cudaMemcpy(d_particles.pos, positions.data(), N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particles.vel, velocities.data(), N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particles.predicted_pos, positions.data(), N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particles.removed, removed.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particles.fixed, fixed.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_particles.damage, 0, N * sizeof(float)));

    // 分配簇数据
    CUDA_CHECK(cudaMalloc(&d_clusters.cluster_starts, C * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clusters.cluster_sizes, C * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clusters.particle_ids, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clusters.rest_offsets, total * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_clusters.rest_cm, C * sizeof(float3)));

    CUDA_CHECK(cudaMemcpy(d_clusters.cluster_starts, cluster_starts.data(), C * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters.cluster_sizes, cluster_sizes.data(), C * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters.particle_ids, particle_ids.data(), total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters.rest_offsets, rest_offsets.data(), total * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters.rest_cm, rest_cm.data(), C * sizeof(float3), cudaMemcpyHostToDevice));

    // 分配 delta buffers (shape-match)
    CUDA_CHECK(cudaMalloc(&d_deltas, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_delta_counts, N * sizeof(int)));
    // Separate delta buffers for self-collision (FleX constraint groups, Macklin 2014 Sec 4)
    CUDA_CHECK(cudaMalloc(&d_deltas_sc, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_delta_counts_sc, N * sizeof(int)));

    // Self-collision hash grid buffers
    if (config.self_collision_enabled) {
        CUDA_CHECK(cudaMalloc(&d_cell_hash, N * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_particle_index, N * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_cell_hash_sorted, N * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_particle_index_sorted, N * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_cell_start, HASH_TABLE_SIZE * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_cell_end, HASH_TABLE_SIZE * sizeof(unsigned int)));

        // Query CUB temp storage size
        cub_temp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, cub_temp_bytes,
            d_cell_hash, d_cell_hash_sorted,
            d_particle_index, d_particle_index_sorted,
            N, 0, 15);
        CUDA_CHECK(cudaMalloc(&d_cub_temp, cub_temp_bytes));
        printf("[GPU] Self-collision hash grid: table=%d, cub_temp=%zu bytes\n",
               HASH_TABLE_SIZE, cub_temp_bytes);
    }

    printf("[GPU] 初始化完成: %d 粒子, %d 簇, %d 总条目\n", N, C, total);

    // 初始化边数据为空
    d_edges.num_edges = 0;
    d_edges.particle_a = nullptr;
}

void ShapeMatchingGPUSolver::init_edges(
    const std::vector<int>& edge_a,
    const std::vector<int>& edge_b,
    const std::vector<float>& edge_pf_rest)
{
    int E = edge_a.size();
    d_edges.num_edges = E;

    CUDA_CHECK(cudaMalloc(&d_edges.particle_a, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edges.particle_b, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edges.pf_rest, E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_edges.damage, E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_edges.broken, E * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_edges.particle_a, edge_a.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edges.particle_b, edge_b.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edges.pf_rest, edge_pf_rest.data(), E * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_edges.damage, 0, E * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_edges.broken, 0, E * sizeof(int)));

    printf("[GPU] 边数据初始化: %d 条边\n", E);
}

// ============== Rigid Body SDF Collision ==============

// SDF collision kernel: trilinear interpolation + push out + friction
__global__ void sdf_collision_kernel(
    float3* pred_pos, float3* old_pos, int* removed, int* fixed,
    float* sdf, int res_x, int res_y, int res_z,
    float3 origin, float3 cell_size,
    float mu_s, float mu_k,  // static and kinetic friction coefficients
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i] || fixed[i]) return;

    float3 p = pred_pos[i];

    // Map to grid coordinates (continuous)
    float gx = (p.x - origin.x) / cell_size.x - 0.5f;
    float gy = (p.y - origin.y) / cell_size.y - 0.5f;
    float gz = (p.z - origin.z) / cell_size.z - 0.5f;

    // Clamp to grid bounds
    if (gx < 0 || gx >= res_x - 1 || gy < 0 || gy >= res_y - 1 || gz < 0 || gz >= res_z - 1)
        return;  // outside SDF grid, no collision

    // Integer cell + fractional
    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    float fx = gx - ix, fy = gy - iy, fz = gz - iz;

    // Trilinear interpolation of SDF
    #define SDF_IDX(x,y,z) sdf[(z)*res_x*res_y + (y)*res_x + (x)]
    float c000 = SDF_IDX(ix,   iy,   iz  );
    float c100 = SDF_IDX(ix+1, iy,   iz  );
    float c010 = SDF_IDX(ix,   iy+1, iz  );
    float c110 = SDF_IDX(ix+1, iy+1, iz  );
    float c001 = SDF_IDX(ix,   iy,   iz+1);
    float c101 = SDF_IDX(ix+1, iy,   iz+1);
    float c011 = SDF_IDX(ix,   iy+1, iz+1);
    float c111 = SDF_IDX(ix+1, iy+1, iz+1);

    float c00 = c000*(1-fx) + c100*fx;
    float c10 = c010*(1-fx) + c110*fx;
    float c01 = c001*(1-fx) + c101*fx;
    float c11 = c011*(1-fx) + c111*fx;
    float c0 = c00*(1-fy) + c10*fy;
    float c1 = c01*(1-fy) + c11*fy;
    float dist = c0*(1-fz) + c1*fz;

    if (dist >= 0.0f) return;  // outside rigid body, no collision

    // Inside (dist < 0): compute gradient of SDF for push direction
    // Central differences
    float eps = 1.0f;  // one grid cell
    float gx_m = gx - eps, gx_p = gx + eps;
    float gy_m = gy - eps, gy_p = gy + eps;
    float gz_m = gz - eps, gz_p = gz + eps;

    // Clamp gradient sample positions
    gx_m = fmaxf(gx_m, 0.0f); gx_p = fminf(gx_p, (float)(res_x - 1) - 0.01f);
    gy_m = fmaxf(gy_m, 0.0f); gy_p = fminf(gy_p, (float)(res_y - 1) - 0.01f);
    gz_m = fmaxf(gz_m, 0.0f); gz_p = fminf(gz_p, (float)(res_z - 1) - 0.01f);

    // Sample SDF at offset positions (nearest-neighbor for speed)
    #define SDF_SAMPLE(gx_, gy_, gz_) SDF_IDX( \
        min(max((int)(gx_), 0), res_x-1), \
        min(max((int)(gy_), 0), res_y-1), \
        min(max((int)(gz_), 0), res_z-1))

    float nx = SDF_SAMPLE(gx_p, gy, gz) - SDF_SAMPLE(gx_m, gy, gz);
    float ny = SDF_SAMPLE(gx, gy_p, gz) - SDF_SAMPLE(gx, gy_m, gz);
    float nz = SDF_SAMPLE(gx, gy, gz_p) - SDF_SAMPLE(gx, gy, gz_m);
    #undef SDF_SAMPLE
    #undef SDF_IDX

    float len = sqrtf(nx*nx + ny*ny + nz*nz);
    if (len < 1e-10f) return;
    nx /= len; ny /= len; nz /= len;

    // Push particle to surface: move by |dist| along gradient
    float push = -dist;  // dist is negative, push is positive
    pred_pos[i].x += push * nx;
    pred_pos[i].y += push * ny;
    pred_pos[i].z += push * nz;

    // Coulomb friction: reduce tangential sliding
    if (mu_s > 0.0f || mu_k > 0.0f) {
        // Displacement this step
        float3 dp = make_float3(pred_pos[i].x - old_pos[i].x,
                                pred_pos[i].y - old_pos[i].y,
                                pred_pos[i].z - old_pos[i].z);

        // Tangential component (remove normal part)
        float dp_n = dp.x*nx + dp.y*ny + dp.z*nz;
        float3 dp_t = make_float3(dp.x - dp_n*nx, dp.y - dp_n*ny, dp.z - dp_n*nz);
        float len_t = sqrtf(dp_t.x*dp_t.x + dp_t.y*dp_t.y + dp_t.z*dp_t.z);

        if (len_t > 1e-10f) {
            if (len_t < mu_s * push) {
                // Static friction: cancel all tangential motion
                pred_pos[i].x -= dp_t.x;
                pred_pos[i].y -= dp_t.y;
                pred_pos[i].z -= dp_t.z;
            } else {
                // Kinetic friction: reduce tangential motion
                float reduce = fminf(mu_k * push / len_t, 1.0f);
                pred_pos[i].x -= dp_t.x * reduce;
                pred_pos[i].y -= dp_t.y * reduce;
                pred_pos[i].z -= dp_t.z * reduce;
            }
        }
    }
}

// Apply R^T (transpose/inverse of rotation) to a vector
__device__ float3 apply_rot_transpose(const float* R, float3 v) {
    return make_float3(
        R[0]*v.x + R[3]*v.y + R[6]*v.z,  // R^T row 0 = R column 0
        R[1]*v.x + R[4]*v.y + R[7]*v.z,
        R[2]*v.x + R[5]*v.y + R[8]*v.z);
}

// Apply R (forward rotation) to a vector
__device__ float3 apply_rot(const float* R, float3 v) {
    return make_float3(
        R[0]*v.x + R[1]*v.y + R[2]*v.z,
        R[3]*v.x + R[4]*v.y + R[5]*v.z,
        R[6]*v.x + R[7]*v.y + R[8]*v.z);
}

// Tool SDF collision: transform particle to tool local frame, query SDF
__global__ void tool_sdf_collision_kernel(
    float3* pred_pos, int* removed, int* fixed,
    float* sdf, int res_x, int res_y, int res_z,
    float3 sdf_origin, float3 sdf_cell_size,
    float3 tool_pos, const float* tool_rot, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i] || fixed[i]) return;

    // Transform to tool local frame: p_local = R^T * (p_world - tool_pos)
    float3 d = make_float3(
        pred_pos[i].x - tool_pos.x,
        pred_pos[i].y - tool_pos.y,
        pred_pos[i].z - tool_pos.z);
    float3 p_local = apply_rot_transpose(tool_rot, d);

    // Map to grid coordinates
    float gx = (p_local.x - sdf_origin.x) / sdf_cell_size.x - 0.5f;
    float gy = (p_local.y - sdf_origin.y) / sdf_cell_size.y - 0.5f;
    float gz = (p_local.z - sdf_origin.z) / sdf_cell_size.z - 0.5f;

    if (gx < 0 || gx >= res_x-1 || gy < 0 || gy >= res_y-1 || gz < 0 || gz >= res_z-1)
        return;

    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    float fx = gx - ix, fy = gy - iy, fz = gz - iz;

    #define SDF_IDX(x,y,z) sdf[(z)*res_x*res_y + (y)*res_x + (x)]
    float c00 = SDF_IDX(ix,iy,iz)*(1-fx) + SDF_IDX(ix+1,iy,iz)*fx;
    float c10 = SDF_IDX(ix,iy+1,iz)*(1-fx) + SDF_IDX(ix+1,iy+1,iz)*fx;
    float c01 = SDF_IDX(ix,iy,iz+1)*(1-fx) + SDF_IDX(ix+1,iy,iz+1)*fx;
    float c11 = SDF_IDX(ix,iy+1,iz+1)*(1-fx) + SDF_IDX(ix+1,iy+1,iz+1)*fx;
    float dist = (c00*(1-fy)+c10*fy)*(1-fz) + (c01*(1-fy)+c11*fy)*fz;

    if (dist >= 0.0f) return;

    // Gradient in local frame
    #define SDF_SAMPLE(gx_, gy_, gz_) SDF_IDX( \
        min(max((int)(gx_), 0), res_x-1), \
        min(max((int)(gy_), 0), res_y-1), \
        min(max((int)(gz_), 0), res_z-1))
    float nx = SDF_SAMPLE(gx+1, gy, gz) - SDF_SAMPLE(gx-1, gy, gz);
    float ny = SDF_SAMPLE(gx, gy+1, gz) - SDF_SAMPLE(gx, gy-1, gz);
    float nz = SDF_SAMPLE(gx, gy, gz+1) - SDF_SAMPLE(gx, gy, gz-1);
    #undef SDF_SAMPLE
    #undef SDF_IDX

    float len = sqrtf(nx*nx + ny*ny + nz*nz);
    if (len < 1e-10f) return;
    nx /= len; ny /= len; nz /= len;

    // Transform gradient from local frame to world frame: n_world = R * n_local
    float3 n_local = make_float3(nx, ny, nz);
    float3 n_world = apply_rot(tool_rot, n_local);

    float push = -dist;
    pred_pos[i].x += push * n_world.x;
    pred_pos[i].y += push * n_world.y;
    pred_pos[i].z += push * n_world.z;
}

// Tool SDF damage: compute distance from edge midpoint to tool surface
__global__ void tool_sdf_damage_kernel(
    float3* pred_pos, int* removed,
    int* edge_a, int* edge_b, float* pf_rest, float* damage, int* broken,
    float* sdf, int res_x, int res_y, int res_z,
    float3 sdf_origin, float3 sdf_cell_size,
    float3 tool_pos, const float* tool_rot, float alpha_min, int num_edges)
{
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges || broken[eid]) return;

    int a = edge_a[eid], b = edge_b[eid];
    if (removed[a] || removed[b]) return;

    // Edge midpoint in world → local frame: R^T * (mid_world - tool_pos)
    float3 mid_world = make_float3(
        (pred_pos[a].x + pred_pos[b].x) * 0.5f - tool_pos.x,
        (pred_pos[a].y + pred_pos[b].y) * 0.5f - tool_pos.y,
        (pred_pos[a].z + pred_pos[b].z) * 0.5f - tool_pos.z);
    float3 mid = apply_rot_transpose(tool_rot, mid_world);

    // Map to grid
    float gx = (mid.x - sdf_origin.x) / sdf_cell_size.x - 0.5f;
    float gy = (mid.y - sdf_origin.y) / sdf_cell_size.y - 0.5f;
    float gz = (mid.z - sdf_origin.z) / sdf_cell_size.z - 0.5f;

    if (gx < 0 || gx >= res_x-1 || gy < 0 || gy >= res_y-1 || gz < 0 || gz >= res_z-1)
        return;

    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    float fx = gx - ix, fy = gy - iy, fz = gz - iz;

    #define SDF_IDX(x,y,z) sdf[(z)*res_x*res_y + (y)*res_x + (x)]
    float c00 = SDF_IDX(ix,iy,iz)*(1-fx) + SDF_IDX(ix+1,iy,iz)*fx;
    float c10 = SDF_IDX(ix,iy+1,iz)*(1-fx) + SDF_IDX(ix+1,iy+1,iz)*fx;
    float c01 = SDF_IDX(ix,iy,iz+1)*(1-fx) + SDF_IDX(ix+1,iy,iz+1)*fx;
    float c11 = SDF_IDX(ix,iy+1,iz+1)*(1-fx) + SDF_IDX(ix+1,iy+1,iz+1)*fx;
    float sdist = (c00*(1-fy)+c10*fy)*(1-fz) + (c01*(1-fy)+c11*fy)*fz;
    #undef SDF_IDX

    // Edge circumradius
    float ex = pred_pos[a].x - pred_pos[b].x;
    float ey = pred_pos[a].y - pred_pos[b].y;
    float ez = pred_pos[a].z - pred_pos[b].z;
    float radius = sqrtf(ex*ex + ey*ey + ez*ez) * 0.5f;
    if (radius < 1e-10f) return;

    // Only process edges near tool surface (|sdf| < 3 * radius)
    if (fabsf(sdist) > radius * 3.0f) return;

    // P(F) from signed distance (same formula as sphere version)
    float pf = 1.0f / (1.0f + expf(-alpha_min * sdist / radius));

    float rest = pf_rest[eid];
    if (rest > 0.01f) {
        float target = fmaxf(0.0f, 1.0f - pf / rest);
        damage[eid] = fmaxf(damage[eid], target);
    }

    if (damage[eid] > 0.5f) {
        broken[eid] = 1;
    }
}

// Tool SDF aspiration: remove high-damage particles inside tool
__global__ void tool_sdf_aspirate_kernel(
    float3* pred_pos, float* particle_damage, int* removed,
    float* sdf, int res_x, int res_y, int res_z,
    float3 sdf_origin, float3 sdf_cell_size,
    float3 tool_pos, const float* tool_rot, float threshold, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i]) return;

    float3 d = make_float3(
        pred_pos[i].x - tool_pos.x,
        pred_pos[i].y - tool_pos.y,
        pred_pos[i].z - tool_pos.z);
    float3 p_local = apply_rot_transpose(tool_rot, d);

    float gx = (p_local.x - sdf_origin.x) / sdf_cell_size.x - 0.5f;
    float gy = (p_local.y - sdf_origin.y) / sdf_cell_size.y - 0.5f;
    float gz = (p_local.z - sdf_origin.z) / sdf_cell_size.z - 0.5f;

    if (gx < 0 || gx >= res_x-1 || gy < 0 || gy >= res_y-1 || gz < 0 || gz >= res_z-1)
        return;

    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    float fx = gx - ix, fy = gy - iy, fz = gz - iz;

    #define SDF_IDX(x,y,z) sdf[(z)*res_x*res_y + (y)*res_x + (x)]
    float c00 = SDF_IDX(ix,iy,iz)*(1-fx) + SDF_IDX(ix+1,iy,iz)*fx;
    float c10 = SDF_IDX(ix,iy+1,iz)*(1-fx) + SDF_IDX(ix+1,iy+1,iz)*fx;
    float c01 = SDF_IDX(ix,iy,iz+1)*(1-fx) + SDF_IDX(ix+1,iy,iz+1)*fx;
    float c11 = SDF_IDX(ix,iy+1,iz+1)*(1-fx) + SDF_IDX(ix+1,iy+1,iz+1)*fx;
    float dist = (c00*(1-fy)+c10*fy)*(1-fz) + (c01*(1-fy)+c11*fy)*fz;
    #undef SDF_IDX

    if (dist < 0.0f && particle_damage[i] > threshold) {
        removed[i] = 1;
    }
}

// SDF debug: count particles inside rigid body
__global__ void sdf_debug_kernel(
    float3* positions, int* removed, int* fixed,
    float* sdf, int res_x, int res_y, int res_z,
    float3 origin, float3 cell_size, int N,
    int* counts)  // [0]=inside, [1]=near(sdf<0.001)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i] || fixed[i]) return;

    float3 p = positions[i];
    float gx = (p.x - origin.x) / cell_size.x - 0.5f;
    float gy = (p.y - origin.y) / cell_size.y - 0.5f;
    float gz = (p.z - origin.z) / cell_size.z - 0.5f;

    if (gx < 0 || gx >= res_x - 1 || gy < 0 || gy >= res_y - 1 || gz < 0 || gz >= res_z - 1)
        return;

    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    float fx = gx - ix, fy = gy - iy, fz = gz - iz;

    #define SDF_IDX(x,y,z) sdf[(z)*res_x*res_y + (y)*res_x + (x)]
    float c00 = SDF_IDX(ix,iy,iz)*(1-fx) + SDF_IDX(ix+1,iy,iz)*fx;
    float c10 = SDF_IDX(ix,iy+1,iz)*(1-fx) + SDF_IDX(ix+1,iy+1,iz)*fx;
    float c01 = SDF_IDX(ix,iy,iz+1)*(1-fx) + SDF_IDX(ix+1,iy,iz+1)*fx;
    float c11 = SDF_IDX(ix,iy+1,iz+1)*(1-fx) + SDF_IDX(ix+1,iy+1,iz+1)*fx;
    float dist = (c00*(1-fy)+c10*fy)*(1-fz) + (c01*(1-fy)+c11*fy)*fz;
    #undef SDF_IDX

    if (dist < 0.0f) atomicAdd(&counts[0], 1);
    if (dist < 0.001f) atomicAdd(&counts[1], 1);
}

void ShapeMatchingGPUSolver::init_sdf(
    SDFData& sdf, const std::vector<float>& sdf_data,
    int res_x, int res_y, int res_z,
    float3 origin, float3 cell_size, const char* label)
{
    sdf.res_x = res_x;
    sdf.res_y = res_y;
    sdf.res_z = res_z;
    sdf.origin = origin;
    sdf.cell_size = cell_size;
    sdf.active = 1;

    int total = res_x * res_y * res_z;
    CUDA_CHECK(cudaMalloc(&sdf.d_sdf, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(sdf.d_sdf, sdf_data.data(),
                           total * sizeof(float), cudaMemcpyHostToDevice));

    int inside = 0;
    for (int i = 0; i < total; i++) if (sdf_data[i] < 0) inside++;
    printf("[GPU] %s SDF: %dx%dx%d, cell=(%.5f,%.5f,%.5f), %d/%d inside voxels\n",
           label, res_x, res_y, res_z, cell_size.x, cell_size.y, cell_size.z, inside, total);
}

// ============== Cutting Host Functions ==============

void ShapeMatchingGPUSolver::gpu_update_damage() {
    if (d_edges.num_edges == 0 || !config.knife_active) return;
    int E = d_edges.num_edges;
    int blocks_e = (E + 255) / 256;

    if (config.tool_type == 1 && d_tool_sdf.active && d_tool_sdf.d_sdf) {
        tool_sdf_damage_kernel<<<blocks_e, 256>>>(
            d_particles.predicted_pos, d_particles.removed,
            d_edges.particle_a, d_edges.particle_b,
            d_edges.pf_rest, d_edges.damage, d_edges.broken,
            d_tool_sdf.d_sdf, d_tool_sdf.res_x, d_tool_sdf.res_y, d_tool_sdf.res_z,
            d_tool_sdf.origin, d_tool_sdf.cell_size,
            config.knife_pos, d_knife_rot, alpha_min, E);
    } else {
        update_edge_damage_kernel<<<blocks_e, 256>>>(
            d_particles.predicted_pos, d_particles.removed,
            d_edges.particle_a, d_edges.particle_b,
            d_edges.pf_rest, d_edges.damage, d_edges.broken,
            config.knife_pos, config.knife_radius, alpha_min, E);
    }
}

void ShapeMatchingGPUSolver::gpu_update_particle_damage() {
    if (d_edges.num_edges == 0) return;
    int N = d_particles.N;
    int E = d_edges.num_edges;
    int blocks_n = (N + 255) / 256;
    int blocks_e = (E + 255) / 256;

    // Zero per-step accumulators (sum + count)
    CUDA_CHECK(cudaMemset(d_particles.damage_sum,   0, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_particles.damage_count, 0, N * sizeof(float)));

    // Pass 1: atomicAdd edge damage → per-particle sum + count
    accumulate_particle_damage_kernel<<<blocks_e, 256>>>(
        d_particles.damage_sum, d_particles.damage_count,
        d_particles.removed,
        d_edges.particle_a, d_edges.particle_b,
        d_edges.damage, E);

    // Pass 2: mean = sum / count, then monotonic update on d_particles.damage
    finalize_particle_damage_kernel<<<blocks_n, 256>>>(
        d_particles.damage,
        d_particles.damage_sum, d_particles.damage_count,
        d_particles.removed, N);
}

void ShapeMatchingGPUSolver::gpu_aspirate() {
    if (!config.knife_active) return;
    int N = d_particles.N;
    int blocks_n = (N + 255) / 256;

    if (config.tool_type == 1 && d_tool_sdf.active && d_tool_sdf.d_sdf) {
        tool_sdf_aspirate_kernel<<<blocks_n, 256>>>(
            d_particles.predicted_pos, d_particles.damage, d_particles.removed,
            d_tool_sdf.d_sdf, d_tool_sdf.res_x, d_tool_sdf.res_y, d_tool_sdf.res_z,
            d_tool_sdf.origin, d_tool_sdf.cell_size,
            config.knife_pos, d_knife_rot, aspiration_threshold, N);
    } else {
        aspirate_kernel<<<blocks_n, 256>>>(
            d_particles.predicted_pos, d_particles.damage, d_particles.removed,
            config.knife_pos, config.knife_radius, aspiration_threshold, N);
    }
}

// ============== Simulation Step ==============

void ShapeMatchingGPUSolver::step() {
    int N = d_particles.N;
    int C = d_clusters.num_clusters;
    int blocks_n = (N + 255) / 256;

    // Upload knife rotation to GPU
    if (!d_knife_rot) {
        CUDA_CHECK(cudaMalloc(&d_knife_rot, 9 * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_knife_rot, config.knife_rot, 9 * sizeof(float), cudaMemcpyHostToDevice));

    // 1. Predict
    predict_kernel<<<blocks_n, 256>>>(
        d_particles.pos, d_particles.vel,
        d_particles.predicted_pos, d_particles.old_pos,
        d_particles.removed, d_particles.fixed,
        config.gravity, config.dt, config.damping, N);

    // 2. Collision (predict 后) — floor + sphere tool
    collision_kernel<<<blocks_n, 256>>>(
        d_particles.predicted_pos, d_particles.old_pos,
        d_particles.removed, d_particles.fixed,
        config.floor_y, config.knife_pos, config.knife_radius,
        (config.tool_type == 0) ? config.knife_active : 0,     // sphere only if tool_type==0
        config.friction_s, config.friction_k, N);              // Coulomb 摩擦（复用 bone SDF 的 μ_s/μ_k）

    // 2a. Tool SDF collision (if mesh tool)
    if (config.tool_type == 1 && config.knife_active && d_tool_sdf.active && d_tool_sdf.d_sdf) {
        tool_sdf_collision_kernel<<<blocks_n, 256>>>(
            d_particles.predicted_pos, d_particles.removed, d_particles.fixed,
            d_tool_sdf.d_sdf, d_tool_sdf.res_x, d_tool_sdf.res_y, d_tool_sdf.res_z,
            d_tool_sdf.origin, d_tool_sdf.cell_size, config.knife_pos, d_knife_rot, N);
    }

    // 2b. Rigid SDF collision
    if (d_rigid_sdf.active && d_rigid_sdf.d_sdf) {
        sdf_collision_kernel<<<blocks_n, 256>>>(
            d_particles.predicted_pos, d_particles.old_pos,
            d_particles.removed, d_particles.fixed,
            d_rigid_sdf.d_sdf, d_rigid_sdf.res_x, d_rigid_sdf.res_y, d_rigid_sdf.res_z,
            d_rigid_sdf.origin, d_rigid_sdf.cell_size,
            config.friction_s, config.friction_k, N);
    }

    // 3. Jacobi 迭代 — FleX constraint groups (Macklin 2014 Sec 4):
    //    Group 1: shape-matching (cluster_solve → apply) — soft constraint, Jacobi-averaged
    //    Group 2: self-collision (self_collision_kernel → apply_sc) — separate delta buffer
    //             so contacts aren't diluted by ~60 shape-match contributions per particle
    //    Group 3: hard contacts (knife/floor/tool SDF/bone SDF) — direct write to pred_pos
    for (int iter = 0; iter < config.num_iterations; iter++) {
        // ---- Group 1: shape matching ----
        clear_deltas_kernel<<<blocks_n, 256>>>(d_deltas, d_delta_counts, N);

        // Solve all clusters（每簇一个 block，32 线程/warp）
        cluster_solve_kernel<<<C, 32>>>(
            d_particles.predicted_pos, d_particles.removed,
            d_clusters.cluster_starts, d_clusters.cluster_sizes,
            d_clusters.particle_ids, d_clusters.rest_offsets,
            d_deltas, d_delta_counts, C);

        // Apply shape-match delta (avg over ~60 cluster memberships)
        apply_kernel<<<blocks_n, 256>>>(
            d_particles.predicted_pos, d_deltas, d_delta_counts,
            d_particles.damage, d_particles.removed, d_particles.fixed,
            config.stiffness, N);

        // ---- Group 2: self-collision (separate delta buffer, no shape-match dilution) ----
        if (config.self_collision_enabled && d_cell_hash) {
            clear_deltas_kernel<<<blocks_n, 256>>>(d_deltas_sc, d_delta_counts_sc, N);
            build_hash_grid();
            float cell_size = config.self_collision_radius * 2.0f;
            self_collision_kernel<<<blocks_n, 256>>>(
                d_particles.predicted_pos, d_particles.removed, d_particles.fixed,
                d_particle_index_sorted, d_cell_hash_sorted,
                d_cell_start, d_cell_end,
                d_deltas_sc, d_delta_counts_sc,      // SEPARATE buffer (was d_deltas)
                config.self_collision_radius, config.self_collision_stiffness,
                cell_size, HASH_TABLE_SIZE, N);
            // Apply self-collision delta (avg over only the colliding pairs, ~2-8)
            apply_self_collision_kernel<<<blocks_n, 256>>>(
                d_particles.predicted_pos, d_deltas_sc, d_delta_counts_sc,
                d_particles.removed, d_particles.fixed, N);
        }

        // ---- Group 3: hard contacts (direct write to pred_pos) ----
        // Collision (每次迭代) — floor (带 Coulomb 摩擦) + sphere tool
        collision_kernel<<<blocks_n, 256>>>(
            d_particles.predicted_pos, d_particles.old_pos,
            d_particles.removed, d_particles.fixed,
            config.floor_y, config.knife_pos, config.knife_radius,
            (config.tool_type == 0) ? config.knife_active : 0,
            config.friction_s, config.friction_k, N);

        // Tool SDF collision (每次迭代)
        if (config.tool_type == 1 && config.knife_active && d_tool_sdf.active && d_tool_sdf.d_sdf) {
            tool_sdf_collision_kernel<<<blocks_n, 256>>>(
                d_particles.predicted_pos, d_particles.removed, d_particles.fixed,
                d_tool_sdf.d_sdf, d_tool_sdf.res_x, d_tool_sdf.res_y, d_tool_sdf.res_z,
                d_tool_sdf.origin, d_tool_sdf.cell_size, config.knife_pos, d_knife_rot, N);
        }

        // Rigid SDF collision (每次迭代)
        if (d_rigid_sdf.active && d_rigid_sdf.d_sdf) {
            sdf_collision_kernel<<<blocks_n, 256>>>(
                d_particles.predicted_pos, d_particles.old_pos,
                d_particles.removed, d_particles.fixed,
                d_rigid_sdf.d_sdf, d_rigid_sdf.res_x, d_rigid_sdf.res_y, d_rigid_sdf.res_z,
                d_rigid_sdf.origin, d_rigid_sdf.cell_size,
                config.friction_s, config.friction_k, N);
        }
    }

    // SDF collision debug (every 500 steps)
    if (d_rigid_sdf.active && d_rigid_sdf.d_sdf && (step_count % 500 == 0)) {
        int* d_counts;
        cudaMalloc(&d_counts, 2 * sizeof(int));
        cudaMemset(d_counts, 0, 2 * sizeof(int));
        sdf_debug_kernel<<<blocks_n, 256>>>(
            d_particles.pos, d_particles.removed, d_particles.fixed,
            d_rigid_sdf.d_sdf, d_rigid_sdf.res_x, d_rigid_sdf.res_y, d_rigid_sdf.res_z,
            d_rigid_sdf.origin, d_rigid_sdf.cell_size, N, d_counts);
        int h[2]; cudaMemcpy(h, d_counts, 2*sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_counts);
        fprintf(stderr, "[Step %d] SDF collision: %d inside, %d near (sdf<0.001)\n",
                step_count, h[0], h[1]);
    }

    // 4. Cutting pipeline（每 cutting_update_interval 步）
    if (config.knife_active && d_edges.num_edges > 0 &&
        (step_count % cutting_update_interval == 0)) {
        gpu_update_damage();
        gpu_update_particle_damage();
        gpu_aspirate();
    }
    step_count++;

    // 5. Update velocity
    float inv_dt = 1.0f / config.dt;
    update_velocity_kernel<<<blocks_n, 256>>>(
        d_particles.pos, d_particles.vel,
        d_particles.predicted_pos, d_particles.old_pos,
        d_particles.removed, d_particles.fixed, inv_dt, N);
}

void ShapeMatchingGPUSolver::download_positions(std::vector<float3>& positions) {
    int N = d_particles.N;
    positions.resize(N);
    CUDA_CHECK(cudaMemcpy(positions.data(), d_particles.pos, N * sizeof(float3), cudaMemcpyDeviceToHost));
}

void ShapeMatchingGPUSolver::download_removed(std::vector<int>& removed) {
    int N = d_particles.N;
    removed.resize(N);
    CUDA_CHECK(cudaMemcpy(removed.data(), d_particles.removed, N * sizeof(int), cudaMemcpyDeviceToHost));
}

void ShapeMatchingGPUSolver::download_damage(std::vector<float>& damage) {
    int N = d_particles.N;
    damage.resize(N);
    CUDA_CHECK(cudaMemcpy(damage.data(), d_particles.damage, N * sizeof(float), cudaMemcpyDeviceToHost));
}

void ShapeMatchingGPUSolver::download_edge_broken(std::vector<int>& broken) {
    int E = d_edges.num_edges;
    broken.resize(E);
    if (E > 0) CUDA_CHECK(cudaMemcpy(broken.data(), d_edges.broken, E * sizeof(int), cudaMemcpyDeviceToHost));
}

void ShapeMatchingGPUSolver::download_edge_damage(std::vector<float>& damage) {
    int E = d_edges.num_edges;
    damage.resize(E);
    if (E > 0) CUDA_CHECK(cudaMemcpy(damage.data(), d_edges.damage, E * sizeof(float), cudaMemcpyDeviceToHost));
}

void ShapeMatchingGPUSolver::upload_removed(const std::vector<int>& removed) {
    int N = d_particles.N;
    CUDA_CHECK(cudaMemcpy(d_particles.removed, removed.data(), N * sizeof(int), cudaMemcpyHostToDevice));
}

void ShapeMatchingGPUSolver::reupload_clusters(
    const std::vector<int>& cluster_starts,
    const std::vector<int>& cluster_sizes,
    const std::vector<int>& particle_ids,
    const std::vector<float3>& rest_offsets,
    const std::vector<float3>& rest_cm)
{
    // 释放旧 buffers
    cudaFree(d_clusters.cluster_starts);
    cudaFree(d_clusters.cluster_sizes);
    cudaFree(d_clusters.particle_ids);
    cudaFree(d_clusters.rest_offsets);
    cudaFree(d_clusters.rest_cm);

    int C = (int)cluster_starts.size();
    int total = (int)particle_ids.size();
    d_clusters.num_clusters = C;
    d_clusters.total_entries = total;

    CUDA_CHECK(cudaMalloc(&d_clusters.cluster_starts, C * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clusters.cluster_sizes, C * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clusters.particle_ids, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_clusters.rest_offsets, total * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_clusters.rest_cm, C * sizeof(float3)));

    CUDA_CHECK(cudaMemcpy(d_clusters.cluster_starts, cluster_starts.data(), C * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters.cluster_sizes, cluster_sizes.data(), C * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters.particle_ids, particle_ids.data(), total * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters.rest_offsets, rest_offsets.data(), total * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_clusters.rest_cm, rest_cm.data(), C * sizeof(float3), cudaMemcpyHostToDevice));
}

void ShapeMatchingGPUSolver::reupload_edges(
    const std::vector<int>& edge_a,
    const std::vector<int>& edge_b,
    const std::vector<float>& pf_rest,
    const std::vector<float>& damage,
    const std::vector<int>& broken)
{
    if (d_edges.particle_a) {
        cudaFree(d_edges.particle_a);
        cudaFree(d_edges.particle_b);
        cudaFree(d_edges.pf_rest);
        cudaFree(d_edges.damage);
        cudaFree(d_edges.broken);
    }

    int E = (int)edge_a.size();
    d_edges.num_edges = E;

    if (E == 0) { d_edges.particle_a = nullptr; return; }

    CUDA_CHECK(cudaMalloc(&d_edges.particle_a, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edges.particle_b, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_edges.pf_rest, E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_edges.damage, E * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_edges.broken, E * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_edges.particle_a, edge_a.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edges.particle_b, edge_b.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edges.pf_rest, pf_rest.data(), E * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edges.damage, damage.data(), E * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_edges.broken, broken.data(), E * sizeof(int), cudaMemcpyHostToDevice));
}

void ShapeMatchingGPUSolver::build_hash_grid() {
    int N = d_particles.N;
    int blocks = (N + 255) / 256;
    float cell_size = config.self_collision_radius * 2.0f;

    // 1. Compute per-particle cell hash
    compute_cell_hash_kernel<<<blocks, 256>>>(
        d_particles.predicted_pos, d_particles.removed,
        d_cell_hash, d_particle_index,
        cell_size, HASH_TABLE_SIZE, N);

    // 2. Sort by hash (CUB radix sort)
    cub::DeviceRadixSort::SortPairs(
        d_cub_temp, cub_temp_bytes,
        d_cell_hash, d_cell_hash_sorted,
        d_particle_index, d_particle_index_sorted,
        N, 0, 15);  // only sort lower 15 bits for TABLE_SIZE=32768

    // 3. Find cell boundaries
    cudaMemset(d_cell_start, 0xFF, HASH_TABLE_SIZE * sizeof(unsigned int));
    cudaMemset(d_cell_end, 0xFF, HASH_TABLE_SIZE * sizeof(unsigned int));
    find_cell_bounds_kernel<<<blocks, 256>>>(
        d_cell_hash_sorted, d_cell_start, d_cell_end, N);
}

void ShapeMatchingGPUSolver::cleanup() {
    cudaFree(d_particles.pos);
    cudaFree(d_particles.vel);
    cudaFree(d_particles.predicted_pos);
    cudaFree(d_particles.old_pos);
    cudaFree(d_particles.damage);
    cudaFree(d_particles.damage_sum);
    cudaFree(d_particles.damage_count);
    cudaFree(d_particles.removed);
    cudaFree(d_particles.fixed);
    cudaFree(d_clusters.cluster_starts);
    cudaFree(d_clusters.cluster_sizes);
    cudaFree(d_clusters.particle_ids);
    cudaFree(d_clusters.rest_offsets);
    cudaFree(d_clusters.rest_cm);
    cudaFree(d_deltas);
    cudaFree(d_delta_counts);
    cudaFree(d_deltas_sc);
    cudaFree(d_delta_counts_sc);
    if (d_edges.particle_a) {
        cudaFree(d_edges.particle_a);
        cudaFree(d_edges.particle_b);
        cudaFree(d_edges.pf_rest);
        cudaFree(d_edges.damage);
        cudaFree(d_edges.broken);
    }
    if (d_cell_hash) {
        cudaFree(d_cell_hash);
        cudaFree(d_particle_index);
        cudaFree(d_cell_hash_sorted);
        cudaFree(d_particle_index_sorted);
        cudaFree(d_cell_start);
        cudaFree(d_cell_end);
        cudaFree(d_cub_temp);
    }
}
