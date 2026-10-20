#include "shape_matching_gpu.h"
#include "svd3_cuda.h"
#include "tool_channels.cuh"   // Step 1: unified tool query (sphere)
#include <cub/cub.cuh>
#include <cstdio>
#include <iostream>
#include <algorithm>

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
    float floor_y, ToolDesc tool, int knife_active,
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

    // 刀具（球形 SDF）— Step 1: via tool_query.cuh (same closed-form projection)
    if (knife_active) {
        float3 p = pred_pos[i];
        if (tool_project_out(tool, p)) pred_pos[i] = p;
    }
}

__device__ __forceinline__ void cluster_solve_body(
    int cid,
    float3* positions, int* removed, float* damage,
    int* cluster_starts, int* cluster_sizes,
    int* particle_ids, float3* rest_offsets,
    float3* deltas, int* delta_counts,
    float damage_shape_weight);

__global__ void cluster_solve_kernel(
    float3* positions, int* removed, float* damage,
    int* cluster_starts, int* cluster_sizes,
    int* particle_ids, float3* rest_offsets,
    float3* deltas, int* delta_counts,
    float damage_shape_weight,    // α: 0=no weighting (debulk), 1=PF-weighted (cut_only)
    int num_clusters)
{
    int cid = blockIdx.x;
    if (cid >= num_clusters) return;
    cluster_solve_body(cid, positions, removed, damage, cluster_starts, cluster_sizes, particle_ids,
                       rest_offsets, deltas, delta_counts, damage_shape_weight);
}

// Body of the legacy whole-cluster shape match (also used by R1 for clusters that are not split).
__device__ __forceinline__ void cluster_solve_body(
    int cid,
    float3* positions, int* removed, float* damage,
    int* cluster_starts, int* cluster_sizes,
    int* particle_ids, float3* rest_offsets,
    float3* deltas, int* delta_counts,
    float damage_shape_weight)
{
    int tid = threadIdx.x;

    int start = cluster_starts[cid];
    int size  = cluster_sizes[cid];
    if (size < 2) return;

    // 1. Weighted centroid. w_i = max(0.05, 1 - α·damage_i) — Müller 2005 effective mass.
    //   debulk α=0: w=1 always (旧行为 byte-identical, removed 仍跳过)
    //   cut_only α=1: damaged particles contribute proportionally less to cluster shape,
    //                 但仍在 cluster 里(不被排除) → SVD 不退化,健康粒子 shape-match 正常
    float cx = 0, cy = 0, cz = 0;
    float tw = 0;
    for (int j = tid; j < size; j += 32) {
        int pid = particle_ids[start + j];
        if (removed[pid]) continue;
        float w = 1.0f - damage_shape_weight * damage[pid];
        if (w < 0.05f) w = 0.05f;  // floor 防 total_w → 0
        cx += w * positions[pid].x;
        cy += w * positions[pid].y;
        cz += w * positions[pid].z;
        tw += w;
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

    // 2a. Weighted rest centroid (same weights as cm — Müller 2005 mass-weighted)
    float qcx = 0, qcy = 0, qcz = 0;
    for (int j = tid; j < size; j += 32) {
        int pid = particle_ids[start + j];
        if (removed[pid]) continue;
        float w = 1.0f - damage_shape_weight * damage[pid];
        if (w < 0.05f) w = 0.05f;
        qcx += w * rest_offsets[start + j].x;
        qcy += w * rest_offsets[start + j].y;
        qcz += w * rest_offsets[start + j].z;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        qcx += __shfl_down_sync(0xFFFFFFFF, qcx, offset);
        qcy += __shfl_down_sync(0xFFFFFFFF, qcy, offset);
        qcz += __shfl_down_sync(0xFFFFFFFF, qcz, offset);
    }
    qcx = __shfl_sync(0xFFFFFFFF, qcx, 0) / total_w;
    qcy = __shfl_sync(0xFFFFFFFF, qcy, 0) / total_w;
    qcz = __shfl_sync(0xFFFFFFFF, qcz, 0) / total_w;

    // 2b. Weighted covariance A_pq (Müller 2005: A = Σ m_i (p_i - cm)(q_i - qcm)^T)
    float a11=0,a12=0,a13=0,a21=0,a22=0,a23=0,a31=0,a32=0,a33=0;
    for (int j = tid; j < size; j += 32) {
        int pid = particle_ids[start + j];
        if (removed[pid]) continue;
        float w = 1.0f - damage_shape_weight * damage[pid];
        if (w < 0.05f) w = 0.05f;
        float px = positions[pid].x - cx;
        float py = positions[pid].y - cy;
        float pz = positions[pid].z - cz;
        float qx = rest_offsets[start + j].x - qcx;
        float qy = rest_offsets[start + j].y - qcy;
        float qz = rest_offsets[start + j].z - qcz;
        a11 += w*px*qx; a12 += w*px*qy; a13 += w*px*qz;
        a21 += w*py*qx; a22 += w*py*qy; a23 += w*py*qz;
        a31 += w*pz*qx; a32 += w*pz*qy; a33 += w*pz*qz;
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

    // 4. Goal scatter — UNWEIGHTED: every non-removed particle gets goal applied.
    //    apply_kernel does damage-based stiffness gating (1 - 0.95*damage),
    //    no need to also weight here (would double-count damage gating).
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

// ============== R1: per-cluster local connectivity (sub-clusters) ==============
// A cluster is split by a cut when its OWN members are no longer connected through intact
// cutting edges that lie inside the cluster — whether or not the whole body has separated.
// cluster_label_kernel: one warp per cluster, label propagation (min) + pointer jumping over the
// cluster's internal intact edges; labels[entry] = local index of the sub-cluster representative,
// -1 for removed members. Recomputed every step from broken[] / removed[].
__global__ void cluster_label_kernel(
    const int* cluster_starts, const int* cluster_sizes, const int* particle_ids, const int* removed,
    const int* cl_edge_start, const int* cl_edge_eid, const int2* cl_edge_uv, const int* broken,
    int* labels, int num_clusters, const int* topo_dirty, int* cluster_multi)
{
    extern __shared__ int lab[];
    int cid = blockIdx.x, tid = threadIdx.x;
    if (cid >= num_clusters || *topo_dirty == 0) return;   // topology unchanged since last labelling
    int start = cluster_starts[cid], size = cluster_sizes[cid];
    for (int j = tid; j < size; j += 32) lab[j] = removed[particle_ids[start + j]] ? -1 : j;
    __syncwarp();
    int e0 = cl_edge_start[cid], e1 = cl_edge_start[cid + 1];
    bool changed = true;
    while (__any_sync(0xFFFFFFFF, changed)) {
        changed = false;
        for (int k = e0 + tid; k < e1; k += 32) {
            if (broken[cl_edge_eid[k]]) continue;
            int2 uv = cl_edge_uv[k];
            int lu = lab[uv.x], lv = lab[uv.y];
            if (lu < 0 || lv < 0 || lu == lv) continue;
            int m = min(lu, lv);
            atomicMin(&lab[uv.x], m); atomicMin(&lab[uv.y], m);
            changed = true;
        }
        __syncwarp();
        for (int j = tid; j < size; j += 32) {           // pointer jumping
            int l = lab[j];
            if (l >= 0) { int ll = lab[l]; if (ll < l) { lab[j] = ll; changed = true; } }
        }
        __syncwarp();
    }
    bool multi = false;
    int first = -1;
    for (int j = 0; j < size; j++) { if (lab[j] >= 0) { first = lab[j]; break; } }
    for (int j = tid; j < size; j += 32) {
        labels[start + j] = lab[j];
        if (lab[j] >= 0 && lab[j] != first) multi = true;
    }
    multi = __any_sync(0xFFFFFFFF, multi);
    if (tid == 0) cluster_multi[cid] = multi ? 1 : 0;
}

__global__ void clear_flag_kernel(int* flag) { *flag = 0; }

// Shape matching per sub-cluster (R1). Same arithmetic as cluster_solve_kernel restricted to the
// members of one label; the rest shape of a sub-cluster is its members' ORIGINAL rest offsets
// (re-centred), i.e. the reference configuration. A cluster with a single label uses the legacy
// threshold (total weight >= 1.5) and therefore reproduces cluster_solve_kernel bit for bit;
// in a split cluster, sub-clusters with fewer than min_sub members get no goal from it (their
// particles keep their other memberships).
__global__ void cluster_solve_sub_kernel(
    float3* positions, int* removed, float* damage,
    int* cluster_starts, int* cluster_sizes,
    int* particle_ids, float3* rest_offsets, const int* labels,
    float3* deltas, int* delta_counts,
    float damage_shape_weight, int min_sub,
    int num_clusters, const int* cluster_multi)
{
    int cid = blockIdx.x;
    if (cid >= num_clusters) return;
    if (!cluster_multi[cid]) {          // not split: the legacy whole-cluster shape match
        cluster_solve_body(cid, positions, removed, damage, cluster_starts, cluster_sizes, particle_ids,
                           rest_offsets, deltas, delta_counts, damage_shape_weight);
        return;
    }
    int tid = threadIdx.x;
    int start = cluster_starts[cid];
    int size  = cluster_sizes[cid];
    if (size < 2) return;

    // number of labels: 1 → legacy path
    int l0 = 0x7fffffff;
    for (int j = tid; j < size; j += 32) { int l = labels[start + j]; if (l >= 0 && l < l0) l0 = l; }
    for (int o = 16; o > 0; o >>= 1) l0 = min(l0, __shfl_down_sync(0xFFFFFFFF, l0, o));
    l0 = __shfl_sync(0xFFFFFFFF, l0, 0);
    if (l0 == 0x7fffffff) return;
    bool multi = false;
    for (int j = tid; j < size; j += 32) { int l = labels[start + j]; if (l >= 0 && l != l0) multi = true; }
    multi = __any_sync(0xFFFFFFFF, multi);

    int prev = -1;
    while (true) {
        int L = 0x7fffffff;
        for (int j = tid; j < size; j += 32) { int l = labels[start + j]; if (l > prev && l < L) L = l; }
        for (int o = 16; o > 0; o >>= 1) L = min(L, __shfl_down_sync(0xFFFFFFFF, L, o));
        L = __shfl_sync(0xFFFFFFFF, L, 0);
        if (L == 0x7fffffff) break;
        prev = L;

        int cnt = 0;
        float cx = 0, cy = 0, cz = 0, tw = 0;
        for (int j = tid; j < size; j += 32) {
            int pid = particle_ids[start + j];
            if (removed[pid] || labels[start + j] != L) continue;
            float w = 1.0f - damage_shape_weight * damage[pid];
            if (w < 0.05f) w = 0.05f;
            cx += w * positions[pid].x; cy += w * positions[pid].y; cz += w * positions[pid].z;
            tw += w; cnt++;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            cx += __shfl_down_sync(0xFFFFFFFF, cx, offset);
            cy += __shfl_down_sync(0xFFFFFFFF, cy, offset);
            cz += __shfl_down_sync(0xFFFFFFFF, cz, offset);
            tw += __shfl_down_sync(0xFFFFFFFF, tw, offset);
            cnt += __shfl_down_sync(0xFFFFFFFF, cnt, offset);
        }
        float total_w = __shfl_sync(0xFFFFFFFF, tw, 0);
        cnt = __shfl_sync(0xFFFFFFFF, cnt, 0);
        if (multi ? (cnt < min_sub) : (total_w < 1.5f)) continue;
        cx = __shfl_sync(0xFFFFFFFF, cx, 0) / total_w;
        cy = __shfl_sync(0xFFFFFFFF, cy, 0) / total_w;
        cz = __shfl_sync(0xFFFFFFFF, cz, 0) / total_w;

        float qcx = 0, qcy = 0, qcz = 0;
        for (int j = tid; j < size; j += 32) {
            int pid = particle_ids[start + j];
            if (removed[pid] || labels[start + j] != L) continue;
            float w = 1.0f - damage_shape_weight * damage[pid];
            if (w < 0.05f) w = 0.05f;
            qcx += w * rest_offsets[start + j].x; qcy += w * rest_offsets[start + j].y; qcz += w * rest_offsets[start + j].z;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            qcx += __shfl_down_sync(0xFFFFFFFF, qcx, offset);
            qcy += __shfl_down_sync(0xFFFFFFFF, qcy, offset);
            qcz += __shfl_down_sync(0xFFFFFFFF, qcz, offset);
        }
        qcx = __shfl_sync(0xFFFFFFFF, qcx, 0) / total_w;
        qcy = __shfl_sync(0xFFFFFFFF, qcy, 0) / total_w;
        qcz = __shfl_sync(0xFFFFFFFF, qcz, 0) / total_w;

        float a11=0,a12=0,a13=0,a21=0,a22=0,a23=0,a31=0,a32=0,a33=0;
        for (int j = tid; j < size; j += 32) {
            int pid = particle_ids[start + j];
            if (removed[pid] || labels[start + j] != L) continue;
            float w = 1.0f - damage_shape_weight * damage[pid];
            if (w < 0.05f) w = 0.05f;
            float px = positions[pid].x - cx, py = positions[pid].y - cy, pz = positions[pid].z - cz;
            float qx = rest_offsets[start + j].x - qcx, qy = rest_offsets[start + j].y - qcy, qz = rest_offsets[start + j].z - qcz;
            a11 += w*px*qx; a12 += w*px*qy; a13 += w*px*qz;
            a21 += w*py*qx; a22 += w*py*qy; a23 += w*py*qz;
            a31 += w*pz*qx; a32 += w*pz*qy; a33 += w*pz*qz;
        }
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
        float u11,u12,u13,u21,u22,u23,u31,u32,u33;
        float s11,s12,s13,s21,s22,s23,s31,s32,s33;
        float v11,v12,v13,v21,v22,v23,v31,v32,v33;
        float r11,r12,r13,r21,r22,r23,r31,r32,r33;
        if (tid == 0) {
            svd(a11,a12,a13,a21,a22,a23,a31,a32,a33,
                u11,u12,u13,u21,u22,u23,u31,u32,u33,
                s11,s12,s13,s21,s22,s23,s31,s32,s33,
                v11,v12,v13,v21,v22,v23,v31,v32,v33);
            multAB(u11,u12,u13,u21,u22,u23,u31,u32,u33,
                   v11,v21,v31,v12,v22,v32,v13,v23,v33,
                   r11,r12,r13,r21,r22,r23,r31,r32,r33);
            float det = r11*(r22*r33-r23*r32) - r12*(r21*r33-r23*r31) + r13*(r21*r32-r22*r31);
            if (det < 0) {
                u13 = -u13; u23 = -u23; u33 = -u33;
                multAB(u11,u12,u13,u21,u22,u23,u31,u32,u33,
                       v11,v21,v31,v12,v22,v32,v13,v23,v33,
                       r11,r12,r13,r21,r22,r23,r31,r32,r33);
            }
        }
        r11 = __shfl_sync(0xFFFFFFFF, r11, 0); r12 = __shfl_sync(0xFFFFFFFF, r12, 0); r13 = __shfl_sync(0xFFFFFFFF, r13, 0);
        r21 = __shfl_sync(0xFFFFFFFF, r21, 0); r22 = __shfl_sync(0xFFFFFFFF, r22, 0); r23 = __shfl_sync(0xFFFFFFFF, r23, 0);
        r31 = __shfl_sync(0xFFFFFFFF, r31, 0); r32 = __shfl_sync(0xFFFFFFFF, r32, 0); r33 = __shfl_sync(0xFFFFFFFF, r33, 0);

        for (int j = tid; j < size; j += 32) {
            int pid = particle_ids[start + j];
            if (removed[pid] || labels[start + j] != L) continue;
            float qx = rest_offsets[start + j].x - qcx, qy = rest_offsets[start + j].y - qcy, qz = rest_offsets[start + j].z - qcz;
            float gx = r11*qx + r12*qy + r13*qz + cx;
            float gy = r21*qx + r22*qy + r23*qz + cy;
            float gz = r31*qx + r32*qy + r33*qz + cz;
            atomicAdd(&deltas[pid].x, gx - positions[pid].x);
            atomicAdd(&deltas[pid].y, gy - positions[pid].y);
            atomicAdd(&deltas[pid].z, gz - positions[pid].z);
            atomicAdd(&delta_counts[pid], 1);
        }
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

// Per-particle stiffness variant (2026-05-24).
// Same physics as apply_kernel but stiffness sourced from per-particle array.
// Used when ShapeMatchingGPUSolver.per_particle_stiffness=true AND
// d_stiffness_per_particle != nullptr.
__global__ void apply_kernel_per_particle_stiff(
    float3* pred_pos, float3* deltas, int* delta_counts,
    float* damage, int* removed, int* fixed,
    const float* stiffness_per_particle, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i] || fixed[i] || delta_counts[i] == 0) return;

    float k_i = stiffness_per_particle[i];
    float local_stiff = k_i * (1.0f - damage[i] * 0.95f);
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

// Edge damage from the tool's CUT channel (tool_channels.cuh). One kernel for every intruder set
// (sphere / grid-SDF surface, sharp-edge curve); replaces update_edge_damage_kernel (sphere),
// tool_sdf_damage_kernel (grid) and sharp_edge_damage_kernel (curve) with the same arithmetic.
// Material fields apply to every channel:
//   alpha_min_per_edge  nullable per-edge α (nullptr → alpha_min_scalar)
//   damage_model        0 = sharpness (legacy), 1 = toughness (growth rate / α)
__global__ void tool_damage_kernel(
    float3* pred_pos, int* removed,
    int* edge_a, int* edge_b, float* pf_rest, float* damage, int* broken,
    CutChannel channel,
    float alpha_min_scalar,
    const float* alpha_min_per_edge,
    float damage_growth_rate, int num_edges,
    int damage_model, int* topo_dirty)
{
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges || broken[eid]) return;

    int a = edge_a[eid], b = edge_b[eid];
    if (removed[a] || removed[b]) return;

    float alpha_min = (alpha_min_per_edge != nullptr) ? alpha_min_per_edge[eid]
                                                       : alpha_min_scalar;

    // 边中点
    float3 mid;
    mid.x = (pred_pos[a].x + pred_pos[b].x) * 0.5f;
    mid.y = (pred_pos[a].y + pred_pos[b].y) * 0.5f;
    mid.z = (pred_pos[a].z + pred_pos[b].z) * 0.5f;

    // 边半径（circumsphere）
    float ex = pred_pos[a].x - pred_pos[b].x;
    float ey = pred_pos[a].y - pred_pos[b].y;
    float ez = pred_pos[a].z - pred_pos[b].z;
    float radius = sqrtf(ex*ex + ey*ey + ez*ez) * 0.5f;
    if (radius < 1e-10f) return;

    // intruder distance of the cut channel (also the search band: outside → no update)
    float sdist;
    if (!cut_channel_sdist(channel, mid, radius, sdist)) return;
    float pf = 1.0f / (1.0f + expf(-alpha_min * sdist / radius));

    // Damage 更新 — capped growth (rate=1.0 → 原 max-update 行为, byte-identical debulk)
    float rest = pf_rest[eid];
    if (rest > 0.01f) {
        float target = fmaxf(0.0f, 1.0f - pf / rest);
        float gap = target - damage[eid];
        if (gap > 0.0f) {
            // Toughness model (damage_model=1): high α slows damage growth.
            // Sharpness model (damage_model=0): unchanged (legacy).
            float effective_rate = damage_growth_rate;
            if (damage_model == 1) {
                effective_rate = damage_growth_rate / fmaxf(alpha_min, 0.1f);
            }
            float inc = gap < effective_rate ? gap : effective_rate;
            damage[eid] = damage[eid] + inc;
        }
    }

    // 断裂检查
    if (damage[eid] > 0.5f) {
        broken[eid] = 1;
        if (topo_dirty) *topo_dirty = 1;     // R1: sub-cluster labels must be recomputed
    }
}

// R3 — straddle fracture. The tool body may not coexist with intact material: an intact edge
// whose SEGMENT the tool separates (its endpoints lie on opposite sides of the tangent plane at
// the deepest intrusion point, see tool_edge_contact_kernel) is under fracture loading. Damage
// uses the same P(F) model with the edge segment as the element and the tool solid as the
// intruder: sdist = phi_seg (penetration of the tool into the segment, < 0), pf = sigma(alpha *
// sdist / r_e), and the same capped growth / damage model as tool_damage_kernel. Same-side
// intrusions (the tool presses into the edge from one side) are contact, not fracture.
__global__ void tool_straddle_damage_kernel(
    float3* pred_pos, int* removed,
    int* edge_a, int* edge_b, float* pf_rest, float* damage, int* broken,
    ToolDesc tool, float sample_h, float eps,
    float alpha_min_scalar, const float* alpha_min_per_edge,
    float damage_growth_rate, int num_edges, int damage_model, int* topo_dirty)
{
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges || broken[eid]) return;
    int a = edge_a[eid], b = edge_b[eid];
    if (removed[a] || removed[b]) return;
    float3 pa = pred_pos[a], pb = pred_pos[b];
    float t, phi; float3 n;
    if (!tool_edge_deepest(tool, pa, pb, sample_h, t, phi, n)) return;
    float3 x = make_float3(pa.x + t*(pb.x - pa.x), pa.y + t*(pb.y - pa.y), pa.z + t*(pb.z - pa.z));
    // Separation: the tool lies BETWEEN the particles when their outward tool normals point into
    // opposite half-spaces (n_a . n_b < 0): crossing a blade or a needle shaft gives ~ -1, an edge
    // lying on a face +1. (Not the tangent plane at the deepest point: at an interior minimum of
    // phi along the segment the gradient is orthogonal to the segment, so that test degenerates.)
    // Topology is decided only beyond the representation's resolution: penetration deeper than eps.
    (void)x;
    if (phi > -eps) return;
    float3 na = tool_query(tool, pa, true).n, nb = tool_query(tool, pb, true).n;
    if (na.x*nb.x + na.y*nb.y + na.z*nb.z >= 0.0f) return;   // same side (or a normal undefined)
    float ex = pa.x - pb.x, ey = pa.y - pb.y, ez = pa.z - pb.z;
    float radius = sqrtf(ex*ex + ey*ey + ez*ez) * 0.5f;
    if (radius < 1e-10f) return;
    float alpha_min = (alpha_min_per_edge != nullptr) ? alpha_min_per_edge[eid] : alpha_min_scalar;
    float pf = 1.0f / (1.0f + expf(-alpha_min * phi / radius));
    float rest = pf_rest[eid];
    if (rest > 0.01f) {
        float target = fmaxf(0.0f, 1.0f - pf / rest);
        float gap = target - damage[eid];
        if (gap > 0.0f) {
            float effective_rate = damage_growth_rate;
            if (damage_model == 1) effective_rate = damage_growth_rate / fmaxf(alpha_min, 0.1f);
            damage[eid] += gap < effective_rate ? gap : effective_rate;
        }
    }
    if (damage[eid] > 0.5f) {
        broken[eid] = 1;
        if (topo_dirty) *topo_dirty = 1;
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
// Per-particle τ support: if tau_per_particle != nullptr, read per-particle threshold;
// else use scalar threshold (legacy, byte-identical).
__global__ void aspirate_kernel(
    float3* pred_pos, float* particle_damage, int* removed,
    ToolDesc tool,
    float threshold_scalar,
    const float* tau_per_particle,    // nullable: nullptr → use scalar
    int N, int* topo_dirty)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i]) return;

    if (!tool_contains(tool, pred_pos[i])) return;   // Step 1: tool op (sphere: |d|^2 <= R^2)

    float threshold = (tau_per_particle != nullptr) ? tau_per_particle[i]
                                                    : threshold_scalar;
    if (particle_damage[i] > threshold) {
        removed[i] = 1;
        if (topo_dirty) *topo_dirty = 1;
    }
}

// ============== Host Functions ==============

// Step 2: grid-SDF tool description for tool_query.cuh (built per launch on the host; the
// rotation is copied by value).
static ToolDesc grid_tool_desc(const SimConfigGPU& c, const SDFData& sdf) {
    ToolDesc t = {};
    t.type = 1; t.pos = c.knife_pos; t.radius = c.knife_radius; t.reach = c.aspiration_reach;
    for (int k = 0; k < 9; k++) t.rot[k] = c.knife_rot[k];
    t.sdf = sdf.d_sdf; t.res_x = sdf.res_x; t.res_y = sdf.res_y; t.res_z = sdf.res_z;
    t.origin = sdf.origin; t.cell = sdf.cell_size;
    t.normal_mode = c.sdf_normal_mode;
    return t;
}

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
    // Edge-contact delta buffers (Step 3c, used only when config.edge_contact)
    CUDA_CHECK(cudaMalloc(&d_deltas_ec, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_delta_counts_ec, N * sizeof(int)));

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

// Upload per-particle τ (aspiration threshold) field. Per-particle gradient
// for toughness experiments. Length must equal d_particles.N.
void ShapeMatchingGPUSolver::upload_per_particle_tau(const std::vector<float>& tau_per_particle) {
    int N = d_particles.N;
    if ((int)tau_per_particle.size() != N) {
        std::cerr << "[GPU] upload_per_particle_tau size mismatch: "
                  << tau_per_particle.size() << " vs N=" << N << std::endl;
        return;
    }
    if (d_tau_per_particle == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_tau_per_particle, N * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_tau_per_particle, tau_per_particle.data(),
                           N * sizeof(float), cudaMemcpyHostToDevice));
    tau_per_particle_cpu = tau_per_particle;
    printf("[GPU] per-particle τ uploaded: %d particles, range [%.3f, %.3f]\n",
           N, *std::min_element(tau_per_particle.begin(), tau_per_particle.end()),
           *std::max_element(tau_per_particle.begin(), tau_per_particle.end()));
}

// Upload per-particle stiffness field (2026-05-24). Per-particle elasticity
// heterogeneity for forward sensitivity probe. Length must equal d_particles.N.
void ShapeMatchingGPUSolver::upload_per_particle_stiffness(const std::vector<float>& stiff_per_particle) {
    int N = d_particles.N;
    if ((int)stiff_per_particle.size() != N) {
        std::cerr << "[GPU] upload_per_particle_stiffness size mismatch: "
                  << stiff_per_particle.size() << " vs N=" << N << std::endl;
        return;
    }
    if (d_stiffness_per_particle == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_stiffness_per_particle, N * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_stiffness_per_particle, stiff_per_particle.data(),
                           N * sizeof(float), cudaMemcpyHostToDevice));
    stiffness_per_particle_cpu = stiff_per_particle;
    printf("[GPU] per-particle stiffness uploaded: %d particles, range [%.3f, %.3f]\n",
           N, *std::min_element(stiff_per_particle.begin(), stiff_per_particle.end()),
           *std::max_element(stiff_per_particle.begin(), stiff_per_particle.end()));
}

// Upload (or re-upload) per-edge α field. Called by viewer after init_edges if
// config.per_edge_alpha = true. Length must equal d_edges.num_edges.
void ShapeMatchingGPUSolver::upload_per_edge_alpha(const std::vector<float>& alpha_per_edge) {
    int E = d_edges.num_edges;
    if ((int)alpha_per_edge.size() != E) {
        std::cerr << "[GPU] upload_per_edge_alpha size mismatch: "
                  << alpha_per_edge.size() << " vs num_edges=" << E << std::endl;
        return;
    }
    if (d_edges.alpha_min_per_edge == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_edges.alpha_min_per_edge, E * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_edges.alpha_min_per_edge, alpha_per_edge.data(),
                           E * sizeof(float), cudaMemcpyHostToDevice));
    alpha_per_edge_cpu = alpha_per_edge;   // keep CPU copy for viewer
    printf("[GPU] per-edge α uploaded: %d edges, range [%.3f, %.3f]\n",
           E, *std::min_element(alpha_per_edge.begin(), alpha_per_edge.end()),
           *std::max_element(alpha_per_edge.begin(), alpha_per_edge.end()));
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

// Tool SDF collision: transform particle to tool local frame, query SDF
__global__ void tool_sdf_collision_kernel(
    float3* pred_pos, int* removed, int* fixed, ToolDesc tool, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || removed[i] || fixed[i]) return;
    // Step 2: via tool_query.cuh (legacy push arithmetic; normal per tool.normal_mode)
    float3 p = pred_pos[i];
    if (tool_project_out(tool, p)) pred_pos[i] = p;
}

// Step 3c — edge contact (FleX-style constraint group with its own delta buffer).
// For every intact material edge whose interior dips into the tool (tool_edge_deepest) with both
// particles on the same side of it,
// push the deepest point x(t) out along the tool normal by -phi. The correction is shared by the
// two particles with the point-on-edge weights wa = 1 - t, wb = t, scaled by s = 1/(wa^2 + wb^2)
// so that x(t) itself moves by exactly -phi * n (minimal-norm distribution, equal masses).
// Edges with a fixed / removed particle and broken edges are skipped.
__global__ void tool_edge_contact_kernel(
    float3* pred_pos, int* removed, int* fixed,
    int* edge_a, int* edge_b, int* broken,
    ToolDesc tool, float sample_h,
    float3* deltas, int* counts, int num_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges || broken[e]) return;
    int a = edge_a[e], b = edge_b[e];
    if (removed[a] || removed[b] || fixed[a] || fixed[b]) return;
    float t, phi; float3 n;
    if (!tool_edge_deepest(tool, pred_pos[a], pred_pos[b], sample_h, t, phi, n)) return;
    // Straddling edge: the tool lies BETWEEN the two particles — the endpoints are on opposite
    // sides of the tangent plane at the deepest point. No non-penetrating state is reachable by
    // moving the edge locally; pushing it out drags the far particle through the tool against the
    // particle collision. Such an edge can only be resolved by fracture (sharp feature: cut
    // channel; blunt body: strain tearing), so contact leaves it alone. (Uses only the normal at
    // the deepest point, which is inside the baked box; endpoint normals are undefined outside it.)
    float3 pa = pred_pos[a], pb = pred_pos[b];
    float3 x = make_float3(pa.x + t*(pb.x - pa.x), pa.y + t*(pb.y - pa.y), pa.z + t*(pb.z - pa.z));
    float sa = (pa.x - x.x)*n.x + (pa.y - x.y)*n.y + (pa.z - x.z)*n.z;
    float sb = (pb.x - x.x)*n.x + (pb.y - x.y)*n.y + (pb.z - x.z)*n.z;
    if (sa * sb < 0.0f) return;
    float wa = 1.0f - t, wb = t;
    float sc = -phi / (wa * wa + wb * wb);
    atomicAdd(&deltas[a].x, wa * sc * n.x); atomicAdd(&deltas[a].y, wa * sc * n.y); atomicAdd(&deltas[a].z, wa * sc * n.z);
    atomicAdd(&counts[a], 1);
    atomicAdd(&deltas[b].x, wb * sc * n.x); atomicAdd(&deltas[b].y, wb * sc * n.y); atomicAdd(&deltas[b].z, wb * sc * n.z);
    atomicAdd(&counts[b], 1);
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

    // Cut channel (tool_channels.cuh), same priority as before Step 3:
    //   1. use_sharp_edge_damage=1 → CUT_CURVE on the sharp features (set_cut_segments; none → no cut)
    //   2. mesh SDF tool           → CUT_SURFACE on the grid SDF (legacy volumetric damage)
    //   3. fallback                → CUT_SURFACE on the analytic sphere
    CutChannel ch = {};
    if (config.use_sharp_edge_damage) {
        sync_cut_segments();
        ch.kind = CUT_CURVE;
        ch.tool.pos = config.knife_pos;
        memcpy(ch.tool.rot, config.knife_rot, sizeof(ch.tool.rot));
        ch.seg = d_cut_segments;
        ch.num_seg = (int)h_cut_segments.size() / 2;
        if (ch.num_seg == 0) ch.kind = -1;    // blunt tool: no sharp feature → no feature cut
    } else if (config.tool_type == 1 && d_tool_sdf.active && d_tool_sdf.d_sdf) {
        ch.kind = CUT_SURFACE;
        ch.tool = grid_tool_desc(config, d_tool_sdf);
        ch.subtract_re = config.sdf_pf_edge_radius;
    } else {
        ch.kind = CUT_SURFACE;
        ch.tool = make_sphere_tool(config.knife_pos, config.knife_radius);
        ch.subtract_re = 1;
    }
    // Material fields (per-edge α, damage model) apply to every channel.
    const float* alpha_arr = (per_edge_alpha && d_edges.alpha_min_per_edge != nullptr)
                              ? d_edges.alpha_min_per_edge : nullptr;
    if (ch.kind >= 0)
    tool_damage_kernel<<<blocks_e, 256>>>(
        d_particles.predicted_pos, d_particles.removed,
        d_edges.particle_a, d_edges.particle_b,
        d_edges.pf_rest, d_edges.damage, d_edges.broken,
        ch, alpha_min, alpha_arr,
        config.damage_growth_rate, E,
        damage_model_mode, local_split ? d_topo_dirty : nullptr);

    // R3: intact edges the tool body separates (straddle) are under fracture loading
    if (config.straddle_fracture) {
        ToolDesc tool;
        float h = 0.0f, eps;
        if (config.tool_type == 1 && d_tool_sdf.active && d_tool_sdf.d_sdf) {
            tool = grid_tool_desc(config, d_tool_sdf);
            h = 0.5f * fminf(tool.cell.x, fminf(tool.cell.y, tool.cell.z));
            eps = h;                          // grid: half the smallest SDF cell
        } else {
            tool = make_sphere_tool(config.knife_pos, config.knife_radius);
            eps = 1e-5f * config.knife_radius;  // analytic sphere: float precision only
        }
        tool_straddle_damage_kernel<<<blocks_e, 256>>>(
            d_particles.predicted_pos, d_particles.removed,
            d_edges.particle_a, d_edges.particle_b,
            d_edges.pf_rest, d_edges.damage, d_edges.broken,
            tool, h, eps, alpha_min, alpha_arr,
            config.damage_growth_rate, E, damage_model_mode, local_split ? d_topo_dirty : nullptr);
    }
}

// Upload the CUT_CURVE features when set_cut_segments changed them.
void ShapeMatchingGPUSolver::sync_cut_segments() {
    if (!cut_segments_dirty) return;
    size_t n = h_cut_segments.size();
    if (n > d_cut_segments_cap) {
        if (d_cut_segments) cudaFree(d_cut_segments);
        CUDA_CHECK(cudaMalloc(&d_cut_segments, n * sizeof(float3)));
        d_cut_segments_cap = n;
    }
    if (n > 0) CUDA_CHECK(cudaMemcpy(d_cut_segments, h_cut_segments.data(), n * sizeof(float3), cudaMemcpyHostToDevice));
    cut_segments_dirty = false;
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

    // Aspirate channel: the tool's contains test (grid SDF or sphere, incl. aspiration_reach for
    // both) and the material threshold (per-particle τ for every tool type).
    ToolDesc tool = (config.tool_type == 1 && d_tool_sdf.active && d_tool_sdf.d_sdf)
                        ? grid_tool_desc(config, d_tool_sdf)
                        : make_sphere_tool(config.knife_pos, config.knife_radius, config.aspiration_reach);
    const float* tau_arr = (per_particle_tau && d_tau_per_particle != nullptr)
                            ? d_tau_per_particle : nullptr;
    aspirate_kernel<<<blocks_n, 256>>>(
        d_particles.predicted_pos, d_particles.damage, d_particles.removed,
        tool, aspiration_threshold, tau_arr, N, local_split ? d_topo_dirty : nullptr);
}

// ============== R1: sub-cluster setup ==============
void ShapeMatchingGPUSolver::setup_local_split(
    const std::vector<int>& cluster_starts, const std::vector<int>& cluster_sizes,
    const std::vector<int>& particle_ids,
    const std::vector<int>& edge_a, const std::vector<int>& edge_b)
{
    int C = (int)cluster_starts.size(), N = d_particles.N, E = (int)edge_a.size();
    std::vector<std::vector<int>> inc(N);                 // particle -> incident cutting edges
    for (int e = 0; e < E; e++) { inc[edge_a[e]].push_back(e); inc[edge_b[e]].push_back(e); }
    std::vector<int> local(N, -1);                         // particle -> member index in current cluster
    std::vector<int> start(C + 1, 0), eid; std::vector<int2> uv;
    max_cluster_size = 0;
    for (int c = 0; c < C; c++) {
        start[c] = (int)eid.size();
        int s0 = cluster_starts[c], n = cluster_sizes[c];
        max_cluster_size = std::max(max_cluster_size, n);
        for (int j = 0; j < n; j++) local[particle_ids[s0 + j]] = j;
        for (int j = 0; j < n; j++) {
            int p = particle_ids[s0 + j];
            for (int e : inc[p]) {
                if (edge_a[e] != p) continue;              // each edge once (from its a-end)
                int q = edge_b[e];
                if (local[q] < 0) continue;                // other end not in this cluster
                eid.push_back(e); uv.push_back(make_int2(j, local[q]));
            }
        }
        for (int j = 0; j < n; j++) local[particle_ids[s0 + j]] = -1;
    }
    start[C] = (int)eid.size();
    CUDA_CHECK(cudaMalloc(&d_cl_edge_start, (C + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cl_edge_eid, std::max<size_t>(1, eid.size()) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cl_edge_uv, std::max<size_t>(1, uv.size()) * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_labels, d_clusters.total_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_topo_dirty, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_multi, C * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cluster_multi, 0, C * sizeof(int)));
    int one = 1;
    CUDA_CHECK(cudaMemcpy(d_topo_dirty, &one, sizeof(int), cudaMemcpyHostToDevice));   // first labelling
    CUDA_CHECK(cudaMemcpy(d_cl_edge_start, start.data(), (C + 1) * sizeof(int), cudaMemcpyHostToDevice));
    if (!eid.empty()) {
        CUDA_CHECK(cudaMemcpy(d_cl_edge_eid, eid.data(), eid.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cl_edge_uv, uv.data(), uv.size() * sizeof(int2), cudaMemcpyHostToDevice));
    }
    local_split = true;
    update_cluster_labels();
    printf("[R1] local cluster split ON: %d clusters, %zu internal cluster edges (%.1f / cluster), max cluster size %d\n",
           C, eid.size(), (double)eid.size() / std::max(C, 1), max_cluster_size);
}

void ShapeMatchingGPUSolver::download_labels(std::vector<int>& labels) {
    labels.clear();
    if (!local_split) return;
    labels.resize(d_clusters.total_entries);
    CUDA_CHECK(cudaMemcpy(labels.data(), d_labels, labels.size() * sizeof(int), cudaMemcpyDeviceToHost));
}

void ShapeMatchingGPUSolver::update_cluster_labels() {
    int C = d_clusters.num_clusters;
    cluster_label_kernel<<<C, 32, max_cluster_size * sizeof(int)>>>(
        d_clusters.cluster_starts, d_clusters.cluster_sizes, d_clusters.particle_ids, d_particles.removed,
        d_cl_edge_start, d_cl_edge_eid, d_cl_edge_uv, d_edges.broken, d_labels, C, d_topo_dirty, d_cluster_multi);
    clear_flag_kernel<<<1, 1>>>(d_topo_dirty);
}

// ============== Simulation Step ==============

void ShapeMatchingGPUSolver::step() {
    int N = d_particles.N;
    int C = d_clusters.num_clusters;
    int blocks_n = (N + 255) / 256;

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
        config.floor_y, make_sphere_tool(config.knife_pos, config.knife_radius),
        (config.tool_type == 0) ? config.knife_active : 0,     // sphere only if tool_type==0
        config.friction_s, config.friction_k, N);              // Coulomb 摩擦（复用 bone SDF 的 μ_s/μ_k）

    // 2a. Tool SDF collision (if mesh tool)
    if (config.tool_type == 1 && config.knife_active && d_tool_sdf.active && d_tool_sdf.d_sdf) {
        tool_sdf_collision_kernel<<<blocks_n, 256>>>(
            d_particles.predicted_pos, d_particles.removed, d_particles.fixed,
            grid_tool_desc(config, d_tool_sdf), N);
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

    // R1: sub-cluster labels from the current broken edges / removed particles
    if (local_split) update_cluster_labels();

    // 3. Jacobi 迭代 — FleX constraint groups (Macklin 2014 Sec 4):
    //    Group 1: shape-matching (cluster_solve → apply) — soft constraint, Jacobi-averaged
    //    Group 2: self-collision (self_collision_kernel → apply_sc) — separate delta buffer
    //             so contacts aren't diluted by ~60 shape-match contributions per particle
    //    Group 3: hard contacts (knife/floor/tool SDF/bone SDF) — direct write to pred_pos
    for (int iter = 0; iter < config.num_iterations; iter++) {
        // ---- Group 1: shape matching ----
        clear_deltas_kernel<<<blocks_n, 256>>>(d_deltas, d_delta_counts, N);

        // Solve all clusters（每簇一个 block，32 线程/warp）; R1: per sub-cluster
        if (local_split) {
            cluster_solve_sub_kernel<<<C, 32>>>(
                d_particles.predicted_pos, d_particles.removed, d_particles.damage,
                d_clusters.cluster_starts, d_clusters.cluster_sizes,
                d_clusters.particle_ids, d_clusters.rest_offsets, d_labels,
                d_deltas, d_delta_counts,
                config.damage_shape_weight, 4,
                C, d_cluster_multi);
        } else {
            cluster_solve_kernel<<<C, 32>>>(
                d_particles.predicted_pos, d_particles.removed, d_particles.damage,
                d_clusters.cluster_starts, d_clusters.cluster_sizes,
                d_clusters.particle_ids, d_clusters.rest_offsets,
                d_deltas, d_delta_counts,
                config.damage_shape_weight,
                C);
        }

        // Apply shape-match delta (avg over ~60 cluster memberships)
        // Day (2026-05-24): dispatch to per-particle stiffness variant when enabled.
        // Global path (else branch) is BYTE-IDENTICAL to original behavior.
        if (per_particle_stiffness && d_stiffness_per_particle != nullptr) {
            apply_kernel_per_particle_stiff<<<blocks_n, 256>>>(
                d_particles.predicted_pos, d_deltas, d_delta_counts,
                d_particles.damage, d_particles.removed, d_particles.fixed,
                d_stiffness_per_particle, N);
        } else {
            apply_kernel<<<blocks_n, 256>>>(
                d_particles.predicted_pos, d_deltas, d_delta_counts,
                d_particles.damage, d_particles.removed, d_particles.fixed,
                config.stiffness, N);
        }

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
            config.floor_y, make_sphere_tool(config.knife_pos, config.knife_radius),
            (config.tool_type == 0) ? config.knife_active : 0,
            config.friction_s, config.friction_k, N);

        // Tool SDF collision (每次迭代)
        if (config.tool_type == 1 && config.knife_active && d_tool_sdf.active && d_tool_sdf.d_sdf) {
            tool_sdf_collision_kernel<<<blocks_n, 256>>>(
                d_particles.predicted_pos, d_particles.removed, d_particles.fixed,
                grid_tool_desc(config, d_tool_sdf), N);
        }

        // Edge contact (Step 3c, 每次迭代): intact edges passing through the tool body
        if (config.edge_contact && config.knife_active && d_edges.num_edges > 0) {
            ToolDesc tool;
            float h = 0.0f;
            if (config.tool_type == 1 && d_tool_sdf.active && d_tool_sdf.d_sdf) {
                tool = grid_tool_desc(config, d_tool_sdf);
                h = 0.5f * fminf(tool.cell.x, fminf(tool.cell.y, tool.cell.z));
            } else {
                tool = make_sphere_tool(config.knife_pos, config.knife_radius);
            }
            int blocks_e = (d_edges.num_edges + 255) / 256;
            clear_deltas_kernel<<<blocks_n, 256>>>(d_deltas_ec, d_delta_counts_ec, N);
            tool_edge_contact_kernel<<<blocks_e, 256>>>(
                d_particles.predicted_pos, d_particles.removed, d_particles.fixed,
                d_edges.particle_a, d_edges.particle_b, d_edges.broken,
                tool, h, d_deltas_ec, d_delta_counts_ec, d_edges.num_edges);
            apply_self_collision_kernel<<<blocks_n, 256>>>(      // same averaged apply, no stiffness
                d_particles.predicted_pos, d_deltas_ec, d_delta_counts_ec,
                d_particles.removed, d_particles.fixed, N);
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
    // cut_only 模式: 仍跑 edge damage 更新 + particle damage 聚合(stiffness 软化用),
    //               但跳过 gpu_aspirate() — 粒子不被 CUSA 吸走。
    //               连通性切断通过 CPU 端的 cluster split (cpu_bfs_split Phase 2) 实现。
    if (config.knife_active && d_edges.num_edges > 0 &&
        (step_count % cutting_update_interval == 0)) {
        gpu_update_damage();
        gpu_update_particle_damage();
        if (config.enable_aspiration) {
            gpu_aspirate();
        }
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

void ShapeMatchingGPUSolver::download_predicted_positions(std::vector<float3>& positions) {
    int N = d_particles.N;
    positions.resize(N);
    CUDA_CHECK(cudaMemcpy(positions.data(), d_particles.predicted_pos, N * sizeof(float3), cudaMemcpyDeviceToHost));
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
    if (local_split) { int one = 1; CUDA_CHECK(cudaMemcpy(d_topo_dirty, &one, sizeof(int), cudaMemcpyHostToDevice)); }
}

void ShapeMatchingGPUSolver::reupload_clusters(
    const std::vector<int>& cluster_starts,
    const std::vector<int>& cluster_sizes,
    const std::vector<int>& particle_ids,
    const std::vector<float3>& rest_offsets,
    const std::vector<float3>& rest_cm)
{
    if (local_split) {   // R1 labels / internal-edge lists index the fixed cluster CSR
        fprintf(stderr, "[R1] reupload_clusters refused: the cluster CSR is fixed while local_split is on\n");
        return;
    }
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
    if (d_cut_segments) { cudaFree(d_cut_segments); d_cut_segments = nullptr; }
    if (d_cl_edge_start) { cudaFree(d_cl_edge_start); cudaFree(d_cl_edge_eid); cudaFree(d_cl_edge_uv); cudaFree(d_labels); cudaFree(d_topo_dirty); cudaFree(d_cluster_multi); }
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
    cudaFree(d_deltas_ec);
    cudaFree(d_delta_counts_ec);
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
