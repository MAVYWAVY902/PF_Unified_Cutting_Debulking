#pragma once

#include "particles.h"
#include "uniform_grid.h"
#include <vector>
#include <numeric>
#include <cassert>

struct Cluster {
    std::vector<int> particle_ids;

    // 静息状态（初始化时计算，之后不变）
    Vec3 rest_cm;                          // 静息质心
    std::vector<Vec3> rest_offsets;        // 每个粒子相对静息质心的偏移

    // P(F) 权重（每帧更新）—— 初始化为 1.0
    std::vector<double> pf_weights;
    int original_size = 0;  // 初始粒子数（用于结构弱化判断）

    // 用粒子当前位置初始化静息状态
    void init(const Particles& particles) {
        int n = static_cast<int>(particle_ids.size());
        assert(n > 0);

        // 初始化 P(F) 权重为 1.0
        pf_weights.assign(n, 1.0);
        rest_offsets.resize(n);
        if (original_size == 0) original_size = n;  // 只在第一次 init 时记录

        // 计算静息质心（质量加权）
        rest_cm = Vec3::Zero();
        double total_m = 0.0;
        for (int j = 0; j < n; j++) {
            int pid = particle_ids[j];
            rest_cm += particles[pid].mass * particles[pid].pos;
            total_m += particles[pid].mass;
        }
        rest_cm /= total_m;

        // 计算每个粒子的静息偏移
        for (int j = 0; j < n; j++) {
            int pid = particle_ids[j];
            rest_offsets[j] = particles[pid].pos - rest_cm;
        }
    }

    // P(F) 加权 shape matching 求解
    // 输入：当前预测位置
    // 输出：累积 deltas 和 counts（Jacobi 模式）
    void solve(const std::vector<Vec3>& positions,
               std::vector<Vec3>& deltas,
               std::vector<int>& delta_counts) const
    {
        int n = static_cast<int>(particle_ids.size());
        if (n == 0) return;

        // 1. P(F) 加权质心
        Vec3 cm = Vec3::Zero();
        double total_w = 0.0;
        for (int j = 0; j < n; j++) {
            int pid = particle_ids[j];
            double w = pf_weights[j];
            cm += w * positions[pid];
            total_w += w;
        }
        if (total_w < 1e-12) return;  // 所有权重都是 0，跳过
        cm /= total_w;

        // 2. P(F) 加权协方差矩阵 A_pq
        Mat3 A = Mat3::Zero();
        for (int j = 0; j < n; j++) {
            int pid = particle_ids[j];
            double w = pf_weights[j];
            Vec3 p = positions[pid] - cm;
            Vec3 q = rest_offsets[j];
            A += w * p * q.transpose();
        }

        // 3. 极分解提取旋转矩阵 R（用 SVD，比 eigendecomposition 更快更稳定）
        Eigen::JacobiSVD<Mat3> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Mat3 R = svd.matrixU() * svd.matrixV().transpose();

        // 确保是正旋转（det > 0），处理反射
        if (R.determinant() < 0) {
            Mat3 U = svd.matrixU();
            U.col(2) = -U.col(2);
            R = U * svd.matrixV().transpose();
        }

        // 4. 计算 goal positions，累积 delta
        //    不在这里做弱化——在 solver 的 Apply 步骤用 per-particle damage stiffness
        for (int j = 0; j < n; j++) {
            int pid = particle_ids[j];
            Vec3 goal = R * rest_offsets[j] + cm;
            Vec3 correction = goal - positions[pid];

            deltas[pid] += correction;
            delta_counts[pid]++;
        }
    }
};

// 构建重叠簇：对每个粒子，以半径 R 为范围创建一个簇
// 使用 UniformGrid 加速邻域搜索，O(N) 替代 O(N²)
inline std::vector<Cluster> build_overlapping_clusters(
    const Particles& particles,
    double radius)
{
    int N = particles.size();

    // 用簇半径作为 cell_size，保证邻域查询只需搜索相邻几个 cell
    UniformGrid grid;
    grid.build(particles, radius);

    std::vector<Cluster> clusters(N);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        clusters[i].particle_ids = grid.query_sphere(particles[i].pos, radius, particles);
        if (clusters[i].particle_ids.size() >= 2) {
            clusters[i].init(particles);
        }
    }

    // 移除无效簇（粒子数 < 2）
    std::vector<Cluster> result;
    result.reserve(N);
    for (auto& c : clusters) {
        if (c.particle_ids.size() >= 2) {
            result.push_back(std::move(c));
        }
    }

    return result;
}
