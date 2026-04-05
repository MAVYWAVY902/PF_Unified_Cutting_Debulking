#pragma once

#include "particles.h"
#include "uniform_grid.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <iostream>

class PFCalculator {
public:
    struct Face { int a, b, c; };

    float alpha_min = 10.0f;              // sigmoid 陡峭度

    std::vector<Face> candidate_faces;     // 候选面列表
    std::vector<float> pf_values;          // 每个面的 P(F)

    // 从簇的粒子关系生成候选面（用于纯粒子模式，没有 tet mesh）
    void init_from_clusters(const std::vector<Cluster>& clusters, int num_particles) {
        std::set<std::tuple<int,int,int>> face_set;
        std::vector<Face> all_faces;

        // 对每个簇中的粒子三元组生成候选面
        for (const auto& cluster : clusters) {
            const auto& ids = cluster.particle_ids;
            int n = (int)ids.size();
            // 对小簇枚举所有三元组，大簇只取部分
            int max_triplets = std::min(n * (n-1) * (n-2) / 6, 50);
            int count = 0;
            for (int i = 0; i < n && count < max_triplets; i++) {
                for (int j = i+1; j < n && count < max_triplets; j++) {
                    for (int k = j+1; k < n && count < max_triplets; k++) {
                        int a = ids[i], b = ids[j], c = ids[k];
                        // 排序去重
                        if (a > b) std::swap(a, b);
                        if (b > c) std::swap(b, c);
                        if (a > b) std::swap(a, b);
                        auto key = std::make_tuple(a, b, c);
                        if (face_set.insert(key).second) {
                            all_faces.push_back({a, b, c});
                            count++;
                        }
                    }
                }
            }
        }

        candidate_faces = std::move(all_faces);
        pf_values.assign(candidate_faces.size(), 1.0f);

        m_particle_faces.resize(num_particles);
        for (int i = 0; i < (int)candidate_faces.size(); i++) {
            m_particle_faces[candidate_faces[i].a].push_back(i);
            m_particle_faces[candidate_faces[i].b].push_back(i);
            m_particle_faces[candidate_faces[i].c].push_back(i);
        }

        std::cout << "PFCalculator (from clusters): " << candidate_faces.size() << " 候选面, "
                  << num_particles << " 粒子" << std::endl;
    }

    // 从 tet mesh 的四面体面初始化候选面
    void init_from_tets(const std::vector<std::array<int, 4>>& tets, int num_particles) {
        // 从所有四面体中提取所有面（含内部面），去重
        std::set<std::tuple<int,int,int>> face_set;
        std::vector<Face> all_faces;

        for (const auto& tet : tets) {
            int v[4] = {tet[0], tet[1], tet[2], tet[3]};
            int fi[4][3] = {
                {v[0], v[1], v[2]},
                {v[0], v[1], v[3]},
                {v[0], v[2], v[3]},
                {v[1], v[2], v[3]}
            };
            for (int f = 0; f < 4; f++) {
                int a = fi[f][0], b = fi[f][1], c = fi[f][2];
                // 排序用于去重
                if (a > b) std::swap(a, b);
                if (b > c) std::swap(b, c);
                if (a > b) std::swap(a, b);
                auto key = std::make_tuple(a, b, c);
                if (face_set.insert(key).second) {
                    all_faces.push_back({a, b, c});
                }
            }
        }

        candidate_faces = std::move(all_faces);
        pf_values.assign(candidate_faces.size(), 1.0f);

        // 构建每个粒子关联哪些面（加速 get_particle_pf）
        m_particle_faces.resize(num_particles);
        for (int i = 0; i < (int)candidate_faces.size(); i++) {
            m_particle_faces[candidate_faces[i].a].push_back(i);
            m_particle_faces[candidate_faces[i].b].push_back(i);
            m_particle_faces[candidate_faces[i].c].push_back(i);
        }

        std::cout << "PFCalculator: " << candidate_faces.size() << " 候选面, "
                  << num_particles << " 粒子" << std::endl;
    }

    // 保存初始 P(F) 值（用于归一化）
    std::vector<float> pf_rest_values;

    // 保存初始 P(F)，之后的权重 = P(F) / P(F)_rest，初始状态下权重全为 1.0
    void save_rest_state() {
        pf_rest_values = pf_values;
        std::cout << "P(F) rest state saved (" << pf_rest_values.size() << " 面)" << std::endl;
    }

    // 获取归一化后的粒子 P(F)（相对初始状态的比值）
    // 用最小值而不是平均值，确保局部变化不被淹没
    float get_particle_pf_normalized(int pid) const {
        if (pid < 0 || pid >= (int)m_particle_faces.size()) return 1.0f;
        const auto& faces = m_particle_faces[pid];
        if (faces.empty()) return 1.0f;

        float min_ratio = 1.0f;
        for (int fi : faces) {
            // 跳过 rest 值太小的面（这些面在初始状态下就不稳定，不应影响权重）
            if (fi >= (int)pf_rest_values.size() || pf_rest_values[fi] < 0.01f) continue;
            float ratio = pf_values[fi] / pf_rest_values[fi];
            min_ratio = std::min(min_ratio, ratio);
        }
        return std::max(0.0f, std::min(min_ratio, 1.0f));  // clamp to [0, 1]
    }

    // 全量更新所有面的 P(F)
    void update_all(const Particles& particles) {
        // 构建空间网格
        // cell_size 用平均面边长的估计
        double avg_edge = 0;
        int sample = std::min(100, (int)candidate_faces.size());
        for (int i = 0; i < sample; i++) {
            const auto& f = candidate_faces[i];
            avg_edge += (particles[f.a].pos - particles[f.b].pos).norm();
        }
        avg_edge /= sample;
        m_grid.build(particles, avg_edge * 2.0);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < (int)candidate_faces.size(); i++) {
            pf_values[i] = _compute_face_pf(i, particles);
        }
    }

    // 局部更新：只更新 center 附近 radius 范围内的面
    int update_local(const Particles& particles, const Vec3& knife_pos, double radius) {
        // 重建空间网格
        double avg_edge = 0;
        int sample = std::min(100, (int)candidate_faces.size());
        for (int i = 0; i < sample; i++) {
            const auto& f = candidate_faces[i];
            avg_edge += (particles[f.a].pos - particles[f.b].pos).norm();
        }
        avg_edge /= sample;
        m_grid.build(particles, avg_edge * 2.0);

        double r2 = radius * radius;
        int updated = 0;

        #pragma omp parallel for schedule(dynamic) reduction(+:updated)
        for (int i = 0; i < (int)candidate_faces.size(); i++) {
            const auto& f = candidate_faces[i];
            Vec3 face_center = (particles[f.a].pos + particles[f.b].pos + particles[f.c].pos) / 3.0;
            if ((face_center - knife_pos).squaredNorm() > r2) continue;
            pf_values[i] = _compute_face_pf(i, particles);
            updated++;
        }
        return updated;
    }

    // 获取粒子的 P(F)（关联面的平均值）
    float get_particle_pf(int pid) const {
        if (pid < 0 || pid >= (int)m_particle_faces.size()) return 1.0f;
        const auto& faces = m_particle_faces[pid];
        if (faces.empty()) return 1.0f;

        float sum = 0.0f;
        for (int fi : faces) {
            sum += pf_values[fi];
        }
        return sum / faces.size();
    }

    int num_faces() const { return (int)candidate_faces.size(); }

private:
    UniformGrid m_grid;
    std::vector<std::vector<int>> m_particle_faces;  // 粒子 → 面索引列表

    // 计算三点外接球球心
    // 公式：center = A + ((|c_a|² * b_a - |b_a|² * c_a) × (b_a × c_a)) / (2 * |b_a × c_a|²)
    static Vec3 _circumcenter(const Vec3& A, const Vec3& B, const Vec3& C) {
        Vec3 ba = B - A;
        Vec3 ca = C - A;
        Vec3 cross_ba_ca = ba.cross(ca);
        double denom = 2.0 * cross_ba_ca.squaredNorm();

        if (denom < 1e-20) {
            // 退化（共线），返回中点
            return (A + B + C) / 3.0;
        }

        Vec3 num = (ca.squaredNorm() * ba - ba.squaredNorm() * ca).cross(cross_ba_ca);
        return A + num / denom;
    }

    // 计算单个面的 P(F)（DMesh++ 原版逻辑：只看最近非成员点）
    float _compute_face_pf(int face_idx, const Particles& particles) const {
        const auto& f = candidate_faces[face_idx];
        Vec3 pA = particles[f.a].pos;
        Vec3 pB = particles[f.b].pos;
        Vec3 pC = particles[f.c].pos;

        // 1. 外接球
        Vec3 center = _circumcenter(pA, pB, pC);
        double radius = (pA - center).norm();

        if (radius < 1e-10) return 1.0f;

        // 2. 查询外接球附近的粒子，找最近的非成员点
        //    搜索范围比外接球稍大，确保能找到最近的点
        auto neighbors = m_grid.query_sphere(center, radius * 2.0, particles);

        double min_dist = 1e10;
        for (int k : neighbors) {
            if (k == f.a || k == f.b || k == f.c) continue;
            double d_k = (particles[k].pos - center).norm();
            if (d_k < min_dist) {
                min_dist = d_k;
            }
        }

        if (min_dist > 1e9) return 1.0f;  // 没有邻居

        // 3. signed distance：正值=最近点在球外（面应存在），负值=在球内（面不应存在）
        double sdist = min_dist - radius;

        // 4. sigmoid：DMesh++ 原版 P(F) = sigmoid(sdist / sdist_unit / sigmoid_T)
        //    sdist_unit 用 radius 归一化，sigmoid_T 控制陡峭度
        //    简化为：P(F) = sigmoid(alpha_min * sdist / radius)
        float pf = 1.0f / (1.0f + std::exp(-(float)(alpha_min * sdist / radius)));

        return pf;
    }
};
