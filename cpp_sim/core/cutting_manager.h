#pragma once

#include "particles.h"
#include "cluster.h"
#include "uniform_grid.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cmath>
#include <iostream>

// 簇内的一条边（两个粒子之间的连接）
struct ClusterEdge {
    int particle_a, particle_b;
    int cluster_idx;
    float pf_rest;       // 初始 P(F)
    float pf_current;    // 当前 P(F)
    float damage;        // 损伤值 [0, 1]，0=完好，1=断裂
    bool broken;         // 是否已断裂
};

class CuttingManager {
public:
    float tau_low = 0.05f;        // P(F) 归一化值低于此 → 边断裂
    float alpha_min = 10.0f;      // sigmoid 陡峭度
    float update_radius_factor = 3.0f;  // 只更新刀具附近 N 倍半径内的边
    int min_fragment_size = 5;    // 小于此粒子数的碎片被移除（debulking）

    std::vector<ClusterEdge> edges;
    std::unordered_map<int, std::vector<int>> m_cluster_edges;
    int total_splits = 0;
    int total_removed = 0;        // 累计移除的粒子数

    // 初始化：提取簇内的**近邻边**（不是所有粒子对，只取距离小于阈值的）
    void init(const std::vector<Cluster>& clusters, const Particles& particles) {
        edges.clear();
        m_particle_edges.clear();

        // 用平均最近邻距离作为边长阈值
        UniformGrid init_grid;
        double avg_spacing = 0;
        int sample_count = 0;
        for (int i = 0; i < std::min(200, (int)particles.size()); i++) {
            auto nbrs = init_grid.query_sphere(particles[i].pos, 0.2, particles);
            double min_d = 1e10;
            for (int j : nbrs) {
                if (j == i) continue;
                double d = (particles[i].pos - particles[j].pos).norm();
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9) { avg_spacing += min_d; sample_count++; }
        }
        // 如果采样失败，用 UniformGrid 重新采样
        if (sample_count == 0) {
            init_grid.build(particles, 0.1);
            for (int i = 0; i < std::min(200, (int)particles.size()); i++) {
                auto nbrs = init_grid.query_sphere(particles[i].pos, 0.2, particles);
                double min_d = 1e10;
                for (int j : nbrs) {
                    if (j == i) continue;
                    double d = (particles[i].pos - particles[j].pos).norm();
                    if (d < min_d) min_d = d;
                }
                if (min_d < 1e9) { avg_spacing += min_d; sample_count++; }
            }
        }
        avg_spacing = (sample_count > 0) ? avg_spacing / sample_count : 0.05;
        double max_edge_len = avg_spacing * 2.5;  // 只保留短边

        // 全局去重：同一对粒子只保留一条边（属于第一个遇到的簇）
        std::set<std::pair<int,int>> edge_set;

        for (int ci = 0; ci < (int)clusters.size(); ci++) {
            const auto& ids = clusters[ci].particle_ids;
            int n = (int)ids.size();
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double dist = (particles[ids[i]].pos - particles[ids[j]].pos).norm();
                    if (dist > max_edge_len) continue;  // 跳过太长的边

                    int a = std::min(ids[i], ids[j]);
                    int b = std::max(ids[i], ids[j]);
                    if (!edge_set.insert({a, b}).second) continue;  // 去重

                    int ei = (int)edges.size();
                    ClusterEdge e;
                    e.particle_a = a;
                    e.particle_b = b;
                    e.cluster_idx = ci;
                    e.pf_rest = 0;
                    e.pf_current = 0;
                    e.damage = 0;
                    e.broken = false;
                    edges.push_back(e);
                    m_particle_edges[a].push_back(ei);
                    m_particle_edges[b].push_back(ei);
                }
            }
        }

        // 构建空间网格，计算初始 P(F)
        _build_grid(particles);
        for (auto& e : edges) {
            e.pf_rest = _compute_edge_pf(e, particles);
            e.pf_current = e.pf_rest;
        }

        std::cout << "CuttingManager: " << edges.size() << " 条簇边, "
                  << clusters.size() << " 个簇" << std::endl;

        // 统计初始 P(F) 分布
        float min_pf = 1, max_pf = 0, avg_pf = 0;
        for (auto& e : edges) {
            min_pf = std::min(min_pf, e.pf_rest);
            max_pf = std::max(max_pf, e.pf_rest);
            avg_pf += e.pf_rest;
        }
        avg_pf /= edges.size();
        std::cout << "  初始边 P(F): min=" << min_pf << " avg=" << avg_pf << " max=" << max_pf << std::endl;

        // 建立簇 → 边索引
        m_cluster_edges.clear();
        for (int i = 0; i < (int)edges.size(); i++) {
            m_cluster_edges[edges[i].cluster_idx].push_back(i);
        }
    }

    // 更新边 P(F) 和损伤（只更新刀具附近）
    void update(const Particles& particles, const Vec3& knife_pos, double knife_radius) {
        _build_grid(particles);

        double search_r2 = std::pow(knife_radius * update_radius_factor, 2);
        int updated = 0;

        for (auto& e : edges) {
            if (e.broken) continue;

            // 只更新刀具附近的边
            Vec3 mid = (particles[e.particle_a].pos + particles[e.particle_b].pos) * 0.5;
            if ((mid - knife_pos).squaredNorm() > search_r2) continue;

            // 计算当前 P(F)
            e.pf_current = _compute_edge_pf(e, particles);

            // 计算损伤（只增不减，不可逆）
            if (e.pf_rest > 0.01f) {
                float ratio = e.pf_current / e.pf_rest;
                float target_damage = std::max(0.0f, 1.0f - ratio);
                e.damage = std::max(e.damage, target_damage);
            }

            // 检查是否断裂
            if (e.pf_current < tau_low * e.pf_rest || e.damage > 0.5f) {
                e.broken = true;
            }

            updated++;
        }
    }

    // 检查断裂边，执行簇分裂
    bool check_and_split(std::vector<Cluster>& clusters, const Particles& particles) {
        // 收集每个簇的断裂边
        std::unordered_map<int, std::vector<std::pair<int,int>>> broken_per_cluster;
        for (auto& e : edges) {
            if (e.broken) {
                broken_per_cluster[e.cluster_idx].push_back({e.particle_a, e.particle_b});
            }
        }

        if (broken_per_cluster.empty()) return false;

        bool any_split = false;
        std::vector<Cluster> new_clusters;

        for (auto& [ci, broken_edges] : broken_per_cluster) {
            if (ci >= (int)clusters.size()) continue;
            auto& cluster = clusters[ci];
            if (cluster.particle_ids.size() < 2) continue;

            // 用被跟踪的未断裂边检查连通性
            auto components = _find_connected_components(ci, cluster);

            if (components.size() > 1) {
                // 簇需要分裂
                for (auto& comp : components) {
                    if (comp.size() < 2) continue;
                    Cluster new_c;
                    new_c.particle_ids = comp;
                    new_c.init(particles);
                    new_clusters.push_back(std::move(new_c));
                }
                // 标记原簇为空（后续清理）
                cluster.particle_ids.clear();
                any_split = true;
                total_splits++;
            }
        }

        if (any_split) {
            // 收集被分裂的簇索引
            std::unordered_set<int> split_cluster_indices;
            for (auto& [ci, _] : broken_per_cluster) {
                if (ci < (int)clusters.size() && clusters[ci].particle_ids.empty()) {
                    split_cluster_indices.insert(ci);
                }
            }

            // 移除空簇，添加新簇
            clusters.erase(
                std::remove_if(clusters.begin(), clusters.end(),
                    [](const Cluster& c) { return c.particle_ids.empty(); }),
                clusters.end()
            );
            for (auto& nc : new_clusters) {
                clusters.push_back(std::move(nc));
            }

            // 构建粒子 → 簇的映射（用于重新分配边）
            std::unordered_map<int, int> particle_to_cluster;
            for (int ci = 0; ci < (int)clusters.size(); ci++) {
                for (int pid : clusters[ci].particle_ids) {
                    particle_to_cluster[pid] = ci;
                }
            }

            // 重新分配边：不生成新边，只重新映射簇索引 + 移除跨组件的断裂边
            std::vector<ClusterEdge> new_edges;
            new_edges.reserve(edges.size());
            for (auto& e : edges) {
                if (e.broken) continue;  // 断裂边直接丢弃

                // 查找两端粒子现在属于哪个簇
                auto it_a = particle_to_cluster.find(e.particle_a);
                auto it_b = particle_to_cluster.find(e.particle_b);
                if (it_a == particle_to_cluster.end() || it_b == particle_to_cluster.end()) continue;

                // 更新簇索引（用 particle_a 所在的簇）
                e.cluster_idx = it_a->second;
                new_edges.push_back(e);
            }

            edges = std::move(new_edges);

            // 重建索引
            m_particle_edges.clear();
            m_cluster_edges.clear();
            for (int i = 0; i < (int)edges.size(); i++) {
                m_particle_edges[edges[i].particle_a].push_back(i);
                m_particle_edges[edges[i].particle_b].push_back(i);
                m_cluster_edges[edges[i].cluster_idx].push_back(i);
            }

            std::cout << "[CuttingManager] 簇分裂! 现有 " << clusters.size()
                      << " 个簇, " << edges.size() << " 条边, 累计分裂 "
                      << total_splits << " 次" << std::endl;
        }

        return any_split;
    }

    // 获取粒子的最大损伤值（用于可视化）
    // CUSA 风格碎片处理：主动移除高损伤区粒子（不等全局断开）
    // 返回本次移除的粒子数
    int process_fragments(std::vector<Cluster>& clusters, Particles& particles,
                          const Vec3& knife_pos, double knife_radius) {
        int removed_count = 0;
        double search_r2 = std::pow(knife_radius * update_radius_factor, 2);
        float aspiration_threshold = 0.3f;  // 渐进软化由 stiffness 提供，吸走在 0.3

        // 1. 刀具内部 + 有损伤的粒子 → 被吸走
        double knife_r2 = knife_radius * knife_radius;
        for (int pid = 0; pid < particles.size(); pid++) {
            if (particles[pid].removed) continue;

            // 粒子必须在刀具球体内部（被工具直接接触）
            double dist2 = (particles[pid].pos - knife_pos).squaredNorm();
            if (dist2 > knife_r2) continue;

            // 检查是否有任何损伤的边
            auto it = m_particle_edges.find(pid);
            if (it == m_particle_edges.end()) continue;

            float max_damage = 0;
            for (int ei : it->second) {
                max_damage = std::max(max_damage, edges[ei].damage);
            }

            // 在刀具内 + 最大损伤 > 阈值 → 吸走
            if (max_damage > aspiration_threshold) {
                particles[pid].removed = true;
                removed_count++;
            }
        }

        if (removed_count == 0) return 0;

        // 2. 从所有簇中移除 removed 粒子
        for (auto& cluster : clusters) {
            int old_size = (int)cluster.particle_ids.size();
            cluster.particle_ids.erase(
                std::remove_if(cluster.particle_ids.begin(), cluster.particle_ids.end(),
                    [&](int pid) { return particles[pid].removed; }),
                cluster.particle_ids.end()
            );
            if (cluster.particle_ids.size() != old_size && cluster.particle_ids.size() >= 2) {
                cluster.init(particles);  // 重算 rest state（步骤 9）
            }
        }

        // 移除空簇
        clusters.erase(
            std::remove_if(clusters.begin(), clusters.end(),
                [](const Cluster& c) { return c.particle_ids.size() < 2; }),
            clusters.end()
        );

        // 3. 清理涉及 removed 粒子的边
        edges.erase(
            std::remove_if(edges.begin(), edges.end(),
                [&](const ClusterEdge& e) {
                    return particles[e.particle_a].removed || particles[e.particle_b].removed;
                }),
            edges.end()
        );

        // 重建索引
        m_particle_edges.clear();
        m_cluster_edges.clear();
        std::unordered_map<int, int> particle_to_cluster;
        for (int ci = 0; ci < (int)clusters.size(); ci++) {
            for (int pid : clusters[ci].particle_ids) {
                particle_to_cluster[pid] = ci;
            }
        }
        for (auto& e : edges) {
            auto it2 = particle_to_cluster.find(e.particle_a);
            if (it2 != particle_to_cluster.end()) e.cluster_idx = it2->second;
        }
        for (int i = 0; i < (int)edges.size(); i++) {
            m_particle_edges[edges[i].particle_a].push_back(i);
            m_particle_edges[edges[i].particle_b].push_back(i);
            m_cluster_edges[edges[i].cluster_idx].push_back(i);
        }

        // 4. 清除孤立粒子（不属于任何有效簇的粒子）
        std::unordered_set<int> in_cluster;
        for (auto& cluster : clusters) {
            for (int pid : cluster.particle_ids) {
                in_cluster.insert(pid);
            }
        }
        int orphan_count = 0;
        for (int i = 0; i < particles.size(); i++) {
            if (!particles[i].removed && !in_cluster.count(i)) {
                particles[i].removed = true;
                orphan_count++;
            }
        }
        removed_count += orphan_count;

        total_removed += removed_count;
        std::cout << "[CUSA] 吸走 " << removed_count << " 粒子 (含 " << orphan_count << " 孤立), "
                  << "剩余 " << clusters.size() << " 簇 "
                  << edges.size() << " 边, "
                  << "累计移除 " << total_removed << "/" << particles.size() << std::endl;

        return removed_count;
    }

    float get_particle_damage(int pid) const {
        float max_damage = 0;
        auto it = m_particle_edges.find(pid);
        if (it != m_particle_edges.end()) {
            for (int ei : it->second) {
                max_damage = std::max(max_damage, edges[ei].damage);
            }
        }
        return max_damage;
    }

private:
    UniformGrid m_grid;
public:
    std::unordered_map<int, std::vector<int>> m_particle_edges;  // 粒子 → 边索引
private:

    void _build_grid(const Particles& particles) {
        // 用平均边长作为 cell size
        double avg_len = 0;
        int sample = std::min(100, (int)edges.size());
        for (int i = 0; i < sample; i++) {
            avg_len += (particles[edges[i].particle_a].pos - particles[edges[i].particle_b].pos).norm();
        }
        avg_len /= std::max(sample, 1);
        m_grid.build(particles, avg_len * 1.5);
    }

    // 边级 P(F)：两点的中球，找最近闯入者
    float _compute_edge_pf(const ClusterEdge& edge, const Particles& particles) const {
        Vec3 pA = particles[edge.particle_a].pos;
        Vec3 pB = particles[edge.particle_b].pos;

        Vec3 center = (pA + pB) * 0.5;
        double radius = (pA - pB).norm() * 0.5;

        if (radius < 1e-10) return 1.0f;

        // 查找附近粒子
        auto neighbors = m_grid.query_sphere(center, radius * 2.0, particles);

        // 找最近非成员点
        double min_dist = 1e10;
        for (int k : neighbors) {
            if (k == edge.particle_a || k == edge.particle_b) continue;
            double d = (particles[k].pos - center).norm();
            if (d < min_dist) min_dist = d;
        }

        if (min_dist > 1e9) return 1.0f;

        // DMesh++ 原版：sigmoid(alpha * sdist / radius)
        double sdist = min_dist - radius;
        float pf = 1.0f / (1.0f + std::exp(-(float)(alpha_min * sdist / radius)));
        return pf;
    }

    // BFS 找连通分量：只用被跟踪的未断裂边作为连接
    std::vector<std::vector<int>> _find_connected_components(
        int cluster_idx,
        const Cluster& cluster) const
    {
        const auto& ids = cluster.particle_ids;
        std::unordered_map<int, std::vector<int>> adj;

        for (int pid : ids) adj[pid];

        // 用预建索引，只看该簇的边
        auto it = m_cluster_edges.find(cluster_idx);
        if (it != m_cluster_edges.end()) {
            for (int ei : it->second) {
                const auto& e = edges[ei];
                if (e.broken) continue;
                adj[e.particle_a].push_back(e.particle_b);
                adj[e.particle_b].push_back(e.particle_a);
            }
        }

        // BFS
        std::unordered_set<int> visited;
        std::vector<std::vector<int>> components;

        for (int pid : ids) {
            if (visited.count(pid)) continue;
            std::vector<int> comp;
            std::queue<int> q;
            q.push(pid);
            visited.insert(pid);
            while (!q.empty()) {
                int v = q.front(); q.pop();
                comp.push_back(v);
                for (int nb : adj[v]) {
                    if (!visited.count(nb)) {
                        visited.insert(nb);
                        q.push(nb);
                    }
                }
            }
            components.push_back(std::move(comp));
        }

        return components;
    }

    // 簇分裂后重建边索引（带距离过滤，和 init 一致）
    void _rebuild_edge_indices(const std::vector<Cluster>& clusters, const Particles& particles) {
        edges.clear();
        m_particle_edges.clear();

        // 计算平均最近邻距离
        double avg_spacing = 0;
        int sample_count = 0;
        UniformGrid tmp_grid;
        tmp_grid.build(particles, 0.1);
        for (int i = 0; i < std::min(200, (int)particles.size()); i++) {
            auto nbrs = tmp_grid.query_sphere(particles[i].pos, 0.2, particles);
            double min_d = 1e10;
            for (int j : nbrs) {
                if (j == i) continue;
                double d = (particles[i].pos - particles[j].pos).norm();
                if (d < min_d) min_d = d;
            }
            if (min_d < 1e9) { avg_spacing += min_d; sample_count++; }
        }
        avg_spacing = (sample_count > 0) ? avg_spacing / sample_count : 0.05;
        double max_edge_len = avg_spacing * 2.5;

        // 全局去重
        std::set<std::pair<int,int>> edge_set;

        for (int ci = 0; ci < (int)clusters.size(); ci++) {
            const auto& ids = clusters[ci].particle_ids;
            int n = (int)ids.size();
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    double dist = (particles[ids[i]].pos - particles[ids[j]].pos).norm();
                    if (dist > max_edge_len) continue;

                    int a = std::min(ids[i], ids[j]);
                    int b = std::max(ids[i], ids[j]);
                    if (!edge_set.insert({a, b}).second) continue;

                    int ei = (int)edges.size();
                    ClusterEdge e;
                    e.particle_a = a;
                    e.particle_b = b;
                    e.cluster_idx = ci;
                    e.pf_current = _compute_edge_pf(e, particles);
                    e.pf_rest = e.pf_current;
                    e.damage = 0;
                    e.broken = false;
                    edges.push_back(e);
                    m_particle_edges[a].push_back(ei);
                    m_particle_edges[b].push_back(ei);
                }
            }
        }
    }
};
