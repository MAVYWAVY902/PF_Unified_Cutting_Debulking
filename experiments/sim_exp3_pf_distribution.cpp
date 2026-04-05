// Sim-Exp 3: P(F) 空间分布
// 证明 P(F) 能自动区分三层组织（无需标注）
// 输出 CSV: particle_id, x, y, z, layer_id, avg_pf, avg_edge_pf

#include "../cpp_sim/core/particles.h"
#include "../cpp_sim/core/cluster.h"
#include "../cpp_sim/core/pf_calculator.h"
#include "../cpp_sim/core/cutting_manager.h"
#include "../cpp_sim/scene/heterogeneous_body.h"
#include <iostream>
#include <fstream>
#include <cmath>

int main() {
    std::cout << "=== Sim-Exp 3: P(F) 空间分布 ===" << std::endl;

    // 构建三层异质体
    HeterogeneousBodyConfig cfg;
    HeterogeneousBody body;
    body.build(cfg);

    Particles& particles = body.particles;
    std::vector<Cluster>& clusters = body.clusters;
    int N = particles.size();

    // 初始化面级 P(F)（从簇关系）
    PFCalculator pf_calc;
    pf_calc.alpha_min = 10.0f;
    pf_calc.init_from_clusters(clusters, N);
    pf_calc.update_all(particles);

    // 初始化边级 P(F)（从 CuttingManager）
    CuttingManager cutting_mgr;
    cutting_mgr.alpha_min = 10.0f;
    cutting_mgr.init(clusters, particles);

    // 输出 CSV
    std::string filename = "exp3_pf_distribution.csv";
    std::ofstream file(filename);
    file << "particle_id,x,y,z,layer_id,face_pf,edge_pf" << std::endl;

    for (int i = 0; i < N; i++) {
        float face_pf = pf_calc.get_particle_pf(i);
        float edge_pf = 0;

        // 从 CuttingManager 的边获取边级 P(F)
        auto it = cutting_mgr.m_particle_edges.find(i);
        if (it != cutting_mgr.m_particle_edges.end() && !it->second.empty()) {
            float sum = 0;
            for (int ei : it->second) {
                sum += cutting_mgr.edges[ei].pf_rest;
            }
            edge_pf = sum / it->second.size();
        }

        file << i << ","
             << particles[i].pos.x() << ","
             << particles[i].pos.y() << ","
             << particles[i].pos.z() << ","
             << particles[i].layer_id << ","
             << face_pf << ","
             << edge_pf << std::endl;
    }
    file.close();
    std::cout << "输出 " << N << " 个粒子到 " << filename << std::endl;

    // 逐层统计
    for (int lid = 0; lid < 3; lid++) {
        float min_fp = 1, max_fp = 0, sum_fp = 0;
        float min_ep = 1, max_ep = 0, sum_ep = 0;
        int count = 0;

        for (int i = 0; i < N; i++) {
            if (particles[i].layer_id != lid) continue;
            count++;

            float fp = pf_calc.get_particle_pf(i);
            min_fp = std::min(min_fp, fp);
            max_fp = std::max(max_fp, fp);
            sum_fp += fp;

            auto it = cutting_mgr.m_particle_edges.find(i);
            if (it != cutting_mgr.m_particle_edges.end() && !it->second.empty()) {
                float sum = 0;
                for (int ei : it->second) sum += cutting_mgr.edges[ei].pf_rest;
                float ep = sum / it->second.size();
                min_ep = std::min(min_ep, ep);
                max_ep = std::max(max_ep, ep);
                sum_ep += ep;
            }
        }

        const char* names[] = {"A(dense/capsule)", "B(medium/core)", "C(sparse/nerve)"};
        printf("  层 %s: %d 粒子\n", names[lid], count);
        printf("    Face P(F): min=%.4f avg=%.4f max=%.4f\n", min_fp, sum_fp/count, max_fp);
        printf("    Edge P(F): min=%.4f avg=%.4f max=%.4f\n", min_ep, sum_ep/count, max_ep);
    }

    return 0;
}
