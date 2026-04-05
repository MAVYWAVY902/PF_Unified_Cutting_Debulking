#pragma once

#include "../core/particles.h"
#include "../core/cluster.h"
#include "../core/uniform_grid.h"
#include <iostream>
#include <cmath>

struct HeterogeneousBodyConfig {
    // 整体尺寸
    double width = 1.0;    // x
    double height = 0.6;   // y（三层总高度）
    double depth = 1.0;    // z

    // 三层粒子间距（密→疏）
    double spacing_A = 0.04;   // 层A：肿瘤核心（密集）
    double spacing_B = 0.06;   // 层B：粘连界面（中等）
    double spacing_C = 0.08;   // 层C：神经鞘（稀疏）

    // 三层高度比例（各占 1/3）
    double layer_A_top = 0.2;   // 层A: y ∈ [0, 0.2]
    double layer_B_top = 0.4;   // 层B: y ∈ [0.2, 0.4]
    // 层C: y ∈ [0.4, 0.6]

    // 初始位置偏移
    Vec3 offset = Vec3(0, 0.5, 0);

    // 簇构建半径（相对于局部 spacing 的倍数）
    double cluster_radius_factor = 1.8;
};

struct HeterogeneousBody {
    Particles particles;
    std::vector<Cluster> clusters;
    int layer_A_start = 0, layer_A_end = 0;
    int layer_B_start = 0, layer_B_end = 0;
    int layer_C_start = 0, layer_C_end = 0;

    void build(const HeterogeneousBodyConfig& cfg) {
        particles.data.clear();

        // 层 A：肿瘤核心（密集，y ∈ [0, layer_A_top]）
        layer_A_start = particles.size();
        _fill_layer(cfg, 0, cfg.layer_A_top, cfg.spacing_A, 0);
        layer_A_end = particles.size();

        // 层 B：粘连界面（中等，y ∈ [layer_A_top, layer_B_top]）
        layer_B_start = particles.size();
        _fill_layer(cfg, cfg.layer_A_top, cfg.layer_B_top, cfg.spacing_B, 1);
        layer_B_end = particles.size();

        // 层 C：神经鞘（稀疏，y ∈ [layer_B_top, height]）
        layer_C_start = particles.size();
        _fill_layer(cfg, cfg.layer_B_top, cfg.height, cfg.spacing_C, 2);
        layer_C_end = particles.size();

        std::cout << "三层异质体: "
                  << "A(" << (layer_A_end - layer_A_start) << " 粒子, spacing=" << cfg.spacing_A << ") "
                  << "B(" << (layer_B_end - layer_B_start) << " 粒子, spacing=" << cfg.spacing_B << ") "
                  << "C(" << (layer_C_end - layer_C_start) << " 粒子, spacing=" << cfg.spacing_C << ") "
                  << "总计=" << particles.size() << std::endl;

        // 构建重叠簇：用全局最大 spacing 作为半径
        double max_spacing = std::max({cfg.spacing_A, cfg.spacing_B, cfg.spacing_C});
        double cluster_radius = max_spacing * cfg.cluster_radius_factor;
        std::cout << "构建簇 (radius=" << cluster_radius << ")..." << std::flush;
        clusters = build_overlapping_clusters(particles, cluster_radius);
        std::cout << " 完成 (" << clusters.size() << " 个簇)" << std::endl;

        // 统计每层的平均簇大小
        double avg_A = 0, avg_B = 0, avg_C = 0;
        int count_A = 0, count_B = 0, count_C = 0;
        for (auto& c : clusters) {
            // 用簇中第一个粒子的 layer_id 判断簇属于哪一层
            int first_layer = particles[c.particle_ids[0]].layer_id;
            if (first_layer == 0) { avg_A += c.particle_ids.size(); count_A++; }
            else if (first_layer == 1) { avg_B += c.particle_ids.size(); count_B++; }
            else if (first_layer == 2) { avg_C += c.particle_ids.size(); count_C++; }
        }
        if (count_A > 0) std::cout << "  层A 平均簇大小: " << avg_A / count_A << std::endl;
        if (count_B > 0) std::cout << "  层B 平均簇大小: " << avg_B / count_B << std::endl;
        if (count_C > 0) std::cout << "  层C 平均簇大小: " << avg_C / count_C << std::endl;
    }

private:
    void _fill_layer(const HeterogeneousBodyConfig& cfg,
                     double y_min, double y_max, double spacing, int layer_id) {
        int nx = std::max(1, (int)std::ceil(cfg.width / spacing));
        int ny = std::max(1, (int)std::ceil((y_max - y_min) / spacing));
        int nz = std::max(1, (int)std::ceil(cfg.depth / spacing));

        for (int ix = 0; ix <= nx; ix++) {
            for (int iy = 0; iy <= ny; iy++) {
                for (int iz = 0; iz <= nz; iz++) {
                    double x = ix * spacing;
                    double y = y_min + iy * spacing;
                    double z = iz * spacing;

                    // 边界检查
                    if (x > cfg.width || y > y_max || z > cfg.depth) continue;

                    Vec3 pos = cfg.offset + Vec3(x, y, z);
                    particles.add(pos, 1.0, layer_id);
                }
            }
        }
    }
};
