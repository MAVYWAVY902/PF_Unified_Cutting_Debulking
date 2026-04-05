#include "core/particles.h"
#include "core/cluster.h"
#include "core/tet_mesh.h"
#include "scene/heterogeneous_body.h"
#include "solver/shape_matching_cpu.h"
#include "render/viewer.h"
#include <iostream>
#include <cmath>
#include <cstring>

// 创建均质立方体粒子网格（无 tet mesh 时的回退）
void create_cube(Particles& particles, double spacing, int side, Vec3 offset, int layer_id = 0) {
    for (int x = 0; x < side; x++)
        for (int y = 0; y < side; y++)
            for (int z = 0; z < side; z++)
                particles.add(offset + Vec3(x, y, z) * spacing, 1.0, layer_id);
}

void print_usage() {
    std::cout << "用法:" << std::endl;
    std::cout << "  ./pf_sim <N>              # 粒子模式 (N^3 立方体)" << std::endl;
    std::cout << "  ./pf_sim <file.msh>       # Tet mesh 模式" << std::endl;
    std::cout << "  ./pf_sim hetero           # 三层异质体模式" << std::endl;
    std::cout << "  添加 --headless 可无头运行" << std::endl;
    std::cout << "\n按键:" << std::endl;
    std::cout << "  LMB: 旋转  RMB: 平移  滚轮: 缩放" << std::endl;
    std::cout << "  K: 开关刀具  Space+鼠标: 移动刀具  Space+滚轮: 前后" << std::endl;
    std::cout << "  P: 暂停  1: 切换面  2: 切换线框  3: 切换粒子点" << std::endl;
    std::cout << "  T: 相机追踪  R: 重置刀具  ESC: 退出" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 0;
    }

    bool headless = false;
    std::string input = argv[1];
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--headless") == 0) headless = true;
    }

    ShapeMatchingSolver solver;
    TetMesh tet_mesh;
    bool use_tet = false;

    // 判断输入是数字还是文件名
    bool is_number = !input.empty() && std::all_of(input.begin(), input.end(), ::isdigit);

    if (input == "hetero") {
        // 三层异质体模式
        std::cout << "=== P(F) Shape Matching (三层异质体模式) ===" << std::endl;

        HeterogeneousBodyConfig cfg;
        HeterogeneousBody body;
        body.build(cfg);

        solver.particles = body.particles;
        solver.clusters = body.clusters;

        // 固定底面粒子
        double fix_threshold = 0.05;
        int fixed_count = 0;
        for (int i = 0; i < solver.particles.size(); i++) {
            if (solver.particles[i].pos.y() < cfg.offset.y() + fix_threshold) {
                solver.particles[i].fixed = true;
                fixed_count++;
            }
        }
        std::cout << "固定底面粒子: " << fixed_count << " 个" << std::endl;

    } else if (!is_number && input.size() > 4) {
        // Tet mesh 模式
        std::cout << "=== P(F) Shape Matching (Tet Mesh 模式) ===" << std::endl;

        if (!tet_mesh.load_msh(input)) {
            std::cerr << "加载失败: " << input << std::endl;
            return 1;
        }
        use_tet = true;

        // 计算 mesh 最低点，确保整体在地板上方
        double min_y = 1e10;
        for (auto& v : tet_mesh.vertices) min_y = std::min(min_y, v.y());
        double y_offset = -min_y + 0.001;  // 几乎贴地，不落下

        // 将 tet mesh 顶点转为粒子（带 y 偏移）
        for (int i = 0; i < tet_mesh.num_vertices(); i++) {
            Vec3 pos = tet_mesh.vertices[i] + Vec3(0, y_offset, 0);
            tet_mesh.vertices[i] = pos;  // 同步更新 tet mesh 的顶点
            solver.particles.add(pos, 1.0, 0);
        }
        int N = solver.particles.size();

        // 固定底面粒子（y 接近地板的粒子）
        double fix_threshold = 0.05;  // 距地板 5cm 以内的粒子固定
        int fixed_count = 0;
        for (int i = 0; i < solver.particles.size(); i++) {
            if (solver.particles[i].pos.y() < fix_threshold) {
                solver.particles[i].fixed = true;
                fixed_count++;
            }
        }
        std::cout << "固定底面粒子: " << fixed_count << " 个" << std::endl;

        // 从 tet 邻接关系构建簇
        std::cout << "从 tet 邻接关系构建簇..." << std::flush;
        solver.clusters = tet_mesh.build_clusters_from_tets(solver.particles, 2);  // rings=2, 更大的簇 → 更硬
        std::cout << " 完成 (" << solver.clusters.size() << " 个簇)" << std::endl;

        // 统计
        double avg_size = 0;
        for (auto& c : solver.clusters) avg_size += c.particle_ids.size();
        avg_size /= solver.clusters.size();
        std::cout << "平均簇大小: " << avg_size << std::endl;

    } else {
        // 粒子模式
        int side = is_number ? std::atoi(input.c_str()) : 10;
        std::cout << "=== P(F) Shape Matching (粒子模式) ===" << std::endl;
        std::cout << "网格: " << side << "^3 = " << side*side*side << " 粒子" << std::endl;

        double spacing = 0.02;
        create_cube(solver.particles, spacing, side, Vec3(0, 0.2, 0), 0);
        int N = solver.particles.size();

        double cluster_radius = spacing * 2.5;
        std::cout << "构建簇..." << std::flush;
        solver.clusters = build_overlapping_clusters(solver.particles, cluster_radius);
        std::cout << " 完成 (" << solver.clusters.size() << " 个簇)" << std::endl;
    }

    // P(F) 计算器（所有模式都启用）
    PFCalculator pf_calc;
    pf_calc.alpha_min = 10.0f;
    if (use_tet) {
        pf_calc.init_from_tets(tet_mesh.tets, solver.particles.size());
    } else {
        pf_calc.init_from_clusters(solver.clusters, solver.particles.size());
    }
    solver.pf_calc = &pf_calc;
    pf_calc.update_all(solver.particles);
    pf_calc.save_rest_state();
    {
        float min_pf = 1.0f, max_pf = 0.0f, avg_pf = 0.0f;
        for (float v : pf_calc.pf_values) {
            min_pf = std::min(min_pf, v);
            max_pf = std::max(max_pf, v);
            avg_pf += v;
        }
        avg_pf /= std::max((int)pf_calc.pf_values.size(), 1);
        std::cout << "初始 P(F): min=" << min_pf << " max=" << max_pf << " avg=" << avg_pf << std::endl;
    }

    // 切割管理器
    CuttingManager cutting_mgr;
    cutting_mgr.alpha_min = 10.0f;
    cutting_mgr.tau_low = 0.05f;
    cutting_mgr.init(solver.clusters, solver.particles);
    solver.cutting_mgr = &cutting_mgr;

    // 配置
    int N = solver.particles.size();
    solver.config.dt = 0.005;
    solver.config.num_iterations = 4;
    solver.config.stiffness = 0.6;
    solver.config.damping = 0.98;
    solver.config.gravity = Vec3(0, -9.81, 0);

    if (headless) {
        for (int step = 0; step < 200; step++) {
            solver.step();
            for (int i = 0; i < N; i++) {
                if (solver.particles[i].pos.y() < 0) {
                    solver.particles[i].pos.y() = 0;
                    solver.particles[i].vel.y() = -solver.particles[i].vel.y() * 0.3;
                }
            }
        }
        std::cout << "无头模式完成 200 步" << std::endl;
        return 0;
    }

    // 可视化
    Viewer viewer;
    viewer.solver = &solver;
    viewer.pf_calc = &pf_calc;
    viewer.cutting_mgr = &cutting_mgr;
    if (use_tet) {
        viewer.tet_mesh = &tet_mesh;
        viewer.draw_surface = true;
        viewer.draw_wireframe = false;
        viewer.draw_points = true;
    } else {
        // 粒子模式和异质体模式：只显示粒子点
        viewer.draw_points = true;
    }

    // 自动调整相机：根据 mesh bounding box
    Vec3 bb_min(1e10, 1e10, 1e10), bb_max(-1e10, -1e10, -1e10);
    for (int i = 0; i < N; i++) {
        bb_min = bb_min.cwiseMin(solver.particles[i].pos);
        bb_max = bb_max.cwiseMax(solver.particles[i].pos);
    }
    Vec3 center = (bb_min + bb_max) * 0.5;
    double extent = (bb_max - bb_min).norm();

    std::cout << "Bounding box: (" << bb_min.x() << "," << bb_min.y() << "," << bb_min.z()
              << ") → (" << bb_max.x() << "," << bb_max.y() << "," << bb_max.z() << ")" << std::endl;
    std::cout << "Center: (" << center.x() << "," << center.y() << "," << center.z() << ")" << std::endl;
    std::cout << "Extent: " << extent << std::endl;

    // 相机紧跟物体质心，距离为物体大小的 2.5 倍
    viewer.set_camera(
        Eigen::Vector3f((float)center.x(), (float)center.y(), (float)center.z()),
        (float)(extent * 1.5)
    );

    viewer.knife_pos = Vec3(bb_max.x() + extent * 5.0, center.y(), center.z());
    viewer.knife_radius = extent * 0.08;  // 小刀具

    if (!viewer.init(1280, 720, "P(F) Shape Matching")) {
        std::cerr << "Viewer 初始化失败" << std::endl;
        return 1;
    }

    print_usage();
    viewer.run();

    return 0;
}
