// GPU 版本入口 — 先做 headless 性能测试
#include "solver/shape_matching_gpu.h"
#include "core/particles.h"
#include "core/cluster.h"
#include "core/tet_mesh.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>

int main(int argc, char** argv) {
    std::string input = (argc > 1) ? argv[1] : "10";
    bool is_number = !input.empty() && std::all_of(input.begin(), input.end(), ::isdigit);

    // 加载 mesh 或生成粒子
    Particles particles;
    std::vector<Cluster> clusters;
    TetMesh tet_mesh;

    if (!is_number && input.size() > 4) {
        // Tet mesh 模式
        if (!tet_mesh.load_msh(input)) {
            std::cerr << "加载失败: " << input << std::endl;
            return 1;
        }
        for (int i = 0; i < tet_mesh.num_vertices(); i++) {
            particles.add(tet_mesh.vertices[i] + Vec3(0, 0.5, 0), 1.0, 0);
        }
        clusters = tet_mesh.build_clusters_from_tets(particles, 2);
    } else {
        // 粒子模式
        int side = std::atoi(input.c_str());
        double spacing = 0.02;
        for (int x = 0; x < side; x++)
            for (int y = 0; y < side; y++)
                for (int z = 0; z < side; z++)
                    particles.add(Vec3(x, y + 0.5, z) * spacing, 1.0, 0);
        clusters = build_overlapping_clusters(particles, spacing * 2.5);
    }

    int N = particles.size();
    int C = (int)clusters.size();
    std::cout << "=== GPU Shape Matching ===" << std::endl;
    std::cout << "粒子: " << N << ", 簇: " << C << std::endl;

    // 打包数据为 GPU 友好格式
    std::vector<float3> positions(N), velocities(N);
    std::vector<int> removed(N, 0), fixed(N, 0);

    for (int i = 0; i < N; i++) {
        positions[i] = make_float3((float)particles[i].pos.x(),
                                    (float)particles[i].pos.y(),
                                    (float)particles[i].pos.z());
        velocities[i] = make_float3(0, 0, 0);
    }

    // 固定底面
    for (int i = 0; i < N; i++) {
        if (particles[i].pos.y() < 0.55) fixed[i] = 1;
    }

    // 打包簇为 CSR 格式
    std::vector<int> cluster_starts(C), cluster_sizes(C);
    std::vector<int> all_particle_ids;
    std::vector<float3> all_rest_offsets;
    std::vector<float3> all_rest_cm(C);

    int offset = 0;
    for (int ci = 0; ci < C; ci++) {
        cluster_starts[ci] = offset;
        cluster_sizes[ci] = (int)clusters[ci].particle_ids.size();
        for (int j = 0; j < cluster_sizes[ci]; j++) {
            all_particle_ids.push_back(clusters[ci].particle_ids[j]);
            all_rest_offsets.push_back(make_float3(
                (float)clusters[ci].rest_offsets[j].x(),
                (float)clusters[ci].rest_offsets[j].y(),
                (float)clusters[ci].rest_offsets[j].z()));
        }
        all_rest_cm[ci] = make_float3(
            (float)clusters[ci].rest_cm.x(),
            (float)clusters[ci].rest_cm.y(),
            (float)clusters[ci].rest_cm.z());
        offset += cluster_sizes[ci];
    }

    std::cout << "CSR 总条目: " << all_particle_ids.size() << std::endl;

    // 初始化 GPU solver
    ShapeMatchingGPUSolver gpu_solver;
    gpu_solver.config.dt = 0.005f;
    gpu_solver.config.gravity = make_float3(0, -9.81f, 0);
    gpu_solver.config.num_iterations = 4;
    gpu_solver.config.stiffness = 0.6f;
    gpu_solver.config.damping = 0.98f;
    gpu_solver.config.floor_y = 0.0f;
    gpu_solver.config.knife_pos = make_float3(100, 100, 100);
    gpu_solver.config.knife_radius = 0.1f;
    gpu_solver.config.knife_active = 0;

    gpu_solver.init(positions, velocities, removed, fixed,
                    cluster_starts, cluster_sizes,
                    all_particle_ids, all_rest_offsets, all_rest_cm);

    // 性能测试
    int num_steps = 500;
    std::cout << "运行 " << num_steps << " 步..." << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < num_steps; step++) {
        gpu_solver.step();
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "完成!" << std::endl;
    std::cout << "总耗时: " << elapsed << "s" << std::endl;
    std::cout << "每步: " << (elapsed / num_steps * 1000) << " ms" << std::endl;
    std::cout << "FPS: " << (num_steps / elapsed) << std::endl;

    // 下载位置验证
    gpu_solver.download_positions(positions);
    float3 cm = make_float3(0, 0, 0);
    for (int i = 0; i < N; i++) {
        cm.x += positions[i].x;
        cm.y += positions[i].y;
        cm.z += positions[i].z;
    }
    cm.x /= N; cm.y /= N; cm.z /= N;
    printf("最终质心: (%.3f, %.3f, %.3f)\n", cm.x, cm.y, cm.z);

    gpu_solver.cleanup();
    return 0;
}
