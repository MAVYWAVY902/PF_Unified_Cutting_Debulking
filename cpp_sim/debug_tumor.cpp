// Headless debug: tumor mesh shape matching 诊断
#include "solver/shape_matching_gpu.h"
#include "core/particles.h"
#include "core/cluster.h"
#include "core/tet_mesh.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

int main(int argc, char** argv) {
    std::string input = (argc > 1) ? argv[1] : "../../resources/tumor/tumor_7800.msh";

    TetMesh tet_mesh;
    Particles particles;
    if (!tet_mesh.load_msh(input)) { std::cerr << "加载失败\n"; return 1; }

    // 同 main_gpu_viewer 一样的初始化
    double min_y = 1e10, max_y = -1e10;
    for (auto& v : tet_mesh.vertices) {
        min_y = std::min(min_y, v.y());
        max_y = std::max(max_y, v.y());
    }
    double mesh_height = max_y - min_y;
    double y_offset = -min_y + 0.001;

    for (int i = 0; i < tet_mesh.num_vertices(); i++) {
        Vec3 pos = tet_mesh.vertices[i] + Vec3(0, y_offset, 0);
        tet_mesh.vertices[i] = pos;
        particles.add(pos, 1.0, 0);
    }

    // 固定底面
    double fixed_threshold = 0.001 + mesh_height * 0.10;
    int fixed_count = 0;
    for (int i = 0; i < particles.size(); i++) {
        if (particles[i].pos.y() < fixed_threshold) {
            particles[i].fixed = true;
            fixed_count++;
        }
    }

    auto clusters = tet_mesh.build_clusters_from_tets(particles, 2);
    int N = particles.size();
    int C = (int)clusters.size();

    printf("N=%d, C=%d, fixed=%d, mesh_height=%.6f\n", N, C, fixed_count, mesh_height);

    // Cluster size 统计
    int min_cs=999999, max_cs=0; double sum_cs=0;
    for (auto& c : clusters) {
        int s = (int)c.particle_ids.size();
        min_cs = std::min(min_cs, s);
        max_cs = std::max(max_cs, s);
        sum_cs += s;
    }
    printf("Cluster size: min=%d max=%d avg=%.1f\n", min_cs, max_cs, sum_cs/C);

    // 打包GPU数据
    std::vector<float3> positions(N), velocities(N);
    std::vector<int> removed(N, 0), fixed_arr(N, 0);
    for (int i = 0; i < N; i++) {
        positions[i] = make_float3((float)particles[i].pos.x(),
                                    (float)particles[i].pos.y(),
                                    (float)particles[i].pos.z());
        velocities[i] = make_float3(0, 0, 0);
        fixed_arr[i] = particles[i].fixed ? 1 : 0;
    }

    std::vector<int> cluster_starts(C), cluster_sizes(C);
    std::vector<int> all_particle_ids;
    std::vector<float3> all_rest_offsets, all_rest_cm(C);
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

    // 计算avg spacing
    float avg_spacing = 0;
    {
        int sample = std::min(N, 500);
        double total = 0; int cnt = 0;
        for (int i = 0; i < sample; i++) {
            float best = 1e10f;
            for (int j = 0; j < N; j++) {
                if (j == i) continue;
                float dx = positions[i].x - positions[j].x;
                float dy = positions[i].y - positions[j].y;
                float dz = positions[i].z - positions[j].z;
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < best) best = d2;
            }
            total += sqrtf(best); cnt++;
        }
        avg_spacing = (float)(total / cnt);
    }
    printf("avg_spacing=%.6f\n", avg_spacing);

    // 初始化 GPU solver
    ShapeMatchingGPUSolver gpu_solver;
    gpu_solver.config.dt = 0.001f;
    gpu_solver.config.gravity = make_float3(0, -9.81f, 0);
    gpu_solver.config.num_iterations = 16;   // 更多迭代补偿高overlap
    gpu_solver.config.stiffness = 1.0f;    // 满stiffness
    gpu_solver.config.damping = 0.98f;
    gpu_solver.config.floor_y = 0.0f;
    gpu_solver.config.knife_pos = make_float3(100, 100, 100);
    gpu_solver.config.knife_radius = avg_spacing * 8.0f;
    gpu_solver.config.knife_active = 0;
    gpu_solver.config.self_collision_radius = avg_spacing * 0.45f;
    gpu_solver.config.self_collision_stiffness = 0.8f;
    gpu_solver.config.self_collision_enabled = 0;  // 先关掉self-collision简化debug

    gpu_solver.init(positions, velocities, removed, fixed_arr,
                    cluster_starts, cluster_sizes,
                    all_particle_ids, all_rest_offsets, all_rest_cm);

    printf("\n--- Running 20 steps ---\n");
    printf("dt=%.4f, stiffness=%.2f, iters=%d, gravity_y=%.2f\n",
           gpu_solver.config.dt, gpu_solver.config.stiffness,
           gpu_solver.config.num_iterations, gpu_solver.config.gravity.y);
    printf("grav_displacement/step = %.8f\n",
           0.5f * 9.81f * 0.005f * 0.005f);

    std::vector<float3> cpu_pos(N);

    for (int step = 0; step < 20; step++) {
        gpu_solver.step();
        cudaDeviceSynchronize();

        gpu_solver.download_positions(cpu_pos);

        // 统计非fixed粒子的Y位移和速度
        double sum_dy = 0, max_dy = 0;
        double min_py = 1e10, max_py = -1e10;
        int active = 0;
        for (int i = 0; i < N; i++) {
            if (fixed_arr[i]) continue;
            active++;
            double dy = cpu_pos[i].y - positions[i].y;  // 相对初始位置
            sum_dy += dy;
            double ady = fabs(dy);
            if (ady > max_dy) max_dy = ady;
            if (cpu_pos[i].y < min_py) min_py = cpu_pos[i].y;
            if (cpu_pos[i].y > max_py) max_py = cpu_pos[i].y;
        }
        if (active > 0) sum_dy /= active;
        printf("Step %2d: avg_dY=%.8f  max_|dY|=%.8f  Y=[%.6f, %.6f]  active=%d\n",
               step, sum_dy, max_dy, min_py, max_py, active);
    }

    gpu_solver.cleanup();
    return 0;
}
