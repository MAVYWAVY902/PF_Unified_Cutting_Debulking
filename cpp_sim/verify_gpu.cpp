// CPU vs GPU 逐步对比验证
#include "solver/shape_matching_gpu.h"
#include "solver/shape_matching_cpu.h"
#include "core/particles.h"
#include "core/cluster.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

int main() {
    // === Step 0: CUDA 设备检查 ===
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    printf("=== CUDA 设备检查 ===\n");
    printf("CUDA 设备数量: %d\n", device_count);
    if (device_count == 0) {
        printf("ERROR: 没有 CUDA 设备!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("设备: %s\n", prop.name);
    printf("SM: %d.%d\n", prop.major, prop.minor);
    printf("全局内存: %zu MB\n", prop.totalGlobalMem / (1024*1024));
    printf("SM 数量: %d\n", prop.multiProcessorCount);
    printf("最大线程/block: %d\n", prop.maxThreadsPerBlock);

    // 验证 kernel 在 GPU 上运行
    {
        float* d_test;
        cudaMalloc(&d_test, sizeof(float));
        float val = 42.0f;
        cudaMemcpy(d_test, &val, sizeof(float), cudaMemcpyHostToDevice);
        float result = 0;
        cudaMemcpy(&result, d_test, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_test);
        printf("CUDA 内存读写测试: write=42.0 read=%.1f %s\n",
               result, (result == 42.0f) ? "PASS" : "FAIL");
    }

    // === Step 1: 构建测试场景（小 cube，方便对比）===
    printf("\n=== 构建测试场景 ===\n");
    int side = 5;
    double spacing = 0.1;

    // CPU solver
    ShapeMatchingSolver cpu_solver;
    for (int x = 0; x < side; x++)
        for (int y = 0; y < side; y++)
            for (int z = 0; z < side; z++)
                cpu_solver.particles.add(Vec3(x, y + 2.0, z) * spacing, 1.0, 0);

    cpu_solver.clusters = build_overlapping_clusters(cpu_solver.particles, spacing * 2.5);
    cpu_solver.config.dt = 0.005;
    cpu_solver.config.num_iterations = 4;
    cpu_solver.config.stiffness = 0.6;
    cpu_solver.config.damping = 0.98;
    cpu_solver.config.gravity = Vec3(0, -9.81, 0);

    int N = cpu_solver.particles.size();
    int C = (int)cpu_solver.clusters.size();
    printf("粒子: %d, 簇: %d\n", N, C);

    // GPU solver — 打包数据
    std::vector<float3> positions(N), velocities(N);
    std::vector<int> removed(N, 0), fixed(N, 0);

    for (int i = 0; i < N; i++) {
        positions[i] = make_float3((float)cpu_solver.particles[i].pos.x(),
                                    (float)cpu_solver.particles[i].pos.y(),
                                    (float)cpu_solver.particles[i].pos.z());
        velocities[i] = make_float3(0, 0, 0);
    }

    std::vector<int> cluster_starts(C), cluster_sizes(C);
    std::vector<int> all_particle_ids;
    std::vector<float3> all_rest_offsets;
    std::vector<float3> all_rest_cm(C);

    int offset = 0;
    for (int ci = 0; ci < C; ci++) {
        cluster_starts[ci] = offset;
        cluster_sizes[ci] = (int)cpu_solver.clusters[ci].particle_ids.size();
        for (int j = 0; j < cluster_sizes[ci]; j++) {
            all_particle_ids.push_back(cpu_solver.clusters[ci].particle_ids[j]);
            all_rest_offsets.push_back(make_float3(
                (float)cpu_solver.clusters[ci].rest_offsets[j].x(),
                (float)cpu_solver.clusters[ci].rest_offsets[j].y(),
                (float)cpu_solver.clusters[ci].rest_offsets[j].z()));
        }
        all_rest_cm[ci] = make_float3(
            (float)cpu_solver.clusters[ci].rest_cm.x(),
            (float)cpu_solver.clusters[ci].rest_cm.y(),
            (float)cpu_solver.clusters[ci].rest_cm.z());
        offset += cluster_sizes[ci];
    }

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

    // === Step 2: 逐步对比 ===
    printf("\n=== CPU vs GPU 逐步对比 ===\n");

    std::vector<float3> gpu_pos(N);
    int num_test_steps = 50;

    for (int step = 0; step < num_test_steps; step++) {
        cpu_solver.step();
        gpu_solver.step();
        cudaDeviceSynchronize();

        // 检查 CUDA 错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA ERROR at step %d: %s\n", step, cudaGetErrorString(err));
            break;
        }

        gpu_solver.download_positions(gpu_pos);

        // 比较
        double max_diff = 0;
        double avg_diff = 0;
        int worst_id = -1;

        for (int i = 0; i < N; i++) {
            double dx = cpu_solver.particles[i].pos.x() - gpu_pos[i].x;
            double dy = cpu_solver.particles[i].pos.y() - gpu_pos[i].y;
            double dz = cpu_solver.particles[i].pos.z() - gpu_pos[i].z;
            double diff = std::sqrt(dx*dx + dy*dy + dz*dz);
            avg_diff += diff;
            if (diff > max_diff) {
                max_diff = diff;
                worst_id = i;
            }
        }
        avg_diff /= N;

        if (step < 10 || step % 10 == 0) {
            // CPU/GPU 质心
            Vec3 cpu_cm = Vec3::Zero();
            float3 gpu_cm = make_float3(0, 0, 0);
            for (int i = 0; i < N; i++) {
                cpu_cm += cpu_solver.particles[i].pos;
                gpu_cm.x += gpu_pos[i].x;
                gpu_cm.y += gpu_pos[i].y;
                gpu_cm.z += gpu_pos[i].z;
            }
            cpu_cm /= N;
            gpu_cm.x /= N; gpu_cm.y /= N; gpu_cm.z /= N;

            printf("[step %3d] max_diff=%.6f avg_diff=%.6f | CPU cm=(%.4f,%.4f,%.4f) GPU cm=(%.4f,%.4f,%.4f)\n",
                step, max_diff, avg_diff,
                cpu_cm.x(), cpu_cm.y(), cpu_cm.z(),
                gpu_cm.x, gpu_cm.y, gpu_cm.z);
        }
    }

    // === Step 3: 性能对比 ===
    printf("\n=== 性能对比 (100 steps) ===\n");

    // CPU
    ShapeMatchingSolver cpu2;
    for (int x = 0; x < side; x++)
        for (int y = 0; y < side; y++)
            for (int z = 0; z < side; z++)
                cpu2.particles.add(Vec3(x, y + 2.0, z) * spacing, 1.0, 0);
    cpu2.clusters = build_overlapping_clusters(cpu2.particles, spacing * 2.5);
    cpu2.config = cpu_solver.config;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < 100; s++) cpu2.step();
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // GPU — 重新初始化
    gpu_solver.cleanup();
    for (int i = 0; i < N; i++) {
        positions[i] = make_float3((float)(i % side) * (float)spacing,
                                    (float)((i / side) % side) * (float)spacing + 2.0f,
                                    (float)(i / (side*side)) * (float)spacing);
        velocities[i] = make_float3(0, 0, 0);
    }
    // 重新打包原始位置
    for (int i = 0; i < N; i++) {
        int x = i % side, y = (i / side) % side, z = i / (side * side);
        positions[i] = make_float3(x * (float)spacing, y * (float)spacing + 2.0f, z * (float)spacing);
        velocities[i] = make_float3(0, 0, 0);
    }
    gpu_solver.init(positions, velocities, removed, fixed,
                    cluster_starts, cluster_sizes,
                    all_particle_ids, all_rest_offsets, all_rest_cm);

    // 预热
    for (int s = 0; s < 10; s++) gpu_solver.step();
    cudaDeviceSynchronize();

    t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < 100; s++) gpu_solver.step();
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("CPU: %.2f ms (%.1f FPS)\n", cpu_ms, 100.0 / (cpu_ms / 1000.0));
    printf("GPU: %.2f ms (%.1f FPS)\n", gpu_ms, 100.0 / (gpu_ms / 1000.0));
    printf("加速比: %.1fx\n", cpu_ms / gpu_ms);

    gpu_solver.cleanup();
    printf("\n=== 验证完成 ===\n");
    return 0;
}
