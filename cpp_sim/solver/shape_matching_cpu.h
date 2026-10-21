#pragma once

#include "../core/particles.h"
#include "../core/cluster.h"
#include "../core/pf_calculator.h"
#include "../core/cutting_manager.h"
#include <vector>
#include <algorithm>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

struct SimConfig {
    double dt = 0.005;
    Vec3 gravity = Vec3(0, -9.81, 0);
    int num_iterations = 5;
    double stiffness = 1.0;
    double damping = 0.98;
    double floor_y = 0.0;
    bool enable_floor = true;
};

class ShapeMatchingSolver {
public:
    Particles particles;
    std::vector<Cluster> clusters;
    SimConfig config;

    // P(F) 可视化
    PFCalculator* pf_calc = nullptr;

    // 切割管理器
    CuttingManager* cutting_mgr = nullptr;
    int cutting_update_interval = 200;  // 每 200 步检查一次切割（先验证能跑）

    // 刀具碰撞
    Vec3 knife_pos = Vec3::Zero();
    double knife_radius = 0;
    bool knife_collision = false;

    int step_count = 0;
    bool pf_rest_saved = false;

    void step() {
        int N = particles.size();
        double dt = config.dt;

        // 保存旧位置
        std::vector<Vec3> old_pos(N);
        for (int i = 0; i < N; i++) {
            old_pos[i] = particles[i].pos;
        }

        // 1. 外力 + 阻尼 + 预测（跳过 removed 和 fixed 粒子）
        for (int i = 0; i < N; i++) {
            if (particles[i].removed || particles[i].fixed) continue;
            particles[i].vel *= config.damping;          // damp first: gravity is not scaled by damping
            particles[i].vel += config.gravity * dt;
            particles[i].predicted_pos = particles[i].pos + particles[i].vel * dt;
        }

        // 收集预测位置
        std::vector<Vec3> pred_pos(N);
        for (int i = 0; i < N; i++) {
            pred_pos[i] = particles[i].predicted_pos;
        }

        // 2. 约束求解 (Jacobi)
        std::vector<Vec3> deltas(N, Vec3::Zero());
        std::vector<int> counts(N, 0);

        int num_clusters = static_cast<int>(clusters.size());
        int num_threads = 1;
        #ifdef _OPENMP
        num_threads = omp_get_max_threads();
        #endif

        std::vector<std::vector<Vec3>> thread_deltas(num_threads, std::vector<Vec3>(N, Vec3::Zero()));
        std::vector<std::vector<int>> thread_counts(num_threads, std::vector<int>(N, 0));

        for (int iter = 0; iter < config.num_iterations; iter++) {
            for (int t = 0; t < num_threads; t++) {
                std::fill(thread_deltas[t].begin(), thread_deltas[t].end(), Vec3::Zero());
                std::fill(thread_counts[t].begin(), thread_counts[t].end(), 0);
            }

            #pragma omp parallel for schedule(static)
            for (int c = 0; c < num_clusters; c++) {
                int tid = 0;
                #ifdef _OPENMP
                tid = omp_get_thread_num();
                #endif
                clusters[c].solve(pred_pos, thread_deltas[tid], thread_counts[tid]);
            }

            // 合并
            std::fill(deltas.begin(), deltas.end(), Vec3::Zero());
            std::fill(counts.begin(), counts.end(), 0);
            for (int t = 0; t < num_threads; t++) {
                for (int i = 0; i < N; i++) {
                    deltas[i] += thread_deltas[t][i];
                    counts[i] += thread_counts[t][i];
                }
            }

            // Apply（per-particle damage → 局部 stiffness 降低）
            for (int i = 0; i < N; i++) {
                if (particles[i].removed || particles[i].fixed || counts[i] == 0) continue;
                float dmg = cutting_mgr ? cutting_mgr->get_particle_damage(i) : 0.0f;
                float local_stiff = config.stiffness * (1.0f - dmg * 0.95f);
                pred_pos[i] += local_stiff * deltas[i] / counts[i];
            }

            // 碰撞约束（每次迭代都执行，标准 PBD 做法）
            _apply_collision_constraints(pred_pos, N);
        }

        // 3. 切割检测（P(F) 边损伤 + 簇分裂 + 碎片处理）
        if (cutting_mgr && knife_collision && (step_count % cutting_update_interval == 0)) {
            cutting_mgr->update(particles, knife_pos, knife_radius);
            cutting_mgr->check_and_split(clusters, particles);
            cutting_mgr->process_fragments(clusters, particles, knife_pos, knife_radius);

            // 详细统计（逐层）
            if (step_count % 400 == 0) {
                // 总体统计
                int total_active = 0, total_removed = 0;
                int layer_active[3] = {0, 0, 0};
                int layer_removed[3] = {0, 0, 0};
                float layer_max_dmg[3] = {0, 0, 0};
                int layer_damaged[3] = {0, 0, 0};

                for (int i = 0; i < (int)particles.size(); i++) {
                    int lid = particles[i].layer_id;
                    if (lid < 0 || lid > 2) lid = 0;
                    if (particles[i].removed) {
                        total_removed++;
                        layer_removed[lid]++;
                    } else {
                        total_active++;
                        layer_active[lid]++;
                        float dmg = cutting_mgr->get_particle_damage(i);
                        layer_max_dmg[lid] = std::max(layer_max_dmg[lid], dmg);
                        if (dmg > 0.1f) layer_damaged[lid]++;
                    }
                }

                printf("[step %d] active=%d removed=%d | "
                       "A(dense): %d left, %d removed, %d damaged, maxDmg=%.2f | "
                       "B(medium): %d left, %d removed, %d damaged, maxDmg=%.2f | "
                       "C(nerve): %d left, %d removed, %d damaged, maxDmg=%.2f\n",
                    step_count, total_active, total_removed,
                    layer_active[0], layer_removed[0], layer_damaged[0], layer_max_dmg[0],
                    layer_active[1], layer_removed[1], layer_damaged[1], layer_max_dmg[1],
                    layer_active[2], layer_removed[2], layer_damaged[2], layer_max_dmg[2]);
            }
        }
        step_count++;

        // 4. 速度更新（跳过 fixed/removed）
        for (int i = 0; i < N; i++) {
            if (particles[i].removed || particles[i].fixed) continue;
            particles[i].vel = (pred_pos[i] - old_pos[i]) / dt;
            particles[i].pos = pred_pos[i];
            particles[i].predicted_pos = pred_pos[i];
        }
    }

private:
    // 碰撞约束：地板 + 刀具（标准 PBD：每次迭代都投影）
    void _apply_collision_constraints(std::vector<Vec3>& pred_pos, int N) {
        // 地板
        if (config.enable_floor) {
            for (int i = 0; i < N; i++) {
                if (pred_pos[i].y() < config.floor_y) {
                    pred_pos[i].y() = config.floor_y;
                }
            }
        }
        // 刀具（球形 SDF）
        if (knife_collision) {
            for (int i = 0; i < N; i++) {
                Vec3 diff = pred_pos[i] - knife_pos;
                double dist = diff.norm();
                if (dist < knife_radius && dist > 1e-10) {
                    Vec3 n = diff / dist;
                    pred_pos[i] = knife_pos + n * knife_radius;
                }
            }
        }
    }
};
