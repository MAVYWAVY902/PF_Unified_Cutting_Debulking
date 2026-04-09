// GPU + OpenGL Viewer 集成入口
// GPU solver 负责仿真，每帧下载 pos/removed/damage 到 CPU 给 viewer 渲染
#include "solver/shape_matching_gpu.h"
#include "solver/shape_matching_cpu.h"  // for ShapeMatchingSolver (holds CPU particles for viewer)
#include "core/particles.h"
#include "core/cluster.h"
#include "core/tet_mesh.h"
#include "core/pf_calculator.h"
#include "core/cutting_manager.h"
#include "core/sim_config.h"
#include "core/rigid_mesh.h"
#include "render/viewer.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <queue>

// CPU BFS 分裂：下载边状态 → 找连通分量 → 分裂簇 → 移除碎片
// 返回 true 如果有任何分裂或移除发生
bool cpu_bfs_split(
    ShapeMatchingGPUSolver& gpu_solver,
    ShapeMatchingSolver& cpu_solver,
    std::vector<int>& edge_a,
    std::vector<int>& edge_b,
    std::vector<float>& edge_pf_rest,
    std::vector<float3>& cpu_positions,
    std::vector<int>& cpu_removed,
    const std::vector<std::vector<int>>& vertex_neighbors,  // tet mesh 完整邻接
    int min_fragment_size = 5)
{
    int N = cpu_solver.particles.size();

    // 1. 下载边状态
    std::vector<int> edge_broken;
    std::vector<float> edge_damage;
    gpu_solver.download_edge_broken(edge_broken);
    gpu_solver.download_edge_damage(edge_damage);
    int E = (int)edge_a.size();

    // 2. 构建粒子 → 簇映射
    int C = (int)cpu_solver.clusters.size();
    std::vector<int> particle_to_cluster(N, -1);
    for (int ci = 0; ci < C; ci++) {
        for (int pid : cpu_solver.clusters[ci].particle_ids) {
            particle_to_cluster[pid] = ci;
        }
    }

    // 3. 构建 broken 边集合（快速查找）
    auto edge_key = [](int a, int b) -> int64_t {
        if (a > b) std::swap(a, b);
        return (int64_t)a * 100000LL + b;
    };
    std::unordered_set<int64_t> broken_edge_set;
    for (int ei = 0; ei < E; ei++) {
        if (edge_broken[ei]) {
            broken_edge_set.insert(edge_key(edge_a[ei], edge_b[ei]));
        }
    }

    // 4. 全局连通分量分析：找活跃粒子中的连通分量
    //    使用 mesh 邻接（vertex_neighbors），减去 broken edges
    //    小于 min_fragment_size 的分量 → 移除
    bool any_change = false;

    // 构建全局 alive 集合
    std::unordered_set<int> alive_global;
    for (int i = 0; i < N; i++) {
        if (!cpu_removed[i]) alive_global.insert(i);
    }

    // 全局 BFS
    std::unordered_set<int> visited;
    std::vector<std::vector<int>> components;
    for (int pid : alive_global) {
        if (visited.count(pid)) continue;
        std::vector<int> comp;
        std::queue<int> q;
        q.push(pid);
        visited.insert(pid);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            comp.push_back(v);
            if (v < (int)vertex_neighbors.size()) {
                for (int nb : vertex_neighbors[v]) {
                    if (!alive_global.count(nb)) continue;
                    if (visited.count(nb)) continue;
                    if (broken_edge_set.count(edge_key(v, nb))) continue;
                    visited.insert(nb);
                    q.push(nb);
                }
            }
        }
        components.push_back(std::move(comp));
    }

    // 找最大分量，其余标记为 removed
    int largest_idx = 0;
    for (int i = 1; i < (int)components.size(); i++) {
        if (components[i].size() > components[largest_idx].size())
            largest_idx = i;
    }

    for (int i = 0; i < (int)components.size(); i++) {
        if (i == largest_idx) continue;
        if ((int)components[i].size() < min_fragment_size) {
            for (int pid : components[i]) cpu_removed[pid] = 1;
            any_change = true;
        }
    }

    // 重建簇（移除只剩 <2 活跃粒子的簇）
    std::vector<Cluster> new_clusters;
    int kept = 0, shrunk = 0, tiny = 0;
    for (int ci = 0; ci < C; ci++) {
        auto& cluster = cpu_solver.clusters[ci];
        std::vector<int> alive_ids;
        for (int pid : cluster.particle_ids) {
            if (!cpu_removed[pid]) alive_ids.push_back(pid);
        }
        if ((int)alive_ids.size() < 2) {
            for (int pid : alive_ids) { cpu_removed[pid] = 1; any_change = true; }
            tiny++;
            continue;
        }
        if (alive_ids.size() != cluster.particle_ids.size()) {
            Cluster nc;
            nc.particle_ids = alive_ids;
            nc.init(cpu_solver.particles);
            new_clusters.push_back(std::move(nc));
            any_change = true;
            shrunk++;
        } else {
            new_clusters.push_back(cluster);  // copy, 不能 move（可能 return false 不赋值回去）
            kept++;
        }
    }
    printf("[SPLIT DEBUG] C=%d kept=%d shrunk=%d tiny=%d new_clusters=%d components=%d removed_by_bfs=%d\n",
           C, kept, shrunk, tiny, (int)new_clusters.size(), (int)components.size(),
           (int)std::count(cpu_removed.begin(), cpu_removed.end(), 1));

    if (!any_change) return false;

    // 5. 更新簇列表
    cpu_solver.clusters = std::move(new_clusters);
    // 移除只有 0-1 个粒子的簇
    cpu_solver.clusters.erase(
        std::remove_if(cpu_solver.clusters.begin(), cpu_solver.clusters.end(),
            [](const Cluster& c) { return c.particle_ids.size() < 2; }),
        cpu_solver.clusters.end());

    // 6. 同步 particles removed 状态（BFS 中可能新标记了 removed）
    for (int i = 0; i < N; i++) {
        cpu_solver.particles[i].removed = (cpu_removed[i] != 0);
        cpu_solver.particles[i].pos = Vec3(
            cpu_positions[i].x, cpu_positions[i].y, cpu_positions[i].z);
    }

    // 7. 剪枝边（移除断裂的和涉及 removed 粒子的）
    std::vector<int> new_edge_a, new_edge_b;
    std::vector<float> new_pf_rest, new_damage;
    std::vector<int> new_broken;
    for (int ei = 0; ei < E; ei++) {
        if (edge_broken[ei]) continue;
        if (cpu_removed[edge_a[ei]] || cpu_removed[edge_b[ei]]) continue;
        new_edge_a.push_back(edge_a[ei]);
        new_edge_b.push_back(edge_b[ei]);
        new_pf_rest.push_back(edge_pf_rest[ei]);
        new_damage.push_back(edge_damage[ei]);
        new_broken.push_back(0);
    }

    // 8. 重新打包 CSR
    int new_C = (int)cpu_solver.clusters.size();
    std::vector<int> csr_starts(new_C), csr_sizes(new_C);
    std::vector<int> csr_pids;
    std::vector<float3> csr_offsets;
    std::vector<float3> csr_cm(new_C);

    int offset = 0;
    for (int ci = 0; ci < new_C; ci++) {
        csr_starts[ci] = offset;
        csr_sizes[ci] = (int)cpu_solver.clusters[ci].particle_ids.size();
        for (int j = 0; j < csr_sizes[ci]; j++) {
            csr_pids.push_back(cpu_solver.clusters[ci].particle_ids[j]);
            csr_offsets.push_back(make_float3(
                (float)cpu_solver.clusters[ci].rest_offsets[j].x(),
                (float)cpu_solver.clusters[ci].rest_offsets[j].y(),
                (float)cpu_solver.clusters[ci].rest_offsets[j].z()));
        }
        csr_cm[ci] = make_float3(
            (float)cpu_solver.clusters[ci].rest_cm.x(),
            (float)cpu_solver.clusters[ci].rest_cm.y(),
            (float)cpu_solver.clusters[ci].rest_cm.z());
        offset += csr_sizes[ci];
    }

    // 9. 上传到 GPU
    gpu_solver.upload_removed(cpu_removed);
    gpu_solver.reupload_clusters(csr_starts, csr_sizes, csr_pids, csr_offsets, csr_cm);
    gpu_solver.reupload_edges(new_edge_a, new_edge_b, new_pf_rest, new_damage, new_broken);

    // 更新 CPU 端边数据
    edge_a = std::move(new_edge_a);
    edge_b = std::move(new_edge_b);
    edge_pf_rest = std::move(new_pf_rest);

    printf("[SPLIT] clusters: %d, edges: %d, removed: %d\n",
           new_C, (int)edge_a.size(),
           (int)std::count(cpu_removed.begin(), cpu_removed.end(), 1));

    return true;
}

int main(int argc, char** argv) {
    std::string input = (argc > 1) ? argv[1] : "10";

    // 加载配置
    SimConfigYAML cfg;
    bool config_loaded = false;
    if (input.size() > 5 && input.substr(input.size()-5) == ".yaml") {
        if (!cfg.load(input)) return 1;
        config_loaded = true;
        std::cout << "加载配置: " << input << std::endl;
    }

    // 判断输入类型
    bool is_mesh = false;
    if (config_loaded) {
        is_mesh = !cfg.mesh_path.empty() && !cfg.use_particle_grid;
    } else {
        bool is_number = !input.empty() && std::all_of(input.begin(), input.end(), ::isdigit);
        if (!is_number && input.size() > 4) {
            is_mesh = true;
            cfg.mesh_path = input;
        } else {
            cfg.use_particle_grid = true;
            cfg.grid_side = std::atoi(input.c_str());
        }
    }

    // CPU 端数据（viewer 读取用）
    ShapeMatchingSolver cpu_solver;
    TetMesh tet_mesh;
    bool use_tet = false;
    double mesh_height = 0.5;
    double deformable_y_offset = 0.0;

    if (is_mesh) {
        // Tet mesh 模式
        std::cout << "=== GPU Shape Matching + Viewer (Tet Mesh) ===" << std::endl;
        if (!tet_mesh.load_msh(cfg.mesh_path)) {
            std::cerr << "加载失败: " << cfg.mesh_path << std::endl;
            return 1;
        }
        use_tet = true;

        double min_y = 1e10, max_y = -1e10;
        for (auto& v : tet_mesh.vertices) {
            min_y = std::min(min_y, v.y());
            max_y = std::max(max_y, v.y());
        }
        double y_offset = -min_y + 0.001;
        deformable_y_offset = y_offset;
        mesh_height = max_y - min_y;

        for (int i = 0; i < tet_mesh.num_vertices(); i++) {
            Vec3 pos = tet_mesh.vertices[i] + Vec3(0, y_offset, 0);
            tet_mesh.vertices[i] = pos;
            cpu_solver.particles.add(pos, 1.0, 0);
        }

        // 固定底面
        int fixed_count = 0;
        if (cfg.fix_bottom) {
            double fixed_threshold = 0.001 + mesh_height * cfg.fixed_ratio;
            for (int i = 0; i < cpu_solver.particles.size(); i++) {
                if (cpu_solver.particles[i].pos.y() < fixed_threshold) {
                    cpu_solver.particles[i].fixed = true;
                    fixed_count++;
                }
            }
            std::cout << "固定底面粒子: " << fixed_count << " / " << cpu_solver.particles.size()
                      << " (阈值: " << fixed_threshold << ")" << std::endl;
        } else {
            std::cout << "底面固定: 关闭" << std::endl;
        }

        cpu_solver.clusters = tet_mesh.build_clusters_from_tets(cpu_solver.particles, cfg.cluster_rings);
        std::cout << "粒子: " << cpu_solver.particles.size()
                  << ", 簇: " << cpu_solver.clusters.size() << std::endl;
    } else {
        // 粒子模式
        std::cout << "=== GPU Shape Matching + Viewer (粒子模式) ===" << std::endl;
        double spacing = cfg.grid_spacing;
        int side = cfg.grid_side;
        for (int x = 0; x < side; x++)
            for (int y = 0; y < side; y++)
                for (int z = 0; z < side; z++)
                    cpu_solver.particles.add(Vec3(x, y + 0.5, z) * spacing, 1.0, 0);

        cpu_solver.clusters = build_overlapping_clusters(cpu_solver.particles, spacing * 2.5);
        std::cout << "粒子: " << cpu_solver.particles.size()
                  << ", 簇: " << cpu_solver.clusters.size() << std::endl;
    }

    int N = cpu_solver.particles.size();
    int C = (int)cpu_solver.clusters.size();

    // --- 打包数据为 GPU 格式 ---
    std::vector<float3> positions(N), velocities(N);
    std::vector<int> removed(N, 0), fixed(N, 0);

    for (int i = 0; i < N; i++) {
        positions[i] = make_float3((float)cpu_solver.particles[i].pos.x(),
                                    (float)cpu_solver.particles[i].pos.y(),
                                    (float)cpu_solver.particles[i].pos.z());
        velocities[i] = make_float3(0, 0, 0);
        fixed[i] = cpu_solver.particles[i].fixed ? 1 : 0;
    }

    // CSR 格式
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

    // --- 初始化 edges（cutting 用）---
    // 从 CuttingManager 获取边数据
    PFCalculator pf_calc;
    pf_calc.alpha_min = cfg.alpha_min;
    if (use_tet) {
        pf_calc.init_from_tets(tet_mesh.tets, N);
    } else {
        pf_calc.init_from_clusters(cpu_solver.clusters, N);
    }

    CuttingManager cutting_mgr;
    cutting_mgr.alpha_min = cfg.alpha_min;
    cutting_mgr.tau_low = 0.05f;
    cutting_mgr.init(cpu_solver.clusters, cpu_solver.particles);

    // 提取边数据给 GPU
    std::vector<int> edge_a, edge_b;
    std::vector<float> edge_pf_rest;
    for (auto& e : cutting_mgr.edges) {
        edge_a.push_back(e.particle_a);
        edge_b.push_back(e.particle_b);
        edge_pf_rest.push_back(e.pf_rest);
    }
    int num_edges = (int)edge_a.size();
    std::cout << "边数: " << num_edges << std::endl;

    // --- 计算mesh统计量，用于auto参数 ---
    float avg_spacing = 0.0f;
    {
        int sample = std::min(N, 1000);
        double total_nn = 0.0; int count = 0;
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
            total_nn += sqrtf(best); count++;
        }
        avg_spacing = (float)(total_nn / count);
    }

    double avg_cluster_membership = 0;
    {
        std::vector<int> membership(N, 0);
        for (auto& c : cpu_solver.clusters)
            for (int pid : c.particle_ids) membership[pid]++;
        double sum = 0; int cnt = 0;
        for (int i = 0; i < N; i++) {
            if (!cpu_solver.particles[i].fixed && membership[i] > 0) {
                sum += membership[i]; cnt++;
            }
        }
        avg_cluster_membership = cnt > 0 ? sum / cnt : 1.0;
    }
    std::cout << "平均粒子间距: " << avg_spacing
              << ", cluster membership: " << avg_cluster_membership << std::endl;

    // --- Resolve auto 参数 ---
    cfg.resolve_auto(avg_spacing, mesh_height, avg_cluster_membership);

    // --- 初始化 GPU solver ---
    ShapeMatchingGPUSolver gpu_solver;
    gpu_solver.config.dt = cfg.dt;
    gpu_solver.config.gravity = make_float3(cfg.gravity[0], cfg.gravity[1], cfg.gravity[2]);
    gpu_solver.config.num_iterations = cfg.num_iterations;
    gpu_solver.config.stiffness = cfg.stiffness;
    gpu_solver.config.damping = cfg.damping;
    gpu_solver.config.floor_y = cfg.floor_y;
    gpu_solver.config.knife_pos = make_float3(100, 100, 100);
    gpu_solver.config.knife_radius = cfg.knife_radius;
    gpu_solver.config.knife_active = 0;
    gpu_solver.config.self_collision_radius = cfg.self_collision_radius;
    gpu_solver.config.self_collision_stiffness = cfg.self_collision_stiffness;
    gpu_solver.config.self_collision_enabled = cfg.self_collision_enabled ? 1 : 0;

    int substeps = cfg.substeps;

    gpu_solver.cutting_update_interval = cfg.cutting_interval;
    gpu_solver.alpha_min = cfg.alpha_min;
    gpu_solver.aspiration_threshold = cfg.aspiration_threshold;

    cfg.print();

    gpu_solver.init(positions, velocities, removed, fixed,
                    cluster_starts, cluster_sizes,
                    all_particle_ids, all_rest_offsets, all_rest_cm);

    if (num_edges > 0) {
        gpu_solver.init_edges(edge_a, edge_b, edge_pf_rest);
    }

    // --- Rigid mesh collision ---
    RigidMesh rigid_mesh;
    if (!cfg.rigid_mesh_path.empty()) {
        if (!rigid_mesh.load_obj(cfg.rigid_mesh_path)) {
            std::cerr << "Rigid mesh加载失败: " << cfg.rigid_mesh_path << std::endl;
        } else {
            // 和deformable mesh做同样的y_offset，保持相对坐标
            if (deformable_y_offset != 0.0) {
                rigid_mesh.translate(Vec3(0, deformable_y_offset, 0));
                std::cout << "Rigid mesh y_offset: " << deformable_y_offset << std::endl;
            }

            // 生成SDF (CPU, 128分辨率)
            rigid_mesh.compute_sdf(128, 0.1);

            // 上传到GPU
            float3 origin = make_float3(
                (float)rigid_mesh.sdf_origin.x(),
                (float)rigid_mesh.sdf_origin.y(),
                (float)rigid_mesh.sdf_origin.z());
            float3 cs = make_float3(
                (float)rigid_mesh.sdf_cell_size.x(),
                (float)rigid_mesh.sdf_cell_size.y(),
                (float)rigid_mesh.sdf_cell_size.z());
            gpu_solver.init_rigid_sdf(rigid_mesh.sdf_data,
                rigid_mesh.sdf_res[0], rigid_mesh.sdf_res[1], rigid_mesh.sdf_res[2],
                origin, cs);
        }
    }

    // --- CPU 端 buffers（GPU → CPU 下载用）---
    std::vector<float3> cpu_positions(N);
    std::vector<int> cpu_removed(N, 0);
    std::vector<float> cpu_damage(N, 0.0f);

    // --- 设置 Viewer ---
    Viewer viewer;
    viewer.solver = &cpu_solver;
    viewer.particle_damage_array = &cpu_damage;
    viewer.num_clusters_display = C;

    // 同步参数到viewer
    viewer.knife_radius = gpu_solver.config.knife_radius;
    viewer.substeps_per_frame = substeps;
    if (rigid_mesh.num_faces() > 0) {
        viewer.rigid_mesh_render = &rigid_mesh;
    }

    if (use_tet) {
        viewer.tet_mesh = &tet_mesh;
        viewer.draw_surface = cfg.draw_surface;
        viewer.draw_wireframe = cfg.draw_wireframe;
        viewer.draw_points = cfg.draw_points;
    } else {
        viewer.draw_points = cfg.draw_points;
    }

    int timing_frame = 0;

    // GPU step 回调：替代 solver->step()
    viewer.on_step = [&](int substeps) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 同步刀具参数到 GPU
        gpu_solver.config.knife_pos = make_float3(
            (float)viewer.knife_pos.x(),
            (float)viewer.knife_pos.y(),
            (float)viewer.knife_pos.z());
        gpu_solver.config.knife_radius = (float)viewer.knife_radius;
        gpu_solver.config.knife_active = viewer.knife_active ? 1 : 0;

        // GPU 仿真
        for (int sub = 0; sub < substeps; sub++) {
            gpu_solver.step();
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        // 下载结果到 CPU
        gpu_solver.download_positions(cpu_positions);
        gpu_solver.download_removed(cpu_removed);
        gpu_solver.download_damage(cpu_damage);

        // 同步到 cpu_solver.particles（viewer 读取用）
        int active = 0, rem = 0;
        for (int i = 0; i < N; i++) {
            cpu_solver.particles[i].pos = Vec3(
                cpu_positions[i].x, cpu_positions[i].y, cpu_positions[i].z);
            cpu_solver.particles[i].removed = (cpu_removed[i] != 0);
            if (cpu_removed[i]) rem++; else active++;
        }
        viewer.num_active_particles = active;
        viewer.num_removed_particles = rem;

        // 边界侵蚀：移除大部分邻居已被移除的孤立粒子
        if (use_tet && rem > 0) {
            bool eroded = false;
            for (int i = 0; i < N; i++) {
                if (cpu_removed[i]) continue;
                const auto& nbrs = tet_mesh.vertex_neighbors[i];
                if (nbrs.empty()) continue;
                int alive_nbrs = 0;
                for (int nb : nbrs) {
                    if (!cpu_removed[nb]) alive_nbrs++;
                }
                if ((float)alive_nbrs / nbrs.size() < 0.4f) {
                    cpu_removed[i] = 1;
                    cpu_solver.particles[i].removed = true;
                    eroded = true;
                }
            }
            if (eroded) {
                gpu_solver.upload_removed(cpu_removed);
                // 重新计数
                active = 0; rem = 0;
                for (int i = 0; i < N; i++) {
                    if (cpu_removed[i]) rem++; else active++;
                }
                viewer.num_active_particles = active;
                viewer.num_removed_particles = rem;
            }
        }

        timing_frame++;

        // 周期性状态输出
        if (timing_frame <= 5 || timing_frame % 300 == 0) {
            double dt_gpu = std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("[frame %d] gpu=%.1fms active=%d removed=%d\n",
                timing_frame, dt_gpu, active, rem);
        }
    };

    // 自动调整相机
    Vec3 bb_min(1e10, 1e10, 1e10), bb_max(-1e10, -1e10, -1e10);
    for (int i = 0; i < N; i++) {
        bb_min = bb_min.cwiseMin(cpu_solver.particles[i].pos);
        bb_max = bb_max.cwiseMax(cpu_solver.particles[i].pos);
    }
    Vec3 center = (bb_min + bb_max) * 0.5;
    double extent = (bb_max - bb_min).norm();

    std::cout << "Bounding box: (" << bb_min.x() << "," << bb_min.y() << "," << bb_min.z()
              << ") → (" << bb_max.x() << "," << bb_max.y() << "," << bb_max.z() << ")" << std::endl;

    viewer.set_camera(
        Eigen::Vector3f((float)center.x(), (float)center.y(), (float)center.z()),
        (float)(extent * 1.5)
    );

    viewer.substeps_per_frame = 2;  // 减少 substeps 提高帧率

    viewer.knife_pos = Vec3(bb_max.x() + extent * 5.0, center.y(), center.z());
    viewer.knife_radius = extent * 0.08;

    if (!viewer.init(1280, 720, "P(F) Shape Matching [GPU]")) {
        std::cerr << "Viewer 初始化失败" << std::endl;
        gpu_solver.cleanup();
        return 1;
    }

    std::cout << "\n按键:" << std::endl;
    std::cout << "  LMB: 旋转  RMB: 平移  滚轮: 缩放" << std::endl;
    std::cout << "  K: 开关刀具  Space+鼠标: 移动刀具  Space+滚轮: 前后" << std::endl;
    std::cout << "  P: 暂停  1: 切换面  2: 切换线框  3: 切换粒子点" << std::endl;
    std::cout << "  T: 相机追踪  R: 重置刀具  ESC: 退出" << std::endl;

    viewer.run();

    gpu_solver.cleanup();
    return 0;
}
