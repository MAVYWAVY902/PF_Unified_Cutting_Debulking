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
#ifdef HAPTICS_ENABLED
#include "core/haply_device.h"
#endif
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
    const TetMesh& tet_mesh,
    int min_fragment_size = 5)
{
    int N = cpu_solver.particles.size();
    int E = (int)edge_a.size();
    bool any_change = false;

    // 1. 下载边状态
    std::vector<int> edge_broken;
    std::vector<float> edge_damage;
    gpu_solver.download_edge_broken(edge_broken);
    gpu_solver.download_edge_damage(edge_damage);

    // 2. 构建 broken 边集合
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

    // 3. BFS用vertex_neighbors作为图，跳过broken edges和removed粒子
    //    旧BFS的逻辑 — 在视觉fragmentation方面更准确
    std::vector<int> component_id(N, -1);
    std::vector<std::vector<int>> components;

    for (int seed = 0; seed < N; seed++) {
        if (cpu_removed[seed] || component_id[seed] != -1) continue;

        int comp_idx = (int)components.size();
        components.push_back({});
        auto& comp = components.back();

        std::queue<int> q;
        q.push(seed);
        component_id[seed] = comp_idx;

        while (!q.empty()) {
            int v = q.front(); q.pop();
            comp.push_back(v);

            if (v < (int)tet_mesh.vertex_neighbors.size()) {
                for (int nb : tet_mesh.vertex_neighbors[v]) {
                    if (cpu_removed[nb]) continue;
                    if (component_id[nb] != -1) continue;
                    if (broken_edge_set.count(edge_key(v, nb))) continue;
                    component_id[nb] = comp_idx;
                    q.push(nb);
                }
            }
        }
    }

    // 找最大分量
    int largest_idx = 0;
    for (int i = 1; i < (int)components.size(); i++) {
        if (components[i].size() > components[largest_idx].size())
            largest_idx = i;
    }

    // 对非最大分量：
    //   size <  min_fragment_size → 标记 removed（clinical = aspirated 小碎末被吸走）
    //   size >= min_fragment_size → 保留，让物理/重力自然处理（掉落、继续和刀交互）
    // 这修正了原版"除最大以外全删"的行为，之前大块分离体会突然消失。
    int dropped_small = 0, kept_big = 0;
    for (int i = 0; i < (int)components.size(); i++) {
        if (i == largest_idx) continue;
        if ((int)components[i].size() < min_fragment_size) {
            for (int pid : components[i]) cpu_removed[pid] = 1;
            dropped_small++;
            any_change = true;
        } else {
            kept_big++;
        }
    }

    // 第 1 部分 — 如果移除了小碎片：同步 CPU particles 状态 + 上传到 GPU
    if (any_change) {
        for (int i = 0; i < N; i++) {
            cpu_solver.particles[i].removed = (cpu_removed[i] != 0);
        }
        gpu_solver.upload_removed(cpu_removed);

        int total_removed = (int)std::count(cpu_removed.begin(), cpu_removed.end(), 1);
        printf("[SPLIT] components=%d dropped_small=%d kept_big=%d total_removed=%d\n",
               (int)components.size(), dropped_small, kept_big, total_removed);
    }

    // 第 2 部分 — 分裂横跨切口的 cluster（关键：消除"隐形弹簧"）
    // 不做这步的话：ring-2 BFS 建的 cluster 会同时含有切口两侧的 alive 粒子，
    // cluster_solve_kernel 对这类 cluster 还是把两侧一起 shape-match，效果
    // 等价于切口两侧每 cluster 一根弹簧把碎片拉回主体。
    // 只有 components > 1 才需要做（单连通图没有跨切口的 cluster）。
    bool clusters_changed = false;
    if ((int)components.size() > 1) {
        std::vector<Cluster> new_clusters;
        new_clusters.reserve(cpu_solver.clusters.size() + 32);
        int split_count = 0;

        for (const auto& c : cpu_solver.clusters) {
            // 按 component_id 分组 cluster 的存活粒子
            std::unordered_map<int, std::vector<int>> by_component;
            for (int pid : c.particle_ids) {
                if (cpu_removed[pid]) continue;
                int cid = component_id[pid];
                if (cid < 0) continue;  // defensive — BFS 已对所有存活粒子赋 id
                by_component[cid].push_back(pid);
            }

            if (by_component.empty()) {
                // 簇全死 → 丢弃
                clusters_changed = true;
                continue;
            }

            if ((int)by_component.size() == 1) {
                // 完全在单一 component 内 → 保持原状（包括原 rest_cm / rest_offsets）
                new_clusters.push_back(c);
            } else {
                // 跨 2+ components → 分裂
                clusters_changed = true;
                split_count++;
                for (auto& [cid, ids] : by_component) {
                    if ((int)ids.size() < 2) continue;  // cluster_solve 要求 size >= 2
                    Cluster nc;
                    nc.particle_ids = ids;
                    std::sort(nc.particle_ids.begin(), nc.particle_ids.end());
                    // 用当前位置作 rest —— 碎片"接受"它分离时刻的形状为新平衡态，
                    // 而不是企图回到原初 tumor 内部的相对位置。
                    nc.init(cpu_solver.particles);
                    new_clusters.push_back(std::move(nc));
                }
            }
        }

        if (clusters_changed) {
            cpu_solver.clusters = std::move(new_clusters);

            // 重建 CSR 扁平数组给 GPU
            int C = (int)cpu_solver.clusters.size();
            std::vector<int> cluster_starts(C), cluster_sizes(C);
            std::vector<int> flat_particle_ids;
            std::vector<float3> flat_rest_offsets;
            std::vector<float3> flat_rest_cm(C);
            flat_particle_ids.reserve(C * 16);   // rings=2 典型 cluster ~15 粒子
            flat_rest_offsets.reserve(C * 16);

            int offset = 0;
            for (int ci = 0; ci < C; ci++) {
                const auto& cl = cpu_solver.clusters[ci];
                cluster_starts[ci] = offset;
                cluster_sizes[ci] = (int)cl.particle_ids.size();
                for (int j = 0; j < cluster_sizes[ci]; j++) {
                    flat_particle_ids.push_back(cl.particle_ids[j]);
                    const Vec3& ro = cl.rest_offsets[j];
                    flat_rest_offsets.push_back(make_float3(
                        (float)ro.x(), (float)ro.y(), (float)ro.z()));
                }
                const Vec3& rc = cl.rest_cm;
                flat_rest_cm[ci] = make_float3((float)rc.x(), (float)rc.y(), (float)rc.z());
                offset += cluster_sizes[ci];
            }

            gpu_solver.reupload_clusters(cluster_starts, cluster_sizes,
                                         flat_particle_ids, flat_rest_offsets,
                                         flat_rest_cm);
            printf("[CLUSTER-SPLIT] split %d clusters across cut → total %d clusters\n",
                   split_count, C);
        }
    }

    return any_change || clusters_changed;
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
    // Identity rotation
    float identity_rot[9] = {1,0,0, 0,1,0, 0,0,1};
    memcpy(gpu_solver.config.knife_rot, identity_rot, 9 * sizeof(float));
    gpu_solver.config.knife_radius = cfg.knife_radius;
    gpu_solver.config.knife_active = 0;
    gpu_solver.config.self_collision_radius = cfg.self_collision_radius;
    gpu_solver.config.self_collision_stiffness = cfg.self_collision_stiffness;
    gpu_solver.config.self_collision_enabled = cfg.self_collision_enabled ? 1 : 0;
    gpu_solver.config.friction_s = cfg.friction_s;
    gpu_solver.config.friction_k = cfg.friction_k;
    gpu_solver.config.tool_type = cfg.tool_mesh_path.empty() ? 0 : 1;

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
            gpu_solver.init_sdf(gpu_solver.d_rigid_sdf, rigid_mesh.sdf_data,
                rigid_mesh.sdf_res[0], rigid_mesh.sdf_res[1], rigid_mesh.sdf_res[2],
                origin, cs, "Bone");
        }
    }

    // --- Tool mesh SDF (optional) ---
    RigidMesh tool_mesh;
    if (!cfg.tool_mesh_path.empty()) {
        if (!tool_mesh.load_obj(cfg.tool_mesh_path)) {
            std::cerr << "Tool mesh加载失败: " << cfg.tool_mesh_path << std::endl;
            gpu_solver.config.tool_type = 0;  // fallback to sphere
        } else {
            // Center tool at origin for local frame SDF
            Vec3 bmin, bmax;
            tool_mesh.get_bounds(bmin, bmax);
            Vec3 center = (bmin + bmax) * 0.5;
            tool_mesh.translate(-center);
            std::cout << "Tool centered at origin, original center: ("
                      << center.x() << ", " << center.y() << ", " << center.z() << ")" << std::endl;

            tool_mesh.compute_sdf(cfg.tool_sdf_resolution, 0.2);

            float3 origin = make_float3(
                (float)tool_mesh.sdf_origin.x(),
                (float)tool_mesh.sdf_origin.y(),
                (float)tool_mesh.sdf_origin.z());
            float3 cs = make_float3(
                (float)tool_mesh.sdf_cell_size.x(),
                (float)tool_mesh.sdf_cell_size.y(),
                (float)tool_mesh.sdf_cell_size.z());
            gpu_solver.init_sdf(gpu_solver.d_tool_sdf, tool_mesh.sdf_data,
                tool_mesh.sdf_res[0], tool_mesh.sdf_res[1], tool_mesh.sdf_res[2],
                origin, cs, "Tool");
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
    if (tool_mesh.num_faces() > 0) {
        viewer.tool_mesh_render = &tool_mesh;
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
        // Sync rotation (row-major 3x3)
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                gpu_solver.config.knife_rot[r * 3 + c] = (float)viewer.knife_rot(r, c);
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

        // Orphan清理：移除拓扑真正孤立的粒子（没有任何存活邻居）
        // 原版用"没有 4-顶点全活的 tet"过于严苛 —— 表面粒子 aspirate 一个邻居就死，
        // 再加 20 次级联，一次刀接触会蒸发 10× 于真正该死的粒子数。
        // 新判据：遍历 vertex_neighbors，至少有 1 个邻居活着就算连接正常。
        // 循环保留（防御性），但新判据下一般 1-2 次就收敛、很少级联。
        if (use_tet && rem > 0) {
            int total_orphans = 0;
            int iter = 0;
            for (; iter < 20; iter++) {
                int round_orphans = 0;
                for (int i = 0; i < N; i++) {
                    if (cpu_removed[i]) continue;
                    if (i >= (int)tet_mesh.vertex_neighbors.size()) continue;
                    const auto& nbrs = tet_mesh.vertex_neighbors[i];
                    if (nbrs.empty()) continue;

                    bool has_alive_neighbor = false;
                    for (int nb : nbrs) {
                        if (!cpu_removed[nb]) { has_alive_neighbor = true; break; }
                    }

                    if (!has_alive_neighbor) {
                        cpu_removed[i] = 1;
                        cpu_solver.particles[i].removed = true;
                        round_orphans++;
                    }
                }
                total_orphans += round_orphans;
                if (round_orphans == 0) break;  // 收敛
            }
            if (total_orphans > 0) {
                if (timing_frame % 60 == 0)
                    printf("[ORPHAN] removed %d in %d iters\n", total_orphans, iter+1);
                gpu_solver.upload_removed(cpu_removed);
                active = 0; rem = 0;
                for (int i = 0; i < N; i++) {
                    if (cpu_removed[i]) rem++; else active++;
                }
                viewer.num_active_particles = active;
                viewer.num_removed_particles = rem;
            }
        }

        // BFS 连通分量分析：小碎片 aspirated、大碎片保留（详见 cpu_bfs_split）
        // 频率从每30帧 → 每5帧：小碎片一生成（< 5 粒子）就被吸走，避免"延迟消失"跳变
        if (use_tet && rem > 0 && timing_frame % 5 == 0) {
            bool changed = cpu_bfs_split(
                gpu_solver, cpu_solver,
                edge_a, edge_b, edge_pf_rest,
                cpu_positions, cpu_removed,
                tet_mesh, 5);
            if (changed) {
                active = 0; rem = 0;
                for (int i = 0; i < N; i++) {
                    if (cpu_removed[i]) rem++; else active++;
                }
                viewer.num_active_particles = active;
                viewer.num_removed_particles = rem;
                C = (int)cpu_solver.clusters.size();
                viewer.num_clusters_display = C;
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

    // 相机：45度俯视（手术台视角）
    // yaw=0 → 正面看, pitch=0.75 → ~43度俯角
    viewer.set_camera(
        Eigen::Vector3f((float)center.x(), (float)center.y(), (float)center.z()),
        (float)(extent * 1.5),
        0.0f,    // yaw: 正对手术区域
        0.75f    // pitch: ~43度俯角（类似CardinalSim的45度）
    );

    // NOTE: viewer.substeps_per_frame 已在上方（line 519）从 cfg.substeps 同步过。
    //   之前这里硬编码 = 2，把 substeps_auto 算出来的补偿值（对小 mesh 可达 40）彻底覆盖，
    //   导致每 render frame 只推进 2×dt ≈ 1ms 仿真时间 → 看起来"重力很弱、下落很慢"。
    //   删掉这行硬编码，让 cfg 的值（yaml 显式或 auto 推导）生效。

    // 刀具初始位置：在手术区域上方，靠近tumor
    viewer.knife_pos = Vec3(center.x(), bb_max.y() + extent * 0.3, center.z());
    // knife_radius 已经在前面从YAML config同步过 (line 431)，不再覆盖

    if (!viewer.init(1280, 720, "P(F) Shape Matching [GPU]")) {
        std::cerr << "Viewer 初始化失败" << std::endl;
        gpu_solver.cleanup();
        return 1;
    }

    std::cout << "\n按键:" << std::endl;
    std::cout << "  LMB: 旋转  RMB: 平移  滚轮: 缩放" << std::endl;
    std::cout << "  K: 开关刀具  Space+鼠标: 移动刀具  Space+滚轮: 前后" << std::endl;
    std::cout << "  X/Y/Z: 旋转刀具 (Shift+X/Y/Z: 反向)" << std::endl;
    std::cout << "  P: 暂停  1: 切换面  2: 切换线框  3: 切换粒子点" << std::endl;
    std::cout << "  T: 相机追踪  R: 重置刀具+旋转  ESC: 退出" << std::endl;

    // --- Haptic device (optional) ---
#ifdef HAPTICS_ENABLED
    std::unique_ptr<HaplyDevice> haply;
    if (cfg.haptics_enabled) {
        haply = std::make_unique<HaplyDevice>();
        HaplyDevice::Config hcfg;
        hcfg.ws_url = cfg.haptics_ws_url;
        hcfg.workspace_center = Vec3(cfg.haptics_workspace_center[0],
                                      cfg.haptics_workspace_center[1],
                                      cfg.haptics_workspace_center[2]);
        hcfg.sim_offset = Vec3(cfg.haptics_sim_offset[0],
                                cfg.haptics_sim_offset[1],
                                cfg.haptics_sim_offset[2]);
        hcfg.scale = cfg.haptics_scale;
        hcfg.versagrip_correction = cfg.haptics_versagrip_correction;
        hcfg.button_activates_knife = cfg.haptics_button_activates_knife;
        for (int i = 0; i < 3; i++) {
            hcfg.pos_axes[i] = cfg.haptics_pos_axes[i];
            hcfg.pos_signs[i] = cfg.haptics_pos_signs[i];
        }

        if (!haply->connect(hcfg)) {
            std::cerr << "Haply device not connected, using keyboard/mouse" << std::endl;
            haply.reset();
        }
    }

    viewer.on_pre_step = [&]() {
        if (haply && haply->is_connected()) {
            Vec3 pos;
            Eigen::Matrix3d rot;
            bool button;
            if (haply->get_state(pos, rot, button)) {
                viewer.knife_pos = pos;
                viewer.knife_rot = rot;
                if (cfg.haptics_button_activates_knife) {
                    viewer.knife_active = button;
                }
            }
        }
    };
#endif

    viewer.run();

#ifdef HAPTICS_ENABLED
    if (haply) haply->disconnect();
#endif

    gpu_solver.cleanup();
    return 0;
}
