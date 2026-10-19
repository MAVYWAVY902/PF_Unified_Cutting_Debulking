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
#include "core/sim_recorder.h"   // Step 0.5a: --record / --stats (observation only)
#include "core/frame_pipeline.h" // Step 0.5d: shared per-frame pipeline (viewer + tool_bench)
#include <filesystem>
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

int main(int argc, char** argv) {
    std::string input = (argc > 1) ? argv[1] : "10";

    // Step 0.5a: optional observation outputs (no effect on the simulation)
    //   --record <file>  per-frame tool pose fed to the solver (replayable, from frame 0)
    //   --stats  <file>  per-frame state after the full pipeline (removed, contact, artifacts)
    //   --replay <file>  drive the tool from a recording instead of mouse/keyboard/Haply (0.5b)
    //   --replay-allow-mismatch  replay even if config/mesh/tool differ from the recording
    std::string record_path, stats_path, replay_path;
    bool replay_allow_mismatch = false;
    for (int ai = 2; ai < argc; ai++) {
        std::string a = argv[ai];
        if (a == "--record" && ai + 1 < argc) record_path = argv[++ai];
        else if (a == "--stats" && ai + 1 < argc) stats_path = argv[++ai];
        else if (a == "--replay" && ai + 1 < argc) replay_path = argv[++ai];
        else if (a == "--replay-allow-mismatch") replay_allow_mismatch = true;
        else {
            std::cerr << "unknown argument: " << a << "  (supported: --record <file> --stats <file> "
                         "--replay <file> --replay-allow-mismatch)" << std::endl;
            return 1;
        }
    }

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
    // cut_only 模式: 关闭 CUSA aspirate (gpu_aspirate 跳过),粒子不被吸走。
    gpu_solver.config.enable_aspiration = (cfg.cutting_mode == "debulk") ? 1 : 0;
    // Damage-weighted shape matching: α=1 in cut_only enables Müller 2005 effective-mass
    //   shape matching with P(F) as mass — damaged particles contribute less to cluster
    //   shape but stay in cluster (SVD stays stable, healthy particles still pulled).
    //   α=0 in debulk reproduces original unweighted behavior exactly.
    gpu_solver.config.damage_shape_weight =
        (cfg.cutting_mode == "cut_only") ? 1.0f : 0.0f;

    // Sharp-edge damage: limits damage to a line segment (the blade edge) instead of
    //   the full SDF volume. Fixes "horizontal blade motion deletes faces" — only
    //   sharp edge cuts, blade body just collides (push-out).
    gpu_solver.config.use_sharp_edge_damage = cfg.use_sharp_edge_damage ? 1 : 0;
    gpu_solver.config.sharp_edge_a = make_float3(
        cfg.sharp_edge_a[0], cfg.sharp_edge_a[1], cfg.sharp_edge_a[2]);
    gpu_solver.config.sharp_edge_b = make_float3(
        cfg.sharp_edge_b[0], cfg.sharp_edge_b[1], cfg.sharp_edge_b[2]);
    gpu_solver.config.damage_growth_rate = cfg.damage_growth_rate;
    if (cfg.use_sharp_edge_damage) {
        printf("[CUT] sharp_edge: A=(%.4f,%.4f,%.4f) B=(%.4f,%.4f,%.4f) growth_rate=%.3f\n",
               cfg.sharp_edge_a[0], cfg.sharp_edge_a[1], cfg.sharp_edge_a[2],
               cfg.sharp_edge_b[0], cfg.sharp_edge_b[1], cfg.sharp_edge_b[2],
               cfg.damage_growth_rate);
    }

    int substeps = cfg.substeps;

    gpu_solver.cutting_update_interval = cfg.cutting_interval;
    gpu_solver.alpha_min = cfg.alpha_min;
    gpu_solver.aspiration_threshold = cfg.aspiration_threshold;
    gpu_solver.damage_model_mode = (cfg.damage_model == "toughness") ? 1 : 0;

    cfg.print();

    gpu_solver.init(positions, velocities, removed, fixed,
                    cluster_starts, cluster_sizes,
                    all_particle_ids, all_rest_offsets, all_rest_cm);

    if (num_edges > 0) {
        gpu_solver.init_edges(edge_a, edge_b, edge_pf_rest);

        // Per-edge α field (Stage 1 validation): if enabled, allocate and
        // compute a linear-gradient α field along selected axis.
        gpu_solver.per_edge_alpha = cfg.per_edge_alpha;
        if (cfg.per_edge_alpha) {
            // Determine axis index
            int axis = 0;
            if (cfg.alpha_gradient_axis == "y") axis = 1;
            else if (cfg.alpha_gradient_axis == "z") axis = 2;

            // Compute bbox of positions along that axis
            float axis_min =  1e30f;
            float axis_max = -1e30f;
            for (const auto& p : positions) {
                float v = (axis == 0) ? p.x : (axis == 1) ? p.y : p.z;
                if (v < axis_min) axis_min = v;
                if (v > axis_max) axis_max = v;
            }
            float axis_range = axis_max - axis_min;
            if (axis_range < 1e-6f) axis_range = 1.0f;  // safety

            // Per-edge α: linear interpolation based on midpoint position
            std::vector<float> alpha_per_edge(num_edges);
            for (int e = 0; e < num_edges; e++) {
                int a = edge_a[e], b = edge_b[e];
                float mid;
                if (axis == 0)      mid = 0.5f * (positions[a].x + positions[b].x);
                else if (axis == 1) mid = 0.5f * (positions[a].y + positions[b].y);
                else                mid = 0.5f * (positions[a].z + positions[b].z);
                float t = (mid - axis_min) / axis_range;
                alpha_per_edge[e] = cfg.alpha_min_low
                                     + t * (cfg.alpha_min_high - cfg.alpha_min_low);
            }
            gpu_solver.upload_per_edge_alpha(alpha_per_edge);
            printf("[CUT] per_edge_alpha ENABLED: axis=%s, α range [%.3f, %.3f]\n",
                   cfg.alpha_gradient_axis.c_str(),
                   cfg.alpha_min_low, cfg.alpha_min_high);
        } else {
            printf("[CUT] per_edge_alpha DISABLED: global alpha_min=%.3f\n",
                   cfg.alpha_min);
        }
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
    // Scheme B: per-tet broken flag (monotonic). Used by rebuild_surface_faces
    // to skip cut tets → 切口在 viewer 中变成可见的开口。
    std::vector<int> tet_broken(use_tet ? tet_mesh.num_tets() : 0, 0);

    // --- Per-particle averaged α (visualization for Stage 1 validation) ---
    // mean of incident edge α values per particle. Static (α field doesn't change
    // mid-sim), so compute once and pass pointer to viewer.
    std::vector<float> cpu_particle_alpha;
    if (cfg.per_edge_alpha && num_edges > 0) {
        cpu_particle_alpha.assign(N, 0.0f);
        std::vector<int> count(N, 0);
        for (int e = 0; e < num_edges; e++) {
            int a = edge_a[e], b = edge_b[e];
            float ae = gpu_solver.alpha_per_edge_cpu[e];
            cpu_particle_alpha[a] += ae; count[a]++;
            cpu_particle_alpha[b] += ae; count[b]++;
        }
        for (int i = 0; i < N; i++) {
            if (count[i] > 0) cpu_particle_alpha[i] /= count[i];
            else cpu_particle_alpha[i] = cfg.alpha_min;
        }
    }

    // --- 设置 Viewer ---
    Viewer viewer;
    viewer.solver = &cpu_solver;
    viewer.particle_damage_array = &cpu_damage;
    if (cfg.per_edge_alpha && !cpu_particle_alpha.empty()) {
        viewer.particle_alpha_array = &cpu_particle_alpha;
        viewer.particle_alpha_min = cfg.alpha_min_low;
        viewer.particle_alpha_max = cfg.alpha_min_high;
        printf("[VIEWER] Stage 1 mode: particles colored by α field (jet colormap, "
               "range [%.2f, %.2f])\n", cfg.alpha_min_low, cfg.alpha_min_high);
    }
    viewer.num_clusters_display = C;
    // Scheme B: cut_only 模式让 viewer 知道 tet_broken,触发 surface 重建
    if (cfg.cutting_mode == "cut_only" && use_tet) {
        viewer.tet_broken_ptr = &tet_broken;
    }

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
    simrec_frame::FramePipeline pipeline{gpu_solver, cpu_solver, tet_mesh,
                                         edge_a, edge_b, edge_pf_rest,
                                         cpu_positions, cpu_removed, cpu_damage, tet_broken,
                                         use_tet, cfg.cutting_mode, N};

    // --- Step 0.5a: recorder / per-frame stats (observation only) ---
    simrec::TrajectoryRecorder recorder;
    simrec::FrameStatsLogger frame_stats;
    simrec::TrajectoryReader replay;
    simrec::RecordMeta meta;
    {
        auto absp = [](const std::string& q) {
            return q.empty() ? q : std::filesystem::absolute(q).lexically_normal().string();
        };
        meta.config_path    = config_loaded ? absp(input) : std::string("-");
        meta.mesh_path      = absp(cfg.mesh_path);
        meta.tool_mesh_path = (gpu_solver.config.tool_type == 1) ? absp(cfg.tool_mesh_path) : std::string();
        meta.dt             = gpu_solver.config.dt;
        meta.substeps       = substeps;
        meta.num_iterations = gpu_solver.config.num_iterations;
        meta.knife_radius   = gpu_solver.config.knife_radius;
        meta.tool_type      = gpu_solver.config.tool_type;
        meta.N              = N;
    }
    std::string provenance;
    if (!replay_path.empty()) {
        if (!replay.load(replay_path)) return 1;
        auto bad = replay.check(meta);
        for (auto& b : bad) std::cerr << "[replay] MISMATCH " << b << std::endl;
        if (!bad.empty() && !replay_allow_mismatch) {
            std::cerr << "[replay] refusing to replay: setup differs from the recording "
                         "(pass --replay-allow-mismatch to override on purpose)" << std::endl;
            return 1;
        }
        provenance = "# replay " + std::filesystem::absolute(replay_path).lexically_normal().string() + "\n";
        for (auto& b : bad) provenance += "# replay_mismatch " + b + "\n";
        std::printf("[replay] driving the tool from %s (%zu frames); mouse/keyboard/Haply tool input ignored\n",
                    replay_path.c_str(), replay.frames().size());
    }
    if (!record_path.empty() && !recorder.open(record_path, meta)) return 1;
    if (!stats_path.empty() && !frame_stats.open(stats_path, positions, avg_spacing, provenance)) return 1;

    // GPU step 回调：替代 solver->step()
    viewer.on_step = [&](int substeps) {
        // Step 0.5b: replay overrides the tool pose AFTER all live inputs (on_pre_step/Haply,
        // mouse, keyboard), so the solver sees exactly the recorded pose for this frame.
        if (!replay_path.empty()) {
            const auto& fr = replay.frames();
            if (timing_frame >= (int)fr.size()) {
                if (timing_frame == (int)fr.size())
                    std::printf("[replay] finished %zu frames, closing\n", fr.size());
                viewer.request_close();
                timing_frame = (int)fr.size() + 1;
                return;
            }
            const auto& r = fr[timing_frame];
            viewer.knife_pos = Vec3(r.kp.x, r.kp.y, r.kp.z);
            for (int a = 0; a < 3; a++)
                for (int b = 0; b < 3; b++) viewer.knife_rot(a, b) = r.rot[a * 3 + b];
            viewer.knife_active = (r.active != 0);
        }
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
        // Step 0.5a: exactly what the solver receives this frame (all inputs converge here)
        auto t_rec0 = std::chrono::steady_clock::now();
        recorder.write(timing_frame, gpu_solver.config.knife_pos, gpu_solver.config.knife_rot,
                       gpu_solver.config.knife_active, substeps);
        double rec_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_rec0).count();

        // Step 0.5d: shared per-frame pipeline (GPU step → download → orphan cleanup → BFS split).
        // Same code as tool_bench --replay, so headless regressions measure what the app does.
        auto t_gpu0 = std::chrono::high_resolution_clock::now();
        simrec_frame::FrameResult fr_res = simrec_frame::advance_frame(pipeline, substeps, timing_frame,
                                                                       viewer.knife_active);
        auto t1 = t_gpu0 + std::chrono::duration_cast<std::chrono::high_resolution_clock::duration>(
                               std::chrono::duration<double, std::milli>(fr_res.gpu_ms));
        int active = fr_res.active, rem = fr_res.removed;
        viewer.num_active_particles = active;
        viewer.num_removed_particles = rem;
        if (fr_res.bfs_ran) viewer.tet_broken_count = fr_res.tet_broken_count;
        if (fr_res.clusters_changed) {
            C = (int)cpu_solver.clusters.size();
            viewer.num_clusters_display = C;
        }
        (void)t0;

        // Step 0.5a: per-frame state after the FULL pipeline (GPU + orphan + BFS)
        frame_stats.write(timing_frame, fr_res.gpu_ms,
                          gpu_solver.config.knife_pos, gpu_solver.config.knife_radius,
                          gpu_solver.config.knife_active, gpu_solver.config.tool_type,
                          cpu_positions, cpu_removed, fixed, rec_us,
                          simrec::FrameSources{fr_res.gpu_new, fr_res.orphan_new, fr_res.bfs_new,
                                               fr_res.orphan_ms, fr_res.bfs_ms});

        timing_frame++;

        // 周期性状态输出
        if (timing_frame <= 5 || timing_frame % 300 == 0) {
            double dt_gpu = fr_res.gpu_ms; (void)t1;
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
    std::cout << "  M: 切换移刀模式（不用按住，远程桌面/TeamViewer 下用这个）" << std::endl;
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
