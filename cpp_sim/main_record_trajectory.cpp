// Headless trajectory recorder.
// Loads cube_tet mesh + config, drives knife along auto trajectory, dumps
// per-frame particle state to binary for Python rendering.
//
// Usage: ./record_trajectory <config.yaml> <output_dir> [n_frames=200]
//
// Output files in output_dir/:
//   meta.bin      — N (int32) + num_frames (int32) + per_edge_alpha (int32: 0/1)
//   alpha.bin     — N float32 per-particle averaged α (only if per_edge_alpha=1)
//   frame_NNNN.bin — N×3 float32 pos + N int32 removed + 3 float32 knife_pos
//
// Trajectory: knife starts at left edge of top surface, moves to right edge.
// Half-embedded means knife center at top y, so half-sphere sticks above surface.

#include "core/sim_config.h"
#include "core/particles.h"
#include "core/cluster.h"
#include "core/tet_mesh.h"
#include "core/cutting_manager.h"
#include "core/pf_calculator.h"
#include "solver/shape_matching_gpu.h"
#include "solver/shape_matching_cpu.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cstring>
#include <cstdio>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <config.yaml> <output_dir> [n_frames=200] [substeps_override=0]"
                  << std::endl;
        return 1;
    }
    std::string config_path = argv[1];
    std::string output_dir = argv[2];
    int n_frames = (argc > 3) ? std::atoi(argv[3]) : 200;
    int substeps_override = (argc > 4) ? std::atoi(argv[4]) : 0;  // 0 → use cfg

    SimConfigYAML cfg;
    if (!cfg.load(config_path)) return 1;
    cfg.print();

    std::filesystem::create_directories(output_dir);

    // ---- Load tet mesh ----
    TetMesh tet_mesh;
    if (!tet_mesh.load_msh(cfg.mesh_path)) {
        std::cerr << "mesh load failed: " << cfg.mesh_path << std::endl;
        return 1;
    }
    int N = tet_mesh.num_vertices();
    printf("loaded %d verts, %d tets\n", N, (int)tet_mesh.num_tets());

    // ---- Build particles (no offset; cube sits on floor at y~=0) ----
    std::vector<float3> positions(N), velocities(N, make_float3(0, 0, 0));
    std::vector<int> removed(N, 0), fixed_arr(N, 0);
    Vec3 bb_min(1e30, 1e30, 1e30), bb_max(-1e30, -1e30, -1e30);
    for (int i = 0; i < N; i++) {
        auto& v = tet_mesh.vertices[i];
        positions[i] = make_float3((float)v.x(), (float)v.y(), (float)v.z());
        bb_min = bb_min.cwiseMin(v);
        bb_max = bb_max.cwiseMax(v);
    }
    printf("bbox: (%.3f,%.3f,%.3f) → (%.3f,%.3f,%.3f)\n",
           bb_min.x(), bb_min.y(), bb_min.z(),
           bb_max.x(), bb_max.y(), bb_max.z());

    // Fix bottom particles
    float fix_threshold = bb_min.y() + (bb_max.y() - bb_min.y()) * cfg.fixed_ratio;
    int n_fixed = 0;
    for (int i = 0; i < N; i++) {
        if (positions[i].y < fix_threshold) {
            fixed_arr[i] = 1;
            n_fixed++;
        }
    }
    printf("fixed %d / %d bottom particles\n", n_fixed, N);

    // ---- Build cpu_solver with particles + clusters ----
    ShapeMatchingSolver cpu_solver;
    for (int i = 0; i < N; i++) {
        cpu_solver.particles.add(tet_mesh.vertices[i], 1.0, 0);
        if (fixed_arr[i]) cpu_solver.particles[i].fixed = true;
    }
    cpu_solver.clusters = tet_mesh.build_clusters_from_tets(cpu_solver.particles, cfg.cluster_rings);
    int C = (int)cpu_solver.clusters.size();
    printf("built %d clusters\n", C);

    // Pack CSR for GPU
    std::vector<int> cluster_starts(C), cluster_sizes(C);
    std::vector<int> all_particle_ids;
    std::vector<float3> all_rest_offsets;
    std::vector<float3> all_rest_cm(C);
    int total = 0;
    for (int ci = 0; ci < C; ci++) {
        cluster_starts[ci] = total;
        cluster_sizes[ci] = (int)cpu_solver.clusters[ci].particle_ids.size();
        for (int j = 0; j < cluster_sizes[ci]; j++) {
            all_particle_ids.push_back(cpu_solver.clusters[ci].particle_ids[j]);
            const auto& off = cpu_solver.clusters[ci].rest_offsets[j];
            all_rest_offsets.push_back(make_float3((float)off.x(), (float)off.y(), (float)off.z()));
        }
        all_rest_cm[ci] = make_float3(
            (float)cpu_solver.clusters[ci].rest_cm.x(),
            (float)cpu_solver.clusters[ci].rest_cm.y(),
            (float)cpu_solver.clusters[ci].rest_cm.z());
        total += cluster_sizes[ci];
    }

    // ---- Build edges from CuttingManager ----
    PFCalculator pf_calc;
    pf_calc.alpha_min = cfg.alpha_min;
    pf_calc.init_from_tets(tet_mesh.tets, N);
    CuttingManager cutting_mgr;
    cutting_mgr.alpha_min = cfg.alpha_min;
    cutting_mgr.tau_low = 0.05f;
    cutting_mgr.init(cpu_solver.clusters, cpu_solver.particles);
    std::vector<int> edge_a, edge_b;
    std::vector<float> edge_pf_rest;
    for (auto& e : cutting_mgr.edges) {
        edge_a.push_back(e.particle_a);
        edge_b.push_back(e.particle_b);
        edge_pf_rest.push_back(e.pf_rest);
    }
    int num_edges = (int)edge_a.size();
    printf("%d edges\n", num_edges);

    // ---- Setup GPU solver ----
    ShapeMatchingGPUSolver gpu_solver;
    gpu_solver.config.dt = cfg.dt;
    gpu_solver.config.gravity = make_float3(cfg.gravity[0], cfg.gravity[1], cfg.gravity[2]);
    gpu_solver.config.num_iterations = cfg.num_iterations;
    gpu_solver.config.stiffness = cfg.stiffness;
    gpu_solver.config.damping = cfg.damping;
    gpu_solver.config.floor_y = cfg.floor_y;
    gpu_solver.config.knife_pos = make_float3(100, 100, 100);  // far away
    float identity_rot[9] = {1,0,0, 0,1,0, 0,0,1};
    memcpy(gpu_solver.config.knife_rot, identity_rot, 9 * sizeof(float));
    gpu_solver.config.knife_radius = cfg.knife_radius;
    gpu_solver.config.knife_active = 0;
    gpu_solver.config.self_collision_radius = cfg.self_collision_radius;
    gpu_solver.config.self_collision_stiffness = cfg.self_collision_stiffness;
    gpu_solver.config.self_collision_enabled = cfg.self_collision_enabled ? 1 : 0;
    gpu_solver.config.friction_s = cfg.friction_s;
    gpu_solver.config.friction_k = cfg.friction_k;
    gpu_solver.config.tool_type = 0;
    gpu_solver.config.enable_aspiration = 1;
    gpu_solver.config.damage_shape_weight = 0.0f;
    gpu_solver.config.use_sharp_edge_damage = 0;
    gpu_solver.config.damage_growth_rate = cfg.damage_growth_rate;

    gpu_solver.cutting_update_interval = cfg.cutting_interval;
    gpu_solver.alpha_min = cfg.alpha_min;
    gpu_solver.aspiration_threshold = cfg.aspiration_threshold;
    gpu_solver.damage_model_mode = (cfg.damage_model == "toughness") ? 1 : 0;
    gpu_solver.per_particle_tau = cfg.per_particle_tau;
    printf("[CUT] damage_model = %s, per_particle_tau = %s\n",
           cfg.damage_model.c_str(),
           cfg.per_particle_tau ? "true" : "false");

    gpu_solver.init(positions, velocities, removed, fixed_arr,
                    cluster_starts, cluster_sizes,
                    all_particle_ids, all_rest_offsets, all_rest_cm);
    gpu_solver.init_edges(edge_a, edge_b, edge_pf_rest);

    // ---- Per-particle τ field (toughness gradient) ----
    std::vector<float> per_particle_tau_field(N, cfg.aspiration_threshold);
    if (cfg.per_particle_tau) {
        int axis = 0;
        if (cfg.tau_gradient_axis == "y") axis = 1;
        else if (cfg.tau_gradient_axis == "z") axis = 2;
        float axis_min = 1e30f, axis_max = -1e30f;
        for (const auto& p : positions) {
            float v = (axis == 0) ? p.x : (axis == 1) ? p.y : p.z;
            if (v < axis_min) axis_min = v;
            if (v > axis_max) axis_max = v;
        }
        float axis_range = axis_max - axis_min;
        if (axis_range < 1e-6f) axis_range = 1.0f;
        for (int i = 0; i < N; i++) {
            float v = (axis == 0) ? positions[i].x
                    : (axis == 1) ? positions[i].y
                    : positions[i].z;
            float t = (v - axis_min) / axis_range;
            per_particle_tau_field[i] = cfg.tau_low + t * (cfg.tau_high - cfg.tau_low);
        }
        gpu_solver.upload_per_particle_tau(per_particle_tau_field);
        printf("[CUT] per_particle_tau ENABLED: axis=%s, τ range [%.3f, %.3f]\n",
               cfg.tau_gradient_axis.c_str(), cfg.tau_low, cfg.tau_high);
    }

    // ---- Per-edge α field ----
    std::vector<float> per_particle_alpha(N, cfg.alpha_min);
    gpu_solver.per_edge_alpha = cfg.per_edge_alpha;
    if (cfg.per_edge_alpha) {
        int axis = 0;
        if (cfg.alpha_gradient_axis == "y") axis = 1;
        else if (cfg.alpha_gradient_axis == "z") axis = 2;
        float axis_min = 1e30f, axis_max = -1e30f;
        for (const auto& p : positions) {
            float v = (axis == 0) ? p.x : (axis == 1) ? p.y : p.z;
            if (v < axis_min) axis_min = v;
            if (v > axis_max) axis_max = v;
        }
        float axis_range = axis_max - axis_min;
        if (axis_range < 1e-6f) axis_range = 1.0f;
        std::vector<float> alpha_per_edge(num_edges);
        for (int e = 0; e < num_edges; e++) {
            int a = edge_a[e], b = edge_b[e];
            float mid;
            if (axis == 0)      mid = 0.5f * (positions[a].x + positions[b].x);
            else if (axis == 1) mid = 0.5f * (positions[a].y + positions[b].y);
            else                mid = 0.5f * (positions[a].z + positions[b].z);
            float t = (mid - axis_min) / axis_range;
            alpha_per_edge[e] = cfg.alpha_min_low + t * (cfg.alpha_min_high - cfg.alpha_min_low);
        }
        gpu_solver.upload_per_edge_alpha(alpha_per_edge);

        // Per-particle averaged α for visualization
        std::vector<int> count(N, 0);
        std::fill(per_particle_alpha.begin(), per_particle_alpha.end(), 0.0f);
        for (int e = 0; e < num_edges; e++) {
            per_particle_alpha[edge_a[e]] += alpha_per_edge[e]; count[edge_a[e]]++;
            per_particle_alpha[edge_b[e]] += alpha_per_edge[e]; count[edge_b[e]]++;
        }
        for (int i = 0; i < N; i++) {
            if (count[i] > 0) per_particle_alpha[i] /= count[i];
            else per_particle_alpha[i] = cfg.alpha_min;
        }
        printf("[CUT] per_edge_alpha ENABLED: range [%.2f, %.2f]\n",
               cfg.alpha_min_low, cfg.alpha_min_high);
    } else {
        printf("[CUT] per_edge_alpha DISABLED, global α=%.2f\n", cfg.alpha_min);
    }

    // ---- Trajectory definition ----
    // Knife center at top surface (y = bb_max.y), half-embedded (no offset).
    // x sweeps from left margin to right margin.
    float margin = cfg.knife_radius * 0.5f;   // small margin to maximize α range experienced
    float x_start = bb_min.x() + margin;
    float x_end   = bb_max.x() - margin;
    float knife_y = bb_max.y();   // top surface — half embedded
    float knife_z = 0.5f * (bb_min.z() + bb_max.z());
    printf("trajectory: (%.3f, %.3f, %.3f) → (%.3f, %.3f, %.3f) over %d frames\n",
           x_start, knife_y, knife_z, x_end, knife_y, knife_z, n_frames);
    printf("knife radius = %.3f (half-embedded at y=%.3f)\n", cfg.knife_radius, knife_y);

    // ---- Dump meta + alpha + tau + surface faces ----
    {
        std::ofstream f(output_dir + "/meta.bin", std::ios::binary);
        int32_t N_i = N;
        int32_t nf_i = n_frames;
        // Pack flags: bit 0 = per_edge_alpha, bit 1 = per_particle_tau
        int32_t flags = (cfg.per_edge_alpha ? 1 : 0) | (cfg.per_particle_tau ? 2 : 0);
        int32_t nfaces_i = (int32_t)tet_mesh.surface_faces.size();
        f.write((char*)&N_i, sizeof(int32_t));
        f.write((char*)&nf_i, sizeof(int32_t));
        f.write((char*)&flags, sizeof(int32_t));
        f.write((char*)&nfaces_i, sizeof(int32_t));
        // Field 1: α range (4 floats: low, high, knife_r, _padding)
        float a_low = cfg.alpha_min_low, a_high = cfg.alpha_min_high;
        f.write((char*)&a_low, sizeof(float));
        f.write((char*)&a_high, sizeof(float));
        f.write((char*)&cfg.knife_radius, sizeof(float));
        // Field 2: τ range (2 floats)
        float t_low = cfg.tau_low, t_high = cfg.tau_high;
        f.write((char*)&t_low, sizeof(float));
        f.write((char*)&t_high, sizeof(float));
        // bbox
        float bbox_min[3] = {(float)bb_min.x(), (float)bb_min.y(), (float)bb_min.z()};
        float bbox_max[3] = {(float)bb_max.x(), (float)bb_max.y(), (float)bb_max.z()};
        f.write((char*)bbox_min, 3 * sizeof(float));
        f.write((char*)bbox_max, 3 * sizeof(float));
    }
    {
        std::ofstream f(output_dir + "/alpha.bin", std::ios::binary);
        f.write((char*)per_particle_alpha.data(), N * sizeof(float));
    }
    {
        std::ofstream f(output_dir + "/tau.bin", std::ios::binary);
        f.write((char*)per_particle_tau_field.data(), N * sizeof(float));
    }
    // Surface faces (triangles) for mesh-based rendering. int32 per index, 3 per triangle.
    {
        std::ofstream f(output_dir + "/surface_faces.bin", std::ios::binary);
        for (const auto& face : tet_mesh.surface_faces) {
            int32_t i0 = face[0], i1 = face[1], i2 = face[2];
            f.write((char*)&i0, sizeof(int32_t));
            f.write((char*)&i1, sizeof(int32_t));
            f.write((char*)&i2, sizeof(int32_t));
        }
    }
    printf("meta + alpha + %d surface faces dumped to %s\n",
           (int)tet_mesh.surface_faces.size(), output_dir.c_str());

    // ---- Simulation loop with trajectory ----
    std::vector<float3> cpu_pos(N);
    std::vector<int> cpu_removed(N, 0);
    int substeps_per_frame = (substeps_override > 0) ? substeps_override : cfg.substeps;
    printf("substeps_per_frame = %d (cfg=%d, override=%d)\n",
           substeps_per_frame, cfg.substeps, substeps_override);
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int frame = 0; frame < n_frames; frame++) {
        float t = (float)frame / std::max(n_frames - 1, 1);
        float kx = x_start + t * (x_end - x_start);
        gpu_solver.config.knife_pos = make_float3(kx, knife_y, knife_z);
        gpu_solver.config.knife_active = (frame > 5) ? 1 : 0;  // first 5 frames settle

        for (int sub = 0; sub < substeps_per_frame; sub++) {
            gpu_solver.step();
        }

        // Download state
        gpu_solver.download_positions(cpu_pos);
        gpu_solver.download_removed(cpu_removed);

        // DIAG: max damage + active count (every 20 frames)
        if (frame % 20 == 0 || frame == n_frames - 1) {
            std::vector<float> dmg;
            gpu_solver.download_damage(dmg);
            float max_dmg = 0.0f;
            int n_high = 0;
            for (float d : dmg) {
                if (d > max_dmg) max_dmg = d;
                if (d > 0.3f) n_high++;
            }
            std::vector<float> ed;
            gpu_solver.download_edge_damage(ed);
            float max_ed = 0.0f;
            for (float d : ed) if (d > max_ed) max_ed = d;
            int n_active = 0;
            for (int v : cpu_removed) if (!v) n_active++;
            printf("  diag: knife_x=%.4f  max_particle_dmg=%.3f  max_edge_dmg=%.3f  "
                   "particles>0.3=%d  active=%d/%d\n",
                   kx, max_dmg, max_ed, n_high, n_active, N);
        }

        // Dump frame (+ per-particle damage for live coloring + fake reaction force)
        std::vector<float> cpu_damage_frame;
        gpu_solver.download_damage(cpu_damage_frame);

        // === Fake reaction force on knife (penetration-based spring proxy) ===
        // For each active particle inside knife sphere: F_on_particle = k * penetration * (dir from knife center).
        // Reaction on knife = -sum(F_on_particle).
        // This is NOT physically accurate (no actual contact dynamics in PBD),
        // but captures the right TREND: more particles in knife = more resistance.
        float k_spring = 1000.0f;   // synthetic spring constant (N/m, scale chosen for readable units)
        float fx_knife = 0, fy_knife = 0, fz_knife = 0;
        int n_in_knife = 0;
        for (int i = 0; i < N; i++) {
            if (cpu_removed[i]) continue;
            float dx = cpu_pos[i].x - kx;
            float dy = cpu_pos[i].y - knife_y;
            float dz = cpu_pos[i].z - knife_z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            if (dist < cfg.knife_radius && dist > 1e-6f) {
                float penetration = cfg.knife_radius - dist;
                float f_mag = k_spring * penetration;
                // Force on particle: pushed AWAY from knife center (dx/dist direction)
                // Reaction on knife: opposite
                fx_knife -= f_mag * (dx / dist);
                fy_knife -= f_mag * (dy / dist);
                fz_knife -= f_mag * (dz / dist);
                n_in_knife++;
            }
        }
        float force_magnitude = sqrtf(fx_knife*fx_knife + fy_knife*fy_knife + fz_knife*fz_knife);

        char fname[256];
        snprintf(fname, sizeof(fname), "%s/frame_%04d.bin", output_dir.c_str(), frame);
        std::ofstream f(fname, std::ios::binary);
        f.write((char*)cpu_pos.data(), N * 3 * sizeof(float));
        f.write((char*)cpu_removed.data(), N * sizeof(int32_t));
        float kp[3] = {kx, knife_y, knife_z};
        f.write((char*)kp, 3 * sizeof(float));
        f.write((char*)cpu_damage_frame.data(), N * sizeof(float));
        // NEW: fake force (4 floats: Fx, Fy, Fz, |F|) + n_in_knife (int32)
        float force_buf[4] = {fx_knife, fy_knife, fz_knife, force_magnitude};
        int32_t n_contact = n_in_knife;
        f.write((char*)force_buf, 4 * sizeof(float));
        f.write((char*)&n_contact, sizeof(int32_t));

    }

    auto elapsed = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time).count();
    printf("sim done: %d frames in %.2fs (%.1f FPS)\n",
           n_frames, elapsed, n_frames / elapsed);
    printf("dumped %d frame files to %s/\n", n_frames, output_dir.c_str());

    gpu_solver.cleanup();
    return 0;
}
