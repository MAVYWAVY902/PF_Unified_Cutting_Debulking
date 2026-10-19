#pragma once
// Step 1 — the ONE place that maps a loaded SimConfigYAML onto ShapeMatchingGPUSolver scalars
// (before solver.init). Shared by the viewer and tool_bench so the two programs cannot drift
// apart again (before this, tool_bench forced aspiration ON and damage weighting OFF even for
// cut_only configs). Body MOVED from main_gpu_viewer.cpp without logic changes.
// Also shared: rigid obstacle (bone) SDF and tool-mesh SDF baking — before this, tool_bench
// never loaded the rigid obstacle (only its friction), so tumour configs differed from the viewer.
// Not covered here (still program-specific): heterogeneous per-edge/per-particle fields
// (allocated after init_edges, viewer only), input/pose.

#include "core/sim_config.h"
#include "solver/shape_matching_gpu.h"
#include "core/rigid_mesh.h"
#include <iostream>
#include <cstdio>
#include <cstring>

inline void apply_config_to_solver(const SimConfigYAML& cfg, ShapeMatchingGPUSolver& s) {
    s.config.dt = cfg.dt;
    s.config.gravity = make_float3(cfg.gravity[0], cfg.gravity[1], cfg.gravity[2]);
    s.config.num_iterations = cfg.num_iterations;
    s.config.stiffness = cfg.stiffness;
    s.config.damping = cfg.damping;
    s.config.floor_y = cfg.floor_y;
    s.config.knife_pos = make_float3(100, 100, 100);
    // Identity rotation
    float identity_rot[9] = {1,0,0, 0,1,0, 0,0,1};
    std::memcpy(s.config.knife_rot, identity_rot, 9 * sizeof(float));
    s.config.knife_radius = cfg.knife_radius;
    s.config.knife_active = 0;
    s.config.self_collision_radius = cfg.self_collision_radius;
    s.config.self_collision_stiffness = cfg.self_collision_stiffness;
    s.config.self_collision_enabled = cfg.self_collision_enabled ? 1 : 0;
    s.config.friction_s = cfg.friction_s;
    s.config.friction_k = cfg.friction_k;
    s.config.tool_type = cfg.tool_mesh_path.empty() ? 0 : 1;
    // debulking off (cut_only): 关闭 CUSA aspirate (gpu_aspirate 跳过),粒子不被吸走。
    s.config.enable_aspiration = cfg.debulking ? 1 : 0;
    // Damage-weighted shape matching: α=1 in cut_only enables Müller 2005 effective-mass
    //   shape matching with P(F) as mass — damaged particles contribute less to cluster
    //   shape but stay in cluster (SVD stays stable, healthy particles still pulled).
    //   α=0 in debulk reproduces original unweighted behavior exactly.
    //   (Kept BOUND to the debulking switch for now — user decision 2026-10-19.)
    s.config.damage_shape_weight = cfg.debulking ? 0.0f : 1.0f;

    // Sharp-edge damage: limits damage to a line segment (the blade edge) instead of
    //   the full SDF volume. Fixes "horizontal blade motion deletes faces" — only
    //   sharp edge cuts, blade body just collides (push-out).
    s.config.use_sharp_edge_damage = cfg.use_sharp_edge_damage ? 1 : 0;
    s.config.sharp_edge_a = make_float3(cfg.sharp_edge_a[0], cfg.sharp_edge_a[1], cfg.sharp_edge_a[2]);
    s.config.sharp_edge_b = make_float3(cfg.sharp_edge_b[0], cfg.sharp_edge_b[1], cfg.sharp_edge_b[2]);
    s.config.damage_growth_rate = cfg.damage_growth_rate;
    if (cfg.use_sharp_edge_damage) {
        printf("[CUT] sharp_edge: A=(%.4f,%.4f,%.4f) B=(%.4f,%.4f,%.4f) growth_rate=%.3f\n",
               cfg.sharp_edge_a[0], cfg.sharp_edge_a[1], cfg.sharp_edge_a[2],
               cfg.sharp_edge_b[0], cfg.sharp_edge_b[1], cfg.sharp_edge_b[2],
               cfg.damage_growth_rate);
    }

    s.cutting_update_interval = cfg.cutting_interval;
    s.alpha_min = cfg.alpha_min;
    s.aspiration_threshold = cfg.aspiration_threshold;
    s.damage_model_mode = (cfg.damage_model == "toughness") ? 1 : 0;
}

// Rigid obstacle (cfg.rigid_mesh_path): shifted by the same y_offset as the deformable mesh,
// baked to a 128 SDF and uploaded as d_rigid_sdf. Call after solver.init(). Body MOVED from
// main_gpu_viewer.cpp. rigid_mesh is returned for rendering.
inline void setup_rigid_object(const SimConfigYAML& cfg, ShapeMatchingGPUSolver& s, double y_offset,
                               RigidMesh& rigid_mesh) {
    if (!cfg.rigid_mesh_path.empty()) {
        if (!rigid_mesh.load_obj(cfg.rigid_mesh_path)) {
            std::cerr << "Rigid mesh加载失败: " << cfg.rigid_mesh_path << std::endl;
        } else {
            // 和deformable mesh做同样的y_offset，保持相对坐标
            if (y_offset != 0.0) {
                rigid_mesh.translate(Vec3(0, y_offset, 0));
                std::cout << "Rigid mesh y_offset: " << y_offset << std::endl;
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
            s.init_sdf(s.d_rigid_sdf, rigid_mesh.sdf_data,
                rigid_mesh.sdf_res[0], rigid_mesh.sdf_res[1], rigid_mesh.sdf_res[2],
                origin, cs, "Bone");
        }
    }
}

// Tool mesh (cfg.tool_mesh_path): centred at its bbox centre (tool local frame), baked at
// cfg.tool_sdf_resolution with 20% padding, uploaded as d_tool_sdf. Falls back to the sphere
// (tool_type 0) if the OBJ fails to load. Call after solver.init(). Body MOVED from the viewer.
inline void setup_tool_mesh(const SimConfigYAML& cfg, ShapeMatchingGPUSolver& s, RigidMesh& tool_mesh) {
    if (!cfg.tool_mesh_path.empty()) {
        if (!tool_mesh.load_obj(cfg.tool_mesh_path)) {
            std::cerr << "Tool mesh加载失败: " << cfg.tool_mesh_path << std::endl;
            s.config.tool_type = 0;  // fallback to sphere
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
            s.init_sdf(s.d_tool_sdf, tool_mesh.sdf_data,
                tool_mesh.sdf_res[0], tool_mesh.sdf_res[1], tool_mesh.sdf_res[2],
                origin, cs, "Tool");
        }
    }
}

// --debulk on|off  →  true/false; returns false on a bad value.
inline bool parse_debulk_flag(const char* v, bool& out) {
    if (!std::strcmp(v, "on"))  { out = true;  return true; }
    if (!std::strcmp(v, "off")) { out = false; return true; }
    std::fprintf(stderr, "--debulk must be on|off (got '%s')\n", v);
    return false;
}
