#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <cmath>

struct SimConfigYAML {
    std::string name = "Unnamed";

    // Mesh or particle grid
    std::string mesh_path;
    bool use_particle_grid = false;
    int grid_side = 10;
    double grid_spacing = 0.02;

    // Simulation
    float dt = 0.005f;
    bool dt_auto = false;
    float gravity[3] = {0, -9.81f, 0};
    int num_iterations = 4;
    bool iterations_auto = false;
    float stiffness = 0.6f;
    float damping = 0.98f;
    int substeps = 4;
    bool substeps_auto = false;

    // Boundary
    float floor_y = 0.0f;
    bool fix_bottom = true;       // fix bottom vertices to ground
    float fixed_ratio = 0.10f;    // fix bottom X% of mesh height (only when fix_bottom=true)

    // Clusters
    int cluster_rings = 2;

    // Knife / Tool
    float knife_radius = 0.1f;
    bool knife_radius_auto = false;
    int cutting_interval = 1;   // update damage every N physics steps (was 50 — caused knife tunneling at normal speeds)
    float alpha_min = 3.0f;   // P(F) sigmoid steepness (was 10 — too steep, saturated damage on any contact)
    float aspiration_threshold = 0.3f;
    std::string tool_mesh_path;     // empty = sphere tool, non-empty = mesh SDF tool
    int tool_sdf_resolution = 64;   // SDF grid resolution for tool

    // Self-collision
    bool self_collision_enabled = true;
    float self_collision_radius = 0.012f;
    bool self_collision_radius_auto = false;
    float self_collision_stiffness = 0.8f;

    // Rigid objects
    std::string rigid_mesh_path;
    float rigid_collision_thickness = 0.0f;  // 0 = auto
    bool rigid_collision_thickness_auto = false;
    float friction_s = 0.5f;    // static friction
    float friction_k = 0.3f;    // kinetic friction

    // Haptics (Haply Inverse3 + VersaGrip)
    bool haptics_enabled = false;
    std::string haptics_ws_url = "ws://localhost:10001";
    double haptics_workspace_center[3] = {0.0, -0.18, 0.15};
    double haptics_sim_offset[3] = {-0.017, 0.012, 0.000};
    double haptics_scale = 0.1;
    bool haptics_versagrip_correction = true;
    bool haptics_button_activates_knife = true;
    int haptics_pos_axes[3] = {0, 2, 1};       // axis permutation
    double haptics_pos_signs[3] = {1, 1, -1};  // axis sign flips

    // Viewer
    bool draw_surface = true;
    bool draw_wireframe = false;
    bool draw_points = true;

    // Load from YAML file
    bool load(const std::string& path) {
        YAML::Node config;
        try {
            config = YAML::LoadFile(path);
        } catch (const std::exception& e) {
            std::cerr << "Config加载失败: " << e.what() << std::endl;
            return false;
        }

        if (config["name"]) name = config["name"].as<std::string>();

        // Mesh or particle grid
        if (config["mesh"]) {
            mesh_path = config["mesh"].as<std::string>();
            use_particle_grid = false;
        }
        if (config["particle_grid"]) {
            use_particle_grid = true;
            auto pg = config["particle_grid"];
            if (pg["side"]) grid_side = pg["side"].as<int>();
            if (pg["spacing"]) grid_spacing = pg["spacing"].as<double>();
        }

        // Simulation
        if (config["simulation"]) {
            auto sim = config["simulation"];
            if (sim["dt"]) {
                try {
                    dt = sim["dt"].as<float>();
                } catch (...) {
                    if (sim["dt"].as<std::string>() == "auto") dt_auto = true;
                }
            }
            if (sim["gravity"]) {
                auto g = sim["gravity"];
                gravity[0] = g[0].as<float>();
                gravity[1] = g[1].as<float>();
                gravity[2] = g[2].as<float>();
            }
            if (sim["num_iterations"]) {
                try {
                    num_iterations = sim["num_iterations"].as<int>();
                } catch (...) {
                    if (sim["num_iterations"].as<std::string>() == "auto") iterations_auto = true;
                }
            }
            if (sim["stiffness"]) stiffness = sim["stiffness"].as<float>();
            if (sim["damping"]) damping = sim["damping"].as<float>();
            if (sim["substeps"]) {
                try {
                    substeps = sim["substeps"].as<int>();
                } catch (...) {
                    if (sim["substeps"].as<std::string>() == "auto") substeps_auto = true;
                }
            }
        }

        // Boundary
        if (config["boundary"]) {
            auto b = config["boundary"];
            if (b["floor_y"]) floor_y = b["floor_y"].as<float>();
            if (b["fix_bottom"]) fix_bottom = b["fix_bottom"].as<bool>();
            if (b["fixed_ratio"]) fixed_ratio = b["fixed_ratio"].as<float>();
        }

        // Clusters
        if (config["clusters"]) {
            auto c = config["clusters"];
            if (c["rings"]) cluster_rings = c["rings"].as<int>();
        }

        // Knife
        if (config["knife"]) {
            auto k = config["knife"];
            if (k["radius"]) {
                try {
                    knife_radius = k["radius"].as<float>();
                } catch (...) {
                    if (k["radius"].as<std::string>() == "auto") knife_radius_auto = true;
                }
            }
            if (k["cutting_interval"]) cutting_interval = k["cutting_interval"].as<int>();
            if (k["alpha_min"]) alpha_min = k["alpha_min"].as<float>();
            if (k["aspiration_threshold"]) aspiration_threshold = k["aspiration_threshold"].as<float>();
            if (k["tool_mesh"]) tool_mesh_path = k["tool_mesh"].as<std::string>();
            if (k["tool_sdf_resolution"]) tool_sdf_resolution = k["tool_sdf_resolution"].as<int>();
        }

        // Self-collision
        if (config["self_collision"]) {
            auto sc = config["self_collision"];
            if (sc["enabled"]) self_collision_enabled = sc["enabled"].as<bool>();
            if (sc["radius"]) {
                try {
                    self_collision_radius = sc["radius"].as<float>();
                } catch (...) {
                    if (sc["radius"].as<std::string>() == "auto") self_collision_radius_auto = true;
                }
            }
            if (sc["stiffness"]) self_collision_stiffness = sc["stiffness"].as<float>();
        }

        // Rigid objects
        if (config["rigid_object"]) {
            auto ro = config["rigid_object"];
            if (ro["mesh"]) rigid_mesh_path = ro["mesh"].as<std::string>();
            if (ro["friction_s"]) friction_s = ro["friction_s"].as<float>();
            if (ro["friction_k"]) friction_k = ro["friction_k"].as<float>();
            if (ro["collision_thickness"]) {
                try {
                    rigid_collision_thickness = ro["collision_thickness"].as<float>();
                } catch (...) {
                    if (ro["collision_thickness"].as<std::string>() == "auto")
                        rigid_collision_thickness_auto = true;
                }
            }
        }

        // Haptics
        if (config["haptics"]) {
            auto h = config["haptics"];
            if (h["enabled"]) haptics_enabled = h["enabled"].as<bool>();
            if (h["websocket_url"]) haptics_ws_url = h["websocket_url"].as<std::string>();
            if (h["workspace_center"]) {
                auto wc = h["workspace_center"];
                haptics_workspace_center[0] = wc[0].as<double>();
                haptics_workspace_center[1] = wc[1].as<double>();
                haptics_workspace_center[2] = wc[2].as<double>();
            }
            if (h["sim_offset"]) {
                auto so = h["sim_offset"];
                haptics_sim_offset[0] = so[0].as<double>();
                haptics_sim_offset[1] = so[1].as<double>();
                haptics_sim_offset[2] = so[2].as<double>();
            }
            if (h["scale"]) haptics_scale = h["scale"].as<double>();
            if (h["versagrip_correction"]) haptics_versagrip_correction = h["versagrip_correction"].as<bool>();
            if (h["button_activates_knife"]) haptics_button_activates_knife = h["button_activates_knife"].as<bool>();
            if (h["pos_axes"]) {
                auto pa = h["pos_axes"];
                haptics_pos_axes[0] = pa[0].as<int>();
                haptics_pos_axes[1] = pa[1].as<int>();
                haptics_pos_axes[2] = pa[2].as<int>();
            }
            if (h["pos_signs"]) {
                auto ps = h["pos_signs"];
                haptics_pos_signs[0] = ps[0].as<double>();
                haptics_pos_signs[1] = ps[1].as<double>();
                haptics_pos_signs[2] = ps[2].as<double>();
            }
        }

        // Viewer
        if (config["viewer"]) {
            auto v = config["viewer"];
            if (v["draw_surface"]) draw_surface = v["draw_surface"].as<bool>();
            if (v["draw_wireframe"]) draw_wireframe = v["draw_wireframe"].as<bool>();
            if (v["draw_points"]) draw_points = v["draw_points"].as<bool>();
        }

        return true;
    }

    // Resolve "auto" parameters given mesh stats
    void resolve_auto(float avg_spacing, double mesh_height, double avg_cluster_membership) {
        if (dt_auto) {
            float dt_scale = std::min(1.0f, (float)(mesh_height / 0.5));
            dt = 0.005f * std::max(0.1f, dt_scale);
        }
        if (iterations_auto) {
            num_iterations = std::max(4, (int)(4 * avg_cluster_membership / 10.0));
        }
        if (substeps_auto) {
            substeps = std::max(4, (int)(0.02f / dt));
        }
        if (knife_radius_auto) {
            knife_radius = avg_spacing * 8.0f;
        }
        if (self_collision_radius_auto) {
            self_collision_radius = avg_spacing * 0.45f;
        }
        if (rigid_collision_thickness_auto || (rigid_collision_thickness == 0.0f && !rigid_mesh_path.empty())) {
            rigid_collision_thickness = avg_spacing * 0.8f;
            rigid_collision_thickness_auto = false;
        }
    }

    void print() const {
        std::cout << "=== Config: " << name << " ===" << std::endl;
        if (!mesh_path.empty())
            std::cout << "  mesh: " << mesh_path << std::endl;
        else
            std::cout << "  particle_grid: " << grid_side << "^3, spacing=" << grid_spacing << std::endl;
        std::cout << "  dt=" << dt << ", iters=" << num_iterations
                  << ", stiffness=" << stiffness << ", substeps=" << substeps << std::endl;
        std::cout << "  knife_radius=" << knife_radius
                  << ", self_collision_radius=" << self_collision_radius << std::endl;
        std::cout << "  fixed_ratio=" << fixed_ratio
                  << ", cluster_rings=" << cluster_rings << std::endl;
    }
};
