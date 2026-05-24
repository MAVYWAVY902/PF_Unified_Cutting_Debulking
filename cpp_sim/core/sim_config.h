#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <cmath>

struct SimConfigYAML {
    std::string name = "Unnamed";

    // Cutting mode:
    //   "debulk"   — current behavior: edges break + clusters split + particles ASPIRATED
    //                (CUSA-style fragment removal, particles disappear)
    //   "cut_only" — edges break + clusters split, but NO particle removal
    //                (true topological cut; two halves stay alive, fall under gravity)
    std::string cutting_mode = "debulk";

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

    // Per-edge α field (validation experiment for anisotropic/heterogeneous cutting).
    //   false → use single global alpha_min for all edges (DEFAULT, byte-identical)
    //   true  → each edge gets its own alpha_min[eid] from a linear gradient
    //           along alpha_gradient_axis (x/y/z), interpolating
    //           alpha_min_low (axis min side) → alpha_min_high (axis max side).
    // Physical meaning: directional/spatial heterogeneity in cutting resistance.
    bool per_edge_alpha = false;
    float alpha_min_low  = 1.0f;
    float alpha_min_high = 10.0f;
    std::string alpha_gradient_axis = "x";   // "x" | "y" | "z"

    // Damage model semantics (LEGACY toggle, kept for backward compat).
    // Recommended now: use per_particle_tau (below) for toughness experiments,
    // since τ IS the physical "yield threshold" / fracture toughness parameter.
    std::string damage_model = "sharpness";

    // Per-particle τ (aspiration threshold) field — toughness gradient.
    //   false → use single global aspiration_threshold for all particles (DEFAULT)
    //   true  → each particle gets its own τ from linear gradient.
    //           LOW τ = particle breaks easily (fragile tissue, easy to cut)
    //           HIGH τ = particle resists breaking (tough tissue, hard to cut)
    //   This is the CORRECT modeling of tissue toughness — τ directly controls
    //   the yield/fracture threshold per particle.
    bool per_particle_tau = false;
    float tau_low  = 0.1f;
    float tau_high = 0.5f;
    std::string tau_gradient_axis = "x";    // "x" | "y" | "z"
    std::string tool_mesh_path;     // empty = sphere tool, non-empty = mesh SDF tool
    int tool_sdf_resolution = 64;   // SDF grid resolution for tool

    // Sharp-edge damage (overrides SDF damage when set).
    // Real-physics cutting: only the sharp edge cuts; blade body just pushes.
    // Specify the two endpoints of the sharp edge in the tool's LOCAL coordinate
    // frame (same frame as the OBJ vertices, before tool_pos translation).
    // When use_sharp_edge_damage=true, damage is computed from point-to-line-segment
    // distance instead of SDF volumetric distance.
    bool use_sharp_edge_damage = false;
    float sharp_edge_a[3] = {0.0f, 0.0f, 0.0f};
    float sharp_edge_b[3] = {0.0f, 0.0f, 0.0f};

    // Damage growth rate per substep. 1.0 = instant snap to target (old behavior).
    // 0.02 = ~25 substeps to break from full pressure → user sees ~50ms deformation
    // before cut. Lower for cut_only, default 1.0 preserves debulk.
    float damage_growth_rate = 1.0f;

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
        if (config["cutting_mode"]) {
            cutting_mode = config["cutting_mode"].as<std::string>();
            if (cutting_mode != "debulk" && cutting_mode != "cut_only") {
                std::cerr << "WARNING: unknown cutting_mode='" << cutting_mode
                          << "', falling back to 'debulk'" << std::endl;
                cutting_mode = "debulk";
            }
        }

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
            if (k["sharp_edge_a"]) {
                auto a = k["sharp_edge_a"];
                sharp_edge_a[0] = a[0].as<float>();
                sharp_edge_a[1] = a[1].as<float>();
                sharp_edge_a[2] = a[2].as<float>();
                use_sharp_edge_damage = true;
            }
            if (k["sharp_edge_b"]) {
                auto b = k["sharp_edge_b"];
                sharp_edge_b[0] = b[0].as<float>();
                sharp_edge_b[1] = b[1].as<float>();
                sharp_edge_b[2] = b[2].as<float>();
            }
            if (k["damage_growth_rate"]) damage_growth_rate = k["damage_growth_rate"].as<float>();
            if (k["per_edge_alpha"]) per_edge_alpha = k["per_edge_alpha"].as<bool>();
            if (k["alpha_min_low"])  alpha_min_low  = k["alpha_min_low"].as<float>();
            if (k["alpha_min_high"]) alpha_min_high = k["alpha_min_high"].as<float>();
            if (k["alpha_gradient_axis"]) {
                alpha_gradient_axis = k["alpha_gradient_axis"].as<std::string>();
                if (alpha_gradient_axis != "x" && alpha_gradient_axis != "y" &&
                    alpha_gradient_axis != "z") {
                    std::cerr << "Invalid alpha_gradient_axis: '" << alpha_gradient_axis
                              << "', falling back to 'x'" << std::endl;
                    alpha_gradient_axis = "x";
                }
            }
            if (k["damage_model"]) {
                damage_model = k["damage_model"].as<std::string>();
                if (damage_model != "sharpness" && damage_model != "toughness") {
                    std::cerr << "Invalid damage_model: '" << damage_model
                              << "', falling back to 'sharpness'" << std::endl;
                    damage_model = "sharpness";
                }
            }
            if (k["per_particle_tau"]) per_particle_tau = k["per_particle_tau"].as<bool>();
            if (k["tau_low"])  tau_low  = k["tau_low"].as<float>();
            if (k["tau_high"]) tau_high = k["tau_high"].as<float>();
            if (k["tau_gradient_axis"]) {
                tau_gradient_axis = k["tau_gradient_axis"].as<std::string>();
                if (tau_gradient_axis != "x" && tau_gradient_axis != "y" &&
                    tau_gradient_axis != "z") tau_gradient_axis = "x";
            }
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
        std::cout << "  cutting_mode: " << cutting_mode
                  << (cutting_mode == "cut_only"
                      ? " (NO particle removal — true topological cut)"
                      : " (CUSA-style fragment aspiration)") << std::endl;
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
