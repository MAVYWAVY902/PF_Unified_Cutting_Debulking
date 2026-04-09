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

    // Knife
    float knife_radius = 0.1f;
    bool knife_radius_auto = false;
    int cutting_interval = 50;
    float alpha_min = 10.0f;
    float aspiration_threshold = 0.3f;

    // Self-collision
    bool self_collision_enabled = true;
    float self_collision_radius = 0.012f;
    bool self_collision_radius_auto = false;
    float self_collision_stiffness = 0.8f;

    // Rigid objects
    std::string rigid_mesh_path;
    float rigid_collision_thickness = 0.0f;  // 0 = auto
    bool rigid_collision_thickness_auto = false;

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
            if (ro["collision_thickness"]) {
                try {
                    rigid_collision_thickness = ro["collision_thickness"].as<float>();
                } catch (...) {
                    if (ro["collision_thickness"].as<std::string>() == "auto")
                        rigid_collision_thickness_auto = true;
                }
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
