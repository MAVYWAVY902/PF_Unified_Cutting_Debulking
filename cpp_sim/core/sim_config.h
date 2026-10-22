#pragma once

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <cmath>

struct SimConfigYAML {
    std::string name = "Unnamed";

    // Debulking switch (yaml `debulking: true|false`, DEFAULT false):
    //   true  ("debulk")   — edges break + clusters split + particles ASPIRATED
    //                        (CUSA-style removal, orphan cleanup, small-fragment drop)
    //   false ("cut_only") — edges break + clusters split, but NO particle removal
    //                        (true topological cut; two halves stay alive, fall under gravity)
    // Legacy key `cutting_mode: debulk|cut_only` is still accepted as an alias; giving both
    // keys with different meanings, or an unknown cutting_mode, is a load error.
    // cutting_mode is kept as the derived string the pipeline reads ("debulk" | "cut_only").
    // NOTE: only the yaml-level default is "off"; ShapeMatchingGPUSolver::config.enable_aspiration
    // keeps its own default (1) for programs that build the solver without a config.
    bool debulking = false;
    std::string cutting_mode = "cut_only";

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
    float damping_rate = -1.0f;         // [1/s] if >= 0: damping per substep = exp(-damping_rate * dt)
                                        //   (dt-independent); < 0 → the per-substep factor `damping`
    std::string solver = "pbd";         // tissue elasticity: "pbd" (stiffness per iteration, legacy) or
                                        //   "xpbd" (compliance: dt- and iteration-independent)
    float xpbd_stiffness = 0.0f;        // [1/s^2] xpbd: stiffness / particle mass of each particle-goal link
    float xpbd_damping = 0.0f;          // [s] xpbd: constraint damping time beta/k (Macklin 2016 eq. 26); 0 = off
    int substeps = 4;
    bool substeps_auto = false;
    bool tool_substep_interp = false;   // R4: tool pose interpolated across the substeps of a frame
    bool render_split_tets = false;     // render.split_tets: a cut tet is drawn as one piece per side (TetMesh split rendering)
    bool tool_proxy = false;            // B: the simulated tool follows the input pose at <= one particle
                                        //    radius per substep (virtual coupling)
    float tool_proxy_speed = 0.0f;      // [m/s, sim time] if > 0: proxy step limit = tool_proxy_speed * dt per
                                        //    substep (the same speed limit for any substep count); 0 → particle radius

    // Boundary
    float floor_y = 0.0f;
    bool fix_bottom = true;       // fix bottom vertices to ground
    float fixed_ratio = 0.10f;    // fix bottom X% of mesh height (only when fix_bottom=true)

    // Clusters
    int cluster_rings = 2;
    bool local_split = false;   // R1: shape matching per sub-cluster (cluster's own intact edges)

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

    // Per-particle stiffness field — elasticity heterogeneity.
    //   false → use single global stiffness for all particles (DEFAULT, current behavior)
    //   true  → each particle gets its own stiffness from linear gradient.
    //           LOW stiffness = particle deforms easily (soft tissue)
    //           HIGH stiffness = particle resists deformation (firm tissue, tumor)
    //   Models spatial heterogeneity in Young's modulus.
    bool per_particle_stiffness = false;
    float stiff_low  = 0.5f;
    float stiff_high = 1.5f;
    std::string stiff_gradient_axis = "x";  // "x" | "y" | "z"
    // Spatial pattern of per-particle stiffness:
    //   "gradient"  : linear along stiff_gradient_axis (stiff_low → stiff_high)
    //   "inclusion" : center sphere of stiff_high, outside stiff_low
    std::string stiff_pattern = "gradient";
    float stiff_inclusion_radius_frac = 0.25f;  // fraction of bbox half-extent
    std::string tool_mesh_path;     // empty = sphere tool, non-empty = mesh SDF tool
    int tool_sdf_resolution = 64;   // SDF grid resolution for tool
    // Step 2 (all default to legacy behaviour):
    float tool_scale = 1.0f;        // uniform scale of the tool OBJ about its bbox centre (also
                                    // scales sharp_edge_a/b, which live in the centred tool frame)
    std::string sdf_normal = "legacy";   // "legacy" | "trilinear"  (mesh-SDF tool push normal)
    bool sdf_pf_edge_radius = false;     // mesh-SDF P(F): subtract edge radius (phi - r_e)
    float aspiration_reach = 0.0f;       // [m] 0 = legacy contains test; >0 = phi <= reach
    bool edge_contact = false;           // Step 3c: intact edges collide with the tool body too
    bool straddle_fracture = false;      // R3: edges the tool body separates accumulate P(F) damage
    bool entry_side_contact = false;     // A1: return particles to the side they entered the tool from
    bool contact_iterate = false;        // F1: repeat the mesh-tool projection until the particle is out
    bool selfcol_tool_block = false;     // F3: no self-collision between particles the tool separates
    bool cut_curve_crossing = false;     // cut_curve: crossing — sharp features cut edges they pass between
    bool cut_curve_leading = false;      // cut_curve: leading — crossing + the feature must lead (normal cone)
    bool contact_ccd = false;            // CCD tool contact: paths that cross the tool are stopped at the entry face
    bool straddle_detour = false;        // R3 only if the tool cuts the edge's element (no detour around the tool)
    int straddle_detour_k = 32;           // sphere samples of the detour test (resolution)
    bool edge_sweep_contact = false;     // EXPERIMENTAL, keep off: tissue-edge vs tool-edge CCD push (blunt tool edges push the edges they sweep across); measured worse on cube and tumor (fights the particle contact: oscillation, tunnelling)
    std::string straddle_rule = "segment"; // R3: segment (legacy) | sheet (the tool's sheets cut the element in two, exact)

    // Sharp-edge damage (overrides SDF damage when set).
    // Real-physics cutting: only the sharp edge cuts; blade body just pushes.
    // Specify the two endpoints of the sharp edge in the tool's LOCAL coordinate
    // frame (same frame as the OBJ vertices, before tool_pos translation).
    // When use_sharp_edge_damage=true, damage is computed from point-to-line-segment
    // distance instead of SDF volumetric distance.
    bool use_sharp_edge_damage = false;
    float sharp_edge_a[3] = {0.0f, 0.0f, 0.0f};
    float sharp_edge_b[3] = {0.0f, 0.0f, 0.0f};
    // Step 3b: cut_features: auto = sharp edges / points extracted from the tool mesh
    //   (core/tool_features.h; wedge or cone apex angle < sharp_angle_deg). A mesh without sharp
    //   features then cuts nothing (blunt tool). Mutually exclusive with sharp_edge_a/b (manual).
    std::string cut_features = "none";   // "none" | "auto"   (manual sharp_edge_a/b = legacy)
    float sharp_angle_deg = 30.0f;

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
        {
            int from_mode = -1, from_flag = -1;          // -1 = key absent
            if (config["cutting_mode"]) {
                std::string m = config["cutting_mode"].as<std::string>();
                if (m != "debulk" && m != "cut_only") {
                    std::cerr << "Config error: unknown cutting_mode='" << m
                              << "' (expected debulk | cut_only)" << std::endl;
                    return false;
                }
                from_mode = (m == "debulk") ? 1 : 0;
            }
            if (config["debulking"]) from_flag = config["debulking"].as<bool>() ? 1 : 0;
            if (from_mode >= 0 && from_flag >= 0 && from_mode != from_flag) {
                std::cerr << "Config error: debulking=" << (from_flag ? "true" : "false")
                          << " contradicts cutting_mode=" << (from_mode ? "debulk" : "cut_only") << std::endl;
                return false;
            }
            debulking = (from_flag >= 0) ? (from_flag == 1) : (from_mode >= 0) ? (from_mode == 1) : false;
            cutting_mode = debulking ? "debulk" : "cut_only";
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
            if (sim["damping_rate"]) damping_rate = sim["damping_rate"].as<float>();
            if (sim["solver"]) solver = sim["solver"].as<std::string>();
            if (sim["xpbd_stiffness"]) xpbd_stiffness = sim["xpbd_stiffness"].as<float>();
            if (sim["xpbd_damping"]) xpbd_damping = sim["xpbd_damping"].as<float>();
            if (sim["substeps"]) {
                try {
                    substeps = sim["substeps"].as<int>();
                } catch (...) {
                    if (sim["substeps"].as<std::string>() == "auto") substeps_auto = true;
                }
            }
            if (sim["tool_substep_interp"]) tool_substep_interp = sim["tool_substep_interp"].as<bool>();
            if (sim["tool_proxy"]) tool_proxy = sim["tool_proxy"].as<bool>();
            if (sim["tool_proxy_speed"]) tool_proxy_speed = sim["tool_proxy_speed"].as<float>();
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
            if (c["local_split"]) local_split = c["local_split"].as<bool>();
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
            if (k["tool_scale"]) tool_scale = k["tool_scale"].as<float>();
            if (k["sdf_normal"]) {
                sdf_normal = k["sdf_normal"].as<std::string>();
                if (sdf_normal != "legacy" && sdf_normal != "trilinear") {
                    std::cerr << "Config error: knife.sdf_normal='" << sdf_normal
                              << "' (expected legacy | trilinear)" << std::endl;
                    return false;
                }
            }
            if (k["sdf_pf_edge_radius"]) sdf_pf_edge_radius = k["sdf_pf_edge_radius"].as<bool>();
            if (k["aspiration_reach"]) aspiration_reach = k["aspiration_reach"].as<float>();
            if (k["edge_contact"]) edge_contact = k["edge_contact"].as<bool>();
            if (k["straddle_fracture"]) straddle_fracture = k["straddle_fracture"].as<bool>();
            if (k["entry_side_contact"]) entry_side_contact = k["entry_side_contact"].as<bool>();
            if (k["contact_iterate"]) contact_iterate = k["contact_iterate"].as<bool>();
            if (k["selfcol_tool_block"]) selfcol_tool_block = k["selfcol_tool_block"].as<bool>();
            if (k["contact_ccd"]) contact_ccd = k["contact_ccd"].as<bool>();
            if (k["straddle_detour"]) straddle_detour = k["straddle_detour"].as<bool>();
            if (k["straddle_detour_k"]) straddle_detour_k = k["straddle_detour_k"].as<int>();
            if (k["edge_sweep_contact"]) edge_sweep_contact = k["edge_sweep_contact"].as<bool>();
            if (k["straddle_rule"]) {
                straddle_rule = k["straddle_rule"].as<std::string>();
                if (straddle_rule != "segment" && straddle_rule != "sheet") {
                    std::cerr << "Config error: knife.straddle_rule must be segment|sheet (got " << straddle_rule << ")" << std::endl;
                    return false;
                }
            }
            if (k["cut_curve"]) {
                std::string m = k["cut_curve"].as<std::string>();
                if (m != "proximity" && m != "crossing" && m != "leading") {
                    std::cerr << "Config error: knife.cut_curve must be proximity|crossing|leading (got " << m << ")" << std::endl;
                    return false;
                }
                cut_curve_crossing = (m == "crossing" || m == "leading");
                cut_curve_leading = (m == "leading");
            }
            if (entry_side_contact && !tool_mesh_path.empty() && sdf_normal == "legacy") {
                std::cerr << "Config error: knife.entry_side_contact needs knife.sdf_normal: trilinear" << std::endl;
                return false;
            }
            if (straddle_fracture && !tool_mesh_path.empty() && sdf_normal == "legacy") {
                std::cerr << "Config error: knife.straddle_fracture needs knife.sdf_normal: trilinear "
                             "(the straddle test uses the tool normal; legacy normals are wrong on "
                             "anisotropic SDF grids)" << std::endl;
                return false;
            }
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
            if (k["cut_features"]) {
                cut_features = k["cut_features"].as<std::string>();
                if (cut_features != "none" && cut_features != "auto") {
                    std::cerr << "Config error: knife.cut_features='" << cut_features
                              << "' (expected none | auto)" << std::endl;
                    return false;
                }
                if (cut_features == "auto" && use_sharp_edge_damage) {
                    std::cerr << "Config error: knife.cut_features: auto together with sharp_edge_a/b "
                                 "(choose automatic extraction OR the manual segment)" << std::endl;
                    return false;
                }
                if (cut_features == "auto" && tool_mesh_path.empty() && !k["tool_mesh"]) {
                    std::cerr << "Config error: knife.cut_features: auto needs knife.tool_mesh" << std::endl;
                    return false;
                }
            }
            if (straddle_rule == "sheet" && ( cut_features != "auto")) {
                std::cerr << "Config error: knife.straddle_rule: sheet needs a tool mesh with cut_features: auto "
                             "(the sheets are built on its sharp edges)" << std::endl;
                return false;
            }
            if (edge_sweep_contact && !contact_ccd) {
                std::cerr << "Config error: knife.edge_sweep_contact needs knife.contact_ccd: true "
                             "(it uses the CCD reference positions)" << std::endl;
                return false;
            }
            if (cut_curve_leading && cut_features != "auto") {   // the normal cone comes from the mesh faces
                std::cerr << "Config error: knife.cut_curve: leading needs knife.cut_features: auto" << std::endl;
                return false;
            }
            if (k["sharp_angle_deg"]) sharp_angle_deg = k["sharp_angle_deg"].as<float>();
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
            if (k["per_particle_stiffness"]) per_particle_stiffness = k["per_particle_stiffness"].as<bool>();
            if (k["stiff_low"])  stiff_low  = k["stiff_low"].as<float>();
            if (k["stiff_high"]) stiff_high = k["stiff_high"].as<float>();
            if (k["stiff_gradient_axis"]) {
                stiff_gradient_axis = k["stiff_gradient_axis"].as<std::string>();
                if (stiff_gradient_axis != "x" && stiff_gradient_axis != "y" &&
                    stiff_gradient_axis != "z") stiff_gradient_axis = "x";
            }
            if (k["stiff_pattern"]) {
                stiff_pattern = k["stiff_pattern"].as<std::string>();
                if (stiff_pattern != "gradient" && stiff_pattern != "inclusion") {
                    std::cerr << "Invalid stiff_pattern: '" << stiff_pattern
                              << "', falling back to 'gradient'" << std::endl;
                    stiff_pattern = "gradient";
                }
            }
            if (k["stiff_inclusion_radius_frac"])
                stiff_inclusion_radius_frac = k["stiff_inclusion_radius_frac"].as<float>();
        }

        // Self-collision
        if (config["render"]) {
            auto r = config["render"];
            if (r["split_tets"]) render_split_tets = r["split_tets"].as<bool>();
        }
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

    // Command-line override (--debulk on|off): applied after load(), before solver setup.
    void set_debulking(bool on) { debulking = on; cutting_mode = on ? "debulk" : "cut_only"; }

    void print() const {
        std::cout << "=== Config: " << name << " ===" << std::endl;
        std::cout << "  cut_features: " << (use_sharp_edge_damage ? "manual sharp_edge_a/b" : cut_features)
                  << (cut_features == "auto" ? " (sharp_angle_deg=" + std::to_string(sharp_angle_deg) + ")" : "") << std::endl;
        std::cout << "  tool: scale=" << tool_scale << " sdf_normal=" << sdf_normal
                  << " sdf_pf_edge_radius=" << (sdf_pf_edge_radius ? "on" : "off")
                  << " aspiration_reach=" << aspiration_reach
                  << " edge_contact=" << (edge_contact ? "on" : "off")
                  << " straddle_fracture=" << (straddle_fracture ? "on" : "off")
                  << " entry_side_contact=" << (entry_side_contact ? "on" : "off")
                  << " contact_iterate=" << (contact_iterate ? "on" : "off")
                  << " selfcol_tool_block=" << (selfcol_tool_block ? "on" : "off")
                  << " cut_curve=" << (cut_curve_leading ? "leading" : cut_curve_crossing ? "crossing" : "proximity")
                  << " contact_ccd=" << (contact_ccd ? "on" : "off")
                  << " straddle_detour=" << (straddle_detour ? std::to_string(straddle_detour_k) : std::string("off"))
                  << " straddle_rule=" << straddle_rule
                  << " edge_sweep_contact=" << (edge_sweep_contact ? "on" : "off") << std::endl;
        std::cout << "  debulking: " << (debulking ? "on" : "off") << "  cutting_mode: " << cutting_mode
                  << (cutting_mode == "cut_only"
                      ? " (NO particle removal — true topological cut)"
                      : " (CUSA-style fragment aspiration)") << std::endl;
        if (!mesh_path.empty())
            std::cout << "  mesh: " << mesh_path << std::endl;
        else
            std::cout << "  particle_grid: " << grid_side << "^3, spacing=" << grid_spacing << std::endl;
        std::cout << "  dt=" << dt << ", iters=" << num_iterations
                  << ", stiffness=" << stiffness << ", substeps=" << substeps
                  << ", tool_substep_interp=" << (tool_substep_interp ? "on" : "off")
                  << ", tool_proxy=" << (tool_proxy ? "on" : "off")
                  << ", render.split_tets=" << (render_split_tets ? "on" : "off") << std::endl;
        std::cout << "  knife_radius=" << knife_radius
                  << ", self_collision_radius=" << self_collision_radius << std::endl;
        std::cout << "  fixed_ratio=" << fixed_ratio
                  << ", cluster_rings=" << cluster_rings
                  << ", local_split=" << (local_split ? "on" : "off") << std::endl;
    }
};
