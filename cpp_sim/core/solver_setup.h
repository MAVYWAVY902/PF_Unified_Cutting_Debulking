#pragma once
// Step 1 — the ONE place that maps a loaded SimConfigYAML onto ShapeMatchingGPUSolver scalars
// (before solver.init). Shared by the viewer and tool_bench so the two programs cannot drift
// apart again (before this, tool_bench forced aspiration ON and damage weighting OFF even for
// cut_only configs). Body MOVED from main_gpu_viewer.cpp without logic changes.
// Also shared: rigid obstacle (bone) SDF and tool-mesh SDF baking — before this, tool_bench
// never loaded the rigid obstacle (only its friction), so tumour configs differed from the viewer.
// Also shared since 3a′: heterogeneous material fields (setup_material_fields). Not covered here
// (still program-specific): input/pose.

#include "core/sim_config.h"
#include "solver/shape_matching_gpu.h"
#include "core/rigid_mesh.h"
#include "core/tool_features.h"
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

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
    // sharp edge lives in the centred tool frame → scaled with the tool (tool_scale, Step 2)
    const float ts = cfg.tool_scale;
    s.config.sharp_edge_a = make_float3(ts * cfg.sharp_edge_a[0], ts * cfg.sharp_edge_a[1], ts * cfg.sharp_edge_a[2]);
    s.config.sharp_edge_b = make_float3(ts * cfg.sharp_edge_b[0], ts * cfg.sharp_edge_b[1], ts * cfg.sharp_edge_b[2]);
    if (cfg.use_sharp_edge_damage)                 // manual feature = the one configured segment
        s.set_cut_segments({s.config.sharp_edge_a, s.config.sharp_edge_b});
    // Step 2 tool switches (defaults = legacy)
    s.config.sdf_normal_mode = (cfg.sdf_normal == "trilinear") ? 1 : 0;
    s.config.sdf_pf_edge_radius = cfg.sdf_pf_edge_radius ? 1 : 0;
    s.config.aspiration_reach = cfg.aspiration_reach;
    s.config.edge_contact = cfg.edge_contact ? 1 : 0;
    s.config.straddle_fracture = cfg.straddle_fracture ? 1 : 0;
    s.config.entry_side_contact = cfg.entry_side_contact ? 1 : 0;
    s.config.contact_iterate = cfg.contact_iterate ? 1 : 0;
    s.config.selfcol_tool_block = cfg.selfcol_tool_block ? 1 : 0;
    s.config.cut_curve_crossing = cfg.cut_curve_crossing ? 1 : 0;
    s.config.cut_curve_leading = cfg.cut_curve_leading ? 1 : 0;
    s.config.contact_ccd = cfg.contact_ccd ? 1 : 0;
    s.config.straddle_detour = cfg.straddle_detour ? 1 : 0;
    s.config.straddle_detour_k = cfg.straddle_detour_k;
    s.config.straddle_sheet = cfg.straddle_rule == "sheet" ? 1 : 0;
    s.config.edge_sweep_contact = cfg.edge_sweep_contact ? 1 : 0;
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
// extra_margin [m]: grow the baked box by whole cells (same cell size, same voxel centres, so phi in
// the original box is unchanged): queries up to one tissue edge away from the tool (e.g. the far
// endpoint of an edge crossing a thin blade) get a valid phi and normal. Pass the longest edge.
inline void setup_tool_mesh(const SimConfigYAML& cfg, ShapeMatchingGPUSolver& s, RigidMesh& tool_mesh,
                            double extra_margin = 0.0) {
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
            if (cfg.tool_scale != 1.0f) {          // Step 2: uniform scale about the bbox centre
                for (auto& v : tool_mesh.vertices) v *= (double)cfg.tool_scale;
                std::cout << "Tool scaled by " << cfg.tool_scale << std::endl;
            }

            tool_mesh.compute_sdf(cfg.tool_sdf_resolution, 0.2, extra_margin);

            // Step 3b: cut_features: auto → sharp edges / points of the (centred, scaled) mesh are
            // the CUT_CURVE channel. No feature = blunt tool: contact only, no cutting.
            if (cfg.cut_features == "auto") {
                SharpFeatures f = extract_sharp_features(tool_mesh, cfg.sharp_angle_deg);
                print_sharp_features(f, cfg.sharp_angle_deg);
                std::vector<float3> segs, nrms;
                for (const auto& p : f.seg) segs.push_back(make_float3((float)p.x(), (float)p.y(), (float)p.z()));
                for (const auto& n : f.nrm) nrms.push_back(make_float3((float)n.x(), (float)n.y(), (float)n.z()));
                s.config.use_sharp_edge_damage = 1;
                s.set_cut_segments(segs, nrms);
                if (segs.empty()) std::printf("[cut-features] no sharp feature: the tool is blunt (contact only)\n");
                if (cfg.straddle_rule == "sheet") {          // R3 sheet rule: the 2D intruders on the sharp edges
                    std::vector<ToolSheet> sh = extract_tool_sheets(tool_mesh, f);
                    print_tool_sheets(sh);
                    std::vector<ToolSheetGPU> hs; std::vector<float4> hseg;
                    auto f3 = [](const Vec3& v) { return make_float3((float)v.x(), (float)v.y(), (float)v.z()); };
                    for (const auto& t : sh) {
                        hs.push_back(ToolSheetGPU{f3(t.o), f3(t.m), f3(t.u), f3(t.v), (int)hseg.size(), (int)t.seg.size()});
                        for (const auto& e : t.seg) hseg.push_back(make_float4((float)e[0], (float)e[1], (float)e[2], (float)e[3]));
                    }
                    s.set_tool_sheets(hs, hseg);
                }
            }

            // edge_sweep_contact: the convex edges of the (centred, scaled) mesh
            if (cfg.edge_sweep_contact) {
                SharpFeatures f = extract_sharp_features(tool_mesh, cfg.sharp_angle_deg);
                std::vector<float3> e4; std::vector<int> sh;
                auto f3 = [](const Vec3& v) { return make_float3((float)v.x(), (float)v.y(), (float)v.z()); };
                for (size_t k = 0; k < f.cvx_sharp.size(); k++) {
                    e4.push_back(f3(f.cvx_seg[2*k])); e4.push_back(f3(f.cvx_seg[2*k+1]));
                    e4.push_back(f3(f.cvx_nrm[2*k])); e4.push_back(f3(f.cvx_nrm[2*k+1]));
                    sh.push_back(f.cvx_sharp[k]);
                }
                s.set_tool_contact_edges(e4, sh);
                int ns = 0; for (int x : sh) ns += x;
                std::printf("[edge-sweep-contact] %zu convex tool edges (%d sharp)\n", sh.size(), ns);
            }
            // contact_ccd: exact triangles for the path test (same local frame as the SDF), outward
            // normals oriented by the sign of the enclosed volume; brute-force loop → small meshes only
            if (cfg.contact_ccd || cfg.straddle_detour) {
                double vol = 0.0;
                for (const auto& f : tool_mesh.faces)
                    vol += tool_mesh.vertices[f[0]].dot(tool_mesh.vertices[f[1]].cross(tool_mesh.vertices[f[2]])) / 6.0;
                const double sg = vol >= 0.0 ? 1.0 : -1.0;
                const size_t max_tri = 256;
                if (tool_mesh.faces.size() <= max_tri) {
                    std::vector<float3> tri4;
                    for (const auto& f : tool_mesh.faces) {
                        const Vec3 &a = tool_mesh.vertices[f[0]], &b = tool_mesh.vertices[f[1]], &c = tool_mesh.vertices[f[2]];
                        Vec3 n = (b - a).cross(c - a);
                        if (n.norm() < 1e-30) continue;
                        n = sg * n.normalized();
                        for (const Vec3* v : {&a, &b, &c}) tri4.push_back(make_float3((float)v->x(), (float)v->y(), (float)v->z()));
                        tri4.push_back(make_float3((float)n.x(), (float)n.y(), (float)n.z()));
                    }
                    s.set_tool_triangles(tri4);
                    std::printf("[contact-ccd] exact path test on %zu tool triangles\n", tri4.size() / 4);
                } else {
                    std::printf("[contact-ccd] %zu triangles > %zu: exact path test needs an acceleration structure "
                                "(not implemented) — SDF tracing only, parts thinner than a cell are not seen\n",
                                tool_mesh.faces.size(), max_tri);
                }
            }

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

// B: tool proxy parameters. max step = particle radius (half the mean particle spacing); reach =
// largest distance of a tool point from the tool origin (mesh vertices after centring/scaling, or
// the sphere radius), so |dp| + angle * reach bounds the motion of every tool point.
inline void setup_tool_proxy(const SimConfigYAML& cfg, const ShapeMatchingGPUSolver& s, const RigidMesh& tool_mesh,
                             float avg_spacing, bool& on, double& max_step, double& reach) {
    on = cfg.tool_proxy;
    max_step = 0.5 * (double)avg_spacing;
    reach = (double)s.config.knife_radius;
    if (s.config.tool_type == 1) { reach = 0.0; for (const auto& v : tool_mesh.vertices) reach = std::max(reach, v.norm()); }
    if (on) std::printf("[tool-proxy] max step %.3f mm per substep (particle radius), tool reach %.1f mm\n",
                        max_step * 1e3, reach * 1e3);
}

// Longest cutting edge at the rest positions (the tool-box margin straddle_fracture needs).
inline double longest_edge(const std::vector<float3>& pos, const std::vector<int>& ea, const std::vector<int>& eb) {
    double m = 0.0;
    for (size_t e = 0; e < ea.size(); e++) {
        float3 a = pos[ea[e]], b = pos[eb[e]];
        m = std::max(m, (double)std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z)));
    }
    return m;
}

// --debulk on|off  →  true/false; returns false on a bad value.
inline bool parse_debulk_flag(const char* v, bool& out) {
    if (!std::strcmp(v, "on"))  { out = true;  return true; }
    if (!std::strcmp(v, "off")) { out = false; return true; }
    std::fprintf(stderr, "--debulk must be on|off (got '%s')\n", v);
    return false;
}

// Heterogeneous material fields (yaml per_edge_alpha / per_particle_tau / per_particle_stiffness),
// built from the REST positions and uploaded. Call after solver.init() and init_edges(). Shared by
// the viewer and tool_bench (3a′): before this only main_record_trajectory.cpp uploaded the τ and
// stiffness fields, so the interactive app silently simulated a uniform material for those configs.
// Field formulas are the ones of main_record_trajectory.cpp / the viewer (unchanged):
//   per-edge α:     linear in the edge midpoint along alpha_gradient_axis, alpha_min_low → high
//   per-particle τ: linear along tau_gradient_axis, tau_low → tau_high
//   stiffness:      "inclusion" (stiff_high inside a centred sphere of radius
//                   stiff_inclusion_radius_frac × bbox half-extent, else stiff_low) or linear gradient
inline void setup_material_fields(const SimConfigYAML& cfg, ShapeMatchingGPUSolver& s,
                                  const std::vector<float3>& positions,
                                  const std::vector<int>& edge_a, const std::vector<int>& edge_b) {
    const int N = (int)positions.size();
    auto axis_of = [](const std::string& a) { return a == "y" ? 1 : (a == "z" ? 2 : 0); };
    auto comp = [](const float3& p, int axis) { return axis == 0 ? p.x : (axis == 1 ? p.y : p.z); };
    auto axis_range = [&](int axis, float& lo, float& range) {
        lo = 1e30f; float hi = -1e30f;
        for (const auto& p : positions) { float v = comp(p, axis); lo = std::min(lo, v); hi = std::max(hi, v); }
        range = hi - lo;
        if (range < 1e-6f) range = 1.0f;
    };

    s.per_edge_alpha = cfg.per_edge_alpha;
    if (cfg.per_edge_alpha && !edge_a.empty()) {
        int axis = axis_of(cfg.alpha_gradient_axis); float lo, range; axis_range(axis, lo, range);
        std::vector<float> alpha_per_edge(edge_a.size());
        for (size_t e = 0; e < edge_a.size(); e++) {
            float mid;
            int a = edge_a[e], b = edge_b[e];
            if (axis == 0)      mid = 0.5f * (positions[a].x + positions[b].x);
            else if (axis == 1) mid = 0.5f * (positions[a].y + positions[b].y);
            else                mid = 0.5f * (positions[a].z + positions[b].z);
            float t = (mid - lo) / range;
            alpha_per_edge[e] = cfg.alpha_min_low + t * (cfg.alpha_min_high - cfg.alpha_min_low);
        }
        s.upload_per_edge_alpha(alpha_per_edge);
        printf("[CUT] per_edge_alpha ENABLED: axis=%s, α range [%.3f, %.3f]\n",
               cfg.alpha_gradient_axis.c_str(), cfg.alpha_min_low, cfg.alpha_min_high);
    } else {
        printf("[CUT] per_edge_alpha DISABLED: global alpha_min=%.3f\n", cfg.alpha_min);
    }

    s.per_particle_tau = cfg.per_particle_tau;
    if (cfg.per_particle_tau) {
        int axis = axis_of(cfg.tau_gradient_axis); float lo, range; axis_range(axis, lo, range);
        std::vector<float> tau(N);
        for (int i = 0; i < N; i++) {
            float t = (comp(positions[i], axis) - lo) / range;
            tau[i] = cfg.tau_low + t * (cfg.tau_high - cfg.tau_low);
        }
        s.upload_per_particle_tau(tau);
        printf("[CUT] per_particle_tau ENABLED: axis=%s, τ range [%.3f, %.3f]\n",
               cfg.tau_gradient_axis.c_str(), cfg.tau_low, cfg.tau_high);
    }

    s.per_particle_stiffness = cfg.per_particle_stiffness;
    if (cfg.per_particle_stiffness) {
        float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
        for (const auto& p : positions)
            for (int k = 0; k < 3; k++) { bmin[k] = std::min(bmin[k], comp(p, k)); bmax[k] = std::max(bmax[k], comp(p, k)); }
        std::vector<float> stiff(N);
        if (cfg.stiff_pattern == "inclusion") {
            float c[3] = {0.5f*(bmin[0]+bmax[0]), 0.5f*(bmin[1]+bmax[1]), 0.5f*(bmin[2]+bmax[2])};
            float half_extent = 0.5f * std::max({bmax[0]-bmin[0], bmax[1]-bmin[1], bmax[2]-bmin[2]});
            float incl_r = cfg.stiff_inclusion_radius_frac * half_extent;
            int n_inside = 0;
            for (int i = 0; i < N; i++) {
                float dx = positions[i].x - c[0], dy = positions[i].y - c[1], dz = positions[i].z - c[2];
                bool in = std::sqrt(dx*dx + dy*dy + dz*dz) < incl_r;
                stiff[i] = in ? cfg.stiff_high : cfg.stiff_low;
                n_inside += in;
            }
            printf("[CUT] per_particle_stiffness ENABLED (inclusion): r_frac=%.2f, inside=%d/%d, range [%.3f, %.3f]\n",
                   cfg.stiff_inclusion_radius_frac, n_inside, N, cfg.stiff_low, cfg.stiff_high);
        } else {
            int axis = axis_of(cfg.stiff_gradient_axis);
            float lo = bmin[axis], range = bmax[axis] - bmin[axis];
            if (range < 1e-6f) range = 1.0f;
            for (int i = 0; i < N; i++) {
                float t = (comp(positions[i], axis) - lo) / range;
                stiff[i] = cfg.stiff_low + t * (cfg.stiff_high - cfg.stiff_low);
            }
            printf("[CUT] per_particle_stiffness ENABLED (gradient): axis=%s, range [%.3f, %.3f]\n",
                   cfg.stiff_gradient_axis.c_str(), cfg.stiff_low, cfg.stiff_high);
        }
        s.upload_per_particle_stiffness(stiff);
    }
}
