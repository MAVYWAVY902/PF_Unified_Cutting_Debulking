// tool_bench — Step-0 measurement harness for the SDF-tool generalization.
//
// Headless, GPU solver only (same init as main_gpu_viewer: +1mm y-offset, fixed
// threshold, resolve_auto). Does NOT run the viewer's CPU orphan cleanup / BFS
// split, so absolute removal counts differ from the interactive app; A/B
// comparisons between tool modes are like-for-like.
//
// Usage:
//   tool_bench <config.yaml> <out_dir> [--tool sphere|gridsphere|mesh]
//              [--frames N] [--grid-res R] [--timing] [--depth D]
//
//   --tool sphere      analytic sphere (tool_type 0), radius = knife.radius
//   --tool gridsphere  EXACT sphere SDF baked to a cubic grid, run through the
//                      tool_type 1 (mesh-SDF) kernels. Isolates "SDF code path"
//                      from "geometry": same shape as --tool sphere.
//   --tool mesh        knife.tool_mesh baked like the viewer does
//   --timing           no per-substep downloads; report GPU ms/frame only
//   --depth D          extra penetration below half-embedded (default 0)
//   --write-traj F     also write the scripted tool poses as a replayable recording
//   --replay F         replay a viewer recording through the SHARED viewer pipeline
//                      (core/frame_pipeline.h: GPU + orphan cleanup + BFS split); add
//                      --gpu-only to replay without the CPU stages, --stats F for the
//                      per-frame stats file, --replay-allow-mismatch to force a setup change.
//   NOTE: the scripted (non-replay) mode is a GPU-KERNEL probe (per-substep aspiration
//   instrumentation), it does NOT run orphan cleanup / BFS. For app-faithful numbers write
//   the trajectory with --write-traj and replay it.
//
// Outputs in out_dir/:
//   frames.csv        per-frame stats
//   final_state.bin   N×float3 pos + N×int32 removed (regression reference)
// and a summary on stdout (incl. FNV-1a hash of final_state for bit-identity).

#include "core/sim_config.h"
#include "core/particles.h"
#include "core/cluster.h"
#include "core/tet_mesh.h"
#include <set>
#include "core/cutting_manager.h"
#include "core/pf_calculator.h"
#include "core/rigid_mesh.h"
#include "core/sim_recorder.h"
#include "core/frame_pipeline.h"
#include "core/solver_setup.h"
#include "solver/shape_matching_gpu.h"
#include "solver/shape_matching_cpu.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ULL) {
    const unsigned char* p = (const unsigned char*)data;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <config.yaml> <out_dir> [--tool sphere|gridsphere|mesh] "
                        "[--frames N] [--grid-res R] [--timing] [--depth D]\n", argv[0]);
        return 1;
    }
    std::string config_path = argv[1], out_dir = argv[2];
    std::string tool = "sphere";
    int n_frames = 80, grid_res = 64, n_settle = 60, n_approach = 30;
    bool timing = false, replay_allow_mismatch = false, gpu_only = false;
    std::string replay_path, stats_path, write_traj_path, pass_path, topo_path;
    float depth = 0.0f;
    // --tool mesh orientation: "flat" = identity (old behaviour; the dissector lies flat, its
    // 71 mm length covers the whole tumour); "tipdown" = long axis vertical, blade width along
    // the sweep (x), thickness along z → the tip enters the tissue edge-first.
    std::string mesh_pose = "flat";
    int debulk_override = -1;   // --debulk on|off; -1 = use config
    bool rigid_off = false;     // --rigid off: drop the rigid obstacle (bone)
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--tool" && i + 1 < argc) tool = argv[++i];
        else if (a == "--frames" && i + 1 < argc) n_frames = std::atoi(argv[++i]);
        else if (a == "--settle" && i + 1 < argc) n_settle = std::atoi(argv[++i]);
        else if (a == "--approach" && i + 1 < argc) n_approach = std::atoi(argv[++i]);
        else if (a == "--grid-res" && i + 1 < argc) grid_res = std::atoi(argv[++i]);
        else if (a == "--timing") timing = true;
        else if (a == "--replay" && i + 1 < argc) replay_path = argv[++i];
        else if (a == "--replay-allow-mismatch") replay_allow_mismatch = true;
        else if (a == "--gpu-only") gpu_only = true;
        else if (a == "--write-traj" && i + 1 < argc) write_traj_path = argv[++i];   // scripted pose → recording file   // replay without orphan cleanup / BFS (old behaviour)
        else if (a == "--stats" && i + 1 < argc) stats_path = argv[++i];
        else if (a == "--pass-through" && i + 1 < argc) pass_path = argv[++i];   
        else if (a == "--dump-topology" && i + 1 < argc) topo_path = argv[++i];   // replay: final positions + cutting edges (a,b,broken)   // replay + mesh tool: per-frame tool/edge crossing diagnostic
        else if (a == "--depth" && i + 1 < argc) depth = (float)std::atof(argv[++i]);
        else if (a == "--mesh-pose" && i + 1 < argc) mesh_pose = argv[++i];
        else if (a == "--rigid" && i + 1 < argc) {
            std::string v = argv[++i];
            if (v != "off" && v != "on") { fprintf(stderr, "--rigid must be on|off\n"); return 1; }
            rigid_off = (v == "off");
        }
        else if (a == "--debulk" && i + 1 < argc) {
            bool on; if (!parse_debulk_flag(argv[++i], on)) return 1;
            debulk_override = on ? 1 : 0;
        }
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (tool != "sphere" && tool != "gridsphere" && tool != "mesh") {
        fprintf(stderr, "--tool must be sphere|gridsphere|mesh\n"); return 1;
    }
    if (mesh_pose != "flat" && mesh_pose != "tipdown") { fprintf(stderr, "--mesh-pose must be flat|tipdown\n"); return 1; }

    SimConfigYAML cfg;
    if (!cfg.load(config_path)) return 1;
    if (debulk_override >= 0) cfg.set_debulking(debulk_override == 1);
    if (rigid_off && !cfg.rigid_mesh_path.empty()) {
        printf("[rigid] obstacle %s NOT loaded (--rigid off)\n", cfg.rigid_mesh_path.c_str());
        cfg.rigid_mesh_path.clear();
    }
    std::filesystem::create_directories(out_dir);

    // ---------------- mesh + particles (mirror main_gpu_viewer) ----------------
    TetMesh tet_mesh;
    if (!tet_mesh.load_msh(cfg.mesh_path)) { fprintf(stderr, "mesh load failed\n"); return 1; }
    int N = tet_mesh.num_vertices();
    double min_y = 1e30, max_y = -1e30;
    for (auto& v : tet_mesh.vertices) { min_y = std::min(min_y, v.y()); max_y = std::max(max_y, v.y()); }
    double y_offset = -min_y + 0.001, mesh_height = max_y - min_y;

    ShapeMatchingSolver cpu_solver;
    std::vector<float3> positions(N), velocities(N, make_float3(0, 0, 0));
    std::vector<int> removed(N, 0), fixed_arr(N, 0);
    for (int i = 0; i < N; i++) {
        Vec3 p = tet_mesh.vertices[i] + Vec3(0, y_offset, 0);
        tet_mesh.vertices[i] = p;
        cpu_solver.particles.add(p, 1.0, 0);
        positions[i] = make_float3((float)p.x(), (float)p.y(), (float)p.z());
    }
    int n_fixed = 0;
    if (cfg.fix_bottom) {
        double thr = 0.001 + mesh_height * cfg.fixed_ratio;
        for (int i = 0; i < N; i++)
            if (cpu_solver.particles[i].pos.y() < thr) {
                cpu_solver.particles[i].fixed = true; fixed_arr[i] = 1; n_fixed++;
            }
    }
    cpu_solver.clusters = tet_mesh.build_clusters_from_tets(cpu_solver.particles, cfg.cluster_rings);
    int C = (int)cpu_solver.clusters.size();

    std::vector<int> cs(C), csz(C), pids;
    std::vector<float3> roff, rcm(C);
    {
        int total = 0;
        for (int ci = 0; ci < C; ci++) {
            const auto& cl = cpu_solver.clusters[ci];
            cs[ci] = total; csz[ci] = (int)cl.particle_ids.size();
            for (int j = 0; j < csz[ci]; j++) {
                pids.push_back(cl.particle_ids[j]);
                const auto& o = cl.rest_offsets[j];
                roff.push_back(make_float3((float)o.x(), (float)o.y(), (float)o.z()));
            }
            rcm[ci] = make_float3((float)cl.rest_cm.x(), (float)cl.rest_cm.y(), (float)cl.rest_cm.z());
            total += csz[ci];
        }
    }

    PFCalculator pf_calc; pf_calc.alpha_min = cfg.alpha_min; pf_calc.init_from_tets(tet_mesh.tets, N);
    CuttingManager cm; cm.alpha_min = cfg.alpha_min; cm.tau_low = 0.05f;
    cm.init(cpu_solver.clusters, cpu_solver.particles, &tet_mesh.vertex_neighbors);
    std::vector<int> ea, eb; std::vector<float> epf;
    for (auto& e : cm.edges) { ea.push_back(e.particle_a); eb.push_back(e.particle_b); epf.push_back(e.pf_rest); }
    int E = (int)ea.size();

    // mesh stats → resolve_auto (same sampling as viewer)
    float avg_spacing = 0.0f;
    {
        int sample = std::min(N, 1000); double tot = 0; int cnt = 0;
        for (int i = 0; i < sample; i++) {
            float best = 1e10f;
            for (int j = 0; j < N; j++) {
                if (j == i) continue;
                float dx = positions[i].x - positions[j].x, dy = positions[i].y - positions[j].y,
                      dz = positions[i].z - positions[j].z;
                best = std::min(best, dx*dx + dy*dy + dz*dz);
            }
            tot += sqrtf(best); cnt++;
        }
        avg_spacing = (float)(tot / cnt);
    }
    double membership = 0;
    {
        std::vector<int> m(N, 0);
        for (auto& c : cpu_solver.clusters) for (int pid : c.particle_ids) m[pid]++;
        double s = 0; int c = 0;
        for (int i = 0; i < N; i++) if (!fixed_arr[i] && m[i] > 0) { s += m[i]; c++; }
        membership = c ? s / c : 1.0;
    }
    cfg.resolve_auto(avg_spacing, mesh_height, membership);

    // ---------------- GPU solver ----------------
    ShapeMatchingGPUSolver g;
    // Step 1: same config → solver mapping as the viewer (core/solver_setup.h). Before this,
    // tool_bench forced aspiration ON, damage weighting OFF and sharp-edge damage OFF.
    apply_config_to_solver(cfg, g);
    // tool_bench-specific overrides: --tool picks the tool code path, --mesh-pose the rotation
    float I3[9] = {1,0,0, 0,1,0, 0,0,1};
    // tipdown (row-major, local→world): local x (length) → world y, local y (thickness) → world z,
    // local z (width) → world x
    float TIPDOWN[9] = {0,0,1, 1,0,0, 0,1,0};
    memcpy(g.config.knife_rot, (tool == "mesh" && mesh_pose == "tipdown") ? TIPDOWN : I3, sizeof(I3));
    g.config.tool_type = (tool == "sphere") ? 0 : 1;
    printf("[tool_bench] debulking=%s (enable_aspiration=%d, damage_shape_weight=%.1f, sharp_edge=%d)\n",
           cfg.debulking ? "on" : "off", g.config.enable_aspiration, g.config.damage_shape_weight,
           g.config.use_sharp_edge_damage);
    printf("[tool_bench] tool switches: tool_scale=%.3f sdf_normal=%s sdf_pf_edge_radius=%d aspiration_reach=%.6g\n",
           cfg.tool_scale, g.config.sdf_normal_mode ? "trilinear" : "legacy", g.config.sdf_pf_edge_radius,
           g.config.aspiration_reach);
    g.init(positions, velocities, removed, fixed_arr, cs, csz, pids, roff, rcm);
    g.init_edges(ea, eb, epf);
    setup_material_fields(cfg, g, positions, ea, eb);   // 3a′: same heterogeneous fields as the viewer
    RigidMesh rigid_mesh;                             // Step 1: rigid obstacle (bone) like the viewer
    setup_rigid_object(cfg, g, y_offset, rigid_mesh);

    const float R = cfg.knife_radius;
    RigidMesh tool_rm;   // --tool mesh: the tool mesh in its local frame (centred, scaled), kept for touch_at
    std::vector<float> tool_sdf_host; int tool_res[3] = {0, 0, 0}; float tool_org[3] = {0, 0, 0}, tool_cell[3] = {0, 0, 0};
    if (tool == "gridsphere") {
        // exact sphere SDF on a cubic grid, same 20% padding convention as compute_sdf
        float half = R * (1.0f + 2.0f * 0.2f);
        float cell = 2.0f * half / grid_res;
        std::vector<float> sdf((size_t)grid_res * grid_res * grid_res);
        for (int z = 0; z < grid_res; z++) for (int y = 0; y < grid_res; y++) for (int x = 0; x < grid_res; x++) {
            float px = -half + (x + 0.5f) * cell, py = -half + (y + 0.5f) * cell, pz = -half + (z + 0.5f) * cell;
            sdf[((size_t)z * grid_res + y) * grid_res + x] = sqrtf(px*px + py*py + pz*pz) - R;
        }
        g.init_sdf(g.d_tool_sdf, sdf, grid_res, grid_res, grid_res,
                   make_float3(-half, -half, -half), make_float3(cell, cell, cell), "Tool(gridsphere)");
        printf("[tool] gridsphere R=%.4f res=%d cell=%.5f (cell/R=%.3f)\n", R, grid_res, cell, cell / R);
    } else if (tool == "mesh") {
        if (cfg.tool_mesh_path.empty()) { fprintf(stderr, "--tool mesh needs knife.tool_mesh in yaml\n"); return 1; }
        RigidMesh& tm = tool_rm;
        setup_tool_mesh(cfg, g, tm);                  // Step 1: same baking as the viewer
        if (g.config.tool_type != 1) { fprintf(stderr, "tool mesh failed to load\n"); return 1; }
        tool_sdf_host = tm.sdf_data;
        for (int k = 0; k < 3; k++) { tool_res[k] = tm.sdf_res[k]; tool_org[k] = (float)tm.sdf_origin[k]; tool_cell[k] = (float)tm.sdf_cell_size[k]; }
    }

    // tool_phi(p): host trilinear phi of the baked tool SDF at world point p (same voxel-centre
    // convention as the kernels: local = R^T (p - knife_pos)).
    const float* KR = g.config.knife_rot;
    auto tool_phi = [&](float px, float py, float pz, const float3& kp) -> float {
        float wx = px - kp.x, wy = py - kp.y, wz = pz - kp.z;
        float l[3] = {KR[0]*wx + KR[3]*wy + KR[6]*wz, KR[1]*wx + KR[4]*wy + KR[7]*wz, KR[2]*wx + KR[5]*wy + KR[8]*wz};
        float gc[3]; int i0[3]; float fr[3];
        for (int k = 0; k < 3; k++) {
            gc[k] = (l[k] - tool_org[k]) / tool_cell[k] - 0.5f;
            if (gc[k] < 0.0f || gc[k] >= tool_res[k] - 1) return 1e30f;   // outside the baked box
            i0[k] = (int)floorf(gc[k]); fr[k] = gc[k] - i0[k];
        }
        auto at = [&](int x, int y, int z) { return tool_sdf_host[((size_t)z * tool_res[1] + y) * tool_res[0] + x]; };
        float c00 = at(i0[0],i0[1],i0[2])*(1-fr[0]) + at(i0[0]+1,i0[1],i0[2])*fr[0];
        float c10 = at(i0[0],i0[1]+1,i0[2])*(1-fr[0]) + at(i0[0]+1,i0[1]+1,i0[2])*fr[0];
        float c01 = at(i0[0],i0[1],i0[2]+1)*(1-fr[0]) + at(i0[0]+1,i0[1],i0[2]+1)*fr[0];
        float c11 = at(i0[0],i0[1]+1,i0[2]+1)*(1-fr[0]) + at(i0[0]+1,i0[1]+1,i0[2]+1)*fr[0];
        return (c00*(1-fr[1]) + c10*fr[1])*(1-fr[2]) + (c01*(1-fr[1]) + c11*fr[1])*fr[2];
    };

    // ---------------- Step 0.5c: replay a viewer recording (GPU-only pipeline) ----------------
    // Same tool poses, same per-frame stats format as the viewer (FrameStatsLogger), but WITHOUT
    // the viewer's CPU orphan cleanup and BFS split → the comparison isolates those two steps.
    if (!replay_path.empty()) {
        simrec::TrajectoryReader rp;
        if (!rp.load(replay_path)) return 1;
        auto absp = [](const std::string& q) {
            return q.empty() ? q : std::filesystem::absolute(q).lexically_normal().string();
        };
        simrec::RecordMeta meta;
        meta.config_path = absp(config_path);
        meta.mesh_path = absp(cfg.mesh_path);
        meta.tool_mesh_path = (g.config.tool_type == 1) ? absp(cfg.tool_mesh_path) : std::string();
        meta.dt = g.config.dt; meta.substeps = cfg.substeps; meta.num_iterations = g.config.num_iterations;
        meta.knife_radius = g.config.knife_radius; meta.tool_type = g.config.tool_type; meta.N = N;
        auto bad = rp.check(meta);
        for (auto& b : bad) fprintf(stderr, "[replay] MISMATCH %s\n", b.c_str());
        if (!bad.empty() && !replay_allow_mismatch) {
            fprintf(stderr, "[replay] refusing (pass --replay-allow-mismatch to override on purpose)\n");
            return 1;
        }
        std::string prov = "# replay " + absp(replay_path) + "\n# pipeline " +
                           std::string(gpu_only ? "tool_bench GPU-only (no orphan cleanup, no BFS split)"
                                                : "tool_bench shared viewer pipeline (frame_pipeline.h)") + "\n";
        for (auto& b : bad) prov += "# replay_mismatch " + b + "\n";
        simrec::FrameStatsLogger stats;
        std::string sp = stats_path.empty() ? out_dir + "/replay_stats.csv" : stats_path;
        if (!stats.open(sp, positions, avg_spacing, prov)) return 1;
        std::vector<float3> rpos(N); std::vector<int> rrem(N, 0); std::vector<float> rdmg(N, 0.0f);
        std::vector<int> tet_broken(tet_mesh.num_tets(), 0);
        simrec_frame::FramePipeline pipe{g, cpu_solver, tet_mesh, ea, eb, epf,
                                         rpos, rrem, rdmg, tet_broken, true, cfg.cutting_mode, N};
        // --pass-through: after each active frame (end-of-frame positions, the tool pose of that
        // frame) classify every edge whose two particles are alive by sampling phi along it every
        // half of the smallest SDF cell:
        //   in_unbroken    an endpoint is inside the tool (phi < -pass_tol), edge not broken
        //   pass_unbroken  both endpoints outside, some interior sample inside, edge NOT broken
        //                  → the tool body lies between the two particles and the edge survives
        //   pass_broken    same geometry, edge broken
        // plus the number of alive particles inside the tool and the deepest interior penetration
        // (min phi) of pass_unbroken edges. Only meaningful for --tool mesh.
        FILE* pass_f = nullptr;
        std::vector<int> pbroken;
        if (!pass_path.empty()) {
            if (tool != "mesh") { fprintf(stderr, "--pass-through needs --tool mesh\n"); return 1; }
            pass_f = fopen(pass_path.c_str(), "w");
            if (!pass_f) { fprintf(stderr, "cannot open %s\n", pass_path.c_str()); return 1; }
            fprintf(pass_f, "frame,particles_inside,in_unbroken,pass_unbroken,pass_broken,pass_unbroken_min_phi_mm\n");
        }
        float pass_h = 0.5f * std::min(tool_cell[0], std::min(tool_cell[1], tool_cell[2]));
        // "inside" means phi < -pass_tol: the collision projects particles ONTO the surface, so a
        // strict phi < 0 would count surface particles that are inside only by float rounding.
        float pass_tol = 0.2f * pass_h;
        float tool_rad = 0.0f;   // bounding radius of the baked box about the tool origin
        for (int k = 0; k < 3; k++) {
            float m = std::max(std::fabs(tool_org[k]), std::fabs(tool_org[k] + tool_res[k] * tool_cell[k]));
            tool_rad += m * m;
        }
        tool_rad = std::sqrt(tool_rad);
        // every 10th active frame also dump each crossing edge (pass_path + ".edges.csv"):
        // blade-local position of its deepest sample, broken flag, damage, current length, and the
        // distance from its midpoint to the sharp-edge segment (what sharp_edge_damage_kernel sees)
        FILE* pass_e = nullptr; std::vector<float> pdmg;
        if (pass_f) {
            pass_e = fopen((pass_path + ".edges.csv").c_str(), "w");
            fprintf(pass_e, "frame,edge,kind,lx,ly,lz,broken,damage,len_mm,mid_to_sharp_mm\n");
        }
        auto to_local = [&](float px, float py, float pz, const float3& kp, float l[3]) {
            float wx = px - kp.x, wy = py - kp.y, wz = pz - kp.z;
            l[0] = KR[0]*wx + KR[3]*wy + KR[6]*wz; l[1] = KR[1]*wx + KR[4]*wy + KR[7]*wz; l[2] = KR[2]*wx + KR[5]*wy + KR[8]*wz;
        };
        auto mid_to_sharp = [&](const float m[3]) {
            float3 A = g.config.sharp_edge_a, B = g.config.sharp_edge_b;
            float ab[3] = {B.x-A.x, B.y-A.y, B.z-A.z}, am[3] = {m[0]-A.x, m[1]-A.y, m[2]-A.z};
            float t = (am[0]*ab[0]+am[1]*ab[1]+am[2]*ab[2]) / (ab[0]*ab[0]+ab[1]*ab[1]+ab[2]*ab[2]);
            t = std::min(1.0f, std::max(0.0f, t));
            float d[3] = {am[0]-t*ab[0], am[1]-t*ab[1], am[2]-t*ab[2]};
            return std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
        };
        int pass_count = 0;
        auto pass_through_frame = [&](int frame, const float3& kp) {
            g.download_edge_broken(pbroken);
            bool dump = pass_e && (pass_count++ % 10 == 0);
            if (dump) g.download_edge_damage(pdmg);
            int pin = 0, in_u = 0, pass_u = 0, pass_b = 0; float minphi = 0.0f;
            for (int i = 0; i < N; i++)
                if (!rrem[i] && tool_phi(rpos[i].x, rpos[i].y, rpos[i].z, kp) < -pass_tol) pin++;
            for (int e = 0; e < E; e++) {
                int a = ea[e], b = eb[e];
                if (rrem[a] || rrem[b]) continue;
                float3 pa = rpos[a], pb = rpos[b];
                float mx = 0.5f*(pa.x+pb.x) - kp.x, my = 0.5f*(pa.y+pb.y) - kp.y, mz = 0.5f*(pa.z+pb.z) - kp.z;
                float dx = pb.x-pa.x, dy = pb.y-pa.y, dz = pb.z-pa.z, len = std::sqrt(dx*dx+dy*dy+dz*dz);
                if (std::sqrt(mx*mx+my*my+mz*mz) > tool_rad + 0.5f*len) continue;
                float fa = tool_phi(pa.x, pa.y, pa.z, kp), fb = tool_phi(pb.x, pb.y, pb.z, kp);
                if (fa < -pass_tol || fb < -pass_tol) { if (!pbroken[e]) in_u++; continue; }
                int ns = std::max(2, (int)std::ceil(len / pass_h));
                float mn = 1e30f, tmn = 0.5f;
                for (int k = 1; k < ns; k++) {
                    float t = (float)k / ns;
                    float v = tool_phi(pa.x+t*dx, pa.y+t*dy, pa.z+t*dz, kp);
                    if (v < mn) { mn = v; tmn = t; }
                }
                if (dump && mn < -pass_tol) {
                    float l[3], m[3];
                    to_local(pa.x+tmn*dx, pa.y+tmn*dy, pa.z+tmn*dz, kp, l);
                    to_local(0.5f*(pa.x+pb.x), 0.5f*(pa.y+pb.y), 0.5f*(pa.z+pb.z), kp, m);
                    fprintf(pass_e, "%d,%d,pass,%.4f,%.4f,%.4f,%d,%.3f,%.4f,%.4f\n", frame, e, l[0]*1e3f, l[1]*1e3f, l[2]*1e3f,
                            pbroken[e], pdmg[e], len*1e3f, mid_to_sharp(m)*1e3f);
                }
                if (mn < -pass_tol) {
                    if (pbroken[e]) pass_b++;
                    else { pass_u++; minphi = std::min(minphi, mn); }
                }
            }
            fprintf(pass_f, "%d,%d,%d,%d,%d,%.4f\n", frame, pin, in_u, pass_u, pass_b, minphi * 1e3f);
        };
        int f = 0; bool prev_active = false;
        for (const auto& fr : rp.frames()) {
            g.config.knife_pos = fr.kp;
            memcpy(g.config.knife_rot, fr.rot, sizeof(fr.rot));
            g.config.knife_active = fr.active;
            // --pass-through: the state the tool meets when it switches on (previous frame's
            // positions, this frame's pose), logged as frame -f
            if (pass_f && fr.active && !prev_active && f > 0) pass_through_frame(-f, fr.kp);
            prev_active = fr.active != 0;
            double ms;
            simrec::FrameSources src;   // GPU-only: orphan/BFS columns stay -1 (stage absent)
            if (gpu_only) {
                int rb = 0; for (int v : rrem) rb += (v != 0);
                cudaDeviceSynchronize();
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int s = 0; s < fr.substeps; s++) g.step();
                cudaDeviceSynchronize();
                ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
                g.download_positions(rpos); g.download_removed(rrem);
                int ra = 0; for (int v : rrem) ra += (v != 0);
                src.gpu_new = ra - rb;
            } else {
                simrec_frame::FrameResult fres = simrec_frame::advance_frame(pipe, fr.substeps, f, fr.active != 0);
                ms = fres.gpu_ms;
                src = simrec::FrameSources{fres.gpu_new, fres.orphan_new, fres.bfs_new, fres.orphan_ms, fres.bfs_ms,
                                               fres.components, fres.broken_edges,
                                               fres.bfs_ran ? fres.tet_broken_count : -1};
            }
            stats.write(f, ms, fr.kp, g.config.knife_radius, fr.active, g.config.tool_type, rpos, rrem, fixed_arr,
                        0.0, src);
            if (pass_f && fr.active) pass_through_frame(f, fr.kp);
            f++;
        }
        if (pass_f) fclose(pass_f);
        if (pass_e) fclose(pass_e);
        // --dump-topology F: end-of-replay state for offline connectivity checks.
        //   F.particles.csv  i,x,y,z,removed,fixed      F.edges.csv  a,b,broken  (cutting edges)
        if (!topo_path.empty()) {
            std::vector<int> brk; g.download_edge_broken(brk);
            FILE* fp = fopen((topo_path + ".particles.csv").c_str(), "w");
            fprintf(fp, "i,x,y,z,removed,fixed\n");
            for (int i = 0; i < N; i++)
                fprintf(fp, "%d,%.9g,%.9g,%.9g,%d,%d\n", i, rpos[i].x, rpos[i].y, rpos[i].z, rrem[i], fixed_arr[i]);
            fclose(fp);
            FILE* fe = fopen((topo_path + ".edges.csv").c_str(), "w");
            fprintf(fe, "a,b,broken\n");
            for (int e = 0; e < E; e++) fprintf(fe, "%d,%d,%d\n", ea[e], eb[e], brk[e]);
            fclose(fe);
            FILE* fc = fopen((topo_path + ".clusters.txt").c_str(), "w");   // one cluster per line
            for (const auto& c : cpu_solver.clusters) {
                for (size_t k = 0; k < c.particle_ids.size(); k++) fprintf(fc, k ? " %d" : "%d", c.particle_ids[k]);
                fprintf(fc, "\n");
            }
            fclose(fc);
        }
        int rt = 0; for (int v : rrem) rt += v;
        printf("[replay] %d frames replayed (%s) | removed total %d / %d | stats -> %s\n", f,
               gpu_only ? "GPU-only" : "shared viewer pipeline", rt, N, sp.c_str());
        g.cleanup();
        return 0;
    }

    // ---------------- trajectory: half-embedded sweep along x at z-mid ----------------
    float bmn[3] = {1e30f, 1e30f, 1e30f}, bmx[3] = {-1e30f, -1e30f, -1e30f};
    for (auto& p : positions) {
        bmn[0] = std::min(bmn[0], p.x); bmn[1] = std::min(bmn[1], p.y); bmn[2] = std::min(bmn[2], p.z);
        bmx[0] = std::max(bmx[0], p.x); bmx[1] = std::max(bmx[1], p.y); bmx[2] = std::max(bmx[2], p.z);
    }
    float zmid = 0.5f * (bmn[2] + bmx[2]);
    // ---- tool geometry used by the SCRIPT (not by the solver) ----
    // Ls = length scale of the scripted motion: sphere → R (unchanged); mesh → half the tool's
    // smallest extent (dissector: half the blade thickness). knife_radius is NOT the tool size for
    // tool_type 1 (the GPU mesh-tool kernels never read it).
    float Ls = R;
    // lower envelope of the (rotated) mesh tool: low_y[(ix,iz)] = lowest world-y offset (relative
    // to the tool centre) where phi <= 0 in the vertical column at horizontal offset (dx,dz).
    float fp_lo[2] = {0, 0}, fp_h = 0, y_lo = 0, y_hi = 0; int fp_n[2] = {0, 0};
    std::vector<float> low_y;
    if (tool == "mesh") {
        float clo[3] = {1e30f, 1e30f, 1e30f}, chi[3] = {-1e30f, -1e30f, -1e30f};
        for (int c = 0; c < 8; c++) {           // world AABB of the rotated baked box
            float l[3];
            for (int k = 0; k < 3; k++) l[k] = tool_org[k] + (((c >> k) & 1) ? tool_res[k] * tool_cell[k] : 0.0f);
            for (int r = 0; r < 3; r++) {
                float w = KR[r*3+0]*l[0] + KR[r*3+1]*l[1] + KR[r*3+2]*l[2];
                clo[r] = std::min(clo[r], w); chi[r] = std::max(chi[r], w);
            }
        }
        float cmin = std::min(tool_cell[0], std::min(tool_cell[1], tool_cell[2]));
        fp_h = 0.5f * cmin; y_lo = clo[1]; y_hi = chi[1];
        fp_lo[0] = clo[0]; fp_lo[1] = clo[2];
        fp_n[0] = (int)ceilf((chi[0] - clo[0]) / fp_h); fp_n[1] = (int)ceilf((chi[2] - clo[2]) / fp_h);
        low_y.assign((size_t)fp_n[0] * fp_n[1], 1e30f);
        float ext_min = 1e30f, ext[3] = {-1e30f, -1e30f, -1e30f}, emn[3] = {1e30f, 1e30f, 1e30f};
        const float3 O = make_float3(0, 0, 0);
        for (int ix = 0; ix < fp_n[0]; ix++) for (int iz = 0; iz < fp_n[1]; iz++) {
            float dx = fp_lo[0] + (ix + 0.5f) * fp_h, dz = fp_lo[1] + (iz + 0.5f) * fp_h;
            for (float y = y_lo; y <= y_hi; y += 0.25f * cmin)
                if (tool_phi(dx, y, dz, O) <= 0.0f) {
                    low_y[(size_t)ix * fp_n[1] + iz] = y;
                    emn[0] = std::min(emn[0], dx); ext[0] = std::max(ext[0], dx);
                    emn[2] = std::min(emn[2], dz); ext[2] = std::max(ext[2], dz);
                    break;
                }
        }
        // vertical solid extent: scan the centre column range for the top too
        for (int ix = 0; ix < fp_n[0]; ix++) for (int iz = 0; iz < fp_n[1]; iz++) {
            if (low_y[(size_t)ix * fp_n[1] + iz] > 1e29f) continue;
            float dx = fp_lo[0] + (ix + 0.5f) * fp_h, dz = fp_lo[1] + (iz + 0.5f) * fp_h;
            for (float y = y_hi; y >= y_lo; y -= 0.25f * cmin)
                if (tool_phi(dx, y, dz, O) <= 0.0f) { ext[1] = std::max(ext[1], y); break; }
            emn[1] = std::min(emn[1], low_y[(size_t)ix * fp_n[1] + iz]);
        }
        for (int k = 0; k < 3; k++) ext_min = std::min(ext_min, ext[k] - emn[k]);
        Ls = 0.5f * ext_min;
        printf("[tool] mesh pose=%s | solid world extent (%.4f, %.4f, %.4f) m | footprint table %dx%d (h=%.5f) | script length scale Ls=%.5f\n",
               mesh_pose.c_str(), ext[0] - emn[0], ext[1] - emn[1], ext[2] - emn[2], fp_n[0], fp_n[1], fp_h, Ls);
    }
    float margin = 0.5f * Ls;
    float x0 = bmn[0] + margin, x1 = bmx[0] - margin;
    // local top-surface height along the sweep line (follows curved tops, e.g. tumor)
    float col = std::max(0.5f * Ls, 1.5f * avg_spacing);
    // Height profile is computed ONCE from the SETTLED body (end of the settle phase) and
    // never updated afterwards. (An earlier version re-read the current top each frame; after
    // removal the "top" became the cavity floor and the knife teleported down into its own
    // hole by up to 0.75R per frame — that measured digging, not contact.)
    std::vector<float3> settled_pos = positions;
    // touch_at(x): knife-CENTRE height at which the sphere first touches the settled body
    // when lowered vertically at (x, zmid) = max_i [ y_i + sqrt(R^2 - rho_i^2) ] over particles
    // with horizontal distance rho_i < R. Uses the whole footprint, so curved tops (tumor) are
    // handled; on a flat top it equals top + R, i.e. centre = touch - R is "half-embedded".
    (void)col;
    // touch_mesh(x): tool-CENTRE height at which the tool mesh, lowered vertically at (x, zmid),
    // first touches the settled tissue SURFACE (tet_mesh.surface_faces on settled_pos), computed
    // exactly. Two polyhedral surfaces brought together along -y first meet at a vertex-face or an
    // edge-edge pair, so the answer is the max over: tool vertex above a tissue face, tissue vertex
    // below a tool face, and tool edge / tissue edge crossing in the xz projection.
    // (The previous version tested only tissue PARTICLES inside the tool's xz footprint: a blade
    //  thinner than the particle spacing slid between surface particles and the "touching" pose
    //  was 2 mm inside the tissue for the scalpel.)
    std::vector<float3> tool_w;                 // tool vertices as world offsets R*v
    std::vector<std::array<int, 2>> tool_e;     // unique tool edges
    float tw_lo[2] = {1e30f, 1e30f}, tw_hi[2] = {-1e30f, -1e30f};   // xz bbox of the offsets
    if (tool == "mesh") {
        for (auto& v : tool_rm.vertices) {
            float3 w = make_float3(KR[0]*v.x() + KR[1]*v.y() + KR[2]*v.z(),
                                   KR[3]*v.x() + KR[4]*v.y() + KR[5]*v.z(),
                                   KR[6]*v.x() + KR[7]*v.y() + KR[8]*v.z());
            tool_w.push_back(w);
            tw_lo[0] = std::min(tw_lo[0], w.x); tw_hi[0] = std::max(tw_hi[0], w.x);
            tw_lo[1] = std::min(tw_lo[1], w.z); tw_hi[1] = std::max(tw_hi[1], w.z);
        }
        std::set<std::pair<int,int>> es;
        for (auto& f : tool_rm.faces)
            for (int k = 0; k < 3; k++) es.insert({std::min(f[k], f[(k+1)%3]), std::max(f[k], f[(k+1)%3])});
        for (auto& e : es) tool_e.push_back({e.first, e.second});
    }
    // y of the vertical line through (qx,qz) on triangle (a,b,c); false if it misses (or the
    // triangle is vertical: its contacts are covered by the vertex and edge-edge tests)
    auto vline_tri = [](float qx, float qz, const float3& a, const float3& b, const float3& c, float& y) {
        float d = (b.x - a.x) * (c.z - a.z) - (c.x - a.x) * (b.z - a.z);
        if (fabsf(d) < 1e-14f) return false;
        float u = ((qx - a.x) * (c.z - a.z) - (c.x - a.x) * (qz - a.z)) / d;
        float v = ((b.x - a.x) * (qz - a.z) - (qx - a.x) * (b.z - a.z)) / d;
        if (u < 0.0f || v < 0.0f || u + v > 1.0f) return false;
        y = a.y + u * (b.y - a.y) + v * (c.y - a.y);
        return true;
    };
    // xz-projection crossing of segments p0p1 and q0q1 → parameters s (on p), t (on q)
    auto seg_xz = [](const float3& p0, const float3& p1, const float3& q0, const float3& q1, float& s, float& t) {
        float rx = p1.x - p0.x, rz = p1.z - p0.z, ux = q1.x - q0.x, uz = q1.z - q0.z;
        float den = rx * uz - rz * ux;
        if (fabsf(den) < 1e-14f) return false;
        float wx = q0.x - p0.x, wz = q0.z - p0.z;
        s = (wx * uz - wz * ux) / den; t = (wx * rz - wz * rx) / den;
        return s >= 0.0f && s <= 1.0f && t >= 0.0f && t <= 1.0f;
    };
    auto touch_mesh = [&](float x) {
        const auto& SF = tet_mesh.surface_faces;
        float lo[2] = {x + tw_lo[0], zmid + tw_lo[1]}, hi[2] = {x + tw_hi[0], zmid + tw_hi[1]};
        std::vector<int> cand;                   // surface faces whose xz bbox overlaps the tool's
        for (int fi = 0; fi < (int)SF.size(); fi++) {
            const float3 &a = settled_pos[SF[fi][0]], &b = settled_pos[SF[fi][1]], &c = settled_pos[SF[fi][2]];
            if (std::max({a.x, b.x, c.x}) < lo[0] || std::min({a.x, b.x, c.x}) > hi[0] ||
                std::max({a.z, b.z, c.z}) < lo[1] || std::min({a.z, b.z, c.z}) > hi[1]) continue;
            cand.push_back(fi);
        }
        float h = -1e30f, y;
        const float3 kc = make_float3(x, 0.0f, zmid);
        auto tw = [&](int i) { return make_float3(kc.x + tool_w[i].x, tool_w[i].y, kc.z + tool_w[i].z); };
        // tool vertex above a tissue face
        for (int i = 0; i < (int)tool_w.size(); i++) {
            float3 w = tw(i);
            for (int fi : cand)
                if (vline_tri(w.x, w.z, settled_pos[SF[fi][0]], settled_pos[SF[fi][1]], settled_pos[SF[fi][2]], y))
                    h = std::max(h, y - w.y);
        }
        // tissue vertex below a tool face
        for (int fi : cand) for (int k = 0; k < 3; k++) {
            const float3& p = settled_pos[SF[fi][k]];
            for (auto& tf : tool_rm.faces)
                if (vline_tri(p.x, p.z, tw(tf[0]), tw(tf[1]), tw(tf[2]), y)) h = std::max(h, p.y - y);
        }
        // tool edge vs tissue edge
        for (int fi : cand) for (int k = 0; k < 3; k++) {
            const float3 &q0 = settled_pos[SF[fi][k]], &q1 = settled_pos[SF[fi][(k+1)%3]];
            for (auto& te : tool_e) {
                float3 p0 = tw(te[0]), p1 = tw(te[1]); float sp, tq;
                if (seg_xz(p0, p1, q0, q1, sp, tq))
                    h = std::max(h, (q0.y + tq * (q1.y - q0.y)) - (p0.y + sp * (p1.y - p0.y)));
            }
        }
        return (h > -1e29f) ? h : bmx[1] - y_lo;
    };
    auto touch_at = [&](float x) {
        float h = -1e30f;
        if (tool == "mesh") return touch_mesh(x);
        for (int i = 0; i < N; i++) {
            const float3& p = settled_pos[i];
            float dx = p.x - x, dz = p.z - zmid, rho2 = dx*dx + dz*dz;
            if (rho2 < R * R) h = std::max(h, p.y + sqrtf(R * R - rho2));
        }
        return (h > -1e29f) ? h : bmx[1] + R;
    };
    // Phases: [0,S) settle (knife inactive, far) | [S,S+A) approach: knife active, starts 0.2R
    // ABOVE the settled surface at x0 and descends linearly to (surface - depth) | [S+A, S+A+W)
    // sweep along x at fixed (surface(x) - depth).
    const int S = n_settle, A = n_approach, W = n_frames, F_total = S + A + W;
    float h0 = 0.0f;

    printf("=== tool_bench: %s | tool=%s | N=%d C=%d E=%d fixed=%d | spacing=%.5f | R=%.4f (R/spacing=%.2f)\n",
           cfg.name.c_str(), tool.c_str(), N, C, E, n_fixed, avg_spacing, Ls, Ls / avg_spacing);
    printf("    dt=%.4f iters=%d substeps=%d cut_interval=%d alpha_min=%.2f tau=%.2f growth=%.3f sc=%d(r=%.4f)\n",
           cfg.dt, cfg.num_iterations, cfg.substeps, cfg.cutting_interval, cfg.alpha_min,
           cfg.aspiration_threshold, cfg.damage_growth_rate, (int)cfg.self_collision_enabled,
           cfg.self_collision_radius);

    // ---------------- run ----------------
    // --write-traj: emit the scripted tool poses as a viewer-compatible recording, so the same
    // scripted trajectory can be replayed through the FULL pipeline (tool_bench --replay or the viewer).
    simrec::TrajectoryRecorder traj_out;
    if (!write_traj_path.empty()) {
        auto absp = [](const std::string& q) {
            return q.empty() ? q : std::filesystem::absolute(q).lexically_normal().string();
        };
        simrec::RecordMeta meta;
        meta.config_path = absp(config_path); meta.mesh_path = absp(cfg.mesh_path);
        meta.tool_mesh_path = (g.config.tool_type == 1) ? absp(cfg.tool_mesh_path) : std::string();
        meta.dt = g.config.dt; meta.substeps = cfg.substeps; meta.num_iterations = g.config.num_iterations;
        meta.knife_radius = g.config.knife_radius; meta.tool_type = g.config.tool_type; meta.N = N;
        if (!traj_out.open(write_traj_path, meta)) return 1;
    }
    std::ofstream csv(out_dir + "/frames.csv");
    csv << "frame,kx,ky,kz,active,removed_total,new_removed,broken_edges,max_edge_dmg,n_contact,gpu_ms\n";

    std::vector<float3> pos, pred;
    std::vector<int> rem_prev(N, 0), rem_now, broken;
    std::vector<float> pdmg, edmg;

    // aspiration-rounding stats (sphere-distance classification of each removal)
    const double bins[4] = {1e-6, 1e-3, 1e-2, 1e30};
    long long rem_bin_ph[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};   // [approach|sweep][depth bin]
    int first_contact = -1, first_removal = -1; float max_step_xy = 0.0f, max_step_y = 0.0f;
    float3 kp_prev = make_float3(0, 0, 0); bool have_prev = false;
    long long band_alive_overtau = 0, band_alive_total = 0, asp_substeps = 0;
    long long alive_overtau = 0, alive_overtau_inside = 0; float max_pdmg_alive = 0.0f;
    std::vector<double> gpu_ms_all;

    auto sphere_dist = [&](const float3& p, const float3& k) {
        float dx = p.x - k.x, dy = p.y - k.y, dz = p.z - k.z;  // float, same ops as aspirate_kernel
        return sqrtf(dx*dx + dy*dy + dz*dz);
    };

    for (int f = 0; f < F_total; f++) {
        int phase = (f < S) ? -1 : (f < S + A ? 0 : 1);   // -1 settle, 0 approach, 1 sweep
        if (f == S) {                                                     // outside timed region
            g.download_positions(settled_pos);
            // sweep only where the constant-penetration path has |dH/dx| <= 1, so the knife never
            // moves more vertically than laterally per frame (no "plunge" on steep flanks)
            const int K = 200; std::vector<float> xs(K), H(K);
            for (int k = 0; k < K; k++) { xs[k] = x0 + (x1 - x0) * k / (K - 1); H[k] = touch_at(xs[k]); }
            int kmax = (int)(std::max_element(H.begin(), H.end()) - H.begin()), kl = kmax, kr = kmax;
            float dx = (x1 - x0) / (K - 1);
            while (kl > 0     && fabsf(H[kl - 1] - H[kl]) <= dx) kl--;
            while (kr < K - 1 && fabsf(H[kr + 1] - H[kr]) <= dx) kr++;
            x0 = xs[kl]; x1 = xs[kr];
            h0 = touch_at(x0);
            printf("[traj] sweep range x in [%.4f, %.4f] (%.0f%% of bbox span, slope-limited)\n",
                   x0, x1, 100.0 * (x1 - x0) / (bmx[0] - bmn[0]));
        }
        float kx, ky;
        if (phase < 0)      { kx = 100.0f; ky = 100.0f; }
        // approach: centre from touch+0.2R (clear of the body) down to touch-R-depth (half-embedded + depth)
        // t runs 0..1 over the A approach frames: the FIRST active frame is at touch+0.2R (clear of
        // the body, as intended), the last at the target. (Was (f-S+1)/A: the first active frame
        // was already one step down, which for the scalpel, step 0.083 mm > 0.2R = 0.08 mm, put
        // the edge inside the tissue at switch-on.)
        else if (phase == 0) { float t = (float)(f - S) / std::max(A - 1, 1); kx = x0; ky = (h0 + 0.2f * Ls) + t * (-1.2f * Ls - depth); }
        else                 { float t = (float)(f - S - A) / std::max(W - 1, 1); kx = x0 + t * (x1 - x0); ky = touch_at(kx) - Ls - depth; }
        float3 kp = make_float3(kx, ky, zmid);
        g.config.knife_pos = kp;
        g.config.knife_active = (phase >= 0) ? 1 : 0;
        traj_out.write(f, g.config.knife_pos, g.config.knife_rot, g.config.knife_active, cfg.substeps);
        if (phase >= 0 && have_prev) {
            max_step_y = std::max(max_step_y, fabsf(ky - kp_prev.y));
            max_step_xy = std::max(max_step_xy, fabsf(kx - kp_prev.x));
        }
        if (phase >= 0) { kp_prev = kp; have_prev = true; }

        double ms = 0.0;
        if (timing) {
            cudaDeviceSynchronize();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int s = 0; s < cfg.substeps; s++) g.step();
            cudaDeviceSynchronize();
            ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
        } else {
            for (int s = 0; s < cfg.substeps; s++) {
                int sc_before = g.step_count;
                bool asp = g.config.knife_active && (sc_before % g.cutting_update_interval == 0);
                cudaDeviceSynchronize();
                auto t0 = std::chrono::high_resolution_clock::now();
                g.step();
                cudaDeviceSynchronize();
                ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
                if (!asp || tool == "mesh") continue;
                asp_substeps++;
                g.download_removed(rem_now);
                g.download_predicted_positions(pred);
                g.download_damage(pdmg);
                for (int i = 0; i < N; i++) {
                    float d = sphere_dist(pred[i], kp);
                    double delta = (R - d) / R;  // >0 inside the sphere
                    if (rem_now[i] && !rem_prev[i]) {
                        int b = 0; while (b < 3 && delta >= bins[b]) b++;
                        rem_bin_ph[phase][b]++;
                        if (first_removal < 0) first_removal = f;
                    }
                    if (!rem_now[i] && !fixed_arr[i]) {
                        max_pdmg_alive = std::max(max_pdmg_alive, pdmg[i]);
                        if (pdmg[i] > cfg.aspiration_threshold) { alive_overtau++; if (delta > 0) alive_overtau_inside++; }
                    }
                    if (!(rem_now[i] && !rem_prev[i]) && !rem_now[i] && !fixed_arr[i] && fabs(delta) < 1e-5) {
                        band_alive_total++;
                        if (pdmg[i] > cfg.aspiration_threshold) band_alive_overtau++;
                    }
                }
                rem_prev = rem_now;
            }
        }
        if (phase >= 0) gpu_ms_all.push_back(ms);

        g.download_positions(pos);
        g.download_removed(rem_now);
        int rem_total = 0, new_rem = 0, n_contact = 0;
        for (int i = 0; i < N; i++) {
            rem_total += rem_now[i];
            if (phase >= 0 && !rem_now[i]) {
                if (tool == "mesh") { if (tool_phi(pos[i].x, pos[i].y, pos[i].z, kp) < 1e-4f * Ls) n_contact++; }
                else if (sphere_dist(pos[i], kp) < R * (1.0f + 1e-4f)) n_contact++;
            }
        }
        static int rem_total_prev = 0; new_rem = rem_total - rem_total_prev; rem_total_prev = rem_total;
        if (phase >= 0) n_contact += new_rem;   // removed this frame were on the knife surface (verified by depth bins)
        if (n_contact > 0 && first_contact < 0) first_contact = f;
        if ((timing || tool == "mesh") && new_rem > 0 && first_removal < 0) first_removal = f;
        if (timing) rem_prev = rem_now;
        int nb = 0; float maxd = 0.0f;
        if (!timing) {
            g.download_edge_broken(broken); g.download_edge_damage(edmg);
            for (int e = 0; e < E; e++) { nb += broken[e]; maxd = std::max(maxd, edmg[e]); }
        }
        csv << f << "," << kx << "," << ky << "," << zmid << "," << g.config.knife_active << ","
            << rem_total << "," << new_rem << "," << nb << "," << maxd << "," << n_contact << "," << ms << "\n";

        // P(F) formula gap on the SAME positions: sphere vs SDF-kernel formula
        if (!timing && tool != "mesh" && phase == 1 && ((f - S - A) % 20 == 0 || f == F_total - 1)) {
            g.download_predicted_positions(pred);
            int n_band = 0, sph_break = 0, sdf_break = 0, deep = 0;
            double sph_sum = 0, sdf_sum = 0;
            for (int e = 0; e < E; e++) {
                int a = ea[e], b = eb[e];
                if (rem_now[a] || rem_now[b]) continue;
                float3 m = make_float3(0.5f*(pred[a].x+pred[b].x), 0.5f*(pred[a].y+pred[b].y), 0.5f*(pred[a].z+pred[b].z));
                float re = 0.5f * sqrtf((pred[a].x-pred[b].x)*(pred[a].x-pred[b].x) +
                                        (pred[a].y-pred[b].y)*(pred[a].y-pred[b].y) +
                                        (pred[a].z-pred[b].z)*(pred[a].z-pred[b].z));
                if (re < 1e-10f) continue;
                float phi = sphere_dist(m, kp) - R;            // exact sphere SDF at midpoint
                if (phi > re) continue;                          // only edges whose circumsphere meets the tool
                n_band++;
                float pf_sph = 1.0f / (1.0f + expf(-cfg.alpha_min * (phi - re) / re));
                float pf_sdf = 1.0f / (1.0f + expf(-cfg.alpha_min * phi / re));
                float t_sph = std::max(0.0f, 1.0f - pf_sph), t_sdf = std::max(0.0f, 1.0f - pf_sdf);
                if (fabsf(phi) > 3.0f * re) { deep++; t_sdf = 0.0f; }   // SDF kernel early-rejects |sdist|>3r_e
                sph_sum += t_sph; sdf_sum += t_sdf;
                sph_break += (t_sph > 0.5f); sdf_break += (t_sdf > 0.5f);
            }
            printf("[PF-gap f=%3d] edges whose circumsphere meets tool: %d | mean target dmg sphere=%.3f sdf=%.3f"
                   " | target>0.5: sphere=%d sdf=%d | deep(|phi|>3re, skipped by sdf)=%d\n",
                   f, n_band, n_band ? sph_sum / n_band : 0.0, n_band ? sdf_sum / n_band : 0.0,
                   sph_break, sdf_break, deep);
        }
    }

    // ---------------- summary ----------------
    g.download_positions(pos); g.download_removed(rem_now);
    {
        std::ofstream fs(out_dir + "/final_state.bin", std::ios::binary);
        fs.write((char*)pos.data(), N * sizeof(float3));
        fs.write((char*)rem_now.data(), N * sizeof(int));
    }
    uint64_t h = fnv1a(pos.data(), N * sizeof(float3));
    h = fnv1a(rem_now.data(), N * sizeof(int), h);
    int rem_total = 0; for (int v : rem_now) rem_total += v;

    std::vector<double> act(gpu_ms_all.begin(), gpu_ms_all.end());
    std::sort(act.begin(), act.end());
    auto pct = [&](double q) { return act.empty() ? 0.0 : act[std::min(act.size() - 1, (size_t)(q * act.size()))]; };

    printf("\n=== SUMMARY tool=%s ===\n", tool.c_str());
    printf("removed total: %d / %d\n", rem_total, N);
    printf("phases: settle %d | approach %d (centre from touch+0.2R down to touch-R-%.4f, touch(x0)=%.4f) | sweep %d frames\n", S, A, depth, h0, W);
    printf("first contact frame: %d | first removal frame: %d | max knife step per frame: vertical %.5f m (%.3fR), lateral %.5f m (%.3fR)\n",
           first_contact, first_removal, max_step_y, max_step_y / Ls, max_step_xy, max_step_xy / Ls);
    printf("final_state FNV-1a: %016llx\n", (unsigned long long)h);
    printf("GPU ms/frame (knife active, %d substeps%s): median=%.2f p90=%.2f p99=%.2f max=%.2f\n",
           cfg.substeps, timing ? "" : ", instrumented", pct(0.5), pct(0.9), pct(0.99), act.empty() ? 0.0 : act.back());
    if (!timing && tool != "mesh") {
        const char* nm[2] = {"approach", "sweep"};
        printf("aspiration substeps observed: %lld\n", asp_substeps);
        for (int ph = 0; ph < 2; ph++)
            printf("  [%-8s] removed, by depth inside sphere (R-d)/R: <1e-6 (rounding): %lld | 1e-6..1e-3: %lld | 1e-3..1e-2: %lld | >1e-2: %lld\n",
                   nm[ph], rem_bin_ph[ph][0], rem_bin_ph[ph][1], rem_bin_ph[ph][2], rem_bin_ph[ph][3]);
        printf("  alive after aspiration (summed over substeps): damage>tau: %lld, of which strictly inside sphere: %lld | max alive particle damage: %.3f (tau=%.2f)\n",
               alive_overtau, alive_overtau_inside, max_pdmg_alive, cfg.aspiration_threshold);
        printf("  alive, on the sphere surface (|R-d|<1e-5 R) right after aspiration: %lld, of which damage>tau: %lld\n",
               band_alive_total, band_alive_overtau);
    }
    g.cleanup();
    return 0;
}
