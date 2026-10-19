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
#include "core/cutting_manager.h"
#include "core/pf_calculator.h"
#include "core/rigid_mesh.h"
#include "core/sim_recorder.h"
#include "core/frame_pipeline.h"
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
    std::string replay_path, stats_path, write_traj_path;
    float depth = 0.0f;
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
        else if (a == "--depth" && i + 1 < argc) depth = (float)std::atof(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }
    if (tool != "sphere" && tool != "gridsphere" && tool != "mesh") {
        fprintf(stderr, "--tool must be sphere|gridsphere|mesh\n"); return 1;
    }

    SimConfigYAML cfg;
    if (!cfg.load(config_path)) return 1;
    std::filesystem::create_directories(out_dir);
    if (cfg.per_particle_tau || cfg.per_particle_stiffness || cfg.per_edge_alpha)
        printf("[WARN] heterogeneous fields in config are IGNORED by tool_bench (uniform material)\n");

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
    cm.init(cpu_solver.clusters, cpu_solver.particles);
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
    g.config.dt = cfg.dt;
    g.config.gravity = make_float3(cfg.gravity[0], cfg.gravity[1], cfg.gravity[2]);
    g.config.num_iterations = cfg.num_iterations;
    g.config.stiffness = cfg.stiffness;
    g.config.damping = cfg.damping;
    g.config.floor_y = cfg.floor_y;
    g.config.knife_pos = make_float3(100, 100, 100);
    float I3[9] = {1,0,0, 0,1,0, 0,0,1};
    memcpy(g.config.knife_rot, I3, sizeof(I3));
    g.config.knife_radius = cfg.knife_radius;
    g.config.knife_active = 0;
    g.config.self_collision_radius = cfg.self_collision_radius;
    g.config.self_collision_stiffness = cfg.self_collision_stiffness;
    g.config.self_collision_enabled = cfg.self_collision_enabled ? 1 : 0;
    g.config.friction_s = cfg.friction_s;
    g.config.friction_k = cfg.friction_k;
    g.config.tool_type = (tool == "sphere") ? 0 : 1;
    g.config.enable_aspiration = 1;
    g.config.damage_shape_weight = 0.0f;
    g.config.use_sharp_edge_damage = 0;
    g.config.damage_growth_rate = cfg.damage_growth_rate;
    g.cutting_update_interval = cfg.cutting_interval;
    g.alpha_min = cfg.alpha_min;
    g.aspiration_threshold = cfg.aspiration_threshold;
    g.damage_model_mode = (cfg.damage_model == "toughness") ? 1 : 0;
    g.init(positions, velocities, removed, fixed_arr, cs, csz, pids, roff, rcm);
    g.init_edges(ea, eb, epf);

    const float R = cfg.knife_radius;
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
        RigidMesh tm;
        if (!tm.load_obj(cfg.tool_mesh_path)) return 1;
        Vec3 bmin, bmax; tm.get_bounds(bmin, bmax); tm.translate(-(bmin + bmax) * 0.5);
        tm.compute_sdf(cfg.tool_sdf_resolution, 0.2);
        g.init_sdf(g.d_tool_sdf, tm.sdf_data, tm.sdf_res[0], tm.sdf_res[1], tm.sdf_res[2],
                   make_float3((float)tm.sdf_origin.x(), (float)tm.sdf_origin.y(), (float)tm.sdf_origin.z()),
                   make_float3((float)tm.sdf_cell_size.x(), (float)tm.sdf_cell_size.y(), (float)tm.sdf_cell_size.z()),
                   "Tool(mesh)");
    }

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
        int f = 0;
        for (const auto& fr : rp.frames()) {
            g.config.knife_pos = fr.kp;
            memcpy(g.config.knife_rot, fr.rot, sizeof(fr.rot));
            g.config.knife_active = fr.active;
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
                src = simrec::FrameSources{fres.gpu_new, fres.orphan_new, fres.bfs_new, fres.orphan_ms, fres.bfs_ms};
            }
            stats.write(f, ms, fr.kp, g.config.knife_radius, fr.active, g.config.tool_type, rpos, rrem, fixed_arr,
                        0.0, src);
            f++;
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
    float margin = 0.5f * R;
    float x0 = bmn[0] + margin, x1 = bmx[0] - margin;
    // local top-surface height along the sweep line (follows curved tops, e.g. tumor)
    float col = std::max(0.5f * R, 1.5f * avg_spacing);
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
    auto touch_at = [&](float x) {
        float h = -1e30f;
        for (int i = 0; i < N; i++) {
            const float3& p = settled_pos[i];
            float dx = p.x - x, dz = p.z - zmid, rho2 = dx*dx + dz*dz;
            if (rho2 < R * R) h = std::max(h, p.y + sqrtf(R * R - rho2));
        }
        return (h > -1e29f) ? h : bmx[1] + R;
    };
    // Phases: [0,S) settle (knife inactive, far) | [S,S+A) approach: knife active, starts 1.2R
    // ABOVE the settled surface at x0 and descends linearly to (surface - depth) | [S+A, S+A+W)
    // sweep along x at fixed (surface(x) - depth).
    const int S = n_settle, A = n_approach, W = n_frames, F_total = S + A + W;
    float h0 = 0.0f;

    printf("=== tool_bench: %s | tool=%s | N=%d C=%d E=%d fixed=%d | spacing=%.5f | R=%.4f (R/spacing=%.2f)\n",
           cfg.name.c_str(), tool.c_str(), N, C, E, n_fixed, avg_spacing, R, R / avg_spacing);
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
        else if (phase == 0) { float t = (float)(f - S + 1) / A; kx = x0; ky = (h0 + 0.2f * R) + t * (-1.2f * R - depth); }
        else                 { float t = (float)(f - S - A) / std::max(W - 1, 1); kx = x0 + t * (x1 - x0); ky = touch_at(kx) - R - depth; }
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
            if (phase >= 0 && !rem_now[i] && sphere_dist(pos[i], kp) < R * (1.0f + 1e-4f)) n_contact++;
        }
        static int rem_total_prev = 0; new_rem = rem_total - rem_total_prev; rem_total_prev = rem_total;
        if (phase >= 0) n_contact += new_rem;   // removed this frame were on the knife surface (verified by depth bins)
        if (n_contact > 0 && first_contact < 0) first_contact = f;
        if (timing && new_rem > 0 && first_removal < 0) first_removal = f;
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
           first_contact, first_removal, max_step_y, max_step_y / R, max_step_xy, max_step_xy / R);
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
