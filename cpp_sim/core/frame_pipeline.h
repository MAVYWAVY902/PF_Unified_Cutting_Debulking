#pragma once
// Step 0.5d — the interactive viewer's per-frame simulation pipeline, shared verbatim with
// tool_bench --replay so headless regression measures exactly what the app does:
//   GPU substeps → download positions/removed/damage → sync CPU particles →
//   orphan cleanup (no fully-alive incident tet, 3 cascade rounds) →
//   every 5th frame: BFS connected components (drop small fragments, split clusters across cuts).
// cpu_bfs_split() and the advance_frame() body were MOVED here from main_gpu_viewer.cpp without
// logic changes (only variable plumbing). Viewer-only work (input, knife sync, recording, UI
// counters, stats, rendering) stays in the viewer.

#include "solver/shape_matching_gpu.h"
#include "solver/shape_matching_cpu.h"
#include "core/tet_mesh.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// CPU BFS 分裂：下载边状态 → 找连通分量 → 分裂簇 → 移除碎片
// 返回 true 如果有任何分裂或移除发生
inline bool cpu_bfs_split(
    ShapeMatchingGPUSolver& gpu_solver,
    ShapeMatchingSolver& cpu_solver,
    std::vector<int>& edge_a,
    std::vector<int>& edge_b,
    std::vector<float>& edge_pf_rest,
    std::vector<float3>& cpu_positions,
    std::vector<int>& cpu_removed,
    std::vector<int>& tet_broken,        // OUT: per-tet flag, ≥1 broken edge → 1 (monotonic)
    const TetMesh& tet_mesh,
    int min_fragment_size = 5,
    bool enable_fragment_drop = true,    // false in cut_only mode
    int* out_components = nullptr,       // OUT (optional): connected components of alive particles
    int* out_broken_edges = nullptr,     // OUT (optional): edges with broken flag set
    bool recluster = true,               // false with R1 local_split: clusters are split per
                                         //   sub-cluster on the GPU; no global re-clustering
    std::unordered_set<int64_t>* out_broken_set = nullptr)   // OUT (optional): broken edges (render.split_tets)
{
    int N = cpu_solver.particles.size();
    int E = (int)edge_a.size();
    bool any_change = false;

    // 1. 下载边状态
    std::vector<int> edge_broken;
    std::vector<float> edge_damage;
    gpu_solver.download_edge_broken(edge_broken);
    gpu_solver.download_edge_damage(edge_damage);

    // 2. 构建 broken 边集合
    auto edge_key = [](int a, int b) -> int64_t {
        // Injective for all non-negative 32-bit indices (was a*100000+b: collided once N > 100000).
        if (a > b) std::swap(a, b);
        return (int64_t)(((uint64_t)(uint32_t)a << 32) | (uint32_t)b);
    };
    std::unordered_set<int64_t> broken_edge_set;
    for (int ei = 0; ei < E; ei++) {
        if (edge_broken[ei]) {
            broken_edge_set.insert(edge_key(edge_a[ei], edge_b[ei]));
        }
    }

    // 3. BFS用vertex_neighbors作为图，跳过broken edges和removed粒子
    //    旧BFS的逻辑 — 在视觉fragmentation方面更准确
    std::vector<int> component_id(N, -1);
    std::vector<std::vector<int>> components;

    for (int seed = 0; seed < N; seed++) {
        if (cpu_removed[seed] || component_id[seed] != -1) continue;

        int comp_idx = (int)components.size();
        components.push_back({});
        auto& comp = components.back();

        std::queue<int> q;
        q.push(seed);
        component_id[seed] = comp_idx;

        while (!q.empty()) {
            int v = q.front(); q.pop();
            comp.push_back(v);

            if (v < (int)tet_mesh.vertex_neighbors.size()) {
                for (int nb : tet_mesh.vertex_neighbors[v]) {
                    if (cpu_removed[nb]) continue;
                    if (component_id[nb] != -1) continue;
                    if (broken_edge_set.count(edge_key(v, nb))) continue;
                    component_id[nb] = comp_idx;
                    q.push(nb);
                }
            }
        }
    }

    if (out_components) *out_components = (int)components.size();
    if (out_broken_edges) *out_broken_edges = (int)std::count(edge_broken.begin(), edge_broken.end(), 1);

    // 找最大分量
    int largest_idx = 0;
    for (int i = 1; i < (int)components.size(); i++) {
        if (components[i].size() > components[largest_idx].size())
            largest_idx = i;
    }

    // 对非最大分量：
    //   size <  min_fragment_size → 标记 removed（clinical = aspirated 小碎末被吸走）
    //   size >= min_fragment_size → 保留，让物理/重力自然处理（掉落、继续和刀交互）
    // 这修正了原版"除最大以外全删"的行为，之前大块分离体会突然消失。
    // ★ cut_only 模式 (enable_fragment_drop=false): 跳过整个 drop_small 循环,
    //   小碎片也保留。两半通过 Phase 2 (cluster split) 实现 kinematic 独立。
    int dropped_small = 0, kept_big = 0;
    if (enable_fragment_drop) {
        for (int i = 0; i < (int)components.size(); i++) {
            if (i == largest_idx) continue;
            if ((int)components[i].size() < min_fragment_size) {
                for (int pid : components[i]) cpu_removed[pid] = 1;
                dropped_small++;
                any_change = true;
            } else {
                kept_big++;
            }
        }
    } else {
        // cut_only: 数一下非主分量(用于诊断打印),但不删
        for (int i = 0; i < (int)components.size(); i++) {
            if (i != largest_idx) kept_big++;
        }
    }

    if (out_broken_set) *out_broken_set = broken_edge_set;

    // ★ Scheme B: 计算 tet_broken[T] — 跟踪哪些 tet 含 broken 边,
    //   让 rebuild_surface_faces 跳过这些 tet → 切口处出现可见开口。
    //   Monotonic: 一旦 tet 被标记 broken,永远 broken。
    {
        int newly_broken = 0;
        for (int ti = 0; ti < (int)tet_mesh.tets.size(); ti++) {
            if (tet_broken[ti]) continue;  // 已标记,保持
            const auto& t = tet_mesh.tets[ti];
            bool has_broken_edge = false;
            for (int i = 0; i < 4 && !has_broken_edge; i++) {
                for (int j = i + 1; j < 4; j++) {
                    if (broken_edge_set.count(edge_key(t[i], t[j]))) {
                        has_broken_edge = true;
                        break;
                    }
                }
            }
            if (has_broken_edge) {
                tet_broken[ti] = 1;
                newly_broken++;
            }
        }
        if (newly_broken > 0) {
            int total_broken = (int)std::count(tet_broken.begin(), tet_broken.end(), 1);
            printf("[CUT-TET] newly_broken=%d total_broken=%d / %d tets\n",
                   newly_broken, total_broken, (int)tet_mesh.tets.size());
        }
    }

    // 第 1 部分 — 如果移除了小碎片：同步 CPU particles 状态 + 上传到 GPU
    if (any_change) {
        for (int i = 0; i < N; i++) {
            cpu_solver.particles[i].removed = (cpu_removed[i] != 0);
        }
        gpu_solver.upload_removed(cpu_removed);

        int total_removed = (int)std::count(cpu_removed.begin(), cpu_removed.end(), 1);
        printf("[SPLIT] components=%d dropped_small=%d kept_big=%d total_removed=%d\n",
               (int)components.size(), dropped_small, kept_big, total_removed);
    }

    // 第 2 部分 — 分裂横跨切口的 cluster（关键：消除"隐形弹簧"）
    // 不做这步的话：ring-2 BFS 建的 cluster 会同时含有切口两侧的 alive 粒子，
    // cluster_solve_kernel 对这类 cluster 还是把两侧一起 shape-match，效果
    // 等价于切口两侧每 cluster 一根弹簧把碎片拉回主体。
    // 只有 components > 1 才需要做（单连通图没有跨切口的 cluster）。
    bool clusters_changed = false;
    if (recluster && (int)components.size() > 1) {
        std::vector<Cluster> new_clusters;
        new_clusters.reserve(cpu_solver.clusters.size() + 32);
        int split_count = 0;

        for (const auto& c : cpu_solver.clusters) {
            // 按 component_id 分组 cluster 的存活粒子
            std::unordered_map<int, std::vector<int>> by_component;
            for (int pid : c.particle_ids) {
                if (cpu_removed[pid]) continue;
                int cid = component_id[pid];
                if (cid < 0) continue;  // defensive — BFS 已对所有存活粒子赋 id
                by_component[cid].push_back(pid);
            }

            if (by_component.empty()) {
                // 簇全死 → 丢弃
                clusters_changed = true;
                continue;
            }

            if ((int)by_component.size() == 1) {
                // 完全在单一 component 内 → 保持原状（包括原 rest_cm / rest_offsets）
                new_clusters.push_back(c);
            } else {
                // 跨 2+ components → split per component.
                //   sub-part >= MIN_NEW_CLUSTER_SIZE: becomes its own cluster, re-rested at the
                //     current positions (碎片"接受"它分离时刻的形状为新平衡态).
                //   sub-part <  MIN_NEW_CLUSTER_SIZE: dropped from THIS cluster only (a 2-3 particle
                //     cluster has a degenerate SVD). Its particles keep their other memberships;
                //     particles left with none are handled after the loop.
                // (Was all-or-nothing: any small sub-part kept the WHOLE cluster across the cut. Cut
                //  surfaces leave many 1-3 particle shards, so nearly every straddling cluster was
                //  kept and the two sides stayed shape-matched together — e.g. a fully sliced cube
                //  slab stayed coupled to the body through 2406 clusters.)
                const int MIN_NEW_CLUSTER_SIZE = 4;
                clusters_changed = true;
                split_count++;
                for (auto& [cid, ids] : by_component) {
                    if ((int)ids.size() < MIN_NEW_CLUSTER_SIZE) continue;
                    Cluster nc;
                    nc.particle_ids = ids;
                    std::sort(nc.particle_ids.begin(), nc.particle_ids.end());
                    nc.init(cpu_solver.particles);
                    new_clusters.push_back(std::move(nc));
                }
            }
        }

        // Particles that lost every membership but belong to a component of >= 2 particles get one
        // cluster: the members of their former clusters that are in the same component (if that
        // reaches MIN_NEW_CLUSTER_SIZE). Shards smaller than that stay free (no shape matching).
        if (clusters_changed) {
            const int MIN_NEW_CLUSTER_SIZE = 4;
            std::vector<char> member(N, 0);
            for (const auto& c : new_clusters) for (int pid : c.particle_ids) member[pid] = 1;
            std::vector<std::vector<int>> old_of(N);
            for (int ci = 0; ci < (int)cpu_solver.clusters.size(); ci++)
                for (int pid : cpu_solver.clusters[ci].particle_ids) old_of[pid].push_back(ci);
            int rescued = 0;
            for (int pid = 0; pid < N; pid++) {
                if (cpu_removed[pid] || member[pid] || old_of[pid].empty()) continue;
                std::vector<int> ids;
                for (int ci : old_of[pid])
                    for (int q : cpu_solver.clusters[ci].particle_ids)
                        if (!cpu_removed[q] && component_id[q] == component_id[pid]) ids.push_back(q);
                std::sort(ids.begin(), ids.end());
                ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
                if ((int)ids.size() < MIN_NEW_CLUSTER_SIZE) continue;
                Cluster nc;
                nc.particle_ids = ids;
                nc.init(cpu_solver.particles);
                for (int q : ids) member[q] = 1;
                new_clusters.push_back(std::move(nc));
                rescued++;
            }
            if (rescued > 0) printf("[CLUSTER-SPLIT] %d clusters rebuilt for particles that lost every membership\n", rescued);
        }

        if (clusters_changed) {
            cpu_solver.clusters = std::move(new_clusters);

            // 重建 CSR 扁平数组给 GPU
            int C = (int)cpu_solver.clusters.size();
            std::vector<int> cluster_starts(C), cluster_sizes(C);
            std::vector<int> flat_particle_ids;
            std::vector<float3> flat_rest_offsets;
            std::vector<float3> flat_rest_cm(C);
            flat_particle_ids.reserve(C * 16);   // rings=2 典型 cluster ~15 粒子
            flat_rest_offsets.reserve(C * 16);

            int offset = 0;
            for (int ci = 0; ci < C; ci++) {
                const auto& cl = cpu_solver.clusters[ci];
                cluster_starts[ci] = offset;
                cluster_sizes[ci] = (int)cl.particle_ids.size();
                for (int j = 0; j < cluster_sizes[ci]; j++) {
                    flat_particle_ids.push_back(cl.particle_ids[j]);
                    const Vec3& ro = cl.rest_offsets[j];
                    flat_rest_offsets.push_back(make_float3(
                        (float)ro.x(), (float)ro.y(), (float)ro.z()));
                }
                const Vec3& rc = cl.rest_cm;
                flat_rest_cm[ci] = make_float3((float)rc.x(), (float)rc.y(), (float)rc.z());
                offset += cluster_sizes[ci];
            }

            gpu_solver.reupload_clusters(cluster_starts, cluster_sizes,
                                         flat_particle_ids, flat_rest_offsets,
                                         flat_rest_cm);
            printf("[CLUSTER-SPLIT] split %d clusters across cut → total %d clusters\n",
                   split_count, C);
        }
    }

    return any_change || clusters_changed;
}

namespace simrec_frame {

// R4 — tool pose interpolation across the substeps of a frame. Rotations are row-major 3x3
// (local -> world), interpolated as unit quaternions (slerp); positions linearly.
struct Quatd { double w, x, y, z; };
inline Quatd quat_from_rot(const float R[9]) {           // Shepperd's method
    double m00 = R[0], m01 = R[1], m02 = R[2], m10 = R[3], m11 = R[4], m12 = R[5], m20 = R[6], m21 = R[7], m22 = R[8];
    double tr = m00 + m11 + m22; Quatd q;
    if (tr > 0) { double s = std::sqrt(tr + 1.0) * 2; q = {0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s}; }
    else if (m00 > m11 && m00 > m22) { double s = std::sqrt(1.0 + m00 - m11 - m22) * 2; q = {(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s}; }
    else if (m11 > m22) { double s = std::sqrt(1.0 + m11 - m00 - m22) * 2; q = {(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s}; }
    else { double s = std::sqrt(1.0 + m22 - m00 - m11) * 2; q = {(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s}; }
    double n = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    return {q.w / n, q.x / n, q.y / n, q.z / n};
}
inline void rot_from_quat(const Quatd& q, float R[9]) {
    double w = q.w, x = q.x, y = q.y, z = q.z;
    R[0] = (float)(1 - 2*(y*y + z*z)); R[1] = (float)(2*(x*y - w*z));     R[2] = (float)(2*(x*z + w*y));
    R[3] = (float)(2*(x*y + w*z));     R[4] = (float)(1 - 2*(x*x + z*z)); R[5] = (float)(2*(y*z - w*x));
    R[6] = (float)(2*(x*z - w*y));     R[7] = (float)(2*(y*z + w*x));     R[8] = (float)(1 - 2*(x*x + y*y));
}
inline Quatd slerp(Quatd a, Quatd b, double t) {
    double d = a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
    if (d < 0) { b = {-b.w, -b.x, -b.y, -b.z}; d = -d; }   // shortest arc
    if (d > 0.9995) {                                      // nearly equal: normalized lerp
        Quatd q = {a.w + t*(b.w - a.w), a.x + t*(b.x - a.x), a.y + t*(b.y - a.y), a.z + t*(b.z - a.z)};
        double n = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
        return {q.w / n, q.x / n, q.y / n, q.z / n};
    }
    double th = std::acos(d), s = std::sin(th), ka = std::sin((1 - t) * th) / s, kb = std::sin(t * th) / s;
    return {ka*a.w + kb*b.w, ka*a.x + kb*b.x, ka*a.y + kb*b.y, ka*a.z + kb*b.z};
}

struct FramePipeline {
    ShapeMatchingGPUSolver& gpu;
    ShapeMatchingSolver& cpu;
    TetMesh& tet_mesh;
    std::vector<int>& edge_a;
    std::vector<int>& edge_b;
    std::vector<float>& edge_pf_rest;
    std::vector<float3>& positions;   // downloaded each frame
    std::vector<int>& removed;        // downloaded each frame, then edited by orphan/BFS
    std::vector<float>& damage;
    std::vector<int>& tet_broken;
    bool use_tet;
    std::string cutting_mode;
    int N;
    bool local_split = false;         // R1: GPU sub-clusters; skip cpu_bfs_split re-clustering
    // R4: interpolate the tool pose across the substeps (only while the tool stays active, so
    // switching it on never sweeps it in from the parking position). State of the previous frame:
    bool tool_interp = false;
    bool prev_valid = false;
    float3 prev_pos = {0, 0, 0};
    float prev_rot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    // B — tool proxy (virtual coupling). The input (mouse / Haply / replay) sets a TARGET pose; the
    // simulated tool is a proxy that moves toward it by at most proxy_max_step per substep, measured
    // as the largest displacement of any tool point (|dp| + angle * tool_reach). Contact resolves a
    // particle only if the tool moved less than a particle radius within the substep (CFL): an input
    // jump of tens of mm in one substep (measured: median 8.4, max 260 mm per frame from the mouse)
    // swept the tool through the tissue, which contact cannot push aside, so everything swept was cut.
    // While the tool is off the proxy follows the target exactly (no interaction to protect).
    bool tool_proxy = false;
    double proxy_max_step = 0.0;      // [m] per substep: particle radius = 0.5 * mean particle spacing
    double tool_reach = 0.0;          // [m] largest distance of a tool point from the tool origin
    bool proxy_valid = false;
    float3 proxy_pos = {0, 0, 0};
    float proxy_rot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    // simulation.damping_mode body: free bodies (connected by intact edges, no fixed particle) are
    // relabelled on the GPU whenever the number of broken edges or removed particles changed
    bool body_damping = false;
    int body_broken = -1, body_removed = -1;
};

// damping_mode body: label the bodies (components of the alive particles over the intact edges, the
// same connectivity the shape matching sees with R1 local_split) and upload them. A body containing a
// fixed particle has no free rigid motion and gets -1 (plain damping), like removed particles.
inline void refresh_body_labels(FramePipeline& P) {
    std::vector<int> eb;
    P.gpu.download_edge_broken(eb);
    int nb = 0, nr = 0;
    for (int v : eb) nb += (v != 0);
    for (int i = 0; i < P.N; i++) nr += (P.removed[i] != 0);
    if (nb == P.body_broken && nr == P.body_removed) return;
    P.body_broken = nb; P.body_removed = nr;
    std::vector<int> parent(P.N);
    for (int i = 0; i < P.N; i++) parent[i] = i;
    auto find = [&](int x) { while (parent[x] != x) x = parent[x] = parent[parent[x]]; return x; };
    for (int e = 0; e < (int)eb.size(); e++) {
        const int a = P.edge_a[e], b = P.edge_b[e];
        if (eb[e] || P.removed[a] || P.removed[b]) continue;
        const int ra = find(a), rb = find(b);
        if (ra != rb) parent[ra] = rb;
    }
    std::vector<char> anchored(P.N, 0);
    for (int i = 0; i < P.N; i++)
        if (!P.removed[i] && P.cpu.particles[i].fixed) anchored[find(i)] = 1;
    std::vector<int> id(P.N, -1), label(P.N, -1);
    int n_bodies = 0;
    for (int i = 0; i < P.N; i++) {
        if (P.removed[i]) continue;
        const int r = find(i);
        if (anchored[r]) continue;
        if (id[r] < 0) id[r] = n_bodies++;
        label[i] = id[r];
    }
    P.gpu.upload_body_labels(label, n_bodies);
}

struct FrameResult {
    int active = 0, removed = 0;
    bool bfs_ran = false, clusters_changed = false;
    int tet_broken_count = 0;
    double gpu_ms = 0.0;              // GPU substeps only (incl. sync), excludes CPU stages
    // Per-source removal attribution for THIS frame (all three sum to the frame's new removals):
    int gpu_new = 0;                  // set by the GPU kernels (aspiration) during the substeps
    int orphan_new = 0;               // orphan cleanup (no fully-alive incident tet)
    int bfs_new = 0;                  // BFS small-fragment drop (only on frames where BFS runs)
    double orphan_ms = 0.0;           // CPU time of the orphan cleanup loop (0 when skipped)
    double bfs_ms = 0.0;              // CPU time of cpu_bfs_split (0 when not run this frame)
    // Topology (Step 1, needed when debulking is off and removal stays 0). Only measured on
    // frames where BFS runs; -1 on other frames (the stats writer carries the last value).
    int components = -1, broken_edges = -1;
    double proxy_lag = 0.0;           // [m] B: distance from the proxy to the target after the frame
};

// knife_active: same flag the viewer previously read from viewer.knife_active (gates BFS).
inline FrameResult advance_frame(FramePipeline& P, int substeps, int frame, bool knife_active) {
    FrameResult res;
    const int N = P.N;
        // removed count entering this frame (state after last frame's orphan/BFS upload)
        int rem_before = 0;
        for (int i = 0; i < N; i++) rem_before += (P.removed[i] != 0);
        if (P.body_damping) refresh_body_labels(P);   // topology at the start of this frame
        // GPU 仿真
        auto t0 = std::chrono::high_resolution_clock::now();
        // R4: the caller has set this frame's TARGET pose; substep k uses the pose (k+1)/substeps of
        // the way from the previous frame's pose (last substep = target).
        const float3 tgt_pos = P.gpu.config.knife_pos;
        float tgt_rot[9]; for (int k = 0; k < 9; k++) tgt_rot[k] = P.gpu.config.knife_rot[k];
        const bool proxy = P.tool_proxy && P.proxy_max_step > 0.0;
        const bool interp = !proxy && P.tool_interp && knife_active && P.prev_valid;
        // The proxy chases the target interpolated over the substeps (from the previous frame's
        // target), not the frame's target itself: chasing that, a proxy faster than the input covered
        // the whole frame's motion in the first substep and stood still for the rest, and the tissue,
        // catching up, swung back and forth across the blade edge in the tool frame (measured: the
        // particles under the edge crossed it 180 times in the side push, 67 with smooth motion).
        const bool proxy_interp = proxy && knife_active && P.prev_valid;
        Quatd qa = {1, 0, 0, 0}, qb = {1, 0, 0, 0};
        if (interp || proxy_interp) { qa = quat_from_rot(P.prev_rot); qb = quat_from_rot(tgt_rot); }
        if (proxy && (!knife_active || !P.proxy_valid)) {   // tool off: the proxy sits on the target
            P.proxy_pos = tgt_pos; for (int k = 0; k < 9; k++) P.proxy_rot[k] = tgt_rot[k];
            P.proxy_valid = true;
        }
        // B: the step limit only matters where the tool can touch tissue. Region = AABB of the alive
        // particles (last downloaded positions) grown by the tool reach plus one step; the part of the
        // proxy's path outside it is travelled at once (e.g. the tool brought in from its parking pose).
        double blo[3] = {1e30, 1e30, 1e30}, bhi[3] = {-1e30, -1e30, -1e30};
        if (proxy && knife_active) {
            for (int i = 0; i < N; i++) {
                if (P.removed[i]) continue;
                const float3& p = P.positions[i];
                blo[0] = std::min(blo[0], (double)p.x); blo[1] = std::min(blo[1], (double)p.y); blo[2] = std::min(blo[2], (double)p.z);
                bhi[0] = std::max(bhi[0], (double)p.x); bhi[1] = std::max(bhi[1], (double)p.y); bhi[2] = std::max(bhi[2], (double)p.z);
            }
            double g = P.tool_reach + P.proxy_max_step;
            for (int k = 0; k < 3; k++) { blo[k] -= g; bhi[k] += g; }
        }
        for (int sub = 0; sub < substeps; sub++) {
            if (proxy && knife_active) {
                // this substep's target: (sub+1)/substeps of the way from the previous frame's target
                float3 sub_pos = tgt_pos; float sub_rot[9]; for (int k = 0; k < 9; k++) sub_rot[k] = tgt_rot[k];
                if (proxy_interp && sub + 1 < substeps) {
                    double t = (double)(sub + 1) / substeps;
                    sub_pos = make_float3((float)(P.prev_pos.x + t * (tgt_pos.x - P.prev_pos.x)),
                                          (float)(P.prev_pos.y + t * (tgt_pos.y - P.prev_pos.y)),
                                          (float)(P.prev_pos.z + t * (tgt_pos.z - P.prev_pos.z)));
                    rot_from_quat(slerp(qa, qb, t), sub_rot);
                }
                const Quatd q_sub = quat_from_rot(sub_rot);
                double dx = sub_pos.x - P.proxy_pos.x, dy = sub_pos.y - P.proxy_pos.y, dz = sub_pos.z - P.proxy_pos.z;
                Quatd qp = quat_from_rot(P.proxy_rot);
                // free part of the path: segment proxy -> target against the region (slab test)
                double o[3] = {P.proxy_pos.x, P.proxy_pos.y, P.proxy_pos.z}, dv[3] = {dx, dy, dz};
                double t_in = 0.0, t_out = 1.0;
                for (int k = 0; k < 3 && t_in <= t_out; k++) {
                    if (std::fabs(dv[k]) < 1e-15) { if (o[k] < blo[k] || o[k] > bhi[k]) t_in = 2.0; continue; }
                    double a = (blo[k] - o[k]) / dv[k], b = (bhi[k] - o[k]) / dv[k];
                    if (a > b) std::swap(a, b);
                    t_in = std::max(t_in, a); t_out = std::min(t_out, b);
                }
                double t_free = (t_in > t_out) ? 1.0 : t_in;    // never enters the region: go all the way
                if (t_free > 0.0) {
                    if (t_free >= 1.0) { P.proxy_pos = sub_pos; for (int k = 0; k < 9; k++) P.proxy_rot[k] = sub_rot[k]; }
                    else {
                        P.proxy_pos = make_float3((float)(o[0] + t_free * dx), (float)(o[1] + t_free * dy), (float)(o[2] + t_free * dz));
                        rot_from_quat(slerp(qp, q_sub, t_free), P.proxy_rot);
                    }
                    dx = sub_pos.x - P.proxy_pos.x; dy = sub_pos.y - P.proxy_pos.y; dz = sub_pos.z - P.proxy_pos.z;
                    qp = quat_from_rot(P.proxy_rot);
                }
                double d = std::fabs(qp.w*q_sub.w + qp.x*q_sub.x + qp.y*q_sub.y + qp.z*q_sub.z);
                double ang = 2.0 * std::acos(std::min(1.0, d));
                double sweep = std::sqrt(dx*dx + dy*dy + dz*dz) + ang * P.tool_reach;   // bound on any tool point
                double f = sweep > P.proxy_max_step ? P.proxy_max_step / sweep : 1.0;
                P.proxy_pos = make_float3((float)(P.proxy_pos.x + f * dx), (float)(P.proxy_pos.y + f * dy),
                                          (float)(P.proxy_pos.z + f * dz));
                if (f >= 1.0) for (int k = 0; k < 9; k++) P.proxy_rot[k] = sub_rot[k];
                else rot_from_quat(slerp(qp, q_sub, f), P.proxy_rot);
                P.gpu.config.knife_pos = P.proxy_pos;
                for (int k = 0; k < 9; k++) P.gpu.config.knife_rot[k] = P.proxy_rot[k];
            } else if (interp) {
                double t = (double)(sub + 1) / substeps;
                P.gpu.config.knife_pos = make_float3((float)(P.prev_pos.x + t * (tgt_pos.x - P.prev_pos.x)),
                                                     (float)(P.prev_pos.y + t * (tgt_pos.y - P.prev_pos.y)),
                                                     (float)(P.prev_pos.z + t * (tgt_pos.z - P.prev_pos.z)));
                if (sub + 1 == substeps) { for (int k = 0; k < 9; k++) P.gpu.config.knife_rot[k] = tgt_rot[k]; P.gpu.config.knife_pos = tgt_pos; }
                else rot_from_quat(slerp(qa, qb, t), P.gpu.config.knife_rot);
            }
            P.gpu.step();
        }
        if (proxy) {
            double lx = tgt_pos.x - P.proxy_pos.x, ly = tgt_pos.y - P.proxy_pos.y, lz = tgt_pos.z - P.proxy_pos.z;
            res.proxy_lag = std::sqrt(lx*lx + ly*ly + lz*lz);
        }
        P.prev_valid = knife_active;
        P.prev_pos = tgt_pos;
        for (int k = 0; k < 9; k++) P.prev_rot[k] = tgt_rot[k];
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        // 下载结果到 CPU
        P.gpu.download_positions(P.positions);
        P.gpu.download_removed(P.removed);
        P.gpu.download_damage(P.damage);

        // 同步到 P.cpu.particles（viewer 读取用）
        int active = 0, rem = 0;
        for (int i = 0; i < N; i++) {
            P.cpu.particles[i].pos = Vec3(
                P.positions[i].x, P.positions[i].y, P.positions[i].z);
            P.cpu.particles[i].removed = (P.removed[i] != 0);
            if (P.removed[i]) rem++; else active++;
        }
        res.gpu_new = rem - rem_before;

        // Orphan清理：判据 = "没有 4-顶点全活的 incident tet"
        //
        // 判据理由（Fix I，2026-04-22 reverted from vertex-neighbor lenient version）：
        //   - 渲染（rebuild_surface_faces）使用"tet 全活"作为 surface face 可见判据。
        //   - 若 orphan 用较宽松的"有活边邻居"判据，会留下"桥粒子"——
        //     物理上活着但没有任何 surface face 呈现它 → 用户关闭粒子可视化时看不见，
        //     但这些桥把 BFS 的 connected component 偷偷连在一起 →
        //     真正的 fragment 不被 BFS 识别 → cluster splitting 不触发 → 隐形弹簧问题
        //     → 视觉上"碎片"漂浮但物理上仍被主体拽着。
        //   - 严格判据让 orphan / BFS / render 三者拓扑一致，桥粒子自然被清理，
        //     真正的 fragment 被 BFS 正确识别并 cluster-split 成独立刚体。
        //
        // Cascade 限制到 3 iter（原版 20）：严格判据 + MEAN aggregation (之前修好的)
        // 意味着每次 aspirate 规模已经较小、cascade 不会再失控；3 iter 够收敛正常切割。
        // ★ cut_only 模式: 跳过整个 orphan 清理。cut_only 不删粒子,rem 应该一直 = 0,
        //   理论上这块不会触发,但为了语义干净显式 gate。
        if (P.use_tet && rem > 0 && P.cutting_mode == "debulk") {
            auto t_orph0 = std::chrono::high_resolution_clock::now();
            int total_orphans = 0;
            int iter = 0;
            for (; iter < 3; iter++) {
                int round_orphans = 0;
                for (int i = 0; i < N; i++) {
                    if (P.removed[i]) continue;
                    const auto& my_tets = P.tet_mesh.vertex_to_tets[i];
                    if (my_tets.empty()) continue;

                    bool has_alive_tet = false;
                    for (int ti : my_tets) {
                        const auto& t = P.tet_mesh.tets[ti];
                        bool all_alive = true;
                        for (int j = 0; j < 4; j++) {
                            if (P.removed[t[j]]) { all_alive = false; break; }
                        }
                        if (all_alive) { has_alive_tet = true; break; }
                    }

                    if (!has_alive_tet) {
                        P.removed[i] = 1;
                        P.cpu.particles[i].removed = true;
                        round_orphans++;
                    }
                }
                total_orphans += round_orphans;
                if (round_orphans == 0) break;  // 收敛
            }
            if (total_orphans > 0) {
                if (frame % 60 == 0)
                    printf("[ORPHAN] removed %d in %d iters (cap=3)\n", total_orphans, iter+1);
                P.gpu.upload_removed(P.removed);
                active = 0; rem = 0;
                for (int i = 0; i < N; i++) {
                    if (P.removed[i]) rem++; else active++;
                }
            }
            res.orphan_new = total_orphans;
            res.orphan_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t_orph0).count();
        }

        // BFS 连通分量分析：小碎片 aspirated、大碎片保留（详见 cpu_bfs_split）
        // 频率从每30帧 → 每5帧：小碎片一生成（< 5 粒子）就被吸走，避免"延迟消失"跳变
        // ★ cut_only 模式: rem 永远 = 0,所以原本的 rem > 0 gate 永远不触发。
        //   改成 (rem > 0 || knife_active) — cut_only 下靠 knife_active 触发 BFS,
        //   发现新的 broken 边 → cluster split → 两半 kinematic 独立 + tet_broken 更新。
        bool enable_drop = (P.cutting_mode == "debulk");
        const bool bfs_frame = P.use_tet && (rem > 0 || knife_active) && frame % 5 == 0;
        // render.split_tets: the drawn pieces follow the GPU fracture every frame. The BFS below
        // refreshes the set only every 5th frame (and never after the knife is switched off in
        // cut_only mode), so freshly broken tets were drawn whole, straight through the blade.
        if (P.use_tet && P.tet_mesh.split_render && !bfs_frame) {
            std::vector<int> eb;
            P.gpu.download_edge_broken(eb);
            int nb = 0;
            for (int v : eb) nb += (v != 0);
            if (nb != (int)P.tet_mesh.broken_edges.size()) {
                P.tet_mesh.broken_edges.clear();
                for (int e = 0; e < (int)eb.size(); e++)
                    if (eb[e]) P.tet_mesh.broken_edges.insert(TetMesh::edge_key(P.edge_a[e], P.edge_b[e]));
            }
            res.broken_edges = nb;
        }
        if (bfs_frame) {
            const int rem_pre_bfs = rem;
            auto t_bfs0 = std::chrono::high_resolution_clock::now();
            bool changed = cpu_bfs_split(
                P.gpu, P.cpu,
                P.edge_a, P.edge_b, P.edge_pf_rest,
                P.positions, P.removed,
                P.tet_broken,
                P.tet_mesh, 5, enable_drop, &res.components, &res.broken_edges, !P.local_split,
                P.tet_mesh.split_render ? &P.tet_mesh.broken_edges : nullptr);
            res.bfs_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - t_bfs0).count();
            // Scheme B: 即使 `changed`=false (无 fragment 删除/cluster split),
            //   tet_broken 也可能增加。把 count 同步给 viewer 触发 surface 重建。
            res.bfs_ran = true;
            res.tet_broken_count = (int)std::count(P.tet_broken.begin(), P.tet_broken.end(), 1);

            // DIAG: 每秒打印一次 damage 状态,帮助 debug "no reaction"
            if (frame % 60 == 0) {
                std::vector<float> ed;
                P.gpu.download_edge_damage(ed);
                float max_ed = 0; int n_high = 0; int n_broken_dl = 0;
                for (auto v : ed) {
                    if (v > max_ed) max_ed = v;
                    if (v > 0.1f) n_high++;
                    if (v > 0.5f) n_broken_dl++;
                }
                std::vector<float> pd;
                P.gpu.download_damage(pd);
                float max_pd = 0; int n_p_high = 0;
                for (auto v : pd) {
                    if (v > max_pd) max_pd = v;
                    if (v > 0.1f) n_p_high++;
                }
                printf("[DIAG frame %d] knife_active=%d mode=%s | edge: max=%.3f n>0.1=%d n_broken=%d | particle: max=%.3f n>0.1=%d\n",
                       frame, (int)knife_active, P.cutting_mode.c_str(),
                       max_ed, n_high, n_broken_dl, max_pd, n_p_high);
            }
            if (changed) {
                active = 0; rem = 0;
                for (int i = 0; i < N; i++) {
                    if (P.removed[i]) rem++; else active++;
                }
                res.clusters_changed = true;
            }
            res.bfs_new = rem - rem_pre_bfs;
        }

        res.active = active;
        res.removed = rem;
        res.gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return res;
}

}  // namespace simrec_frame
