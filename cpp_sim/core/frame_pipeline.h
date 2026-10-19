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
    bool enable_fragment_drop = true)    // false in cut_only mode
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
    if ((int)components.size() > 1) {
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
                // 跨 2+ components.
                // ★ All-or-nothing 分裂: 仅当**所有**子组件 size >= MIN_NEW_CLUSTER_SIZE
                //   才分裂(避免退化的 2-3 粒子 cluster SVD 数值不稳定 → spikes)。
                //   否则保留原 cluster, soft damage 权重 (damage_shape_weight) 自然处理
                //   边界粒子 — 它们仍在 cluster 里, SVD 不退化, 健康粒子 shape-match 正常,
                //   damaged 粒子贡献被自动衰减。
                //   旧版 (min=2): 单粒子 sub-component 被静默丢弃 → 粒子失去 cluster 成员资格
                //                → 多个 cluster 同时丢失 → 粒子完全失去 shape-match → 飞出。
                const int MIN_NEW_CLUSTER_SIZE = 4;
                bool all_big_enough = true;
                for (auto& [cid, ids] : by_component) {
                    if ((int)ids.size() < MIN_NEW_CLUSTER_SIZE) {
                        all_big_enough = false;
                        break;
                    }
                }
                if (all_big_enough) {
                    clusters_changed = true;
                    split_count++;
                    for (auto& [cid, ids] : by_component) {
                        Cluster nc;
                        nc.particle_ids = ids;
                        std::sort(nc.particle_ids.begin(), nc.particle_ids.end());
                        // 用当前位置作 rest —— 碎片"接受"它分离时刻的形状为新平衡态。
                        nc.init(cpu_solver.particles);
                        new_clusters.push_back(std::move(nc));
                    }
                } else {
                    // 子组件太小 → 不分裂, 保留原 cluster intact.
                    // 跨切口粒子由 weighted cluster_solve (Müller 2005 with damage as mass)
                    // 自然处理: damaged 侧贡献 ~5% 权重, intact 侧主导 cluster shape。
                    new_clusters.push_back(c);
                }
            }
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
};

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
};

// knife_active: same flag the viewer previously read from viewer.knife_active (gates BFS).
inline FrameResult advance_frame(FramePipeline& P, int substeps, int frame, bool knife_active) {
    FrameResult res;
    const int N = P.N;
        // removed count entering this frame (state after last frame's orphan/BFS upload)
        int rem_before = 0;
        for (int i = 0; i < N; i++) rem_before += (P.removed[i] != 0);
        // GPU 仿真
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int sub = 0; sub < substeps; sub++) {
            P.gpu.step();
        }
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
        if (P.use_tet && (rem > 0 || knife_active) && frame % 5 == 0) {
            const int rem_pre_bfs = rem;
            auto t_bfs0 = std::chrono::high_resolution_clock::now();
            bool changed = cpu_bfs_split(
                P.gpu, P.cpu,
                P.edge_a, P.edge_b, P.edge_pf_rest,
                P.positions, P.removed,
                P.tet_broken,
                P.tet_mesh, 5, enable_drop);
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
