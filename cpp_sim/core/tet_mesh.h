#pragma once

#include "particles.h"
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>

struct TetMesh {
    // 顶点（和 Particles 共享，顶点 i 就是粒子 i）
    std::vector<Vec3> vertices;

    // 四面体（每个四面体 4 个顶点索引）
    std::vector<std::array<int, 4>> tets;

    // 表面三角面（从四面体自动提取）
    std::vector<std::array<int, 3>> surface_faces;

    // 表面法线（逐面）
    std::vector<Vec3> face_normals;

    // 每个顶点的邻居（共享四面体的顶点）
    std::vector<std::vector<int>> vertex_neighbors;

    // 每个顶点所属的tet索引列表（用于orphan检测）
    std::vector<std::vector<int>> vertex_to_tets;

    int num_vertices() const { return (int)vertices.size(); }
    int num_tets() const { return (int)tets.size(); }
    int num_surface_faces() const { return (int)surface_faces.size(); }

    // 从 .msh v4.1 文件加载
    bool load_msh(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return false;
        }

        std::string line;
        std::vector<int> node_tags;
        std::vector<Vec3> node_coords;
        std::vector<std::array<int, 3>> file_triangles;  // .msh 中的三角面
        std::vector<std::array<int, 4>> file_tets;       // .msh 中的四面体

        while (std::getline(file, line)) {
            if (line == "$Nodes") {
                _parse_nodes(file, node_tags, node_coords);
            } else if (line == "$Elements") {
                _parse_elements(file, node_tags, file_triangles, file_tets);
            }
        }

        if (node_coords.empty() || file_tets.empty()) {
            std::cerr << "文件中没有有效的节点或四面体" << std::endl;
            return false;
        }

        // 建立 gmsh tag → 连续索引 的映射
        std::unordered_map<int, int> tag_to_idx;
        for (int i = 0; i < (int)node_tags.size(); i++) {
            tag_to_idx[node_tags[i]] = i;
        }

        // 填充顶点
        vertices = node_coords;

        // 填充四面体（映射索引）
        tets.resize(file_tets.size());
        for (int i = 0; i < (int)file_tets.size(); i++) {
            for (int j = 0; j < 4; j++) {
                auto it = tag_to_idx.find(file_tets[i][j]);
                if (it == tag_to_idx.end()) {
                    std::cerr << "四面体引用了未知节点 tag: " << file_tets[i][j] << std::endl;
                    return false;
                }
                tets[i][j] = it->second;
            }
        }

        // 提取表面三角面
        _extract_surface_faces();

        // 计算法线
        _compute_normals();

        // 构建邻居关系
        _build_vertex_neighbors();

        // 构建顶点→tet索引
        vertex_to_tets.assign(vertices.size(), {});
        for (int ti = 0; ti < (int)tets.size(); ti++) {
            for (int j = 0; j < 4; j++) {
                vertex_to_tets[tets[ti][j]].push_back(ti);
            }
        }

        std::cout << "加载 tet mesh: " << num_vertices() << " 顶点, "
                  << num_tets() << " 四面体, "
                  << num_surface_faces() << " 表面三角面" << std::endl;

        return true;
    }

    // 根据 removed 粒子重建表面（debulk 后调用）
    // 原理：跳过含 removed 顶点的四面体，重新算哪些面是表面
    // ★ Scheme B (cut_only): 也跳过 tet_broken[ti]==1 的 tet → 切口处内部面
    //   变成新 surface,viewer 中切口可见。tet_broken 是 monotonic,从外部传入。
    //   nullptr 表示不启用(原 debulk 行为完全保持)。
    void rebuild_surface_faces(const Particles& particles,
                                const std::vector<int>* tet_broken = nullptr) {
        if (split_render) { _rebuild_surface_split(particles); return; }
        surface_faces.clear();

        // Key = the sorted vertex triple itself (was a*1e10+b*1e5+c: collided once N > 100000).
        using FaceKey = std::array<int, 3>;
        struct FaceKeyHash {
            size_t operator()(const FaceKey& k) const {
                uint64_t h = 1469598103934665603ULL;            // FNV-1a over the three indices
                for (int x : k) { h ^= (uint32_t)x; h *= 1099511628211ULL; }
                return (size_t)h;
            }
        };
        auto encode_face = [](int a, int b, int c) -> FaceKey {
            if (a > b) std::swap(a, b);
            if (b > c) std::swap(b, c);
            if (a > b) std::swap(a, b);
            return {a, b, c};
        };

        // 记录每个面的出现次数、原始顶点顺序、以及第四个顶点（用于法线方向）
        struct FaceInfo {
            std::array<int,3> verts;
            int fourth;  // 所在 tet 的第四个顶点
            int count;
        };
        std::unordered_map<FaceKey, FaceInfo, FaceKeyHash> face_map;
        face_map.reserve(tets.size() * 2);

        for (int ti = 0; ti < (int)tets.size(); ti++) {
            const auto& tet = tets[ti];
            bool has_removed = false;
            for (int j = 0; j < 4; j++) {
                if (particles[tet[j]].removed) { has_removed = true; break; }
            }
            if (has_removed) continue;
            // Scheme B: tet 含 broken 边 → 视为"已切开",跳过 → 内部面变 surface
            if (tet_broken && (*tet_broken)[ti]) continue;

            int v[4] = {tet[0], tet[1], tet[2], tet[3]};
            // 4 个面，每个面的第四个顶点是不在面上的那个
            int face_indices[4][3] = {
                {v[0], v[1], v[2]},  // fourth = v[3]
                {v[0], v[1], v[3]},  // fourth = v[2]
                {v[0], v[2], v[3]},  // fourth = v[1]
                {v[1], v[2], v[3]}   // fourth = v[0]
            };
            int fourth[4] = {v[3], v[2], v[1], v[0]};

            for (int fi = 0; fi < 4; fi++) {
                auto key = encode_face(face_indices[fi][0], face_indices[fi][1], face_indices[fi][2]);
                auto it = face_map.find(key);
                if (it == face_map.end()) {
                    face_map[key] = {{face_indices[fi][0], face_indices[fi][1], face_indices[fi][2]},
                                     fourth[fi], 1};
                } else {
                    it->second.count++;
                }
            }
        }

        surface_faces.reserve(face_map.size() / 3);
        for (auto& [key, info] : face_map) {
            if (info.count == 1) {
                // 内联法线方向修正（避免 O(faces × tets) 的 _fix_face_orientations）
                Vec3 v0 = vertices[info.verts[0]];
                Vec3 v1 = vertices[info.verts[1]];
                Vec3 v2 = vertices[info.verts[2]];
                Vec3 v3 = vertices[info.fourth];
                Vec3 normal = (v1 - v0).cross(v2 - v0);
                if (normal.dot(v3 - v0) > 0) {
                    std::swap(info.verts[1], info.verts[2]);
                }
                surface_faces.push_back(info.verts);
            }
        }

        // 直接计算法线（不再调用 _fix_face_orientations）
        face_normals.resize(surface_faces.size());
        for (int i = 0; i < (int)surface_faces.size(); i++) {
            const auto& f = surface_faces[i];
            Vec3 v0 = vertices[f[0]];
            Vec3 v1 = vertices[f[1]];
            Vec3 v2 = vertices[f[2]];
            face_normals[i] = (v1 - v0).cross(v2 - v0).normalized();
        }
    }

    // 用当前粒子位置更新法线
    void update_normals(const Particles& particles) {
        if (split_render) {
            update_render_positions(particles);
            for (int i = 0; i < num_surface_faces(); i++) {
                const auto& f = surface_faces[i];
                face_normals[i] = (render_pos[f[1]] - render_pos[f[0]]).cross(render_pos[f[2]] - render_pos[f[0]]).normalized();
            }
            return;
        }
        for (int i = 0; i < num_surface_faces(); i++) {
            const auto& f = surface_faces[i];
            Vec3 v0 = particles[f[0]].pos;
            Vec3 v1 = particles[f[1]].pos;
            Vec3 v2 = particles[f[2]].pos;
            face_normals[i] = (v1 - v0).cross(v2 - v0).normalized();
        }
    }

    // ---------------- split rendering (render.split_tets) ----------------
    // A tet whose broken edges separate its four vertices into several groups (connected through
    // its intact edges) is drawn once per group, each piece ending at the cut, instead of being
    // hidden as a whole (virtual node algorithm, Molino et al. 2004, restricted to rendering).
    // Piece of group C = convex hull of the vertices of C and the cut points on the edges from C to
    // the other vertices. A cut point belongs to one side of its edge: the point of edge (u, v) on
    // the side of u is x_u + s R_u (X_v - X_u), with R_u the rotation of the material around u
    // (best-fit rotation, rest -> current, of u and its neighbours through intact edges), so each
    // side's cut face moves with its own fragment and an opening appears where the fragments part.
    // s = 0.5 (the position of the cut along the edge is not recorded). Faces shared by two pieces
    // (same vertex set) are interior and dropped, as for whole tets; the two cut faces of a closed
    // cut are both kept (each side's own face) and coincide while the cut is closed.
    // Vertex ids: 0..N-1 particles, N + k cut point k (cut_verts[k]).
    bool split_render = false;
    std::unordered_set<int64_t> broken_edges;          // key edge_key(a, b), set by the frame pipeline
    struct CutVert { int u, v; };                       // cut point of edge (u, v) on the side of u
    std::vector<CutVert> cut_verts;
    std::vector<Vec3> render_pos;                       // positions of all render vertices (particles + cut points)
    float cut_s = 0.5f;

    static int64_t edge_key(int a, int b) {             // same key as the frame pipeline's broken-edge set
        if (a > b) std::swap(a, b);
        return (int64_t)(((uint64_t)(uint32_t)a << 32) | (uint32_t)b);
    }
    int num_render_vertices() const { return num_vertices() + (int)cut_verts.size(); }
    // owner particle of a render vertex (for removed checks)
    int render_owner(int id) const { return id < num_vertices() ? id : cut_verts[id - num_vertices()].u; }

    void update_render_positions(const Particles& particles) {
        const int N = num_vertices();
        render_pos.resize(N + cut_verts.size());
        for (int i = 0; i < N; i++) render_pos[i] = particles[i].pos;
        std::unordered_map<int, Eigen::Matrix3d> R;     // rotation per cut-point owner
        for (size_t k = 0; k < cut_verts.size(); k++) {
            int u = cut_verts[k].u, v = cut_verts[k].v;
            auto it = R.find(u);
            if (it == R.end()) it = R.emplace(u, _material_rotation(particles, u)).first;
            render_pos[N + k] = particles[u].pos + (double)cut_s * (it->second * (vertices[v] - vertices[u]));
        }
    }

    Eigen::Matrix3d _material_rotation(const Particles& particles, int u) const {
        std::vector<int> S = {u};
        for (int w : vertex_neighbors[u])
            if (!particles[w].removed && !broken_edges.count(edge_key(u, w))) S.push_back(w);
        if (S.size() < 3) return Eigen::Matrix3d::Identity();
        Vec3 c = Vec3::Zero(), C = Vec3::Zero();
        for (int i : S) { c += particles[i].pos; C += vertices[i]; }
        c /= (double)S.size(); C /= (double)S.size();
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        for (int i : S) A += (particles[i].pos - c) * (vertices[i] - C).transpose();
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
        if ((U * V.transpose()).determinant() < 0.0) U.col(2) *= -1.0;
        return U * V.transpose();
    }

    void _rebuild_surface_split(const Particles& particles) {
        const int N = num_vertices();
        cut_verts.clear();
        std::unordered_map<int64_t, int> cut_id;        // directed (u, v) -> render id
        auto cid = [&](int u, int v) {
            int64_t k = (int64_t)(((uint64_t)(uint32_t)u << 32) | (uint32_t)v);
            auto it = cut_id.find(k);
            if (it != cut_id.end()) return it->second;
            int id = N + (int)cut_verts.size();
            cut_verts.push_back({u, v});
            cut_id.emplace(k, id);
            return id;
        };
        auto rest = [&](int id) -> Vec3 {
            if (id < N) return vertices[id];
            const CutVert& c = cut_verts[id - N];
            return vertices[c.u] + (double)cut_s * (vertices[c.v] - vertices[c.u]);
        };
        using Key = std::array<int, 4>;                 // sorted polygon vertex ids, padded with -1
        struct KeyHash {
            size_t operator()(const Key& k) const {
                uint64_t h = 1469598103934665603ULL;
                for (int x : k) { h ^= (uint32_t)x; h *= 1099511628211ULL; }
                return (size_t)h;
            }
        };
        struct Poly { std::vector<int> v; int count; };
        std::unordered_map<Key, Poly, KeyHash> polys;
        polys.reserve(tets.size() * 2);
        auto add_poly = [&](std::vector<int> v, const Vec3& away) {   // orient: normal points away from `away`
            if (v.size() < 3) return;
            Vec3 n = Vec3::Zero(), c = Vec3::Zero();
            for (size_t i = 0; i < v.size(); i++) {      // Newell normal
                Vec3 a = rest(v[i]), b = rest(v[(i + 1) % v.size()]);
                n += a.cross(b); c += a;
            }
            c /= (double)v.size();
            if (n.dot(away - c) > 0.0) std::reverse(v.begin(), v.end());
            Key k = {-1, -1, -1, -1};
            std::vector<int> sv = v; std::sort(sv.begin(), sv.end());
            for (size_t i = 0; i < sv.size() && i < 4; i++) k[i] = sv[i];
            auto it = polys.find(k);
            if (it == polys.end()) polys.emplace(k, Poly{v, 1}); else it->second.count++;
        };
        static const int FACE[4][4] = {{0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 3, 1}, {1, 2, 3, 0}};   // 3 face vertices + opposite
        for (const auto& t : tets) {
            bool rem = false;
            for (int j = 0; j < 4; j++) if (particles[t[j]].removed) { rem = true; break; }
            if (rem) continue;
            int g[4] = {0, 1, 2, 3};                     // groups: union over intact tet edges
            auto find = [&](int x) { while (g[x] != x) x = g[x] = g[g[x]]; return x; };
            for (int i = 0; i < 4; i++)
                for (int j = i + 1; j < 4; j++)
                    if (!broken_edges.count(edge_key(t[i], t[j]))) g[find(i)] = find(j);
            for (int r = 0; r < 4; r++) {
                if (find(r) != r) continue;              // one piece per group root
                bool in[4]; Vec3 cen = Vec3::Zero(); int nc = 0;
                for (int j = 0; j < 4; j++) { in[j] = find(j) == r; if (in[j]) { cen += vertices[t[j]]; nc++; } }
                cen /= (double)nc;
                std::vector<std::pair<int, int>> crack;  // cut-face segments, one per mixed tet face
                for (int f = 0; f < 4; f++) {
                    std::vector<int> v; int cut[2], ncut = 0;
                    for (int k = 0; k < 3; k++) {
                        int p = FACE[f][k], q = FACE[f][(k + 1) % 3];
                        if (in[p]) v.push_back(t[p]);
                        if (in[p] != in[q]) {
                            int id = in[p] ? cid(t[p], t[q]) : cid(t[q], t[p]);
                            v.push_back(id); cut[ncut++] = id;
                        }
                    }
                    add_poly(v, vertices[t[FACE[f][3]]]);
                    if (ncut == 2) crack.push_back({cut[0], cut[1]});
                }
                if (crack.size() >= 3) {                 // chain the segments into the cut-face polygon
                    std::vector<int> loop = {crack[0].first, crack[0].second};
                    std::vector<char> used(crack.size(), 0); used[0] = 1;
                    for (size_t n = 1; n < crack.size(); n++)
                        for (size_t i = 0; i < crack.size(); i++) {
                            if (used[i]) continue;
                            if (crack[i].first == loop.back())  { loop.push_back(crack[i].second); used[i] = 1; break; }
                            if (crack[i].second == loop.back()) { loop.push_back(crack[i].first);  used[i] = 1; break; }
                        }
                    if (loop.size() > 1 && loop.front() == loop.back()) loop.pop_back();
                    add_poly(loop, cen);
                }
            }
        }
        surface_faces.clear();
        for (auto& [k, p] : polys) {
            if (p.count != 1) continue;
            for (size_t i = 1; i + 1 < p.v.size(); i++) surface_faces.push_back({p.v[0], p.v[i], p.v[i + 1]});
        }
        face_normals.assign(surface_faces.size(), Vec3::Zero());
        update_normals(particles);
    }

    // 根据 tet 邻接关系构建簇（比空间半径更精确）
    std::vector<Cluster> build_clusters_from_tets(const Particles& particles, int rings = 1) {
        int N = num_vertices();
        std::vector<Cluster> clusters(N);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < N; i++) {
            // 从顶点 i 出发，收集 `rings` 跳邻居
            std::unordered_set<int> members;
            members.insert(i);
            std::vector<int> frontier = {i};

            for (int r = 0; r < rings; r++) {
                std::vector<int> next_frontier;
                for (int v : frontier) {
                    for (int nb : vertex_neighbors[v]) {
                        if (members.find(nb) == members.end()) {
                            members.insert(nb);
                            next_frontier.push_back(nb);
                        }
                    }
                }
                frontier = next_frontier;
            }

            clusters[i].particle_ids.assign(members.begin(), members.end());
            std::sort(clusters[i].particle_ids.begin(), clusters[i].particle_ids.end());

            if (clusters[i].particle_ids.size() >= 2) {
                clusters[i].init(particles);
            }
        }

        // 移除无效簇
        std::vector<Cluster> result;
        result.reserve(N);
        for (auto& c : clusters) {
            if (c.particle_ids.size() >= 2) {
                result.push_back(std::move(c));
            }
        }
        return result;
    }

private:
    void _parse_nodes(std::ifstream& file,
                      std::vector<int>& tags,
                      std::vector<Vec3>& coords) {
        std::string line;
        std::getline(file, line);
        std::istringstream header(line);
        int num_blocks, total_nodes, min_tag, max_tag;
        header >> num_blocks >> total_nodes >> min_tag >> max_tag;

        tags.reserve(total_nodes);
        coords.reserve(total_nodes);

        for (int b = 0; b < num_blocks; b++) {
            std::getline(file, line);
            std::istringstream block_header(line);
            int entity_dim, entity_tag, parametric, num_nodes;
            block_header >> entity_dim >> entity_tag >> parametric >> num_nodes;

            // 读取 node tags
            std::vector<int> block_tags(num_nodes);
            for (int i = 0; i < num_nodes; i++) {
                file >> block_tags[i];
            }

            // 读取坐标
            for (int i = 0; i < num_nodes; i++) {
                double x, y, z;
                file >> x >> y >> z;
                tags.push_back(block_tags[i]);
                coords.push_back(Vec3(x, y, z));
            }
            std::getline(file, line); // 消耗换行
        }
        std::getline(file, line); // $EndNodes
    }

    void _parse_elements(std::ifstream& file,
                         const std::vector<int>& node_tags,
                         std::vector<std::array<int, 3>>& triangles,
                         std::vector<std::array<int, 4>>& tetrahedra) {
        std::string line;
        std::getline(file, line);
        std::istringstream header(line);
        int num_blocks, total_elems, min_tag, max_tag;
        header >> num_blocks >> total_elems >> min_tag >> max_tag;

        for (int b = 0; b < num_blocks; b++) {
            std::getline(file, line);
            std::istringstream block_header(line);
            int entity_dim, entity_tag, elem_type, num_elems;
            block_header >> entity_dim >> entity_tag >> elem_type >> num_elems;

            for (int i = 0; i < num_elems; i++) {
                int elem_tag;
                file >> elem_tag;

                if (elem_type == 2) {
                    // 三角形 (3 nodes)
                    int a, b, c;
                    file >> a >> b >> c;
                    triangles.push_back({a, b, c});
                } else if (elem_type == 4) {
                    // 四面体 (4 nodes)
                    int a, b, c, d;
                    file >> a >> b >> c >> d;
                    tetrahedra.push_back({a, b, c, d});
                } else {
                    // 其他元素类型（线段等），跳过
                    int nodes_per_elem = _nodes_for_type(elem_type);
                    for (int j = 0; j < nodes_per_elem; j++) {
                        int dummy; file >> dummy;
                    }
                }
            }
            std::getline(file, line); // 消耗换行
        }
        std::getline(file, line); // $EndElements
    }

    int _nodes_for_type(int type) {
        switch (type) {
            case 1: return 2;  // line
            case 2: return 3;  // triangle
            case 3: return 4;  // quad
            case 4: return 4;  // tet
            case 5: return 8;  // hex
            case 15: return 1; // point
            default: return 0;
        }
    }

    // 从四面体中提取表面三角面
    // 原理：只属于一个四面体的面就是表面面
    void _extract_surface_faces() {
        surface_faces.clear();

        // 用 sorted (a,b,c) tuple 作为 key，统计每个面出现次数
        using FaceKey = std::tuple<int,int,int>;
        auto make_key = [](int a, int b, int c) -> FaceKey {
            if (a > b) std::swap(a, b);
            if (b > c) std::swap(b, c);
            if (a > b) std::swap(a, b);
            return {a, b, c};
        };

        std::map<FaceKey, int> face_count;
        std::map<FaceKey, std::array<int,3>> face_original;

        for (const auto& tet : tets) {
            int v[4] = {tet[0], tet[1], tet[2], tet[3]};
            // 每个四面体 4 个面
            int face_indices[4][3] = {
                {v[0], v[1], v[2]},
                {v[0], v[1], v[3]},
                {v[0], v[2], v[3]},
                {v[1], v[2], v[3]}
            };

            for (int fi = 0; fi < 4; fi++) {
                auto key = make_key(face_indices[fi][0], face_indices[fi][1], face_indices[fi][2]);
                face_count[key]++;
                face_original[key] = {face_indices[fi][0], face_indices[fi][1], face_indices[fi][2]};
            }
        }

        // 出现恰好 1 次的面就是表面面
        for (const auto& [key, count] : face_count) {
            if (count == 1) {
                surface_faces.push_back(face_original[key]);
            }
        }

        // 修正法线方向：确保法线朝外
        _fix_face_orientations();
    }

    void _fix_face_orientations() {
        // 对每个表面面，计算法线方向，确保它指向四面体外部
        // 方法：找到包含这个面的四面体，用第四个顶点判断法线方向
        for (auto& face : surface_faces) {
            // 找包含这三个顶点的四面体
            for (const auto& tet : tets) {
                bool has[3] = {false, false, false};
                int fourth = -1;
                for (int j = 0; j < 4; j++) {
                    if (tet[j] == face[0]) has[0] = true;
                    else if (tet[j] == face[1]) has[1] = true;
                    else if (tet[j] == face[2]) has[2] = true;
                    else fourth = tet[j];
                }
                if (has[0] && has[1] && has[2] && fourth >= 0) {
                    // 检查法线是否指向远离第四个顶点的方向
                    Vec3 v0 = vertices[face[0]];
                    Vec3 v1 = vertices[face[1]];
                    Vec3 v2 = vertices[face[2]];
                    Vec3 v3 = vertices[fourth];
                    Vec3 normal = (v1 - v0).cross(v2 - v0);
                    Vec3 to_fourth = v3 - v0;
                    if (normal.dot(to_fourth) > 0) {
                        // 法线朝内，翻转
                        std::swap(face[1], face[2]);
                    }
                    break;
                }
            }
        }
    }

    void _compute_normals() {
        face_normals.resize(surface_faces.size());
        for (int i = 0; i < (int)surface_faces.size(); i++) {
            const auto& f = surface_faces[i];
            Vec3 v0 = vertices[f[0]];
            Vec3 v1 = vertices[f[1]];
            Vec3 v2 = vertices[f[2]];
            face_normals[i] = (v1 - v0).cross(v2 - v0).normalized();
        }
    }

    void _build_vertex_neighbors() {
        int N = num_vertices();
        vertex_neighbors.resize(N);

        for (const auto& tet : tets) {
            // 四面体中的每对顶点互为邻居
            for (int i = 0; i < 4; i++) {
                for (int j = i + 1; j < 4; j++) {
                    vertex_neighbors[tet[i]].push_back(tet[j]);
                    vertex_neighbors[tet[j]].push_back(tet[i]);
                }
            }
        }

        // 去重
        for (auto& nbrs : vertex_neighbors) {
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }
    }
};
