#pragma once
// Step 3b — sharp features of a tool mesh, for the CUT_CURVE channel (solver/tool_channels.cuh).
//
// A blade cuts with the features whose material wedge is thin:
//   sharp EDGE   mesh edge whose interior dihedral (wedge) angle theta < theta_max and convex
//   sharp POINT  vertex whose interior solid angle Omega < Omega_max = 2*pi*(1 - cos(theta_max/2)),
//                i.e. sharper than a cone of apex angle theta_max (needle / trocar tip), and not
//                already an endpoint of a sharp edge
// The same angle threshold therefore defines both: a wedge thinner than theta_max, or a cone with
// an apex angle below theta_max. Scalpel wedge (0.8 mm over 5 mm) = 9.1 deg; its 90 deg end faces
// and the ~85 deg spine edges are not sharp.
//
// Interior solid angle of a vertex of a closed polyhedron (all incident edges convex):
//   Omega = sum_e theta_e - (n - 2) * pi        (area of the spherical polygon whose interior
//                                                angles are the n incident dihedral angles)
// A flat vertex gives 2*pi, a trihedral corner A+B+C-pi.
//
// Requirements: closed, consistently oriented, edge-manifold triangle mesh after welding
// coincident vertices (UV-mapped OBJs duplicate positions along seams). Violations are counted and
// reported; such edges are never marked sharp.

#include "rigid_mesh.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <unordered_map>
#include <vector>

struct SharpFeatures {
    std::vector<Vec3> seg;       // pairs (seg[2k], seg[2k+1]); a point feature has seg[2k] == seg[2k+1]
    std::vector<Vec3> nrm;       // pairs (nrm[2k], nrm[2k+1]): outward normals of the two faces of a sharp
                                 //   edge (they span its normal cone); zero for a point feature
    int n_edges = 0, n_points = 0;
    int boundary_edges = 0, nonmanifold_edges = 0, misoriented_edges = 0;
    double sharp_length = 0.0;   // total length of the sharp edges
};

inline SharpFeatures extract_sharp_features(const RigidMesh& m, double sharp_angle_deg) {
    SharpFeatures out;
    const double PI = 3.14159265358979323846;
    const double theta_max = sharp_angle_deg * PI / 180.0;
    const double omega_max = 2.0 * PI * (1.0 - std::cos(0.5 * theta_max));

    // 1. weld coincident vertices (tolerance relative to the mesh size)
    Vec3 bmin, bmax; m.get_bounds(bmin, bmax);
    const double tol = 1e-7 * (bmax - bmin).norm();
    std::map<std::array<long long, 3>, int> key_to_id;
    std::vector<int> weld(m.vertices.size());
    std::vector<Vec3> P;
    for (size_t i = 0; i < m.vertices.size(); i++) {
        const Vec3& v = m.vertices[i];
        std::array<long long, 3> k = {std::llround(v.x() / tol), std::llround(v.y() / tol), std::llround(v.z() / tol)};
        auto it = key_to_id.find(k);
        if (it == key_to_id.end()) { it = key_to_id.emplace(k, (int)P.size()).first; P.push_back(v); }
        weld[i] = it->second;
    }
    std::vector<std::array<int, 3>> F;
    for (const auto& f : m.faces) {
        std::array<int, 3> g = {weld[f[0]], weld[f[1]], weld[f[2]]};
        if (g[0] == g[1] || g[1] == g[2] || g[0] == g[2]) continue;
        if ((P[g[1]] - P[g[0]]).cross(P[g[2]] - P[g[0]]).norm() < 1e-30) continue;
        F.push_back(g);
    }

    // 2. orientation: outward if the signed volume is positive
    double vol = 0.0;
    for (const auto& f : F) vol += P[f[0]].dot(P[f[1]].cross(P[f[2]])) / 6.0;
    const double sgn = vol >= 0.0 ? 1.0 : -1.0;
    std::vector<Vec3> N(F.size());
    for (size_t i = 0; i < F.size(); i++)
        N[i] = sgn * (P[F[i][1]] - P[F[i][0]]).cross(P[F[i][2]] - P[F[i][0]]).normalized();

    // 3. edge -> incident (face, directed?) list
    struct Inc { int face; bool fwd; };              // fwd: the face traverses a -> b with a < b
    std::map<std::pair<int, int>, std::vector<Inc>> E;
    for (int fi = 0; fi < (int)F.size(); fi++)
        for (int k = 0; k < 3; k++) {
            int a = F[fi][k], b = F[fi][(k + 1) % 3];
            E[{std::min(a, b), std::max(a, b)}].push_back({fi, a < b});
        }

    // 4. dihedral angle of every manifold edge
    std::unordered_map<long long, double> theta_of;  // key a*|P|+b
    auto ekey = [&](int a, int b) { return (long long)std::min(a, b) * (long long)P.size() + std::max(a, b); };
    std::vector<std::vector<int>> vert_edges(P.size());
    std::vector<char> on_sharp_edge(P.size(), 0), vert_ok(P.size(), 1);
    for (const auto& [e, inc] : E) {
        int a = e.first, b = e.second;
        if (inc.size() == 1) { out.boundary_edges++; vert_ok[a] = vert_ok[b] = 0; continue; }
        if (inc.size() > 2)  { out.nonmanifold_edges++; vert_ok[a] = vert_ok[b] = 0; continue; }
        if (inc[0].fwd == inc[1].fwd) { out.misoriented_edges++; vert_ok[a] = vert_ok[b] = 0; continue; }
        const Vec3& n1 = N[inc[0].face];
        const Vec3& n2 = N[inc[1].face];
        int d = -1;                                   // vertex of face 2 not on the edge
        for (int k = 0; k < 3; k++) { int v = F[inc[1].face][k]; if (v != a && v != b) d = v; }
        double phi = std::acos(std::max(-1.0, std::min(1.0, n1.dot(n2))));
        bool convex = n1.dot(P[d] - P[a]) <= 0.0;
        double theta = convex ? PI - phi : PI + phi;
        theta_of[ekey(a, b)] = theta;
        vert_edges[a].push_back(b); vert_edges[b].push_back(a);
        if (!convex) vert_ok[a] = vert_ok[b] = 0;     // solid-angle formula needs a convex cone
        if (convex && theta < theta_max) {
            out.seg.push_back(P[a]); out.seg.push_back(P[b]);
            out.nrm.push_back(n1); out.nrm.push_back(n2);
            out.n_edges++;
            out.sharp_length += (P[a] - P[b]).norm();
            on_sharp_edge[a] = on_sharp_edge[b] = 1;
        }
    }

    // 5. sharp points: convex vertices with a small interior solid angle
    for (int v = 0; v < (int)P.size(); v++) {
        if (!vert_ok[v] || on_sharp_edge[v] || vert_edges[v].size() < 3) continue;
        double sum = 0.0;
        for (int w : vert_edges[v]) sum += theta_of[ekey(v, w)];
        double omega = sum - ((double)vert_edges[v].size() - 2.0) * PI;
        if (omega < omega_max) {
            out.seg.push_back(P[v]); out.seg.push_back(P[v]);
            out.nrm.push_back(Vec3::Zero()); out.nrm.push_back(Vec3::Zero());
            out.n_points++;
        }
    }
    return out;
}

inline void print_sharp_features(const SharpFeatures& f, double sharp_angle_deg, int max_list = 8) {
    std::printf("[cut-features] sharp angle < %.1f deg: %d edges (total length %.5f m), %d points"
                " | mesh issues: boundary %d, non-manifold %d, misoriented %d edges\n",
                sharp_angle_deg, f.n_edges, f.sharp_length, f.n_points,
                f.boundary_edges, f.nonmanifold_edges, f.misoriented_edges);
    int n = (int)f.seg.size() / 2;
    for (int k = 0; k < n && k < max_list; k++) {
        const Vec3 &a = f.seg[2 * k], &b = f.seg[2 * k + 1];
        if (a == b) std::printf("    point (%.5f, %.5f, %.5f)\n", a.x(), a.y(), a.z());
        else {
            std::printf("    edge  (%.5f, %.5f, %.5f) - (%.5f, %.5f, %.5f)", a.x(), a.y(), a.z(), b.x(), b.y(), b.z());
            if ((int)f.nrm.size() == 2 * n)
                std::printf("  face normals (%.3f, %.3f, %.3f) (%.3f, %.3f, %.3f)", f.nrm[2*k].x(), f.nrm[2*k].y(), f.nrm[2*k].z(),
                            f.nrm[2*k+1].x(), f.nrm[2*k+1].y(), f.nrm[2*k+1].z());
            std::printf("\n");
        }
    }
    if (n > max_list) std::printf("    ... %d more\n", n - max_list);
}

// Straddle rule "sheet" — the 2D intruders of a tool (intruder-dimension framework, paper §3.2).
// A sharp edge is the rim of a blade; the blade is the SHEET through the edge on the bisector
// plane of its two faces (normal m = n1 - n2, normalised). The sheet's extent is the cross-section
// of the tool solid with that plane, kept as a set of 2D boundary segments (any number of loops,
// convex or not). An edge element (minimum ball B of the edge, endpoints a, b) is cut in two by
// the tool when a and b lie on opposite sides of the plane and the disk B ∩ plane lies inside the
// cross-section: then B minus the sheet has two components, a in one and b in the other. If any
// part of the disk is outside the cross-section, the material connects around the sheet's rim.
// Since the sheet is part of the solid, a separation by the sheet is a separation by the tool.
struct ToolSheet {
    Vec3 o, m, u, v;                               // plane point, unit normal, in-plane basis (u, v, m right-handed)
    std::vector<std::array<double, 4>> seg;       // cross-section boundary, (u0, v0, u1, v1), oriented
    double area = 0.0;                             // enclosed area (shoelace over the oriented segments)
};

inline std::vector<ToolSheet> extract_tool_sheets(const RigidMesh& mesh, const SharpFeatures& f) {
    std::vector<ToolSheet> out;
    Vec3 bmin, bmax; mesh.get_bounds(bmin, bmax);
    const double tol = 1e-7 * (bmax - bmin).norm();
    double vol = 0.0;                              // outward face normals: sign of the enclosed volume
    for (const auto& fc : mesh.faces)
        vol += mesh.vertices[fc[0]].dot(mesh.vertices[fc[1]].cross(mesh.vertices[fc[2]])) / 6.0;
    const double sg = vol >= 0.0 ? 1.0 : -1.0;
    const int n = (int)f.seg.size() / 2;
    for (int k = 0; k < n; k++) {
        if (f.seg[2 * k] == f.seg[2 * k + 1] || (int)f.nrm.size() < 2 * n) continue;   // point feature
        Vec3 m = f.nrm[2 * k] - f.nrm[2 * k + 1];
        if (m.norm() < 1e-12) continue;
        m.normalize();
        const Vec3 o = f.seg[2 * k];
        bool dup = false;                          // features on one plane share one sheet
        for (const auto& s : out)
            if (std::fabs(std::fabs(s.m.dot(m)) - 1.0) < 1e-9 && std::fabs(s.m.dot(o - s.o)) < tol) { dup = true; break; }
        if (dup) continue;
        ToolSheet s; s.o = o; s.m = m;
        Vec3 a = std::fabs(m.x()) < 0.9 ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
        s.u = (a - m * m.dot(a)).normalized();
        s.v = m.cross(s.u);
        for (const auto& fc : mesh.faces) {
            Vec3 P[3]; double d[3];
            for (int i = 0; i < 3; i++) {
                P[i] = mesh.vertices[fc[i]];
                d[i] = m.dot(P[i] - o);
                if (std::fabs(d[i]) < tol) d[i] = 0.0;   // on the plane: counted on the + side (consistent per vertex)
            }
            Vec3 X[2]; int nx = 0;
            for (int i = 0; i < 3; i++) {
                int j = (i + 1) % 3;
                if ((d[i] < 0.0) != (d[j] < 0.0)) {
                    X[nx++] = d[j] == 0.0 ? P[j] : d[i] == 0.0 ? P[i] : P[i] + (P[j] - P[i]) * (d[i] / (d[i] - d[j]));
                }
            }
            if (nx != 2) continue;
            Vec3 nf = sg * (P[1] - P[0]).cross(P[2] - P[0]);
            if ((X[1] - X[0]).dot(m.cross(nf)) < 0.0) std::swap(X[0], X[1]);   // solid on the left (CCW about m)
            std::array<double, 4> e = {s.u.dot(X[0] - o), s.v.dot(X[0] - o), s.u.dot(X[1] - o), s.v.dot(X[1] - o)};
            if (e[0] == e[2] && e[1] == e[3]) continue;
            s.area += 0.5 * (e[0] * e[3] - e[2] * e[1]);
            s.seg.push_back(e);
        }
        out.push_back(s);
    }
    return out;
}

inline void print_tool_sheets(const std::vector<ToolSheet>& sh) {
    std::printf("[straddle-sheet] %zu sheet(s) from the sharp edges\n", sh.size());
    for (const auto& s : sh) {
        double u0 = 1e30, u1 = -1e30, v0 = 1e30, v1 = -1e30;
        for (const auto& e : s.seg) {
            u0 = std::min({u0, e[0], e[2]}); u1 = std::max({u1, e[0], e[2]});
            v0 = std::min({v0, e[1], e[3]}); v1 = std::max({v1, e[1], e[3]});
        }
        std::printf("    plane n (%.3f, %.3f, %.3f) through (%.5f, %.5f, %.5f): %zu boundary segments,"
                    " area %.2f mm^2, extent %.2f x %.2f mm\n",
                    s.m.x(), s.m.y(), s.m.z(), s.o.x(), s.o.y(), s.o.z(), s.seg.size(), s.area * 1e6,
                    (u1 - u0) * 1e3, (v1 - v0) * 1e3);
    }
}
