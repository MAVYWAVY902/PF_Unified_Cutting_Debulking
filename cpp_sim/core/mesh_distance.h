#pragma once
// R2 — exact signed distance to a closed triangle mesh (host reference implementation).
//   distance: closest point over all triangles (Ericson, same routine as RigidMesh)
//   sign:     generalized winding number (Jacobson et al. 2013): w = sum of signed solid angles
//             / 4 pi, inside when w > 1/2 — exact for closed meshes, robust near the surface and
//             independent of face winding consistency in the sense that it only needs a closed,
//             consistently oriented surface (orientation fixed by the sign of the total volume).
// Used by tool_bench diagnostics to measure what the baked grid SDF misses near thin features
// (the blade-edge strip where the wedge is thinner than a cell).

#include "rigid_mesh.h"
#include <cmath>
#include <vector>

struct ExactMeshSDF {
    std::vector<Vec3> V;
    std::vector<std::array<int, 3>> F;
    double orient = 1.0;                 // +1 if F is outward-wound, -1 otherwise

    void build(const RigidMesh& m) {
        V = m.vertices; F = m.faces;
        double vol = 0.0;
        for (const auto& f : F) vol += V[f[0]].dot(V[f[1]].cross(V[f[2]])) / 6.0;
        orient = vol >= 0.0 ? 1.0 : -1.0;
    }

    static Vec3 closest_on_tri(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {
        Vec3 ab = b - a, ac = c - a, ap = p - a;
        double d1 = ab.dot(ap), d2 = ac.dot(ap);
        if (d1 <= 0 && d2 <= 0) return a;
        Vec3 bp = p - b; double d3 = ab.dot(bp), d4 = ac.dot(bp);
        if (d3 >= 0 && d4 <= d3) return b;
        double vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
        Vec3 cp = p - c; double d5 = ab.dot(cp), d6 = ac.dot(cp);
        if (d6 >= 0 && d5 <= d6) return c;
        double vb = d5 * d2 - d1 * d6;
        if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
        double va = d3 * d6 - d5 * d4;
        if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        double den = 1.0 / (va + vb + vc);
        return a + ab * (vb * den) + ac * (vc * den);
    }

    // generalized winding number at p (Van Oosterom–Strackee solid angle per triangle)
    double winding(const Vec3& p) const {
        double w = 0.0;
        for (const auto& f : F) {
            Vec3 a = V[f[0]] - p, b = V[f[1]] - p, c = V[f[2]] - p;
            double la = a.norm(), lb = b.norm(), lc = c.norm();
            double num = a.dot(b.cross(c));
            double den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
            w += 2.0 * std::atan2(num, den);
        }
        return orient * w / (4.0 * 3.14159265358979323846);
    }

    // signed distance (< 0 inside) and, optionally, the closest surface point
    double phi(const Vec3& p, Vec3* closest = nullptr) const {
        double best = 1e300; Vec3 cbest;
        for (const auto& f : F) {
            Vec3 q = closest_on_tri(p, V[f[0]], V[f[1]], V[f[2]]);
            double d2 = (p - q).squaredNorm();
            if (d2 < best) { best = d2; cbest = q; }
        }
        if (closest) *closest = cbest;
        double d = std::sqrt(best);
        return winding(p) > 0.5 ? -d : d;
    }
};
