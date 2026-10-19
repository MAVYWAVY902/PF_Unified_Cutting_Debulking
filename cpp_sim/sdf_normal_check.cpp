// sdf_normal_check — Step-0 measurement: how wrong is the push direction / depth the
// tool-SDF collision kernel computes?
//
// For each test tool, bake the SDF exactly like main_gpu_viewer does (centre the
// mesh, RigidMesh::compute_sdf(res, 0.2)), then sample points that the kernel
// would actually push (inside the grid, trilinear phi < 0, truly inside the mesh)
// and compare against the exact outward normal / depth from brute-force closest
// point on the triangle mesh.
//
//   KERNEL    : exactly tool_sdf_collision_kernel — index-space central difference
//               of NEAREST-NEIGHBOUR samples at gx±1 (int-truncated), no 1/cell
//   CORRECTED : world-space central difference of TRILINEAR samples, divided by
//               2*cell per axis (what the plan proposes)
//
// Usage: sdf_normal_check <res> <obj> [<obj> ...]     (res = viewer tool_sdf_resolution)

#include "core/rigid_mesh.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

struct Grid {
    std::vector<float> d; int rx, ry, rz; Vec3 o, c;
    float at(int x, int y, int z) const {
        x = std::min(std::max(x, 0), rx - 1); y = std::min(std::max(y, 0), ry - 1); z = std::min(std::max(z, 0), rz - 1);
        return d[((size_t)z * ry + y) * rx + x];
    }
    // continuous grid coords (kernel convention: voxel centres at integer coords)
    void g(const Vec3& p, double& gx, double& gy, double& gz) const {
        gx = (p.x() - o.x()) / c.x() - 0.5; gy = (p.y() - o.y()) / c.y() - 0.5; gz = (p.z() - o.z()) / c.z() - 0.5;
    }
    bool interior(const Vec3& p) const {
        double gx, gy, gz; g(p, gx, gy, gz);
        return !(gx < 0 || gx >= rx - 1 || gy < 0 || gy >= ry - 1 || gz < 0 || gz >= rz - 1);
    }
    double tri(const Vec3& p) const {
        double gx, gy, gz; g(p, gx, gy, gz);
        int ix = (int)floor(gx), iy = (int)floor(gy), iz = (int)floor(gz);
        double fx = gx - ix, fy = gy - iy, fz = gz - iz;
        double c00 = at(ix,iy,iz)*(1-fx) + at(ix+1,iy,iz)*fx, c10 = at(ix,iy+1,iz)*(1-fx) + at(ix+1,iy+1,iz)*fx;
        double c01 = at(ix,iy,iz+1)*(1-fx) + at(ix+1,iy,iz+1)*fx, c11 = at(ix,iy+1,iz+1)*(1-fx) + at(ix+1,iy+1,iz+1)*fx;
        return (c00*(1-fy) + c10*fy)*(1-fz) + (c01*(1-fy) + c11*fy)*fz;
    }
    Vec3 n_kernel(const Vec3& p) const {        // verbatim tool_sdf_collision_kernel
        double gx, gy, gz; g(p, gx, gy, gz);
        auto S = [&](double a, double b, double cc) { return (double)at((int)a, (int)b, (int)cc); };
        Vec3 n(S(gx+1,gy,gz) - S(gx-1,gy,gz), S(gx,gy+1,gz) - S(gx,gy-1,gz), S(gx,gy,gz+1) - S(gx,gy,gz-1));
        double l = n.norm(); return l < 1e-10 ? Vec3::Zero() : Vec3(n / l);
    }
    Vec3 n_corrected(const Vec3& p) const {
        Vec3 n;
        for (int k = 0; k < 3; k++) {
            Vec3 h = Vec3::Zero(); h[k] = c[k];
            n[k] = (tri(p + h) - tri(p - h)) / (2.0 * c[k]);
        }
        double l = n.norm(); return l < 1e-10 ? Vec3::Zero() : Vec3(n / l);
    }
};

static Vec3 closest_on_tri(const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c) {   // Ericson 5.1.5
    Vec3 ab = b - a, ac = c - a, ap = p - a;
    double d1 = ab.dot(ap), d2 = ac.dot(ap); if (d1 <= 0 && d2 <= 0) return a;
    Vec3 bp = p - b; double d3 = ab.dot(bp), d4 = ac.dot(bp); if (d3 >= 0 && d4 <= d3) return b;
    double vc = d1*d4 - d3*d2; if (vc <= 0 && d1 >= 0 && d3 <= 0) return a + ab * (d1 / (d1 - d3));
    Vec3 cp = p - c; double d5 = ab.dot(cp), d6 = ac.dot(cp); if (d6 >= 0 && d5 <= d6) return c;
    double vb = d5*d2 - d1*d6; if (vb <= 0 && d2 >= 0 && d6 <= 0) return a + ac * (d2 / (d2 - d6));
    double va = d3*d6 - d5*d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    double den = 1.0 / (va + vb + vc); return a + ab * (vb * den) + ac * (vc * den);
}

static bool ray_hit(const Vec3& o, const Vec3& dir, const Vec3& a, const Vec3& b, const Vec3& c) {
    Vec3 e1 = b - a, e2 = c - a, h = dir.cross(e2); double det = e1.dot(h);
    if (std::abs(det) < 1e-14) return false;
    double inv = 1.0 / det; Vec3 s = o - a; double u = inv * s.dot(h); if (u < 0 || u > 1) return false;
    Vec3 q = s.cross(e1); double v = inv * dir.dot(q); if (v < 0 || u + v > 1) return false;
    return inv * e2.dot(q) > 1e-12;
}

static void stats(const char* label, std::vector<double> v, const char* unit) {
    if (v.empty()) { printf("    %-34s (no samples)\n", label); return; }
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v[std::min(v.size() - 1, (size_t)(f * v.size()))]; };
    int gt = 0; for (double x : v) gt += (x > 30.0);
    printf("    %-34s median=%7.2f  p90=%7.2f  p99=%7.2f  max=%7.2f %s", label, q(0.5), q(0.9), q(0.99), v.back(), unit);
    if (std::string(unit) == "deg") printf("   >30deg: %.1f%%", 100.0 * gt / v.size());
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <res> <obj> [<obj>...]\n", argv[0]); return 1; }
    int res = std::atoi(argv[1]);
    for (int ai = 2; ai < argc; ai++) {
        std::string path = argv[ai];
        RigidMesh m;
        if (!m.load_obj(path)) continue;
        Vec3 bmin, bmax; m.get_bounds(bmin, bmax); m.translate(-(bmin + bmax) * 0.5);
        m.get_bounds(bmin, bmax);
        m.compute_sdf(res, 0.2);
        Grid G{m.sdf_data, m.sdf_res[0], m.sdf_res[1], m.sdf_res[2], m.sdf_origin, m.sdf_cell_size};
        double cmin = G.c.minCoeff(), cmax = G.c.maxCoeff();
        printf("\n=== %s  extent=(%.4f, %.4f, %.4f) m  grid=%dx%dx%d  cell=(%.5f, %.5f, %.5f) m  anisotropy=%.2f\n",
               path.c_str(), (bmax - bmin).x(), (bmax - bmin).y(), (bmax - bmin).z(),
               G.rx, G.ry, G.rz, G.c.x(), G.c.y(), G.c.z(), cmax / cmin);

        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> U(0.0, 1.0);
        const Vec3 rays[3] = {Vec3(1, 0.0137, 0.0071).normalized(), Vec3(0.0093, 1, 0.0151).normalized(),
                              Vec3(0.0119, 0.0067, 1).normalized()};
        std::vector<double> ang_k, ang_c, depth_err, depth_rel;
        int tried = 0;
        while ((int)ang_k.size() < 3000 && tried < 400000) {
            tried++;
            Vec3 p(bmin.x() + U(rng) * (bmax - bmin).x(), bmin.y() + U(rng) * (bmax - bmin).y(),
                   bmin.z() + U(rng) * (bmax - bmin).z());
            if (!G.interior(p)) continue;
            double phi = G.tri(p);
            if (phi >= 0.0) continue;                         // kernel only acts where phi<0
            // exact: closest point + inside by 3-ray majority
            double best = 1e30; Vec3 cp;
            int votes = 0;
            for (int r = 0; r < 3; r++) {
                int hits = 0;
                for (int f = 0; f < m.num_faces(); f++) {
                    const auto& F = m.faces[f];
                    if (r == 0) {
                        Vec3 q = closest_on_tri(p, m.vertices[F[0]], m.vertices[F[1]], m.vertices[F[2]]);
                        double d = (p - q).squaredNorm(); if (d < best) { best = d; cp = q; }
                    }
                    hits += ray_hit(p, rays[r], m.vertices[F[0]], m.vertices[F[1]], m.vertices[F[2]]);
                }
                votes += (hits & 1);
            }
            if (votes < 2) continue;                          // not truly inside
            double dist = std::sqrt(best);
            if (dist < 1e-9) continue;
            Vec3 n_true = (cp - p) / dist;                    // outward push direction from an inside point
            Vec3 nk = G.n_kernel(p), nc = G.n_corrected(p);
            auto ang = [&](const Vec3& a) { return a.norm() < 0.5 ? 180.0 : std::acos(std::max(-1.0, std::min(1.0, a.dot(n_true)))) * 180.0 / M_PI; };
            ang_k.push_back(ang(nk)); ang_c.push_back(ang(nc));
            depth_err.push_back(std::abs(-phi - dist) / cmin);
            depth_rel.push_back(std::abs(-phi - dist) / dist * 100.0);
        }
        printf("  samples: %zu inside points the kernel would push (tried %d)\n", ang_k.size(), tried);
        stats("normal error, KERNEL (as-is):", ang_k, "deg");
        stats("normal error, CORRECTED:", ang_c, "deg");
        stats("depth error |phi_tri - d_true|:", depth_err, "cells(min)");
    }
    return 0;
}
