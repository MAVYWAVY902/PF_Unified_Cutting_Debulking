#pragma once
// Unified tool query for every tool-dependent GPU kernel (Step 1: sphere, Step 2: grid SDF).
//
// A tool exposes four device-side operations. Kernels call these instead of doing their own
// tool geometry, so a tool representation is implemented in ONE place.
//
//   tool_query(t, p, need_n)          signed distance phi (<0 inside) + outward unit normal n
//   tool_project_out(t, p)            if p is inside, move it to the surface
//   tool_contains(t, p)               "is p at the tool" test used by aspiration
//   tool_outside_damage_band(t, p, r) true = too far from the tool for edge damage to matter
//
// Generic definitions (grid SDF): project = p - phi*n, contains = phi < 0 (legacy, strict) or
// phi <= reach, band = |phi| > 3 r_e. The analytic sphere keeps its CLOSED-FORM versions with
// exactly the float operations the pre-Step-1 kernels used. This is deliberate: Step 0 showed
// that 100% of sphere aspirations happen to particles the collision has just projected onto the
// surface (|d-R|/R < 1e-6), i.e. with reach = 0 removal is decided by float rounding of
// "project, then |d|^2 <= R^2". A differently rounded projection would change removal counts.
//
// Step-2 switches (all default to the legacy behaviour, see SimConfigGPU):
//   normal_mode   0 = legacy: index-space central difference of NEAREST-NEIGHBOUR samples
//                     (wrong direction on anisotropic grids: dissector 64x16x16 median 36 deg)
//                 1 = trilinear: gradient of the trilinear field, central difference with one
//                     cell per axis divided by 2*cell (world units) — sdf_normal_check: 3.8 deg
//   reach         0 = legacy contains test (sphere closed |d|<=R, grid strict phi<0)
//                 >0 = contains is phi <= reach for every tool type: aspiration acts on tissue
//                     within `reach` of the tool surface (suction reach), so it no longer
//                     depends on the rounding of the collision projection
// (The P(F) edge-radius offset is applied in the damage kernel, not here: it is a property of
//  the intruder model, not of the tool geometry.)

#include <cuda_runtime.h>
#include "tool_sheet.h"

struct ToolDesc {
    int    type;          // 0 = analytic sphere, 1 = grid SDF (tool local frame)
    float3 pos;           // tool frame origin in world (sphere: centre)
    float  radius;        // sphere radius (type 0)
    float  reach;         // aspiration reach [m]; 0 = legacy contains test
    // type 1 only
    float  rot[9];        // local -> world, row-major
    const float* sdf;     // [res_x*res_y*res_z], x fastest
    int    res_x, res_y, res_z;
    float3 origin;        // grid box min corner (local frame); voxel centres at origin+(i+0.5)*cell
    float3 cell;
    int    normal_mode;   // 0 legacy, 1 trilinear
    const float3* tri;    // contact_ccd: exact triangles (v0 v1 v2 n per triangle, local frame); nullptr = none
    int    n_tri;
    const ToolSheetGPU* sheet;   // straddle_rule sheet: 2D intruders (core/tool_features.h), local frame
    const float4* sheet_seg;            // cross-section boundary segments (u0, v0, u1, v1) of all sheets
    int    n_sheet;
};


struct ToolSample {
    float  phi;           // signed distance to the tool surface (<0 inside); 1e30 = outside grid
    float3 n;             // outward unit normal in world (zero if undefined / not requested)
};

__host__ __device__ inline ToolDesc make_sphere_tool(float3 c, float r, float reach = 0.0f) {
    ToolDesc t = {};
    t.type = 0; t.pos = c; t.radius = r; t.reach = reach;
    return t;
}

// ---------------- grid helpers (same arithmetic as the legacy tool_sdf_* kernels) ----------------
__device__ inline float3 tq_rot_T(const float* R, float3 v) {   // R^T v
    return make_float3(R[0]*v.x + R[3]*v.y + R[6]*v.z,
                       R[1]*v.x + R[4]*v.y + R[7]*v.z,
                       R[2]*v.x + R[5]*v.y + R[8]*v.z);
}
__device__ inline float3 tq_rot(const float* R, float3 v) {     // R v
    return make_float3(R[0]*v.x + R[1]*v.y + R[2]*v.z,
                       R[3]*v.x + R[4]*v.y + R[5]*v.z,
                       R[6]*v.x + R[7]*v.y + R[8]*v.z);
}
__device__ inline float tq_at(const ToolDesc& t, int x, int y, int z) {
    return t.sdf[z * t.res_x * t.res_y + y * t.res_x + x];
}
__device__ inline float tq_at_clamped(const ToolDesc& t, int x, int y, int z) {
    x = min(max(x, 0), t.res_x - 1); y = min(max(y, 0), t.res_y - 1); z = min(max(z, 0), t.res_z - 1);
    return tq_at(t, x, y, z);
}
// trilinear at continuous grid coords; caller guarantees 0 <= g < res-1 (legacy interior test)
__device__ inline float tq_trilinear(const ToolDesc& t, float gx, float gy, float gz) {
    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    float fx = gx - ix, fy = gy - iy, fz = gz - iz;
    float c00 = tq_at(t,ix,iy,iz)*(1-fx) + tq_at(t,ix+1,iy,iz)*fx;
    float c10 = tq_at(t,ix,iy+1,iz)*(1-fx) + tq_at(t,ix+1,iy+1,iz)*fx;
    float c01 = tq_at(t,ix,iy,iz+1)*(1-fx) + tq_at(t,ix+1,iy,iz+1)*fx;
    float c11 = tq_at(t,ix,iy+1,iz+1)*(1-fx) + tq_at(t,ix+1,iy+1,iz+1)*fx;
    return (c00*(1-fy)+c10*fy)*(1-fz) + (c01*(1-fy)+c11*fy)*fz;
}
// trilinear with index clamping, for the normal stencil (may step one cell past the interior)
__device__ inline float tq_trilinear_clamped(const ToolDesc& t, float gx, float gy, float gz) {
    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    float fx = gx - ix, fy = gy - iy, fz = gz - iz;
    float c00 = tq_at_clamped(t,ix,iy,iz)*(1-fx) + tq_at_clamped(t,ix+1,iy,iz)*fx;
    float c10 = tq_at_clamped(t,ix,iy+1,iz)*(1-fx) + tq_at_clamped(t,ix+1,iy+1,iz)*fx;
    float c01 = tq_at_clamped(t,ix,iy,iz+1)*(1-fx) + tq_at_clamped(t,ix+1,iy,iz+1)*fx;
    float c11 = tq_at_clamped(t,ix,iy+1,iz+1)*(1-fx) + tq_at_clamped(t,ix+1,iy+1,iz+1)*fx;
    return (c00*(1-fy)+c10*fy)*(1-fz) + (c01*(1-fy)+c11*fy)*fz;
}

__device__ inline ToolSample tool_query(const ToolDesc& t, float3 p, bool need_n = true) {
    ToolSample s;
    if (t.type == 0) {
        float dx = p.x - t.pos.x, dy = p.y - t.pos.y, dz = p.z - t.pos.z;
        float d = sqrtf(dx*dx + dy*dy + dz*dz);
        s.phi = d - t.radius;                  // same expression as the legacy damage kernel
        if (need_n && d > 1e-10f) { float inv = 1.0f / d; s.n = make_float3(dx * inv, dy * inv, dz * inv); }
        else                      { s.n = make_float3(0.0f, 0.0f, 0.0f); }
        return s;
    }
    // type 1: grid SDF in the tool local frame
    float3 d = make_float3(p.x - t.pos.x, p.y - t.pos.y, p.z - t.pos.z);
    float3 l = tq_rot_T(t.rot, d);
    float gx = (l.x - t.origin.x) / t.cell.x - 0.5f;
    float gy = (l.y - t.origin.y) / t.cell.y - 0.5f;
    float gz = (l.z - t.origin.z) / t.cell.z - 0.5f;
    s.n = make_float3(0.0f, 0.0f, 0.0f);
    if (gx < 0 || gx >= t.res_x-1 || gy < 0 || gy >= t.res_y-1 || gz < 0 || gz >= t.res_z-1) {
        s.phi = 1e30f;                         // outside the baked box: no contact, no damage
        return s;
    }
    s.phi = tq_trilinear(t, gx, gy, gz);
    if (!need_n) return s;
    float nx, ny, nz;
    if (t.normal_mode == 0) {
        // legacy (verbatim): nearest-neighbour samples at g±1, index-space difference
        nx = tq_at_clamped(t, (int)(gx+1), (int)gy, (int)gz) - tq_at_clamped(t, (int)(gx-1), (int)gy, (int)gz);
        ny = tq_at_clamped(t, (int)gx, (int)(gy+1), (int)gz) - tq_at_clamped(t, (int)gx, (int)(gy-1), (int)gz);
        nz = tq_at_clamped(t, (int)gx, (int)gy, (int)(gz+1)) - tq_at_clamped(t, (int)gx, (int)gy, (int)(gz-1));
    } else {
        // trilinear field gradient, world units (±1 cell along each axis / 2 cells)
        nx = (tq_trilinear_clamped(t, gx+1, gy, gz) - tq_trilinear_clamped(t, gx-1, gy, gz)) / (2.0f * t.cell.x);
        ny = (tq_trilinear_clamped(t, gx, gy+1, gz) - tq_trilinear_clamped(t, gx, gy-1, gz)) / (2.0f * t.cell.y);
        nz = (tq_trilinear_clamped(t, gx, gy, gz+1) - tq_trilinear_clamped(t, gx, gy, gz-1)) / (2.0f * t.cell.z);
    }
    float len = sqrtf(nx*nx + ny*ny + nz*nz);
    if (len < 1e-10f) return s;
    nx /= len; ny /= len; nz /= len;
    s.n = tq_rot(t.rot, make_float3(nx, ny, nz));
    return s;
}

// Returns true if p was moved.
//   sphere: radial projection c + (p-c)*R/|p-c| (legacy formula)
//   grid:   p += (-phi) * n                      (legacy tool_sdf_collision_kernel arithmetic)
__device__ inline bool tool_project_out(const ToolDesc& t, float3& p) {
    if (t.type == 0) {
        float dx = p.x - t.pos.x, dy = p.y - t.pos.y, dz = p.z - t.pos.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dist < t.radius && dist > 1e-10f) {
            float inv = t.radius / dist;
            p.x = t.pos.x + dx * inv;
            p.y = t.pos.y + dy * inv;
            p.z = t.pos.z + dz * inv;
            return true;
        }
        return false;
    }
    ToolSample q = tool_query(t, p, false);
    if (q.phi >= 0.0f) return false;          // also covers "outside grid" (1e30)
    q = tool_query(t, p, true);
    if (q.n.x == 0.0f && q.n.y == 0.0f && q.n.z == 0.0f) return false;
    float push = -q.phi;
    p.x += push * q.n.x;
    p.y += push * q.n.y;
    p.z += push * q.n.z;
    return true;
}

// F1: repeat the push until the point is on/outside the surface. One push of |phi| along the grid
// normal reaches the surface only where the field is locally planar; inside a tool only 1-4 cells
// thick (blade next to the cutting edge, front corners) the trilinear gradient is smeared and one
// push left particles 0.03-0.08 mm inside, so they drifted through the blade over the iterations.
__device__ inline bool tool_project_out_iter(const ToolDesc& t, float3& p, int max_iter) {
    bool moved = false;
    for (int k = 0; k < max_iter; k++) {
        if (!tool_project_out(t, p)) break;
        moved = true;
    }
    return moved;
}

// Aspiration "at the tool" test.
//   reach == 0 (legacy): sphere |p-c|^2 <= R^2 (closed, no sqrt); grid phi < 0 (strict)
//   reach  > 0         : phi <= reach for every tool type
__device__ inline bool tool_contains(const ToolDesc& t, float3 p) {
    if (t.reach > 0.0f) return tool_query(t, p, false).phi <= t.reach;
    if (t.type == 0) {
        float dx = p.x - t.pos.x, dy = p.y - t.pos.y, dz = p.z - t.pos.z;
        return !(dx*dx + dy*dy + dz*dz > t.radius * t.radius);
    }
    return tool_query(t, p, false).phi < 0.0f;
}

// Edge-damage search band (legacy per type):
//   sphere: edge midpoint farther than 3R from the centre (= 2R beyond the surface)
//   grid:   |phi| > 3 r_e (r_e = edge half-length), or outside the baked box
__device__ inline bool tool_outside_damage_band(const ToolDesc& t, float3 p, float r_e) {
    if (t.type == 0) {
        float dx = p.x - t.pos.x, dy = p.y - t.pos.y, dz = p.z - t.pos.z;
        return sqrtf(dx*dx + dy*dy + dz*dz) > t.radius * 3.0f;
    }
    return fabsf(tool_query(t, p, false).phi) > r_e * 3.0f;
}

// Step 3c — deepest point of a tissue EDGE inside the tool (edge contact).
// A tool thinner than the particle spacing can sit between the two particles of an edge, where
// the particle collision never sees it. Returns true if an interior point x(t) = a + t (b - a),
// 0 < t < 1, lies inside the tool, with t, phi(x(t)) < 0 and the outward normal there.
//   sphere: exact closest point of the segment to the centre
//   grid:   samples every h along the edge, then a ternary refinement around the best sample;
//           skipped when phi(mid) > |b - a| / 2 (phi is 1-Lipschitz: the whole edge is outside);
//           a midpoint outside the baked box carries no bound, so the edge is sampled
// Edges whose deepest point is an endpoint are left to the particle collision.
__device__ inline bool tool_edge_deepest(const ToolDesc& t, float3 a, float3 b, float h,
                                         float& t_out, float& phi_out, float3& n_out) {
    float3 d = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    float L2 = d.x*d.x + d.y*d.y + d.z*d.z;
    if (L2 < 1e-20f) return false;
    if (t.type == 0) {
        float s = ((t.pos.x - a.x)*d.x + (t.pos.y - a.y)*d.y + (t.pos.z - a.z)*d.z) / L2;
        if (s <= 0.0f || s >= 1.0f) return false;
        float3 x = make_float3(a.x + s*d.x, a.y + s*d.y, a.z + s*d.z);
        ToolSample q = tool_query(t, x, true);
        if (q.phi >= 0.0f || (q.n.x == 0.0f && q.n.y == 0.0f && q.n.z == 0.0f)) return false;
        t_out = s; phi_out = q.phi; n_out = q.n;
        return true;
    }
    float L = sqrtf(L2);
    float3 mid = make_float3(a.x + 0.5f*d.x, a.y + 0.5f*d.y, a.z + 0.5f*d.z);
    float phi_mid = tool_query(t, mid, false).phi;
    if (phi_mid < 1e29f && phi_mid > 0.5f * L) return false;   // 1e30 = outside the baked box: no bound, sample
    int ns = (int)ceilf(L / h);
    ns = ns < 2 ? 2 : (ns > 64 ? 64 : ns);
    float best = 1e30f, bt = 0.5f;
    for (int k = 1; k < ns; k++) {
        float s = (float)k / ns;
        float v = tool_query(t, make_float3(a.x + s*d.x, a.y + s*d.y, a.z + s*d.z), false).phi;
        if (v < best) { best = v; bt = s; }
    }
    if (best >= 0.0f) return false;
    float lo = fmaxf(bt - 1.0f / ns, 0.0f), hi = fminf(bt + 1.0f / ns, 1.0f);
    for (int it = 0; it < 8; it++) {
        float m1 = lo + (hi - lo) / 3.0f, m2 = hi - (hi - lo) / 3.0f;
        float v1 = tool_query(t, make_float3(a.x + m1*d.x, a.y + m1*d.y, a.z + m1*d.z), false).phi;
        float v2 = tool_query(t, make_float3(a.x + m2*d.x, a.y + m2*d.y, a.z + m2*d.z), false).phi;
        if (v1 < v2) hi = m2; else lo = m1;
    }
    float s = 0.5f * (lo + hi);
    if (s <= 0.0f || s >= 1.0f) return false;
    ToolSample q = tool_query(t, make_float3(a.x + s*d.x, a.y + s*d.y, a.z + s*d.z), true);
    if (q.phi >= 0.0f || (q.n.x == 0.0f && q.n.y == 0.0f && q.n.z == 0.0f)) return false;
    t_out = s; phi_out = q.phi; n_out = q.n;
    return true;
}

// Entry-side contact (A1). tool_project_out moves a particle inside the tool to the NEAREST surface.
// For a tool thinner than a particle's motion within a substep (blade body 0.1-0.3 mm next to the
// cutting edge), a particle pulled past the mid-plane is then pushed out on the far side: it passes
// THROUGH the tool (measured: 85% of the edges broken by a sideways push straddled the blade this
// way). Here the particle is returned to the side it entered from: its path relative to the tool
// over the substep, from old_p (tool pose prev) to p (tool pose cur), is bisected for the entry point
// x* (phi = 0); p is projected onto the tangent plane at x* (normal n*), which removes the
// penetration and keeps sliding along the face. A particle already inside at the start of the
// substep (no entry point) falls back to tool_project_out. Grid tools only (prev.type == cur.type).
__device__ inline float tq_phi_local(const ToolDesc& t, float3 l) {
    float gx = (l.x - t.origin.x) / t.cell.x - 0.5f;
    float gy = (l.y - t.origin.y) / t.cell.y - 0.5f;
    float gz = (l.z - t.origin.z) / t.cell.z - 0.5f;
    if (gx < 0 || gx >= t.res_x-1 || gy < 0 || gy >= t.res_y-1 || gz < 0 || gz >= t.res_z-1) return 1e30f;
    return tq_trilinear(t, gx, gy, gz);
}
__device__ inline bool tool_project_out_entry(const ToolDesc& cur, const ToolDesc& prev, float3& p, float3 old_p,
                                              int max_iter = 1) {
    if (cur.type != 1) return tool_project_out_iter(cur, p, max_iter);
    ToolSample q = tool_query(cur, p, false);
    if (q.phi >= 0.0f) return false;
    float3 ln = tq_rot_T(cur.rot,  make_float3(p.x - cur.pos.x,  p.y - cur.pos.y,  p.z - cur.pos.z));
    float3 lo = tq_rot_T(prev.rot, make_float3(old_p.x - prev.pos.x, old_p.y - prev.pos.y, old_p.z - prev.pos.z));
    // A particle resting ON the surface at the start of the substep (the previous contact put it at
    // phi = 0, rounding / trilinear interpolation leave it at phi slightly < 0) is outside: its side is
    // known. Requiring phi > 0 (as before) classified exactly these contacting particles as "started
    // inside" and fell back to the nearest face — which, once shape matching pulled them past the
    // blade's mid-plane, is the OPPOSITE face (measured: 80% of the edges R3 broke under a sideways
    // push). Tolerance = the representation's resolution, half the smallest SDF cell (as in R3).
    const float eps = 0.5f * fminf(cur.cell.x, fminf(cur.cell.y, cur.cell.z));
    const float phi_o = tq_phi_local(cur, lo);
    if (!(phi_o >= -eps)) return tool_project_out_iter(cur, p, max_iter);   // started inside: no entry side
    float a = 0.0f;
    if (phi_o > 0.0f) {                                                    // phi(lo) > 0, phi(ln) < 0: bisect
        float b = 1.0f;
        for (int it = 0; it < 16; it++) {
            float m = 0.5f * (a + b);
            float3 lm = make_float3(lo.x + m*(ln.x - lo.x), lo.y + m*(ln.y - lo.y), lo.z + m*(ln.z - lo.z));
            if (tq_phi_local(cur, lm) > 0.0f) a = m; else b = m;
        }
    }                                                                      // else: started on the surface, entry = lo
    float3 ls = make_float3(lo.x + a*(ln.x - lo.x), lo.y + a*(ln.y - lo.y), lo.z + a*(ln.z - lo.z));  // entry point (outside side)
    float3 xs = tq_rot(cur.rot, ls); xs.x += cur.pos.x; xs.y += cur.pos.y; xs.z += cur.pos.z;
    ToolSample qs = tool_query(cur, xs, true);
    float3 n = qs.n;
    if (n.x == 0.0f && n.y == 0.0f && n.z == 0.0f) return tool_project_out_iter(cur, p, max_iter);
    float d = (p.x - xs.x)*n.x + (p.y - xs.y)*n.y + (p.z - xs.z)*n.z;      // < 0: behind the entry plane
    if (d < 0.0f) { p.x -= d * n.x; p.y -= d * n.y; p.z -= d * n.z; }     // back onto the entry tangent plane
    // Still inside (curved / wedge surface, or in front of the entry plane at a corner): push along the
    // ENTRY normal, never to the nearest face, so the particle stays on the side it came from.
    for (int k = 0; k < max_iter; k++) {
        float ph = tool_query(cur, p, false).phi;
        if (ph >= 0.0f) break;
        p.x -= ph * n.x; p.y -= ph * n.y; p.z -= ph * n.z;
    }
    return true;
}

// Exact path test of contact_ccd against the tool's triangles (tool local frame). The grid SDF misses
// any part thinner than a cell: the bottom 5.25 mm of the x12 scalpel wedge (thinner than its 0.84 mm
// z-cell) was invisible, and shape matching carried particles 8 mm straight through it. Here the first
// ENTRY crossing of the path lo -> ln (motion against the face's outward normal) gives the entry point
// and its exact face normal. A reference lying on a face (the previous contact put it there) may miss
// its own face numerically: if the path leaves the tool (an exit crossing) without an entry, it started
// on the surface, and the entry face is the triangle closest to lo.
__device__ inline float3 tq_closest_on_tri(float3 p, float3 a, float3 b, float3 c) {   // Ericson 5.1.5
    float3 ab = make_float3(b.x-a.x, b.y-a.y, b.z-a.z), ac = make_float3(c.x-a.x, c.y-a.y, c.z-a.z);
    float3 ap = make_float3(p.x-a.x, p.y-a.y, p.z-a.z);
    float d1 = ab.x*ap.x + ab.y*ap.y + ab.z*ap.z, d2 = ac.x*ap.x + ac.y*ap.y + ac.z*ap.z;
    if (d1 <= 0.0f && d2 <= 0.0f) return a;
    float3 bp = make_float3(p.x-b.x, p.y-b.y, p.z-b.z);
    float d3 = ab.x*bp.x + ab.y*bp.y + ab.z*bp.z, d4 = ac.x*bp.x + ac.y*bp.y + ac.z*bp.z;
    if (d3 >= 0.0f && d4 <= d3) return b;
    float vc = d1*d4 - d3*d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) { float v = d1 / (d1 - d3); return make_float3(a.x+v*ab.x, a.y+v*ab.y, a.z+v*ab.z); }
    float3 cp = make_float3(p.x-c.x, p.y-c.y, p.z-c.z);
    float d5 = ab.x*cp.x + ab.y*cp.y + ab.z*cp.z, d6 = ac.x*cp.x + ac.y*cp.y + ac.z*cp.z;
    if (d6 >= 0.0f && d5 <= d6) return c;
    float vb = d5*d2 - d1*d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) { float w = d2 / (d2 - d6); return make_float3(a.x+w*ac.x, a.y+w*ac.y, a.z+w*ac.z); }
    float va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return make_float3(b.x + w*(c.x-b.x), b.y + w*(c.y-b.y), b.z + w*(c.z-b.z));
    }
    float den = 1.0f / (va + vb + vc), v = vb * den, w = vc * den;
    return make_float3(a.x + ab.x*v + ac.x*w, a.y + ab.y*v + ac.y*w, a.z + ab.z*v + ac.z*w);
}

// Exact point-in-mesh test (closed mesh): parity of the crossings of a ray from l along a fixed,
// generic direction (not aligned with any mesh axis, so it does not run along faces or edges).
__device__ inline bool tq_inside_exact(const ToolDesc& cur, float3 l) {
    const float3 d = make_float3(0.5773f, 0.5774f, 0.5775f);
    int c = 0;
    for (int k = 0; k < cur.n_tri; k++) {
        float3 v0 = cur.tri[4*k], v1 = cur.tri[4*k+1], v2 = cur.tri[4*k+2];
        float3 e1 = make_float3(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z), e2 = make_float3(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z);
        float3 pv = make_float3(d.y*e2.z - d.z*e2.y, d.z*e2.x - d.x*e2.z, d.x*e2.y - d.y*e2.x);
        float det = e1.x*pv.x + e1.y*pv.y + e1.z*pv.z;
        if (fabsf(det) < 1e-30f) continue;
        float inv = 1.0f / det;
        float3 tv = make_float3(l.x-v0.x, l.y-v0.y, l.z-v0.z);
        float u = (tv.x*pv.x + tv.y*pv.y + tv.z*pv.z) * inv;
        if (u < 0.0f || u > 1.0f) continue;
        float3 q = make_float3(tv.y*e1.z - tv.z*e1.y, tv.z*e1.x - tv.x*e1.z, tv.x*e1.y - tv.y*e1.x);
        float v = (d.x*q.x + d.y*q.y + d.z*q.z) * inv;
        if (v < 0.0f || u + v > 1.0f) continue;
        if ((e2.x*q.x + e2.y*q.y + e2.z*q.z) * inv > 0.0f) c++;
    }
    return (c & 1) != 0;
}

__device__ inline bool tool_ccd_exact(const ToolDesc& cur, float3& p, float3 lo, float3 ln, int max_iter) {
    float3 d = make_float3(ln.x - lo.x, ln.y - lo.y, ln.z - lo.z);
    float t_entry = 2.0f; int k_entry = -1; bool exit_hit = false;
    for (int k = 0; k < cur.n_tri; k++) {
        float3 v0 = cur.tri[4*k], v1 = cur.tri[4*k+1], v2 = cur.tri[4*k+2], nk = cur.tri[4*k+3];
        float3 e1 = make_float3(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z), e2 = make_float3(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z);
        float3 pv = make_float3(d.y*e2.z - d.z*e2.y, d.z*e2.x - d.x*e2.z, d.x*e2.y - d.y*e2.x);
        float det = e1.x*pv.x + e1.y*pv.y + e1.z*pv.z;
        if (fabsf(det) < 1e-30f) continue;
        float inv = 1.0f / det;
        float3 tv = make_float3(lo.x-v0.x, lo.y-v0.y, lo.z-v0.z);
        float u = (tv.x*pv.x + tv.y*pv.y + tv.z*pv.z) * inv;
        if (u < 0.0f || u > 1.0f) continue;
        float3 q = make_float3(tv.y*e1.z - tv.z*e1.y, tv.z*e1.x - tv.x*e1.z, tv.x*e1.y - tv.y*e1.x);
        float v = (d.x*q.x + d.y*q.y + d.z*q.z) * inv;
        if (v < 0.0f || u + v > 1.0f) continue;
        float t = (e2.x*q.x + e2.y*q.y + e2.z*q.z) * inv;
        if (t < 0.0f || t > 1.0f) continue;
        if (d.x*nk.x + d.y*nk.y + d.z*nk.z < 0.0f) { if (t < t_entry) { t_entry = t; k_entry = k; } }
        else exit_hit = true;
    }
    float3 xe, n;
    if (k_entry >= 0) {
        xe = make_float3(lo.x + t_entry*d.x, lo.y + t_entry*d.y, lo.z + t_entry*d.z);
        n = cur.tri[4*k_entry+3];
    } else if (exit_hit || tq_inside_exact(cur, ln)) {     // started on the surface: its own face
        // (the common contact case: the tool moved into a particle resting on its face — the entry is
        //  at t ~ 0 and is easily missed in floating point, and the path never leaves the tool)
        float best = 1e30f; int kb = -1;
        for (int k = 0; k < cur.n_tri; k++) {
            float3 c = tq_closest_on_tri(lo, cur.tri[4*k], cur.tri[4*k+1], cur.tri[4*k+2]);
            float dd = (c.x-lo.x)*(c.x-lo.x) + (c.y-lo.y)*(c.y-lo.y) + (c.z-lo.z)*(c.z-lo.z);
            if (dd < best) { best = dd; kb = k; }
        }
        if (kb < 0) return false;
        xe = tq_closest_on_tri(lo, cur.tri[4*kb], cur.tri[4*kb+1], cur.tri[4*kb+2]);
        n = cur.tri[4*kb+3];
    } else {
        return false;                                      // the path never crosses the tool surface
    }
    // back onto the entry plane (local frame), then to world
    float dd = (ln.x - xe.x)*n.x + (ln.y - xe.y)*n.y + (ln.z - xe.z)*n.z;
    if (dd >= 0.0f) return false;
    float3 lp = make_float3(ln.x - dd*n.x, ln.y - dd*n.y, ln.z - dd*n.z);
    float3 nw = tq_rot(cur.rot, n);
    p = tq_rot(cur.rot, lp); p.x += cur.pos.x; p.y += cur.pos.y; p.z += cur.pos.z;
    for (int k = 0; k < max_iter; k++) {                   // still inside (another face): along the entry normal
        float ph = tool_query(cur, p, false).phi;
        if (ph >= 0.0f) break;
        p.x -= ph * nw.x; p.y -= ph * nw.y; p.z -= ph * nw.z;
    }
    return true;
}

// CCD tool contact (contact_ccd). The positions a particle takes within a substep are not checked
// only at the end: its path from a reference known to be outside the tool (ref, expressed in tool
// pose ref_t) to its current position p (pose cur) is traced in the tool local frame. If the path
// enters the tool — whether p ends inside or already beyond the far face — p is put back on the
// tangent plane at the entry point and, if still inside, pushed along the entry normal: the particle
// stays on the side it came from. This is the PBD collision constraint of Mueller et al. 2007 (ray
// x -> p, entry point q_c, normal n_c, C = (p - q_c) . n_c >= 0), evaluated against every update
// (tool motion between substeps, and each shape-matching apply inside the solver loop: one apply
// moved particles 7.7 mm, clean over a 6.3 mm blade, which an end-position test cannot see).
// Tracing: sphere tracing on phi (1-Lipschitz), step max(phi, eps), eps = half the smallest SDF cell
// (the representation's resolution); a reference within eps of the surface counts as outside (the
// previous contact left it at phi ~ 0). A reference deeper than eps inside has no side: nearest face.
__device__ inline bool tool_project_out_ccd(const ToolDesc& cur, const ToolDesc& ref_t, float3& p, float3 ref,
                                            int max_iter) {
    if (cur.type != 1) return tool_project_out_iter(cur, p, max_iter);
    const float eps = 0.5f * fminf(cur.cell.x, fminf(cur.cell.y, cur.cell.z));
    float3 ln = tq_rot_T(cur.rot,   make_float3(p.x - cur.pos.x,   p.y - cur.pos.y,   p.z - cur.pos.z));
    float3 lo = tq_rot_T(ref_t.rot, make_float3(ref.x - ref_t.pos.x, ref.y - ref_t.pos.y, ref.z - ref_t.pos.z));
    float3 dl = make_float3(ln.x - lo.x, ln.y - lo.y, ln.z - lo.z);
    float L = sqrtf(dl.x*dl.x + dl.y*dl.y + dl.z*dl.z);
    float phi_n = tq_phi_local(cur, ln);
    // prefilter: the grid phi can overestimate the true distance by up to half a cell diagonal (a blade
    // thinner than a cell is not in the grid at all), so the exact test must not be skipped within it
    const float err = cur.tri ? 0.5f * sqrtf(cur.cell.x*cur.cell.x + cur.cell.y*cur.cell.y + cur.cell.z*cur.cell.z) : eps;
    if (phi_n > L + err) return false;                     // no point of the path can reach the tool
    float phi_o = tq_phi_local(cur, lo);
    if (phi_o < -eps) return phi_n < 0.0f ? tool_project_out_iter(cur, p, max_iter) : false;   // no side known
    if (cur.tri && cur.n_tri > 0) return tool_ccd_exact(cur, p, lo, ln, max_iter);
    // first entry along lo -> ln
    float t_in = -1.0f, t_out = 0.0f;                      // bracket: phi(t_out) >= 0, phi(t_in) < 0
    if (phi_o <= 0.0f) { t_in = 0.0f; }                    // on the surface: entry = the reference itself
    else if (L > 1e-12f) {
        float t = 0.0f, ph = phi_o;
        while (t < 1.0f) {
            float tn = fminf(1.0f, t + fmaxf(ph, eps) / L);
            float3 lm = make_float3(lo.x + tn*dl.x, lo.y + tn*dl.y, lo.z + tn*dl.z);
            float pn = tq_phi_local(cur, lm);
            if (pn < 0.0f) { t_out = t; t_in = tn; break; }
            t = tn; ph = pn;
        }
        if (t_in > 0.0f) {
            for (int it = 0; it < 16; it++) {
                float m = 0.5f * (t_out + t_in);
                float3 lm = make_float3(lo.x + m*dl.x, lo.y + m*dl.y, lo.z + m*dl.z);
                if (tq_phi_local(cur, lm) >= 0.0f) t_out = m; else t_in = m;
            }
        }
    }
    if (t_in < 0.0f) return false;                         // the path never entered the tool
    float te = (phi_o <= 0.0f) ? 0.0f : t_out;             // entry point on the outside of the surface
    float3 ls = make_float3(lo.x + te*dl.x, lo.y + te*dl.y, lo.z + te*dl.z);
    float3 xs = tq_rot(cur.rot, ls); xs.x += cur.pos.x; xs.y += cur.pos.y; xs.z += cur.pos.z;
    float3 n = tool_query(cur, xs, true).n;
    if (n.x == 0.0f && n.y == 0.0f && n.z == 0.0f) return phi_n < 0.0f ? tool_project_out_iter(cur, p, max_iter) : false;
    float d = (p.x - xs.x)*n.x + (p.y - xs.y)*n.y + (p.z - xs.z)*n.z;      // < 0: behind the entry plane
    bool moved = false;
    if (d < 0.0f) { p.x -= d * n.x; p.y -= d * n.y; p.z -= d * n.z; moved = true; }
    for (int k = 0; k < max_iter; k++) {                   // still inside: along the ENTRY normal only
        float ph = tool_query(cur, p, false).phi;
        if (ph >= 0.0f) break;
        p.x -= ph * n.x; p.y -= ph * n.y; p.z -= ph * n.z; moved = true;
    }
    return moved;
}

// R3 detour test (straddle_detour). R3 loads an edge the tool body SEPARATES. The proxy "segment
// crosses the tool and the endpoint normals oppose" also holds for an edge that only clips a convex
// corner or the outline of the tool (blade ends, spine, cutting edge): there the material goes around
// the tool, nothing is separated (measured: 100% / 94% of the R3 breaks left under a sideways push).
// Separation is decided on the P(F) element of the edge, its minimum ball (centre = midpoint, radius
// r_e): the endpoints a, b are separated only if the tool blocks every detour a -> c -> b through a
// point c of the ball's boundary sphere. K = the number of sphere samples: a resolution, not a
// threshold (results must not depend on it).
// A detour leg is blocked if it passes through the tool's INTERIOR. The legs start at the edge's own
// particles, which rest ON the tool surface after contact: their own face (an exit crossing at t ~ 0
// when the leg leaves outward) and their inside/outside status (a coin flip at phi = 0) must not count.
// Blocked = an entry crossing (motion against the face normal), or an exit crossing clearly after the
// start (the leg went inward and the entry at t ~ 0 was missed in floating point), or the detour point
// itself inside (check_q: q is the detour point c; edge endpoints are never tested).
__device__ inline bool tq_segment_blocked(const ToolDesc& t, float3 p, float3 q, float h, float eps, bool check_q) {
    if (t.type == 1 && t.tri && t.n_tri > 0) {
        float3 lp = tq_rot_T(t.rot, make_float3(p.x - t.pos.x, p.y - t.pos.y, p.z - t.pos.z));
        float3 lq = tq_rot_T(t.rot, make_float3(q.x - t.pos.x, q.y - t.pos.y, q.z - t.pos.z));
        if (check_q && tq_inside_exact(t, lq)) return true;
        float3 d = make_float3(lq.x - lp.x, lq.y - lp.y, lq.z - lp.z);
        float L = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
        const float t_tol = L > 1e-12f ? (1e-3f * eps) / L : 1.0f;   // floating-point tolerance at the start
        for (int k = 0; k < t.n_tri; k++) {
            float3 v0 = t.tri[4*k], v1 = t.tri[4*k+1], v2 = t.tri[4*k+2], nk = t.tri[4*k+3];
            float3 e1 = make_float3(v1.x-v0.x, v1.y-v0.y, v1.z-v0.z), e2 = make_float3(v2.x-v0.x, v2.y-v0.y, v2.z-v0.z);
            float3 pv = make_float3(d.y*e2.z - d.z*e2.y, d.z*e2.x - d.x*e2.z, d.x*e2.y - d.y*e2.x);
            float det = e1.x*pv.x + e1.y*pv.y + e1.z*pv.z;
            if (fabsf(det) < 1e-30f) continue;
            float inv = 1.0f / det;
            float3 tv = make_float3(lp.x-v0.x, lp.y-v0.y, lp.z-v0.z);
            float u = (tv.x*pv.x + tv.y*pv.y + tv.z*pv.z) * inv;
            if (u < 0.0f || u > 1.0f) continue;
            float3 qq = make_float3(tv.y*e1.z - tv.z*e1.y, tv.z*e1.x - tv.x*e1.z, tv.x*e1.y - tv.y*e1.x);
            float v = (d.x*qq.x + d.y*qq.y + d.z*qq.z) * inv;
            if (v < 0.0f || u + v > 1.0f) continue;
            float tt = (e2.x*qq.x + e2.y*qq.y + e2.z*qq.z) * inv;
            if (tt < 0.0f || tt > 1.0f) continue;
            bool entering = d.x*nk.x + d.y*nk.y + d.z*nk.z < 0.0f;
            if (entering || tt > t_tol) return true;
        }
        return false;
    }
    float L = sqrtf((q.x-p.x)*(q.x-p.x) + (q.y-p.y)*(q.y-p.y) + (q.z-p.z)*(q.z-p.z));   // SDF: sampled
    int n = (int)ceilf(L / fmaxf(h, 1e-9f)); n = n < 2 ? 2 : (n > 64 ? 64 : n);
    for (int k = 1; k <= n; k++) {                         // the start (an edge endpoint) is not tested
        if (k == n && !check_q) break;
        float s = (float)k / n;
        if (tool_query(t, make_float3(p.x + s*(q.x-p.x), p.y + s*(q.y-p.y), p.z + s*(q.z-p.z)), false).phi < -eps) return true;
    }
    return false;
}

__device__ inline bool tool_separates_element(const ToolDesc& t, float3 a, float3 b, int K, float h, float eps) {
    // Detour points c on the whole boundary sphere of the element (Fibonacci sphere, K points); both
    // chords a -> c, c -> b lie inside the ball. (Sampling only the equator normal to ab missed the
    // detour around a blade END lying along ab: measured 31 -> 29 breaks only.)
    float L = sqrtf((b.x-a.x)*(b.x-a.x) + (b.y-a.y)*(b.y-a.y) + (b.z-a.z)*(b.z-a.z));
    if (L < 1e-12f) return false;
    float3 m = make_float3(0.5f*(a.x+b.x), 0.5f*(a.y+b.y), 0.5f*(a.z+b.z));
    float re = 0.5f * L;
    const float ga = 2.39996323f;                          // golden angle
    for (int k = 0; k < K; k++) {
        float zc = 1.0f - 2.0f * (k + 0.5f) / K, rr = sqrtf(fmaxf(0.0f, 1.0f - zc*zc)), th = ga * k;
        float3 c = make_float3(m.x + re*rr*cosf(th), m.y + re*rr*sinf(th), m.z + re*zc);
        if (!tq_segment_blocked(t, a, c, h, eps, true) && !tq_segment_blocked(t, b, c, h, eps, true)) return false;   // a free detour
    }
    return true;
}

// Straddle rule "sheet" (core/tool_features.h): the tool cuts the element of edge (pa, pb) — its
// minimum ball B — in two iff, for some sheet, pa and pb lie on opposite sides of the sheet's plane
// and the disk B ∩ plane lies inside the sheet's cross-section (centre inside by ray parity, distance
// to every boundary segment >= the disk radius). Exact for any cross-section polygon (non-convex,
// several loops); no sampling, no tolerance. sdist = distance from the element centre to the sheet
// minus r_e (< 0: the P(F) intruder distance of a 2D intruder through the element), most negative
// over the separating sheets.
__device__ inline bool tool_sheet_separates(const ToolDesc& t, float3 pa, float3 pb, float& sdist) {
    if (t.n_sheet == 0) return false;
    float3 la = tq_rot_T(t.rot, make_float3(pa.x - t.pos.x, pa.y - t.pos.y, pa.z - t.pos.z));
    float3 lb = tq_rot_T(t.rot, make_float3(pb.x - t.pos.x, pb.y - t.pos.y, pb.z - t.pos.z));
    float3 c = make_float3(0.5f * (la.x + lb.x), 0.5f * (la.y + lb.y), 0.5f * (la.z + lb.z));
    float ex = la.x - lb.x, ey = la.y - lb.y, ez = la.z - lb.z;
    float r2 = 0.25f * (ex*ex + ey*ey + ez*ez);
    bool sep = false;
    for (int k = 0; k < t.n_sheet; k++) {
        const ToolSheetGPU& s = t.sheet[k];
        float sa = s.m.x*(la.x - s.o.x) + s.m.y*(la.y - s.o.y) + s.m.z*(la.z - s.o.z);
        float sb = s.m.x*(lb.x - s.o.x) + s.m.y*(lb.y - s.o.y) + s.m.z*(lb.z - s.o.z);
        if (!((sa < 0.0f && sb > 0.0f) || (sa > 0.0f && sb < 0.0f))) continue;
        float3 w = make_float3(c.x - s.o.x, c.y - s.o.y, c.z - s.o.z);
        float dc = s.m.x*w.x + s.m.y*w.y + s.m.z*w.z;
        float rho2 = r2 - dc*dc;                         // > 0: the plane passes between a and b
        float cu = s.u.x*w.x + s.u.y*w.y + s.u.z*w.z;
        float cv = s.v.x*w.x + s.v.y*w.y + s.v.z*w.z;
        bool inside = false, clear = true;
        for (int i = s.seg0; i < s.seg0 + s.nseg; i++) {
            float4 e = t.sheet_seg[i];
            if ((e.y > cv) != (e.w > cv) && cu < e.x + (cv - e.y) * (e.z - e.x) / (e.w - e.y)) inside = !inside;
            float du = e.z - e.x, dv = e.w - e.y;        // squared distance centre -> segment
            float L2 = du*du + dv*dv;
            float q = L2 > 0.0f ? fminf(fmaxf(((cu - e.x)*du + (cv - e.y)*dv) / L2, 0.0f), 1.0f) : 0.0f;
            float pu = e.x + q*du - cu, pv = e.y + q*dv - cv;
            if (pu*pu + pv*pv < rho2) { clear = false; break; }
        }
        if (!inside || !clear) continue;
        float sd = fabsf(dc) - sqrtf(r2);
        sdist = sep ? fminf(sdist, sd) : sd;
        sep = true;
    }
    return sep;
}
