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
__device__ inline float3 tq_rot_T(const float* R, float3 v) {   // R^T v  (== apply_rot_transpose)
    return make_float3(R[0]*v.x + R[3]*v.y + R[6]*v.z,
                       R[1]*v.x + R[4]*v.y + R[7]*v.z,
                       R[2]*v.x + R[5]*v.y + R[8]*v.z);
}
__device__ inline float3 tq_rot(const float* R, float3 v) {     // R v    (== apply_rot)
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
