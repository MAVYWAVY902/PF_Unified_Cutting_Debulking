#pragma once
// Step 1 — unified tool query for every tool-dependent GPU kernel.
//
// A tool exposes four device-side operations. Kernels call these instead of doing their own
// tool geometry, so a new tool representation (Step 2: grid SDF) is added in ONE place.
//
//   tool_query(t, p)                  signed distance phi (<0 inside) + outward unit normal n
//   tool_project_out(t, p)            if p is inside, move it to the closest surface point
//   tool_contains(t, p)               closed inside test (surface counts as inside)
//   tool_outside_damage_band(t, p)    true = too far from the tool for edge damage to matter
//
// Generic definitions (what an SDF tool will use): project = p - phi*n, contains = phi <= 0,
// band = phi > b. The analytic sphere keeps its CLOSED-FORM versions of the first three,
// with exactly the float operations the pre-Step-1 kernels used. This is deliberate, not
// cosmetic: Step 0 showed that 100% of aspirations happen to particles the collision has just
// projected onto the sphere (|d-R|/R < 1e-6), i.e. removal is decided by float rounding of
// "project, then |d|^2 <= R^2". A mathematically equivalent but differently rounded projection
// would flip those particles to "1 ulp outside" and change removal counts. Removing that
// rounding dependence is a separate, switchable fix (later step), not part of this refactor.
//
// Step 1 implements type 0 (analytic sphere) only. Mesh-SDF tools (type 1) still go through
// their own kernels (tool_sdf_*), unchanged, until Step 2.

#include <cuda_runtime.h>

struct ToolDesc {
    int    type;     // 0 = analytic sphere (only type handled here in Step 1)
    float3 pos;      // tool frame origin in world (sphere: centre)
    float  radius;   // sphere radius
};

struct ToolSample {
    float  phi;      // signed distance to the tool surface (<0 inside)
    float3 n;        // outward unit normal (zero if undefined, e.g. at the centre)
};

__host__ __device__ inline ToolDesc make_sphere_tool(float3 c, float r) {
    ToolDesc t; t.type = 0; t.pos = c; t.radius = r; return t;
}

__device__ inline ToolSample tool_query(const ToolDesc& t, float3 p) {
    float dx = p.x - t.pos.x, dy = p.y - t.pos.y, dz = p.z - t.pos.z;
    float d = sqrtf(dx*dx + dy*dy + dz*dz);
    ToolSample s;
    s.phi = d - t.radius;                      // same expression as the legacy damage kernel
    if (d > 1e-10f) { float inv = 1.0f / d; s.n = make_float3(dx * inv, dy * inv, dz * inv); }
    else            { s.n = make_float3(0.0f, 0.0f, 0.0f); }
    return s;
}

// Returns true if p was moved. Sphere: radial projection c + (p-c)*R/|p-c| (legacy formula).
__device__ inline bool tool_project_out(const ToolDesc& t, float3& p) {
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

// Sphere: |p-c|^2 <= R^2 without a sqrt (legacy aspirate test; surface points are inside).
__device__ inline bool tool_contains(const ToolDesc& t, float3 p) {
    float dx = p.x - t.pos.x, dy = p.y - t.pos.y, dz = p.z - t.pos.z;
    return !(dx*dx + dy*dy + dz*dz > t.radius * t.radius);
}

// Sphere: legacy edge-damage search radius 3R around the centre (= band 2R beyond the surface).
__device__ inline bool tool_outside_damage_band(const ToolDesc& t, float3 p) {
    float dx = p.x - t.pos.x, dy = p.y - t.pos.y, dz = p.z - t.pos.z;
    return sqrtf(dx*dx + dy*dy + dz*dz) > t.radius * 3.0f;
}
