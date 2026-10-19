#pragma once
// Tool functional channels (Step 3). A tool = geometry (ToolDesc, tool_query.cuh) + channels:
//
//   contact channel   the whole tool solid pushes tissue out          (collision kernels)
//   cut channel       an INTRUDER SET with a distance function d(x);  (tool_damage_kernel)
//                     every cut channel uses the same P(F) model
//                         pf = sigmoid(-alpha * sdist / r_e),  sdist = d(mid) - r_e
//                     (mid / r_e = edge midpoint / half-length)
//   aspirate channel  removes damaged particles the tool contains     (aspirate_kernel)
//
// Cut-channel intruder sets:
//   CUT_SURFACE  the tool solid itself, d = phi (sphere: |x-c| - R; grid SDF: trilinear phi).
//                Used by the sphere / CUSA tip and by the legacy volumetric mesh-SDF damage.
//                subtract_re = 0 reproduces the legacy grid kernel (sdist = phi, no -r_e).
//   CUT_CURVE    sharp features as line segments in the tool LOCAL frame (blade edge, curette rim,
//                needle tip = a degenerate polyline), d = distance to the nearest segment.
//
// Each set keeps the float operations of the kernel it replaces (sphere update_edge_damage_kernel,
// tool_sdf_damage_kernel, sharp_edge_damage_kernel), including its search band:
//   sphere: |mid-c| > 3R;  grid: |phi| > 3 r_e;  curve: d > 3 r_e.

#include "tool_query.cuh"

enum CutChannelKind { CUT_SURFACE = 0, CUT_CURVE = 1 };

struct CutChannel {
    int kind;
    // CUT_SURFACE
    ToolDesc tool;
    int subtract_re;          // 1: sdist = phi - r_e; 0: legacy grid sdist = phi
    // CUT_CURVE (tool frame = tool.pos / tool.rot, local -> world, row-major)
    const float3* seg;        // seg[2k], seg[2k+1] = endpoints of segment k, tool local frame
    int num_seg;
};

// Signed intruder distance sdist of an edge (midpoint mid, half-length r_e), or false if the edge
// is outside the channel's search band (or the channel has no valid geometry).
__device__ inline bool cut_channel_sdist(const CutChannel& c, float3 mid, float r_e, float& sdist) {
    if (c.kind == CUT_SURFACE) {
        if (c.tool.type == 0) {
            if (tool_outside_damage_band(c.tool, mid, 0.0f)) return false;   // r_e unused for the sphere
            sdist = tool_query(c.tool, mid).phi - r_e;
            return true;
        }
        if (tool_outside_damage_band(c.tool, mid, r_e)) return false;
        float phi = tool_query(c.tool, mid, false).phi;
        sdist = c.subtract_re ? (phi - r_e) : phi;
        return true;
    }
    // CUT_CURVE: midpoint in the tool local frame, distance to the nearest segment
    float3 w = make_float3(mid.x - c.tool.pos.x, mid.y - c.tool.pos.y, mid.z - c.tool.pos.z);
    float3 m = tq_rot_T(c.tool.rot, w);
    float best = 1e30f;
    for (int k = 0; k < c.num_seg; k++) {
        float3 A = c.seg[2 * k], B = c.seg[2 * k + 1];
        float3 ab = make_float3(B.x - A.x, B.y - A.y, B.z - A.z);
        float3 am = make_float3(m.x - A.x, m.y - A.y, m.z - A.z);
        float ab_dot_ab = ab.x*ab.x + ab.y*ab.y + ab.z*ab.z;
        if (ab_dot_ab < 1e-12f) continue;                   // degenerate segment
        float t = (am.x*ab.x + am.y*ab.y + am.z*ab.z) / ab_dot_ab;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        float3 cl = make_float3(A.x + t * ab.x, A.y + t * ab.y, A.z + t * ab.z);
        float dx = m.x - cl.x, dy = m.y - cl.y, dz = m.z - cl.z;
        best = fminf(best, sqrtf(dx*dx + dy*dy + dz*dz + 1e-12f));
    }
    if (best > 1e29f) return false;                         // no valid segment
    if (best > r_e * 3.0f) return false;
    sdist = best - r_e;
    return true;
}
