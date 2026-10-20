#pragma once
// Tool functional channels (Step 3). A tool = geometry (ToolDesc, tool_query.cuh) + channels:
//
//   contact channel   the whole tool solid pushes tissue out          (collision kernels)
//   cut channel       an INTRUDER SET with a distance function d(x);  (tool_damage_kernel)
//                     every cut channel uses the same P(F) model
//                         pf = sigmoid(alpha * sdist / r_e),  sdist = d(mid) - r_e  (sdist < 0 → pf < 1/2)
//                     (mid / r_e = edge midpoint / half-length)
//   aspirate channel  removes damaged particles the tool contains     (aspirate_kernel)
//
// Cut-channel intruder sets:
//   CUT_SURFACE  the tool solid itself, d = phi (sphere: |x-c| - R; grid SDF: trilinear phi).
//                Used by the sphere / CUSA tip and by the legacy volumetric mesh-SDF damage.
//                subtract_re = 0 reproduces the legacy grid kernel (sdist = phi, no -r_e).
//   CUT_CURVE    sharp features in the tool LOCAL frame: line segments (blade edge, curette rim) and
//                points (needle / trocar tip, a segment with equal endpoints); d = distance to the
//                nearest feature (core/tool_features.h extracts them from the tool mesh).
//
// Each set keeps the float operations of the kernel it replaces (sphere update_edge_damage_kernel,
// tool_sdf_damage_kernel, sharp_edge_damage_kernel), including its search band:
//   sphere: |mid-c| > 3R;  grid: |phi| > 3 r_e;  curve: d > 3 r_e.

#include "tool_query.cuh"
#include "cross_rec.h"

enum CutChannelKind { CUT_SURFACE = 0, CUT_CURVE = 1 };


struct CutChannel {
    int kind;
    // CUT_SURFACE
    ToolDesc tool;
    int subtract_re;          // 1: sdist = phi - r_e; 0: legacy grid sdist = phi
    // CUT_CURVE (tool frame = tool.pos / tool.rot, local -> world, row-major)
    const float3* seg;        // seg[2k], seg[2k+1] = endpoints of segment k, tool local frame
    int num_seg;
    // CUT_CURVE crossing mode (cut_curve: crossing): tool pose and particle positions at the
    // previous damage evaluation; prev_pos == nullptr → proximity (legacy)
    ToolDesc prev_tool;
    const float3* prev_pos;
    CrossRec* log; int* log_n; int log_cap;   // optional diagnostics (nullptr = off)
    const float3* nrm;        // leading mode: nrm[2k], nrm[2k+1] = face normals of feature k (local); nullptr = off
};

// Leading test (cut_curve: leading). A sharp edge separates the tissue only when IT meets the tissue
// first, i.e. when the tool's motion relative to the tissue, u, lies in the edge's normal cone (the
// directions whose support point on the convex wedge is the edge): u_perp = a n1 + b n2, a, b >= 0,
// u_perp = u without its component along the edge (sliding along the edge does not change which
// part leads). A blade pushed sideways meets the tissue with its face, moved spine-first with its
// back; a particle that slipped through the blade body crosses the edge line with the face leading.
// For a thin wedge the cone is almost the whole half-space below the edge, so oblique slicing cuts.
__device__ inline bool cut_feature_leads(float3 n1, float3 n2, float3 u) {
    float3 m = make_float3(n1.y*n2.z - n1.z*n2.y, n1.z*n2.x - n1.x*n2.z, n1.x*n2.y - n1.y*n2.x);  // along the edge
    float mm = m.x*m.x + m.y*m.y + m.z*m.z;
    if (mm < 1e-12f) return true;                         // degenerate wedge: no cone information
    float3 c2 = make_float3(u.y*n2.z - u.z*n2.y, u.z*n2.x - u.x*n2.z, u.x*n2.y - u.y*n2.x);   // u x n2
    float3 c1 = make_float3(n1.y*u.z - n1.z*u.y, n1.z*u.x - n1.x*u.z, n1.x*u.y - n1.y*u.x);   // n1 x u
    float a = (c2.x*m.x + c2.y*m.y + c2.z*m.z) / mm;      // u_perp = a n1 + b n2 (component along m drops out)
    float b = (c1.x*m.x + c1.y*m.y + c1.z*m.z) / mm;
    return a >= 0.0f && b >= 0.0f;
}

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
        float t = 0.0f;                                     // point feature (A == B): distance to A
        if (ab_dot_ab >= 1e-12f) {
            t = (am.x*ab.x + am.y*ab.y + am.z*ab.z) / ab_dot_ab;
            if (t < 0.0f) t = 0.0f;
            if (t > 1.0f) t = 1.0f;
        }
        float3 cl = make_float3(A.x + t * ab.x, A.y + t * ab.y, A.z + t * ab.z);
        float dx = m.x - cl.x, dy = m.y - cl.y, dz = m.z - cl.z;
        best = fminf(best, sqrtf(dx*dx + dy*dy + dz*dz + 1e-12f));
    }
    if (best > 1e29f) return false;                         // no feature
    if (best > r_e * 3.0f) return false;
    sdist = best - r_e;
    return true;
}

// CUT_CURVE, crossing mode. A sharp edge cuts a tissue edge when it passes BETWEEN the two
// particles, not when it comes near the edge's midpoint: with the proximity rule every particle the
// blade edge passes through has all its edges' midpoints within r_e of the line, so all of them broke
// and the particle was left with no shape constraint ("sand"). In the tool local frame the feature
// segment AB is fixed; since the previous damage evaluation the tissue edge moved from (a0,b0) to
// (a1,b1), sweeping the patch a0-b0-b1-a1 (two triangles). AB intersecting the patch = the feature
// crossed the edge (tool motion and tissue motion alike). The intrusion is measured where it
// happened: x = the crossing point on the edge, sdist = |x - mid| - r_e (same P(F) element as the
// proximity rule, which measures the CLOSEST approach whether or not the feature crosses).
// Point features (A == B) cannot cross a segment; they keep the proximity rule.
__device__ inline bool cc_seg_tri(float3 A, float3 B, float3 v0, float3 v1, float3 v2, float& bu, float& bv) {
    float3 dir = make_float3(B.x - A.x, B.y - A.y, B.z - A.z);
    float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 p = make_float3(dir.y*e2.z - dir.z*e2.y, dir.z*e2.x - dir.x*e2.z, dir.x*e2.y - dir.y*e2.x);
    float det = e1.x*p.x + e1.y*p.y + e1.z*p.z;
    if (fabsf(det) < 1e-24f) return false;                 // degenerate patch (no relative motion) or parallel
    float inv = 1.0f / det;
    float3 tv = make_float3(A.x - v0.x, A.y - v0.y, A.z - v0.z);
    float u = (tv.x*p.x + tv.y*p.y + tv.z*p.z) * inv;
    if (u < 0.0f || u > 1.0f) return false;
    float3 q = make_float3(tv.y*e1.z - tv.z*e1.y, tv.z*e1.x - tv.x*e1.z, tv.x*e1.y - tv.y*e1.x);
    float v = (dir.x*q.x + dir.y*q.y + dir.z*q.z) * inv;
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = (e2.x*q.x + e2.y*q.y + e2.z*q.z) * inv;      // along AB
    if (t < 0.0f || t > 1.0f) return false;
    bu = u; bv = v;
    return true;
}

__device__ inline bool cut_curve_crossing_sdist(const CutChannel& c, float3 pa0, float3 pb0, float3 pa1, float3 pb1,
                                                float r_e, float& sdist, int eid = -1) {
    float3 a1 = tq_rot_T(c.tool.rot, make_float3(pa1.x - c.tool.pos.x, pa1.y - c.tool.pos.y, pa1.z - c.tool.pos.z));
    float3 b1 = tq_rot_T(c.tool.rot, make_float3(pb1.x - c.tool.pos.x, pb1.y - c.tool.pos.y, pb1.z - c.tool.pos.z));
    float3 a0 = tq_rot_T(c.prev_tool.rot, make_float3(pa0.x - c.prev_tool.pos.x, pa0.y - c.prev_tool.pos.y, pa0.z - c.prev_tool.pos.z));
    float3 b0 = tq_rot_T(c.prev_tool.rot, make_float3(pb0.x - c.prev_tool.pos.x, pb0.y - c.prev_tool.pos.y, pb0.z - c.prev_tool.pos.z));
    float3 m = make_float3(0.5f*(a1.x + b1.x), 0.5f*(a1.y + b1.y), 0.5f*(a1.z + b1.z));
    float mva = sqrtf((a1.x-a0.x)*(a1.x-a0.x) + (a1.y-a0.y)*(a1.y-a0.y) + (a1.z-a0.z)*(a1.z-a0.z));
    float mvb = sqrtf((b1.x-b0.x)*(b1.x-b0.x) + (b1.y-b0.y)*(b1.y-b0.y) + (b1.z-b0.z)*(b1.z-b0.z));
    float reach = r_e + fmaxf(mva, mvb);                   // every patch point is within this of m
    bool found = false; float best = 1e30f; float best_s = -1.0f; int best_k = -1;
    for (int k = 0; k < c.num_seg; k++) {
        float3 A = c.seg[2 * k], B = c.seg[2 * k + 1];
        float3 ab = make_float3(B.x - A.x, B.y - A.y, B.z - A.z);
        float3 am = make_float3(m.x - A.x, m.y - A.y, m.z - A.z);
        float L2 = ab.x*ab.x + ab.y*ab.y + ab.z*ab.z;
        float t = 0.0f;
        if (L2 >= 1e-12f) { t = (am.x*ab.x + am.y*ab.y + am.z*ab.z) / L2; t = fminf(fmaxf(t, 0.0f), 1.0f); }
        float dx = m.x - (A.x + t*ab.x), dy = m.y - (A.y + t*ab.y), dz = m.z - (A.z + t*ab.z);
        float d = sqrtf(dx*dx + dy*dy + dz*dz + 1e-12f);
        if (L2 < 1e-12f) {                                  // point feature: proximity rule
            if (d <= r_e * 3.0f) { best = fminf(best, d - r_e); found = true; }
            continue;
        }
        if (d > reach) continue;                            // the patch cannot reach this feature
        float u, v;
        for (int tri = 0; tri < 2; tri++) {
            bool hit = tri == 0 ? cc_seg_tri(A, B, a0, b0, b1, u, v)    // patch triangle a0 b0 b1: s = u + v
                                : cc_seg_tri(A, B, a0, b1, a1, u, v);   // patch triangle a0 b1 a1: s = u
            if (!hit) continue;
            float s = tri == 0 ? u + v : u;
            if (c.nrm) {                                    // leading mode: tool motion rel. tissue at x
                float3 p0 = make_float3(a0.x + s*(b0.x - a0.x), a0.y + s*(b0.y - a0.y), a0.z + s*(b0.z - a0.z));
                float3 p1 = make_float3(a1.x + s*(b1.x - a1.x), a1.y + s*(b1.y - a1.y), a1.z + s*(b1.z - a1.z));
                if (!cut_feature_leads(c.nrm[2 * k], c.nrm[2 * k + 1],
                                       make_float3(p0.x - p1.x, p0.y - p1.y, p0.z - p1.z))) {
                    if (c.log) {                            // diagnostics: geometric crossing rejected (seg = -2 - k)
                        int i = atomicAdd(c.log_n, 1);
                        if (i < c.log_cap) c.log[i] = CrossRec{eid, -2 - k, s, 0.0f, a0, b0, a1, b1};
                    }
                    continue;
                }
            }
            float sd = r_e * fabsf(2.0f * s - 1.0f) - r_e;
            if (sd < best) { best = sd; best_s = s; best_k = k; } found = true;
        }
    }
    if (!found) return false;
    sdist = best;
    if (c.log && best_k >= 0) {
        int i = atomicAdd(c.log_n, 1);
        if (i < c.log_cap) c.log[i] = CrossRec{eid, best_k, best_s, best, a0, b0, a1, b1};
    }
    return true;
}
