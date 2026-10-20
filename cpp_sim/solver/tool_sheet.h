#pragma once
// straddle_rule sheet: one 2D intruder of the tool (core/tool_features.h ToolSheet), tool local frame.
#include <cuda_runtime.h>

struct ToolSheetGPU {
    float3 o, m, u, v;    // plane point, unit normal, in-plane basis
    int    seg0, nseg;    // range of its cross-section boundary segments in ToolDesc::sheet_seg
};
