#pragma once
#include <cuda_runtime.h>
// Diagnostics (tool_bench --cross-log): one record per edge loaded by a CUT_CURVE crossing,
// endpoints in the tool local frame before (a0, b0) and after (a1, b1) the sweep.
struct CrossRec { int eid; int seg; float s; float sdist; float3 a0, b0, a1, b1; };
