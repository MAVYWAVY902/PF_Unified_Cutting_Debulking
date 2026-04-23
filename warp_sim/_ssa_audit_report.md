# SSA Silent-Zero Audit — remaining solver kernels
**Date:** 2026-04-22
**Scope:** Kernels NOT covered by Phase 1-4 SSA rewrites
**Goal:** classify each as 🟢 GREEN (already safe), 🟡 YELLOW (minor fix), 🔴 RED (major rewrite)

## Criteria

Under Warp 1.12, the following patterns silently return zero gradient:
1. **Local variable re-assignment**: `v = X ... v = Y` (use fresh names + `wp.where` instead)
2. **Conditional array write**: `if cond: arr[i] = X` (use `wp.where` + always-write, or delta buffer)
3. **Read-modify-write on arrays**: `arr[i] = arr[i] + X` (flagged by `verify_autograd_array_access=True`; use `wp.atomic_add`)
4. **`= non_additive_op`**: `v = v / w`, `v = v * w` (NOT covered by the `+=`/`-=` exemption)

Already rewritten (not audited here — confirmed GREEN by cosine=1.000 gate tests):
- `collision_kernel_ssa` + `apply_collision_delta_kernel`
- `collision_kernel_batched_ssa`
- `update_edge_damage_kernel_ssa` + `apply_edge_damage_delta_kernel`
- `update_edge_damage_kernel_batched_ssa`

## Audit results

| # | Kernel | File:line | Status | Issues found | Work |
|---|---|---|---|---|---|
| 1 | `predict_kernel` | kernels.py:9-31 | 🔴 **RED** | `v = v * damping` re-assigns local; `old_pos[i] = pos[i]` written in both branches (conditional array write) | SSA rewrite, ~1 day |
| 2 | `clear_deltas_kernel` | kernels.py:137-144 | 🟢 GREEN | Pure constant writes, no grad concern | None |
| 3 | `cluster_solve_kernel` | kernels.py:149-283 | 🔴 **RED** (+ SVD) | Many `cx = cx / tw` type re-assigns; `R = ...` re-assigned in det<0 branch; SVD backward unstable | **Full rewrite + Müller 2016 port, ~1-2 weeks** |
| 4 | `apply_kernel` | kernels.py:288-305 | 🟡 YELLOW | `pred_pos[i] = pred_pos[i] + ...` — read-modify-write on array | Swap to `wp.atomic_add` via delta kernel, ~2 hrs |
| 5 | `update_velocity_kernel` | kernels.py:310-325 | 🟢 GREEN | All single-assignment writes, no R-M-W | None |
| 6 | `update_particle_damage_kernel` | pf_damage.py:168-194 | 🟢 GREEN | All `wp.atomic_add`, SSA-clean | None |
| 7 | `finalize_particle_damage_kernel` | pf_damage.py:197-216 | 🟡 YELLOW | `particle_damage[i] = particle_damage[i] + ...` — R-M-W | Use delta buffer + atomic_add apply, ~3 hrs |
| 8 | `update_soft_removal_kernel` | pf_damage.py:221-241 | 🟢 GREEN | Single-assignment write to `alpha[i]` | None |

## Detailed findings

### 🔴 RED: `predict_kernel` — the smoking gun

Lines 28-30:
```python
v = vel[i] + gravity * dt
v = v * damping         # ← second assignment of v, silent-zero landmine
vel[i] = v
```

This is the **exact same pattern** as the original `collision_kernel` (re-assigning `p` three times) that caused the original zero-gradient bug we debugged last session. Under Warp 1.12 SSA rule, the `v = v * damping` invalidates any backward flowing through `vel[i]` or `pred_pos[i]` (which reads `v * dt`).

Also:
```python
if alpha[i] < 0.01 or fixed[i] == 1:
    old_pos[i] = pos[i]      # ← write in branch A
    pred_pos[i] = pos[i]
    return

old_pos[i] = pos[i]          # ← write in branch B (same value!)
```
Branch-dependent array writes. Warp can sometimes handle these (both write same value), but safer to unconditionally write once outside branches.

**Fix pattern**: `v_raw = vel[i] + gravity * dt; v_damped = v_raw * damping; vel[i] = wp.where(alive, v_damped, vel[i]); pred_pos[i] = wp.where(alive, pos[i] + v_damped * dt, pos[i]); old_pos[i] = pos[i]` (always write old_pos).

### 🔴 RED: `cluster_solve_kernel` — the big one

Multiple re-assignments:
```python
cx = float(0.0); cy = float(0.0); cz = float(0.0); tw = float(0.0)
for j in range(size):  # loop accumulates — `cx = cx + x` is += pattern, OK per Warp docs
    cx = cx + w * p[0]
    ...

cx = cx / tw                  # ❌ NOT += pattern, direct re-assign → silent-zero
cy = cy / tw                  # ❌
cz = cz / tw                  # ❌
qcx = qcx / tw                # ❌
qcy = qcy / tw                # ❌
qcz = qcz / tw                # ❌

...
R = U * wp.transpose(V)       # first assign
if det < 0.0:
    R = U_fixed * ...          # ❌ second assign
```

Plus: SVD backward instability (well-known, needs Müller 2016 polar decomposition replacement per `reference_diff_rl_levels.md`).

This one is NOT a quick fix. Needs:
- Refactor local accumulators to use fresh names (`cx_sum` → `cx_mean`)
- Replace branch re-assign of `R` with `wp.where` on the whole matrix
- **Port Müller 2016 iterative polar decomposition to replace `wp.svd3` entirely** (this is the real fix for Block 2)

Estimated 1-2 weeks. Handle as a self-contained subproject.

### 🟡 YELLOW: `apply_kernel`

Line 305:
```python
pred_pos[i] = pred_pos[i] + local_stiff * inv_count * deltas[i]
#             ^^^^^^^^^^^ read
# ^^^^^^^^^^^ write — SAME array, SAME index
```

`wp.config.verify_autograd_array_access = True` flags this as write-after-read violation. Fix: use `wp.atomic_add(pred_pos, i, local_stiff * inv_count * deltas[i])` directly in this kernel, OR compute into `apply_delta` buffer and apply via a dedicated kernel (matches `apply_collision_delta_kernel` pattern).

Quick fix: ~2 hours.

### 🟡 YELLOW: `finalize_particle_damage_kernel`

Line 216:
```python
particle_damage[i] = particle_damage[i] + wp.log(1.0 + wp.exp(sp_arg)) / 20.0
```
Same R-M-W pattern. Fix: compute increment into `particle_damage_delta` buffer, apply via `apply_particle_damage_delta_kernel`.

Quick fix: ~3 hours.

## Summary

| Severity | Count | Est. work |
|---|---|---|
| 🔴 RED (major rewrite) | 2 (predict, cluster_solve) | ~1-2 weeks (dominated by cluster_solve + Müller) |
| 🟡 YELLOW (minor fix) | 2 (apply, finalize_particle_damage) | ~5 hours total |
| 🟢 GREEN (no work) | 4 | 0 |

**Total minimum work** (excluding cluster_solve + Müller): **~1.5 days**.
**Full job with cluster_solve + Müller**: **~2-3 weeks**.

## Recommended execution order (staged)

Each stage must pass `test_capsule_task.py` (38/38) before advancing:

1. **Stage A — quick wins** (0.5 day)
   - `apply_kernel` → delta-buffer pattern
   - `finalize_particle_damage_kernel` → delta-buffer pattern
   - Regression: run test_capsule_task.py + Phase 1-4 gate tests

2. **Stage B — predict fix** (1 day)
   - `predict_kernel` SSA rewrite
   - Gate test: FD-cosine vs autograd on `pos.grad` after 1 substep
   - Regression

3. **Stage C — cluster_solve + Müller 2016** (1-2 weeks, self-contained subproject)
   - Port Müller 2016 iterative quaternion polar decomposition
   - Rewrite cluster_solve_kernel in SSA form using Müller polar instead of wp.svd3
   - Validate forward (cosine vs old on 100 random clusters) + backward (FD-cosine on pred_pos.grad)
   - Regression

4. **Stage D — full-substep L2 validation** (2 days)
   - Build integrated tape covering full `_substep()`
   - FD-cosine test: `∂loss/∂action` through full physics step
   - Target: cosine > 0.9 with FD on representative capsule scene

5. **Stage E — Fig. 3 gradient sweep** (2-3 days)
   - Sweep (num_iterations ∈ {1,2,3,4,6,8}, H ∈ {1,2,4,8}, alpha_min ∈ {3,5,10})
   - Per-cell: gradient norm, NaN rate, FD-cosine, wall-clock
   - Paper figure data

**Total: ~3-4 weeks** for full M2 sprint.

## What Stage A-D unlock vs don't

✅ **Unlocked after Stage D (full single-step L2)**:
- C3b sys-ID via tape (∂loss/∂stiffness through cluster_solve)
- Fig. 3 gradient characterization (Stage E)
- Reviewer-proof C2 section ("tape covers full substep, not just one kernel")

❌ **Still blocked after Stage D** (these require Jacobi adjoint / damage chain redesign):
- L3 (H>1 steps through PBD)
- Multi-step tactile features
- SAPO-style policy gradient
