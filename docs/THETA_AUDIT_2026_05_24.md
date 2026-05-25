# Theta Audit — Old (3, 20, 0.5) vs New (3, 100, 0.3) Config B

**Date**: 2026-05-24
**Trigger**: Config B propagation (commit 58e63a1) migrated production sys-ID
to theta=(3, 100, 0.3) with sigmoid bounding. This audit catalogs which
downstream scripts still use the old theta=(3, 20, 0.5) and why.

## Summary

| Category | Count | Action |
|---|---|---|
| Production pipeline | 1 | ✅ Updated to (3,100,0.3) in commit `58e63a1` |
| New verification suite | 2 | ✅ Already use new theta (P1, P3) |
| Historical day_* experiments | 35 | 🚫 Keep — historical context |
| Historical r_* experiments | 10 | 🚫 Keep — Hessian/conditioning studies |
| Historical test_layer_* | 3 | 🚫 Keep — Layer A POC |
| Historical path4_* | 8 | 🚫 Keep — Path 4 falsification |
| POC soft removal | 3 | 🚫 Keep — POC verification |
| Calibration entry points | 3 | 🚫 Keep — each tied to specific θ |
| Stage A baseline verifiers | 2 | 🚫 Keep — historic baseline at old θ |
| Stage B tests | 2 | 🚫 Keep — Stage B baseline at old θ |
| M3 diagnostics / dry runs | 4 | 🚫 Keep — M3 at old θ |
| Visualizations | 7 | 🚫 Keep — tied to npz data |
| Other test/synth | 4 | 🚫 Keep — specific test configs |
| **Total** | **82** | **1 updated + 2 new + 79 keep** |

**Decision rationale**: All "active" candidates have intentional theta choices
tied to their purpose (e.g., calibrate_resect.py validates Day 9 fix-sharpness
LOCK at sharpness=20 — changing it would invalidate the validation result).

The production pipeline (stageA_inclusion_recovery.py) is the canonical entry
that has been updated. All other 81 files keep their original theta because
they document historical results that should remain reproducible.

## Production Pipeline (UPDATED)

### tools/stageA_inclusion_recovery.py
- **OLD**: theta=(3, 20, 0.5), lr=0.01, raw stiff + hard clamp, N_epochs=60
- **NEW**: theta=(3, 100, 0.3), lr=0.03, sigmoid bound + early stop, N_epochs=30
- Updated in commit `58e63a1` (P4 → P4b → P4c chain)

## New Verification Suite (Already Uses New Theta)

### tools/p1_grad_sanity_sharp100.py
Compares grad scaling at old vs new theta. Uses BOTH (3,20,0.5) and (3,100,0.3)
deliberately for comparison.

### tools/p3_fd_parity_sharp100.py
FD parity at theta=(3, 100, 0.3). New theta intentional.

## Historical day_* Experiments (KEEP, 35 files)

Day-numbered scripts document specific experiments at the theta active at
that time. All used (3, 20, 0.5) which was the locked theta from Day 9.

Examples:
- `day3b_stiffness_diagnose.py` — diagnose at theta=(3,20,0.5)
- `day6b_full_forward_torch_op.py` — Day 6 forward op, __main__ test uses (3,20,0.5)
- `day7_step1_lgap_landscape.py` — Day 7 landscape sweep at (3,20,0.5)
- `day8_stable_inner_loop.py` — Day 8 inner loop fix, sharpness=20 fixed
- `day9_*` — Day 9 fix-sharpness lock validation (the SUBJECT of the lock)

**Why keep**: Historical results in memory files reference these scripts
running at (3, 20, 0.5). Changing them invalidates the cross-reference chain.

## Historical r_* Experiments (KEEP, 10 files)

R3/R4 = Hessian conditioning + multi-init basin studies. All conducted at
theta=(3, 20, 0.5) per Day 9 lock.

Examples:
- `r3_multi_init_basin.py`, `r3b_diagnose_all_high_init.py`
- `r4_hessian_*`, `r4_test12_*`, `r4_sanity_*`

**Why keep**: Hessian conditioning is theta-specific. Re-running at sharp=100
would give different Hessian, invalidating the R3/R4 findings documented in
memory files like `project_day8_audit_complete`, `project_day9_fix_sharpness_locked`.

## Historical test_layer_* (KEEP, 3 files)

Layer A POC tests at original theta. Tied to commit history of Layer A
gradient parity work.

## Historical path4_* (KEEP, 8 files)

Path 4 (continuous-damage trajectory opt) falsification tests at original
theta. Each path4_*_test_*.py tests a specific falsifier — theta choice tied
to falsifier definition.

## POC Soft Removal (KEEP, 3 files)

- `poc_soft_removal_ssa.py`
- `poc_soft_removal_ssa_v2_real.py`
- `poc_soft_removal_ssa_v3_full.py`

Verification chain for soft removal SSA refactor (Path B+). Each POC was
verified at theta=(3, 20, 0.5) — the production theta at that time.

## Calibration Entry Points (KEEP)

### tools/calibrate_resect.py
- **Purpose**: Day 9 fix-sharpness validation (sharpness=20 LOCKED)
- **theta**: (3, 20, 0.5) — the VERY POINT being validated
- **Change?**: NO — changing sharpness invalidates the Day 9 lock decision

### tools/calibrate_stage2_autograd.py
- **Purpose**: Stage B autograd validator at default theta
- **theta**: defaults to (3, 20, 0.5) for backward-compat with Stage B Day 5.1 tests
- **Change?**: NO — Stage B verified at sharp=20; re-verify at sharp=100 = new task

### tools/calibrate_stage2_FD.py
- **Purpose**: Stage B FD parity test
- **theta**: same as calibrate_stage2_autograd
- **Change?**: NO — symmetric with above

## Stage A Baseline Verifiers (KEEP)

### tools/stageA_fd_parity_per_particle_stiff.py
- **Purpose**: Stage A FD parity at production theta when Stage A was committed
- **theta**: (3, 20, 0.5) — production at commit time
- **Change?**: NO — P3 (tools/p3_fd_parity_sharp100.py) is the new theta version

### tools/stageA_g1g2_cross_theta_call.py
- **Purpose**: Stage A cross-theta and cross-call verification
- **theta**: (3, 20, 0.5) primary + (5, 15, 0.6) cross-theta perturbation
- **Change?**: NO — historic Stage A verification baseline

## Stage B Tests (KEEP)

### tools/test_stage_b_calibrator_sanity.py
- **Purpose**: Stage B 4-sanity test suite (Day 5.1 milestone)
- **theta**: setup_target() default sharpness=20.0
- **Change?**: NO — Stage B 100% verification done at this theta

### tools/test_stage_b_gradient_parity.py
- **Purpose**: Stage B gradient parity vs legacy
- **theta**: default (3, 20, 0.5)
- **Change?**: NO — historic regression baseline

## M3 Diagnostics (KEEP, 4 files)

- `analyze_stage2_landscape.py` — Stage 2 loss landscape at threshold=0.5
- `diagnostic_m3_covariance.py` — M3 covariance diagnostic at sharpness=20
- `dry_run_m3_extended.py` — M3 forward dry run
- `dry_run_m3_kernel_switch.py` — M3 SSA vs legacy kernel switch test

**Why keep**: M3 milestone (capsule task progress) validated at sharpness=20.

## Visualizations (KEEP, 7 files)

- `animate_config_b_vs_a.py` — explicitly compares OLD vs NEW theta
- `animate_t10_cube.py` — uses old theta for Step 2 viz
- `visualize_cube_cut.py`
- `visualize_soft_cube_sweep.py` — A/B/C config comparison
- `visualize_step2_compare.py`, `visualize_step2_v2.py` — Step 2 viz (old theta data in npz)
- `visualize_t10_deformation.py` — T=10 deformation at old theta

**Why keep**: Tied to npz data files that were generated with specific theta.
Changing the script without re-generating npz would mismatch viz vs data.

## Other (KEEP, 4 files)

### tools/online_residual_synth.py
- **Purpose**: Online residual learning synthesis experiment
- **theta**: explicit theta_true=(3, 20, 0.5) for synth target
- **Change?**: NO — specific synth experiment config

### tools/run_sharpness_sensitivity.py
- **Purpose**: Sharpness sensitivity probe (sharpness=20 control + variants)
- **theta**: comment says "sharpness=20, 1 ckpt (200k), 50 episodes — measure σ"
- **Change?**: NO — sharpness=20 is the CONTROL group

### tools/test_dynamic_loss.py
- **Purpose**: Dynamic loss function test at threshold=0.5
- **Change?**: NO — specific test config

### tools/day2_solver_wrapper.py
- **Purpose**: Solver wrapper (CORE), `__main__` self-test uses (3,20,0.5)
- **API**: theta is a PARAMETER, not hardcoded
- **Change?**: NO — hardcoded value only in self-test, not in API

## Action Items

- [x] Production pipeline updated (commit `58e63a1`)
- [x] New verification suite (P1, P3) uses new theta
- [x] This audit doc created
- [ ] (Future, phantom phase): If/when phantom data calibrates a different theta,
      decide which active candidates need re-running at the new theta. At that
      point, may need to update Stage A/B baseline tests to new theta.

## Phantom-Stage Re-Audit Trigger

When phantom hardware data arrives:
1. Decide new theta based on phantom F/T sensor + position calibration
2. Re-run subset of active candidates at new theta:
   - calibrate_stage2_autograd.py (Stage B re-verify)
   - calibrate_stage2_FD.py
   - stageA_fd_parity_per_particle_stiff.py
   - test_stage_b_* (Stage B regression at new theta)
3. Update this doc with re-audit findings
