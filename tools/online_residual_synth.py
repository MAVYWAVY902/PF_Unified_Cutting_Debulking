#!/usr/bin/env python3
"""
Online residual real-to-sim calibration POC (synth-to-synth).

Adapts Liang et al. 2023 (UCSD, arXiv:2309.11656) Algorithm 1 from the
non-cutting manipulation regime to the cutting/topology-change regime
with 4 cutting-specific modifications:

  Mod 1: Residual on damage state δ_v (irreversible) instead of positions x
         → softplus parameterization enforces Δδ_v ≥ 0
  Mod 2: Three-channel loss (cavity α + axial force F + physical consistency)
  Mod 3: L_traj (trajectory consistency) replaces L_hist (elastic equilibrium)
         since cutting damage is irreversible
  Mod 4: 4 global params (k, κ, s, τ) instead of per-particle k
         → no smoothness loss needed

Architecture (matches Liang Algorithm 1):
   for each frame t along knife trajectory u_t:
       1. Forward sim with current θ → predicted (δ_v^sim, α^sim, F^sim)
       2. Get observation (α^obs, F^obs) — synth from θ_true in this POC
       3. Inner loop: solve Δδ_v ≥ 0 such that α_corrected ≈ α^obs
       4. Outer loop: 1-step Adam on θ using L = L_gap + L_force + L_traj + L_prior

Synth-to-synth validation:
   target trajectory generated with θ_true; init with θ_0 ≠ θ_true;
   verify θ converges toward θ_true within 100 frames.

Reuses from tools/calibrate_stage2_autograd.py:
   setup_synthetic_system, make_params, run_ssa_full, mse_loss_kernel.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

# Diagnostic anchor — mandatory per Stage B engineering rules
wp.config.verify_autograd_array_access = True

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@wp.kernel
def corrected_alpha_sigmoid_kernel(
    delta_v_corrected: wp.array(dtype=wp.float32),
    sharpness:         wp.array(dtype=wp.float32),     # (1,) param
    threshold:         wp.array(dtype=wp.float32),     # (1,) param
    alpha_corrected:   wp.array(dtype=wp.float32),
):
    """Compute α_corrected = sigmoid(-s · (δv_corrected - τ)) per particle.

    Used to implement Liang 2023's L_gap loss inside Warp Tape:
        δv_corrected = δv_sim + Δδ_v  (Δδ_v is constant from inner loop)
        α_corrected = sigmoid(-s·(δv_corrected - τ))
        L_gap = mean (α_corrected - α_obs)²

    Gradient w.r.t. sharpness and threshold flows automatically via Warp;
    gradient w.r.t. α_min flows through δv_sim (forward sim).
    """
    i = wp.tid()
    dv = delta_v_corrected[i]
    s = sharpness[0]
    tau = threshold[0]
    # σ(-s·(dv-τ)) = 1 / (1 + exp(s·(dv-τ)))
    alpha_corrected[i] = 1.0 / (1.0 + wp.exp(s * (dv - tau)))


@wp.kernel
def add_residual_kernel(
    delta_v_sim:        wp.array(dtype=wp.float32),
    delta_v_residual:   wp.array(dtype=wp.float32),    # constant Δδ_v from inner loop
    delta_v_corrected:  wp.array(dtype=wp.float32),
):
    """Compute δv_corrected[i] = δv_sim[i] + Δδ_v[i].
    Δδ_v is treated as constant in the tape (no grad through it).
    """
    i = wp.tid()
    delta_v_corrected[i] = delta_v_sim[i] + delta_v_residual[i]


@wp.kernel
def masked_mse_loss_kernel(
    alpha_sim: wp.array(dtype=wp.float32),
    alpha_target: wp.array(dtype=wp.float32),
    mask: wp.array(dtype=wp.int32),
    loss_buf: wp.array(dtype=wp.float32),
):
    """Per-particle MSE summed only over particles where mask[i]==1.

    Used to mimic real-world observation visibility: only the surface (or
    a subset of the surface) is observable from cameras. Internal particles
    contribute zero to the loss and zero gradient.
    """
    i = wp.tid()
    if mask[i] == 1:
        diff = alpha_sim[i] - alpha_target[i]
        wp.atomic_add(loss_buf, 0, diff * diff)


def compute_observable_mask(sys_dict: dict, mode: str = "surface",
                            margin: float = 0.02) -> np.ndarray:
    """Return (N,) int32 mask: 1 = observable from sensors, 0 = internal/occluded.

    Modes (mimics what cameras can actually see in real setup):
      'all'       — all particles observable (no masking; for baseline comparison)
      'surface'   — all particles touching the surface mesh (≈top + sides)
      'top'       — only y ≥ y_max - margin (D405 from above)
      'top+side'  — top OR x ≥ x_max - margin (D405 + 1 side RGB)
      'top+2side' — top + 2 perpendicular side faces

    Default 'surface' = best case for cube observability.
    """
    pos = sys_dict["pos"]
    N = sys_dict["N"]
    mask = np.zeros(N, dtype=np.int32)

    if mode == "all":
        mask[:] = 1
        return mask

    bbox_lo = pos.min(axis=0); bbox_hi = pos.max(axis=0)

    if mode == "surface":
        sf = sys_dict.get("surface_faces")
        if sf is None:
            raise ValueError("mode='surface' requires sys_dict to have 'surface_faces'")
        surface_verts = np.unique(sf.ravel())
        mask[surface_verts] = 1
        return mask

    if mode == "top":
        mask[pos[:, 1] >= bbox_hi[1] - margin] = 1
        return mask

    if mode == "top+side":
        is_top = pos[:, 1] >= bbox_hi[1] - margin
        is_side = pos[:, 0] >= bbox_hi[0] - margin
        mask[is_top | is_side] = 1
        return mask

    if mode == "top+2side":
        is_top = pos[:, 1] >= bbox_hi[1] - margin
        is_side1 = pos[:, 0] >= bbox_hi[0] - margin
        is_side2 = pos[:, 2] >= bbox_hi[2] - margin
        mask[is_top | is_side1 | is_side2] = 1
        return mask

    raise ValueError(f"unknown observable mask mode: {mode!r}")

# Reuse existing infra
from tools.calibrate_stage2_autograd import (  # noqa: E402
    setup_synthetic_system,
    make_params,
    run_ssa_full,
    mse_loss_kernel,
    Adam2D,
)


# =============================================================================
# 3D cube geometry for synth POC (replaces 1D line in cumulative-regime tests)
# =============================================================================

def setup_cube_system(N_per_dim: int = 8,
                     extent: tuple = (0.1, 0.9),
                     knife_center: tuple = (0.5, 0.5, 0.5)) -> dict:
    """3D cubic lattice of N_per_dim^3 particles in [extent[0], extent[1]]^3
    with 6-neighbor edges (3 edges per particle along +x, +y, +z).

    Returns the same dict shape as setup_synthetic_system so all existing
    kernels and helpers work unchanged. This gives a 3D synth POC that's
    much closer to real cutting (volume mesh, particles fully populate the
    knife's swept volume).

    Sizing:
        N_per_dim=8   →  N=512   E=1344     (recommended for fast POC)
        N_per_dim=10  →  N=1000  E=2700
        N_per_dim=16  →  N=4096  E=11520    (heavy, only if needed)

    The active "informative band" around the knife scales with cube volume
    captured by the knife sweep, which dramatically improves α_min gradient
    signal vs the 1D line (where only 1-2 particles are active per frame).
    """
    lo, hi = extent
    xs = np.linspace(lo, hi, N_per_dim, dtype=np.float32)
    spacing = float(xs[1] - xs[0])

    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    pos_np = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1).astype(np.float32)
    N = N_per_dim ** 3

    def lin(i, j, k):
        return i * N_per_dim * N_per_dim + j * N_per_dim + k

    edge_a_list, edge_b_list = [], []
    for i in range(N_per_dim):
        for j in range(N_per_dim):
            for k in range(N_per_dim):
                a = lin(i, j, k)
                if i + 1 < N_per_dim:
                    edge_a_list.append(a); edge_b_list.append(lin(i + 1, j, k))
                if j + 1 < N_per_dim:
                    edge_a_list.append(a); edge_b_list.append(lin(i, j + 1, k))
                if k + 1 < N_per_dim:
                    edge_a_list.append(a); edge_b_list.append(lin(i, j, k + 1))
    edge_a = np.asarray(edge_a_list, dtype=np.int32)
    edge_b = np.asarray(edge_b_list, dtype=np.int32)
    E = len(edge_a)
    pf_rest = np.full(E, spacing, dtype=np.float32)

    return dict(
        pos=pos_np,
        edge_a=edge_a,
        edge_b=edge_b,
        pf_rest=pf_rest,
        knife_pos=np.array([list(knife_center)], dtype=np.float32),
        N=N, E=E,
    )


def make_cube_knife_trajectory(n_frames: int,
                               x_start: float = 0.2,
                               x_end: float = 0.8,
                               y: float = 0.5,
                               z: float = 0.5) -> np.ndarray:
    """Linear knife sweep along x at constant (y, z).
    Defaults: enter cube at x=0.2, exit at x=0.8, slicing through center.
    """
    xs = np.linspace(x_start, x_end, n_frames)
    traj = np.zeros((n_frames, 3), dtype=np.float32)
    traj[:, 0] = xs
    traj[:, 1] = y
    traj[:, 2] = z
    return traj


def load_msh_system(msh_path: str,
                    knife_center: tuple = (0.15, 0.3, 0.15)) -> dict:
    """Load a tet .msh file and convert to sys_dict for pf_damage kernels.

    Edges are derived from the 6 edges of each tetrahedron and deduplicated.
    pf_rest is the Euclidean rest length of each edge.

    Surface faces: prefer existing 'triangle' cell blocks (gmsh tags them when
    the mesh was built with boundary physical groups). Fallback: derive from
    tets (faces belonging to exactly 1 tet).

    Returns dict with: pos, edge_a, edge_b, pf_rest, knife_pos, N, E, bbox,
    surface_faces (F, 3) int32 — for surface-mesh rendering.
    """
    import meshio
    m = meshio.read(msh_path)
    pos = np.asarray(m.points, dtype=np.float32)
    N = len(pos)
    bbox = (pos.min(axis=0), pos.max(axis=0))

    tets_list = [cb.data for cb in m.cells if cb.type == "tetra"]
    if not tets_list:
        raise ValueError(f"no 'tetra' cells in {msh_path}")
    tets = np.concatenate(tets_list, axis=0).astype(np.int32)
    n_tets = len(tets)

    edge_pairs = np.empty((n_tets * 6, 2), dtype=np.int64)
    p = 0
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        seg = np.stack([tets[:, a], tets[:, b]], axis=1)
        edge_pairs[p:p + n_tets] = seg
        p += n_tets
    edge_pairs_sorted = np.sort(edge_pairs, axis=1)
    unique_edges = np.unique(edge_pairs_sorted, axis=0)
    edge_a = unique_edges[:, 0].astype(np.int32)
    edge_b = unique_edges[:, 1].astype(np.int32)
    E = len(edge_a)
    pf_rest = np.linalg.norm(pos[edge_a] - pos[edge_b], axis=1).astype(np.float32)

    # Surface faces: prefer existing triangle blocks, fallback to tet boundary.
    tri_blocks = [cb.data for cb in m.cells if cb.type == "triangle"]
    if tri_blocks:
        surface_faces = np.concatenate(tri_blocks, axis=0).astype(np.int32)
    else:
        from collections import defaultdict
        face_count = defaultdict(int)
        face_first = {}
        for tet in tets:
            a, b, c, d = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
            for face in [(a, c, b), (a, b, d), (a, d, c), (b, c, d)]:
                key = tuple(sorted(face))
                face_count[key] += 1
                if key not in face_first:
                    face_first[key] = face
        boundary = [face_first[k] for k, cnt in face_count.items() if cnt == 1]
        surface_faces = np.asarray(boundary, dtype=np.int32)

    return dict(
        pos=pos,
        edge_a=edge_a,
        edge_b=edge_b,
        pf_rest=pf_rest,
        knife_pos=np.array([list(knife_center)], dtype=np.float32),
        N=N, E=E,
        bbox=bbox,
        surface_faces=surface_faces,
    )


def make_surface_knife_trajectory(bbox: tuple,
                                  n_frames: int,
                                  surface_axis: str = "y",
                                  margin_fraction: float = 0.15) -> np.ndarray:
    """Slow surface-knife sweep across the top face of a mesh bbox.

    Knife center at axis_max (so half the sphere is exposed above the surface,
    half embedded below). Sweeps along the longer non-up axis, with the other
    non-up axis fixed at center.

    Args:
        bbox            — (lo, hi) tuple of (3,) np arrays
        n_frames        — trajectory length (more = slower per-frame motion)
        surface_axis    — which axis is up ('x', 'y', or 'z')
        margin_fraction — fraction of sweep-axis span left as margin at each end

    Returns:
        (n_frames, 3) np.array of knife center positions.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    up = axis_map[surface_axis]
    lo, hi = bbox
    surface_pos = float(hi[up])

    other = [i for i in range(3) if i != up]
    spans = sorted([(float(hi[i] - lo[i]), i) for i in other], reverse=True)
    sweep_axis, other_axis = spans[0][1], spans[1][1]

    span = float(hi[sweep_axis] - lo[sweep_axis])
    x0 = float(lo[sweep_axis] + margin_fraction * span)
    x1 = float(hi[sweep_axis] - margin_fraction * span)
    other_center = float((hi[other_axis] + lo[other_axis]) / 2.0)

    traj = np.zeros((n_frames, 3), dtype=np.float32)
    traj[:, sweep_axis] = np.linspace(x0, x1, n_frames, dtype=np.float32)
    traj[:, up] = surface_pos
    traj[:, other_axis] = other_center
    return traj
from pf_shape_matching_warp.sim.pf_damage import (  # noqa: E402
    update_edge_damage_kernel_ssa_param_arr,
    apply_edge_damage_ssa_kernel,
    update_particle_damage_kernel,
    finalize_particle_damage_kernel_ssa,
    apply_particle_damage_ssa_kernel,
    update_soft_removal_kernel_ssa_param_arr,
)


# =============================================================================
# Helper: run_ssa returning both alpha and delta_v (mirrors run_ssa_full)
# =============================================================================

def run_ssa_full_with_state(
    params: dict,
    sys_dict: dict,
    n_cycles: int = 1,
    state_in: dict = None,
):
    """Cycle-scoped SSA forward; returns final (alpha, particle_damage, edge_damage) wp.arrays.

    state_in: optional dict carrying frame-to-frame persistent state.
        Keys (all np.ndarray with float32 dtype):
            "alpha":            (N,) — initial alpha for cycle 0
            "edge_damage":      (E,) — initial edge_damage_buffers[0]
            "particle_damage":  (N,) — initial particle_damage_buffers[0]
        If None, initializes to REST: alpha=1, both damages=0 (legacy behavior).

    Detachment property: state_in is loaded via np.ndarray -> fresh wp.array,
    so the Warp Tape sees only the CURRENT call's cycles. Cross-frame gradient
    flow is intentionally blocked (Liang 2023 detached-state pattern; Mod 1
    softplus residual Δδ_v absorbs cumulative state error).

    Returns: (alpha_final_wp, particle_damage_final_wp, edge_damage_final_wp)
    """
    N, E = sys_dict["N"], sys_dict["E"]
    pos = wp.array(sys_dict["pos"], dtype=wp.vec3)
    edge_a = wp.array(sys_dict["edge_a"], dtype=wp.int32)
    edge_b = wp.array(sys_dict["edge_b"], dtype=wp.int32)
    pf_rest = wp.array(sys_dict["pf_rest"], dtype=wp.float32)
    knife_pos = wp.array(sys_dict["knife_pos"], dtype=wp.vec3)
    fixed = wp.array(np.zeros(N, dtype=np.int32), dtype=wp.int32)

    if state_in is None:
        alpha_init_np = np.ones(N, dtype=np.float32)
        edge_dmg_init_np = np.zeros(E, dtype=np.float32)
        part_dmg_init_np = np.zeros(N, dtype=np.float32)
    else:
        alpha_init_np = np.asarray(state_in["alpha"], dtype=np.float32)
        edge_dmg_init_np = np.asarray(state_in["edge_damage"], dtype=np.float32)
        part_dmg_init_np = np.asarray(state_in["particle_damage"], dtype=np.float32)
        assert alpha_init_np.shape == (N,), f"state_in[alpha] shape {alpha_init_np.shape} != ({N},)"
        assert edge_dmg_init_np.shape == (E,), f"state_in[edge_damage] shape {edge_dmg_init_np.shape} != ({E},)"
        assert part_dmg_init_np.shape == (N,), f"state_in[particle_damage] shape {part_dmg_init_np.shape} != ({N},)"

    alpha_buffers = [
        wp.array(alpha_init_np if i == 0 else np.ones(N, dtype=np.float32),
                 dtype=wp.float32, requires_grad=True)
        for i in range(n_cycles + 1)
    ]
    edge_damage_buffers = [
        wp.array(edge_dmg_init_np if i == 0 else np.zeros(E, dtype=np.float32),
                 dtype=wp.float32, requires_grad=True)
        for i in range(n_cycles + 1)
    ]
    particle_damage_buffers = [
        wp.array(part_dmg_init_np if i == 0 else np.zeros(N, dtype=np.float32),
                 dtype=wp.float32, requires_grad=True)
        for i in range(n_cycles + 1)
    ]

    for cycle in range(n_cycles):
        # Edge damage
        edge_damage_delta = wp.zeros(E, dtype=wp.float32, requires_grad=True)
        wp.launch(update_edge_damage_kernel_ssa_param_arr, dim=E, inputs=[
            pos, alpha_buffers[cycle], edge_a, edge_b, pf_rest,
            edge_damage_buffers[cycle],
            knife_pos, params["knife_radius"], params["alpha_min"],
            edge_damage_delta,
        ])
        wp.launch(apply_edge_damage_ssa_kernel, dim=E, inputs=[
            edge_damage_buffers[cycle], edge_damage_delta,
            edge_damage_buffers[cycle + 1],
        ])
        # Particle damage
        damage_sum = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        damage_count = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        wp.launch(update_particle_damage_kernel, dim=E, inputs=[
            damage_sum, damage_count, alpha_buffers[cycle],
            edge_a, edge_b, edge_damage_buffers[cycle + 1],
        ])
        particle_damage_delta = wp.zeros(N, dtype=wp.float32, requires_grad=True)
        wp.launch(finalize_particle_damage_kernel_ssa, dim=N, inputs=[
            damage_sum, damage_count,
            particle_damage_buffers[cycle], particle_damage_delta,
        ])
        wp.launch(apply_particle_damage_ssa_kernel, dim=N, inputs=[
            particle_damage_buffers[cycle], particle_damage_delta,
            particle_damage_buffers[cycle + 1],
        ])
        # Soft removal
        wp.launch(update_soft_removal_kernel_ssa_param_arr, dim=N, inputs=[
            particle_damage_buffers[cycle + 1],
            alpha_buffers[cycle], fixed,
            params["threshold"], params["sharpness"],
            alpha_buffers[cycle + 1],
        ])

    return alpha_buffers[-1], particle_damage_buffers[-1], edge_damage_buffers[-1]


# =============================================================================
# Adam optimizer extended to N-dimensional params
# =============================================================================

class AdamND:
    """N-dimensional Adam, generalization of Adam2D.

    Supports per-parameter learning rate via lr as np.array of shape (N,) or
    scalar.
    """

    def __init__(self, n_dims, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        if np.isscalar(lr):
            self.lr = np.full(n_dims, lr, dtype=np.float64)
        else:
            self.lr = np.asarray(lr, dtype=np.float64)
            assert self.lr.shape == (n_dims,), f"lr shape {self.lr.shape} != ({n_dims},)"
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = np.zeros(n_dims)
        self.v = np.zeros(n_dims)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        """Reset moments — useful when restarting from new init."""
        self.m = np.zeros_like(self.m)
        self.v = np.zeros_like(self.v)
        self.t = 0


# =============================================================================
# Inner loop: damage residual Δδ_v ≥ 0 (numpy / analytic)
# =============================================================================

def _softplus(x):
    """Numerically stable softplus."""
    return np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))


def _softplus_deriv(x):
    """Derivative of softplus = sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def compute_damage_residual(
    delta_v_sim,           # np.array(N,) — current sim damage state
    alpha_obs,             # np.array(N,) — observed cavity
    s, tau,                # current sigmoid params (float scalars)
    K_inner=30,
    lr_inner=0.05,
    verbose=False,
):
    """Inner loop: find Δδ_v ≥ 0 such that
        α_corrected = sigmoid(-s·(δ_v_sim + Δδ_v - τ))  ≈  α_obs

    Uses softplus parameterization: Δδ_v = softplus(δ_raw) ≥ 0
    enforces irreversibility (damage can only INCREASE).

    Analytic gradient (avoids autograd framework dependency at this scale):
        L = ½ Σ (sigmoid(-s·(δ_v + Δδ_v - τ)) - α_obs)²
        ∂L/∂Δδ_v = -(α_corr - α_obs) · α_corr · (1 - α_corr) · s
        ∂L/∂δ_raw = ∂L/∂Δδ_v · softplus'(δ_raw)

    Returns:
        delta_delta_v: np.array(N,) ≥ 0 (final residual)
        L_gap: float (final residual norm)
        L_inner_init: float (initial loss for comparison)
    """
    delta_v_sim = np.asarray(delta_v_sim, dtype=np.float64)
    alpha_obs = np.asarray(alpha_obs, dtype=np.float64)

    N = len(delta_v_sim)
    delta_raw = np.zeros(N, dtype=np.float64)  # Init Δδ_v = softplus(0) = log(2) ≈ 0.69

    # Adam state for this inner loop
    m = np.zeros(N); v = np.zeros(N)
    b1, b2, eps = 0.9, 0.999, 1e-8

    L_inner_init = None
    for k in range(K_inner):
        delta_delta_v = _softplus(delta_raw)
        delta_v_corrected = delta_v_sim + delta_delta_v
        alpha_corr = _sigmoid(-s * (delta_v_corrected - tau))

        # Loss: MSE on alpha
        diff = alpha_corr - alpha_obs
        L = 0.5 * float((diff ** 2).sum())
        if L_inner_init is None:
            L_inner_init = L

        # Analytic gradient ∂L/∂δ_raw
        # ∂α_corr/∂(δ_v + Δδ_v) = -s · α_corr · (1 - α_corr)
        # So ∂L/∂Δδ_v = diff · (-s · α_corr · (1 - α_corr))
        grad_delta = diff * (-s * alpha_corr * (1 - alpha_corr))
        # Chain through softplus: ∂Δδ_v/∂δ_raw = sigmoid(δ_raw)
        grad_raw = grad_delta * _softplus_deriv(delta_raw)

        # Adam update on delta_raw
        m = b1 * m + (1 - b1) * grad_raw
        v = b2 * v + (1 - b2) * grad_raw ** 2
        m_hat = m / (1 - b1 ** (k + 1))
        v_hat = v / (1 - b2 ** (k + 1))
        delta_raw = delta_raw - lr_inner * m_hat / (np.sqrt(v_hat) + eps)

        if verbose and (k % 5 == 0 or k == K_inner - 1):
            print(f"    inner k={k:2d}  L={L:.4e}  |Δδ|={np.linalg.norm(delta_delta_v):.4f}")

    delta_delta_v_final = _softplus(delta_raw)
    delta_v_corrected_final = delta_v_sim + delta_delta_v_final
    alpha_corr_final = _sigmoid(-s * (delta_v_corrected_final - tau))
    L_gap = 0.5 * float(((alpha_corr_final - alpha_obs) ** 2).sum())

    return delta_delta_v_final.astype(np.float32), L_gap, L_inner_init


# =============================================================================
# Outer loop: Warp-tape backward for ∂L/∂θ on (alpha_min, sharpness, threshold)
# =============================================================================

def compute_outer_gradient(
    theta,                # np.array(3,) — current (alpha_min, sharpness, threshold)
    sys_dict,             # dict with single knife_pos
    alpha_obs,            # np.array(N,) — observation
    n_cycles=1,
    state_in: dict = None,    # persistent state from previous frame
    knife_radius: float = 0.1,
    observable_mask: np.ndarray = None,  # (N,) int32: 1=observable, 0=internal/hidden
    delta_delta_v_const: np.ndarray = None,  # (N,) Liang-style residual; if given, use L_gap
):
    """Computes ∂L_direct/∂θ where L_direct = Σ (α_sim - α_obs)²

    Strategy A (detached residual): outer gradient flows ONLY through
    forward sim w.r.t. θ in the CURRENT frame's SSA cycle. When state_in
    is supplied, it serves as the initial condition (loaded as fresh
    wp.array → cross-frame gradient is blocked by detachment).

    Returns:
        loss: float (L_direct, normalized by N)
        grad: np.array(3,) — ∂L/∂(alpha_min, sharpness, threshold)
    """
    N = sys_dict["N"]
    params = make_params(
        alpha_min=float(theta[0]),
        sharpness=float(theta[1]),
        threshold=float(theta[2]),
        knife_radius=knife_radius,
        requires_grad=True,
    )
    target_wp = wp.array(alpha_obs.astype(np.float32), dtype=wp.float32)
    loss_buf = wp.zeros(1, dtype=wp.float32, requires_grad=True)

    # Normalization: by N (default) or by # observable particles (with mask)
    if observable_mask is None:
        n_eff = float(N)
        mask_wp = None
    else:
        n_eff = float(int(observable_mask.sum()))
        if n_eff <= 0:
            raise ValueError("observable_mask has no observable particles (sum=0)")
        mask_wp = wp.array(observable_mask.astype(np.int32), dtype=wp.int32)

    with wp.Tape() as tape:
        if state_in is None:
            # Note: L_gap mode needs δv_sim, so we use state-aware path even when state_in=None
            if delta_delta_v_const is not None:
                alpha_sim_wp, delta_v_sim_wp, _ed_wp = run_ssa_full_with_state(
                    params, sys_dict, n_cycles=n_cycles, state_in=None,
                )
            else:
                alpha_sim_wp = run_ssa_full(params, sys_dict, n_cycles=n_cycles)
                delta_v_sim_wp = None
        else:
            alpha_sim_wp, delta_v_sim_wp, _ed_wp = run_ssa_full_with_state(
                params, sys_dict, n_cycles=n_cycles, state_in=state_in,
            )

        if delta_delta_v_const is not None:
            # L_gap mode (Liang 2023):
            #   α_corrected = sigmoid(-s·(δv_sim + Δδ_v - τ))
            #   L = mean (α_corrected - α_obs)²
            # Δδ_v is constant in tape (np.array → fresh wp.array, no grad).
            dv_resid_wp = wp.array(delta_delta_v_const.astype(np.float32), dtype=wp.float32)
            dv_corrected_wp = wp.zeros(N, dtype=wp.float32, requires_grad=True)
            wp.launch(add_residual_kernel, dim=N, inputs=[
                delta_v_sim_wp, dv_resid_wp, dv_corrected_wp,
            ])
            alpha_corrected_wp = wp.zeros(N, dtype=wp.float32, requires_grad=True)
            wp.launch(corrected_alpha_sigmoid_kernel, dim=N, inputs=[
                dv_corrected_wp, params["sharpness"], params["threshold"],
                alpha_corrected_wp,
            ])
            output_wp = alpha_corrected_wp
        else:
            output_wp = alpha_sim_wp

        if mask_wp is None:
            wp.launch(mse_loss_kernel, dim=N,
                      inputs=[output_wp, target_wp, loss_buf])
        else:
            wp.launch(masked_mse_loss_kernel, dim=N,
                      inputs=[output_wp, target_wp, mask_wp, loss_buf])

    tape.backward(loss=loss_buf)

    loss = float(loss_buf.numpy()[0]) / n_eff
    grad = np.array([
        float(params["alpha_min"].grad.numpy()[0]) / n_eff,
        float(params["sharpness"].grad.numpy()[0]) / n_eff,
        float(params["threshold"].grad.numpy()[0]) / n_eff,
    ])
    return loss, grad


# =============================================================================
# L_traj (P8 Mod 3): trajectory consistency loss
# =============================================================================

def compute_traj_loss(
    theta,                    # current (alpha_min, sharp, tau)
    history,                  # list of dicts from online_calibration_loop
    sys_dict,                 # base sys_dict (knife_pos overridden per snapshot)
    n_samples=4,
    history_window=20,
    n_cycles_per_frame=1,
    rng=None,
    knife_radius=0.1,
    observable_mask=None,
):
    """L_traj = (1/n_samples) Σ_h ‖α^sim(θ; u_h) - α^obs_h‖²

    Enforces trajectory-wide identifiability by demanding that the current
    θ simultaneously explains 4 randomly-sampled past observations.

    Algorithm (mirrors Liang L_hist but for cutting):
      1. Sample n_samples random indices from the LAST history_window snapshots
      2. For each sample h:
         a. Build sys_h with h.knife_pos
         b. Compute L_direct + gradient via Warp tape using current θ
      3. Average loss + gradient over samples

    Returns:
        (L_traj_value, grad_traj) — both scalar / (3,) np.array
        If len(history) < n_samples, returns (0, zeros) (warmup phase).
    """
    if len(history) < n_samples:
        return 0.0, np.zeros(3)

    if rng is None:
        rng = np.random.default_rng()

    # Window: last `history_window` (or all if shorter)
    window = history[-history_window:] if len(history) >= history_window else history[:]
    indices = rng.choice(len(window), size=n_samples, replace=False)

    total_loss = 0.0
    total_grad = np.zeros(3)
    for idx in indices:
        h = window[idx]
        sys_h = dict(sys_dict)
        sys_h["knife_pos"] = np.array([h["knife_pos"]], dtype=np.float32)
        # Reuse compute_outer_gradient — it does its own Warp tape backward.
        # Fix 1: replay from frame_h's INPUT state (state_pre), not rest.
        # In legacy mode (persistent_state=False) state_pre is None → rest behavior preserved.
        L_h, grad_h = compute_outer_gradient(
            theta, sys_h, h["alpha_obs"],
            n_cycles=n_cycles_per_frame,
            state_in=h.get("state_pre"),
            knife_radius=knife_radius,
            observable_mask=observable_mask,
        )
        total_loss += L_h
        total_grad += grad_h

    return total_loss / n_samples, total_grad / n_samples


# =============================================================================
# Synth 'real' trajectory generator
# =============================================================================

def make_knife_trajectory(n_frames: int, x_start: float = 0.4, x_end: float = 0.7):
    """Linear knife sweep along x-axis. y, z fixed at 0.5 (matches
    setup_synthetic_system 1D particle line).

    Returns: np.array shape (n_frames, 3)
    """
    xs = np.linspace(x_start, x_end, n_frames)
    traj = np.zeros((n_frames, 3), dtype=np.float32)
    traj[:, 0] = xs
    traj[:, 1] = 0.5
    traj[:, 2] = 0.5
    return traj


def synth_force_proxy(alpha_arr: np.ndarray, k_bulk: float, dt: float = 0.005,
                      mass: float = 1.0) -> float:
    """Approximate axial reaction force on knife from cavity-particle response.

    For synth POC: F ~ k_bulk × (1 - mean(alpha)) × (some unit factor).
    Real version would integrate shape_delta over knife-contact particles.
    """
    mean_damage = float((1.0 - alpha_arr).mean())
    return k_bulk * mean_damage * (mass / dt) * 0.001  # scale to ~Newton range


def generate_synth_real_trajectory(
    sys_dict,
    theta_true,            # (alpha_min, sharpness, threshold) np array shape (3,)
    knife_trajectory,      # (n_frames, 3) np array of knife positions
    k_bulk_true=1.0,
    n_cycles_per_frame=1,
    accumulate=True,       # NEW: persistent damage across frames
):
    """Run run_ssa_full at each knife position to produce 'real' observations.

    Each frame, the knife at u_t produces a cavity α_t. We record:
      alpha_obs   — the synth ground-truth cavity
      force_obs   — synthetic axial force (proxy formula)
      knife_pos   — the u_t used

    accumulate=True (DEFAULT): damage persists across frames (matches real cutting).
      Frame t's SSA cycle starts from frame t-1's output state. This is the
      regime where Fix 1 (persistent-state online calibration) is the
      correct physics.

    accumulate=False (LEGACY): each frame re-runs from REST. Preserved for
      backward compatibility with POC P1-P11 robustness numbers. Use only
      for reproducing pre-Fix-1 baselines.

    Returns: list of dicts (one per frame).
    """
    N, E = sys_dict["N"], sys_dict["E"]
    frames = []
    state_persistent = None  # None = rest at frame 0
    for t, kp in enumerate(knife_trajectory):
        sys_t = dict(sys_dict)
        sys_t["knife_pos"] = np.array([kp], dtype=np.float32)

        params_true = make_params(
            alpha_min=float(theta_true[0]),
            sharpness=float(theta_true[1]),
            threshold=float(theta_true[2]),
            knife_radius=0.1,
            requires_grad=False,
        )
        if accumulate:
            alpha_wp, part_dmg_wp, edge_dmg_wp = run_ssa_full_with_state(
                params_true, sys_t, n_cycles=n_cycles_per_frame,
                state_in=state_persistent,
            )
            alpha_t = alpha_wp.numpy().copy()
            # Update persistent state for the next frame
            state_persistent = dict(
                alpha=alpha_t.astype(np.float32),
                edge_damage=edge_dmg_wp.numpy().astype(np.float32),
                particle_damage=part_dmg_wp.numpy().astype(np.float32),
            )
        else:
            alpha_t = run_ssa_full(params_true, sys_t, n_cycles=n_cycles_per_frame).numpy().copy()
        force_t = synth_force_proxy(alpha_t, k_bulk_true)

        frames.append(dict(
            t=t,
            knife_pos=kp.copy(),
            alpha_obs=alpha_t,
            force_obs=force_t,
        ))
    return frames


# =============================================================================
# Smoke test — verifies imports + Adam works
# =============================================================================

def smoke_test():
    """Quick verification that all imports work and Adam3D converges on
    a trivial quadratic loss. Should run in < 1s.
    """
    print("[smoke] testing imports + AdamND ...")
    wp.init()

    # Trivial quadratic: minimize ||θ - target||² over 3 params
    target = np.array([3.0, 20.0, 0.5])
    theta = np.array([5.0, 10.0, 0.3])
    opt = AdamND(n_dims=3, lr=np.array([0.1, 1.0, 0.01]))

    for step in range(200):
        grad = 2.0 * (theta - target)
        theta = opt.step(theta, grad)

    err = np.abs(theta - target)
    print(f"[smoke] AdamND converged: θ={theta}, |err|={err}")
    assert np.all(err < 1e-3), f"AdamND failed: err={err}"

    # Sanity check synth system + run_ssa_full still works
    sys_dict = setup_synthetic_system(N=20)
    params = make_params(alpha_min=3.0, sharpness=20.0, requires_grad=False)
    alpha = run_ssa_full(params, sys_dict, n_cycles=1).numpy()
    debulked = float((alpha < 0.5).mean())
    print(f"[smoke] run_ssa_full ran: alpha range=[{alpha.min():.3f}, {alpha.max():.3f}], "
          f"debulked frac={debulked:.3f}")

    print("[smoke] PASS")


def trajectory_test():
    """Phase 2 test: generate a 20-frame synth trajectory and verify
    cavity grows along the knife sweep.
    """
    print("\n[traj] generating 20-frame synth 'real' trajectory ...")
    wp.init()
    sys_dict = setup_synthetic_system(N=20)
    traj = make_knife_trajectory(n_frames=20, x_start=0.4, x_end=0.7)

    theta_true = np.array([3.0, 20.0, 0.5])  # alpha_min, sharpness, threshold
    frames = generate_synth_real_trajectory(
        sys_dict, theta_true, traj,
        k_bulk_true=1.0, n_cycles_per_frame=1,
    )

    # Sanity: as knife moves through particle line, cavity should appear
    # at different positions. Print debulked frac per frame.
    print(f"[traj] {len(frames)} frames generated")
    print(f"       knife_x: {traj[0,0]:.2f} → {traj[-1,0]:.2f}")
    debulked_per_frame = [(f["alpha_obs"] < 0.5).mean() for f in frames]
    force_per_frame = [f["force_obs"] for f in frames]

    for t in [0, 5, 10, 15, 19]:
        kx = frames[t]["knife_pos"][0]
        deb = debulked_per_frame[t]
        F = frames[t]["force_obs"]
        a_min = frames[t]["alpha_obs"].min()
        a_max = frames[t]["alpha_obs"].max()
        print(f"  frame {t:2d}: knife_x={kx:.3f}  debulked={deb:.3f}  "
              f"α=[{a_min:.3f}, {a_max:.3f}]  F={F:.4f}")

    # Check trajectory is non-trivial (cavity varies across frames)
    deb_arr = np.array(debulked_per_frame)
    assert deb_arr.max() > 0.01, "Trajectory generated no damage — knife too far"
    print(f"[traj] PASS  (debulked range: [{deb_arr.min():.3f}, {deb_arr.max():.3f}])")


def inner_loop_test():
    """Phase 3 test: given delta_v from WRONG params, run inner loop to find
    Δδ_v that recovers alpha_obs (from true params).

    Setup:
      - theta_true = (3.0, 20.0, 0.5)   → produces alpha_obs
      - theta_wrong = (5.0, 10.0, 0.5)  → produces delta_v_sim (wrong)
      - Inner loop solves Δδ_v ≥ 0 → check alpha_corrected ≈ alpha_obs
    """
    print("\n[inner] testing damage residual recovery ...")
    wp.init()
    sys_dict = setup_synthetic_system(N=20)
    # Use a knife position that produces partial cavity
    sys_dict["knife_pos"] = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

    # 1. Generate alpha_obs from theta_true
    theta_true = np.array([3.0, 20.0, 0.5])
    params_true = make_params(
        alpha_min=float(theta_true[0]), sharpness=float(theta_true[1]),
        threshold=float(theta_true[2]), requires_grad=False,
    )
    alpha_obs_wp, delta_v_true_wp, _ = run_ssa_full_with_state(params_true, sys_dict, n_cycles=1)
    alpha_obs = alpha_obs_wp.numpy()
    delta_v_true = delta_v_true_wp.numpy()

    # 2. Generate delta_v_sim from theta_wrong (smaller alpha_min → less damage)
    theta_wrong = np.array([5.0, 10.0, 0.5])
    params_wrong = make_params(
        alpha_min=float(theta_wrong[0]), sharpness=float(theta_wrong[1]),
        threshold=float(theta_wrong[2]), requires_grad=False,
    )
    alpha_sim_wp, delta_v_sim_wp, _ = run_ssa_full_with_state(params_wrong, sys_dict, n_cycles=1)
    alpha_sim = alpha_sim_wp.numpy()
    delta_v_sim = delta_v_sim_wp.numpy()

    print(f"[inner] alpha_obs (true): min={alpha_obs.min():.3f}, max={alpha_obs.max():.3f}, "
          f"debulked frac={float((alpha_obs<0.5).mean()):.3f}")
    print(f"[inner] alpha_sim (wrong init): min={alpha_sim.min():.3f}, max={alpha_sim.max():.3f}, "
          f"debulked frac={float((alpha_sim<0.5).mean()):.3f}")
    L_initial = 0.5 * float(((alpha_sim - alpha_obs) ** 2).sum())
    print(f"[inner] initial cavity gap: L0 = {L_initial:.4e}")

    # 3. Run inner loop using the WRONG params' s, tau as the "current"
    s = float(theta_wrong[1]); tau = float(theta_wrong[2])
    delta_delta_v, L_gap, L_init = compute_damage_residual(
        delta_v_sim, alpha_obs, s, tau,
        K_inner=50, lr_inner=0.1, verbose=True,
    )

    # 4. Verify
    delta_v_corrected = delta_v_sim + delta_delta_v
    alpha_corrected = _sigmoid(-s * (delta_v_corrected - tau))
    L_residual = 0.5 * float(((alpha_corrected - alpha_obs) ** 2).sum())

    print(f"[inner] L_init = {L_init:.4e}")
    print(f"[inner] L_final residual = {L_residual:.4e}")
    print(f"[inner] reduction factor: {L_init / max(L_residual, 1e-12):.2e}×")
    print(f"[inner] |Δδ_v|_2 = {np.linalg.norm(delta_delta_v):.4f}")
    print(f"[inner] Δδ_v range: [{delta_delta_v.min():.3f}, {delta_delta_v.max():.3f}]")
    print(f"[inner] alpha_corrected: min={alpha_corrected.min():.3f}, max={alpha_corrected.max():.3f}")

    # Pass criterion: residual reduces by ≥ 10x AND alpha_corrected close to obs
    pass_ratio = (L_init / max(L_residual, 1e-12)) >= 10.0
    pass_irrev = float(delta_delta_v.min()) >= 0.0  # ≥ 0 ALWAYS (softplus guarantee)
    if pass_ratio and pass_irrev:
        print(f"[inner] PASS  (residual reduced {L_init/L_residual:.1f}× and Δδ_v ≥ 0)")
    else:
        print(f"[inner] FAIL  (ratio_pass={pass_ratio}, irrev_pass={pass_irrev})")
        return False
    return True


# =============================================================================
# Main online calibration loop (Phase 5)
# =============================================================================

def online_calibration_loop(
    sys_dict,
    real_frames,                # output of generate_synth_real_trajectory
    theta_0,                    # np.array(3,) initial guess
    K_inner=30,
    lr_inner=0.1,
    lr_outer=np.array([0.1, 1.0, 0.05]),    # per-param (alpha_min, sharp, threshold)
    n_cycles_per_frame=1,
    param_bounds=((0.5, 50.0), (1.0, 200.0), (0.05, 0.95)),
    persistent_state=True,      # Fix 1: detached persistent state across frames. False=legacy.
    knife_radius=0.1,           # propagated to make_params (was previously hardcoded 0.1)
    observable_mask=None,       # (N,) int32 or None. None = all particles in loss; otherwise mask out invisible particles.
    loss_mode='direct',         # 'direct' (current): L = ||α_sim - α_obs||²  (DiSECt-style)
                                # 'gap' (Liang 2023): L = ||α_sim_corrected - α_obs||²
                                #     where α_corrected uses Δδ_v from inner loop as constant
    state_update_mode='D',      # Fix 1 follow-up: state propagation strategy
                                #   'D' (NEW default): state_persistent = raw sim state (no residual).
                                #        Mod 1 residual computed only for L_gap diagnostic, NOT used
                                #        to advance state. Cumulative-regime correct.
                                #   'C' (legacy): state_persistent = sim + residual correction.
                                #        Pegs τ to clamp under cumulative regime. Kept for ablation.
    lambda_prior=0.0,           # P7 Fix 1: L2 anchor toward theta_0. 0=disabled.
    lambda_traj=0.0,            # P8 Mod 3: trajectory consistency. 0=disabled.
    n_traj_samples=4,           # P8: snapshots per outer step (Liang default)
    traj_history_window=20,     # P8: window from which to sample (Liang default)
    obs_noise_std=0.0,          # P10: Gaussian noise on α_obs per frame. 0=clean.
    noise_seed=12345,           # P10: separate RNG for obs noise (deterministic)
    disturbances=None,          # P11: list of (frame, delta_theta) to inject after lock
    reset_adam_on_disturbance=False,  # P11: optional — reset Adam moments at disturbance
    rng_seed=42,                # P8: reproducibility for L_traj sampling
    track_best=True,            # P7: keep best-θ-so-far snapshot
    verbose=True,
    log_every=10,
):
    """Main online residual + Adam calibration loop.

    Algorithm (mirrors Liang Algorithm 1, adapted for cutting):
      for each frame t with observation (α_obs, F_obs):
          1. Forward sim with current θ → (α_sim, δ_v_sim)
          2. Inner loop → Δδ_v (state correction for next frame)
          3. Outer Warp-tape gradient → ∂L/∂θ
          4. (P7 Fix 1) Add L_prior gradient = 2·λ·(θ - θ_0)
          5. Adam step on θ (with per-param lr)
          6. Track best θ by L_direct
          7. Save history snapshot

    Returns: dict with log, history, best_theta, best_L, best_step.
    """
    n_frames = len(real_frames)
    theta = theta_0.copy().astype(np.float64)
    theta_0_const = theta_0.copy().astype(np.float64)  # immutable anchor
    opt = AdamND(n_dims=3, lr=lr_outer)
    history = []
    log = []

    bounds_lo = np.array([b[0] for b in param_bounds])
    bounds_hi = np.array([b[1] for b in param_bounds])

    # Best-so-far tracker
    best_L = float("inf")
    best_theta = theta.copy()
    best_step = 0

    # P8: RNG for L_traj snapshot sampling
    rng = np.random.default_rng(rng_seed)
    # P10: separate RNG for obs noise (independent of sampling RNG)
    noise_rng = np.random.default_rng(noise_seed)

    # Fix 1: detached persistent state across frames. None = rest at frame 0.
    state_persistent = None

    for t, obs in enumerate(real_frames):
        # 1. Build per-frame sys_dict
        sys_t = dict(sys_dict)
        sys_t["knife_pos"] = np.array([obs["knife_pos"]], dtype=np.float32)

        # Fix 1: snapshot the INPUT state for this frame (used by history → L_traj)
        # state_pre is what gets fed into THIS frame's SSA cycle.
        if persistent_state:
            state_pre_this_frame = (None if state_persistent is None
                                    else dict(state_persistent))
        else:
            state_pre_this_frame = None

        # P10: inject fresh Gaussian noise to α_obs (simulates perception noise)
        if obs_noise_std > 0.0:
            noise = noise_rng.normal(0.0, obs_noise_std, obs["alpha_obs"].shape)
            alpha_obs_noisy = np.clip(obs["alpha_obs"] + noise, 0.0, 1.0).astype(np.float32)
        else:
            alpha_obs_noisy = obs["alpha_obs"]

        # 2. Forward sim with current θ → (alpha, delta_v, edge_dmg) for inner loop
        params_curr = make_params(
            alpha_min=float(theta[0]), sharpness=float(theta[1]),
            threshold=float(theta[2]),
            knife_radius=knife_radius,
            requires_grad=False,
        )
        alpha_sim_wp, delta_v_sim_wp, edge_dmg_sim_wp = run_ssa_full_with_state(
            params_curr, sys_t, n_cycles=n_cycles_per_frame,
            state_in=state_pre_this_frame,
        )
        alpha_sim = alpha_sim_wp.numpy().copy()
        delta_v_sim = delta_v_sim_wp.numpy().copy()
        edge_dmg_sim = edge_dmg_sim_wp.numpy().copy()

        # 3. Inner loop: compute Δδ_v on NOISY obs
        delta_delta_v, L_gap, _ = compute_damage_residual(
            delta_v_sim, alpha_obs_noisy,
            float(theta[1]), float(theta[2]),
            K_inner=K_inner, lr_inner=lr_inner, verbose=False,
        )

        # 4a. Outer gradient (Strategy A) — on NOISY obs, with same state_in
        # loss_mode='gap' uses Liang's residual-based loss; 'direct' uses raw α MSE
        ddv_for_loss = delta_delta_v if loss_mode == 'gap' else None
        L_direct, grad_data = compute_outer_gradient(
            theta, sys_t, alpha_obs_noisy, n_cycles=n_cycles_per_frame,
            state_in=state_pre_this_frame,
            knife_radius=knife_radius,
            observable_mask=observable_mask,
            delta_delta_v_const=ddv_for_loss,
        )

        # 4b. P7 Fix 1: L_prior anchor toward theta_0
        if lambda_prior > 0.0:
            grad_prior = 2.0 * lambda_prior * (theta - theta_0_const)
        else:
            grad_prior = np.zeros_like(theta)

        # 4c. P8 Mod 3: L_traj — trajectory consistency on 4 history snapshots
        if lambda_traj > 0.0:
            L_traj, grad_traj = compute_traj_loss(
                theta, history, sys_dict,
                n_samples=n_traj_samples,
                history_window=traj_history_window,
                n_cycles_per_frame=n_cycles_per_frame,
                rng=rng,
                knife_radius=knife_radius,
                observable_mask=observable_mask,
            )
        else:
            L_traj, grad_traj = 0.0, np.zeros_like(theta)

        grad = grad_data + grad_prior + lambda_traj * grad_traj

        # 5. Adam step + clip to bounds
        theta = opt.step(theta, grad)
        theta = np.clip(theta, bounds_lo, bounds_hi)

        # 5b. P11: inject disturbance if scheduled at this frame
        disturbance_applied = None
        if disturbances is not None:
            for d_frame, d_delta in disturbances:
                if t == d_frame:
                    d_delta = np.asarray(d_delta, dtype=np.float64)
                    theta_pre = theta.copy()
                    theta = np.clip(theta + d_delta, bounds_lo, bounds_hi)
                    disturbance_applied = {
                        "delta": d_delta.copy(),
                        "theta_pre": theta_pre,
                        "theta_post": theta.copy(),
                    }
                    if reset_adam_on_disturbance:
                        opt.reset()
                    if verbose:
                        print(f"  ★★★ DISTURBANCE at t={t}: Δθ = {d_delta}  "
                              f"({theta_pre.round(3)} → {theta.round(3)})")

        # 6. Best-so-far tracking
        if track_best and L_direct < best_L:
            best_L = L_direct
            best_theta = theta.copy()
            best_step = t

        # 7. Log + history
        log_entry = dict(
            t=t,
            knife_x=float(obs["knife_pos"][0]),
            L_direct=L_direct,
            L_gap=L_gap,
            alpha_min=float(theta[0]),
            sharpness=float(theta[1]),
            threshold=float(theta[2]),
            grad_a=float(grad_data[0]),
            grad_s=float(grad_data[1]),
            grad_t=float(grad_data[2]),
            grad_prior_a=float(grad_prior[0]),
            grad_prior_s=float(grad_prior[1]),
            grad_prior_t=float(grad_prior[2]),
            L_traj=float(L_traj),
            grad_traj_a=float(grad_traj[0]),
            grad_traj_s=float(grad_traj[1]),
            grad_traj_t=float(grad_traj[2]),
            disturbance=disturbance_applied,
        )
        log.append(log_entry)

        # History (used by L_traj on subsequent frames) — STORE NOISY obs
        # so L_traj resamples are consistent with what was actually observed.
        # Fix 1: also store state_pre (input state to this frame) so L_traj
        # replay starts from the correct cumulative state, not rest.
        history.append(dict(
            knife_pos=obs["knife_pos"].copy(),
            alpha_obs=alpha_obs_noisy.copy(),
            delta_v_corrected=(delta_v_sim + delta_delta_v).copy(),
            state_pre=state_pre_this_frame,   # None in legacy mode
        ))

        # Fix 1: advance persistent state for next frame.
        # Two strategies (see state_update_mode docstring above):
        #   'D' (DEFAULT): raw sim state, no residual injection. Empirically
        #        gives clean convergence in cumulative regime when active
        #        particles per frame ≥ ~50.
        #   'C' (legacy): residual-corrected state. Mod 1's Δδ_v added to dv;
        #        α recomputed via current-θ sigmoid. Pegs τ to clamp.
        if persistent_state:
            if state_update_mode == 'D':
                state_persistent = dict(
                    alpha=alpha_sim.astype(np.float32),
                    edge_damage=edge_dmg_sim.astype(np.float32),
                    particle_damage=delta_v_sim.astype(np.float32),
                )
            elif state_update_mode == 'C':
                delta_v_next = (delta_v_sim + delta_delta_v).astype(np.float32)
                alpha_next = _sigmoid(
                    -float(theta[1]) * (delta_v_next - float(theta[2]))
                ).astype(np.float32)
                state_persistent = dict(
                    alpha=alpha_next,
                    edge_damage=edge_dmg_sim.astype(np.float32),
                    particle_damage=delta_v_next,
                )
            else:
                raise ValueError(f"unknown state_update_mode={state_update_mode!r}, expected 'C' or 'D'")

        if verbose and (t % log_every == 0 or t == n_frames - 1):
            best_tag = " ★" if t == best_step else ""
            traj_tag = f"  L_traj={L_traj:.4e}" if lambda_traj > 0 else ""
            print(f"[t={t:3d}] L_dir={L_direct:.4e}  L_gap={L_gap:.4e}{traj_tag}  "
                  f"θ=({theta[0]:.3f}, {theta[1]:.3f}, {theta[2]:.3f}){best_tag}")

    return dict(
        log=log,
        history=history,
        best_theta=best_theta,
        best_L=best_L,
        best_step=best_step,
        final_theta=theta.copy(),
        final_L=float(log[-1]["L_direct"]),
        theta_0=theta_0_const,
        lambda_prior=lambda_prior,
        lambda_traj=lambda_traj,
        obs_noise_std=obs_noise_std,
        persistent_state=persistent_state,
        state_update_mode=state_update_mode,
        loss_mode=loss_mode,
    )


def persistent_state_test():
    """Fix 1 sanity: persistent-state synth trajectory should show cavity
    growing monotonically across frames (no resets to rest).

    Test design:
      1. Build N=20 1D particle line, knife sweep x=0.35 → 0.65, 40 frames.
      2. Generate trajectories TWO ways:
           (a) accumulate=True  (Fix 1, new default)
           (b) accumulate=False (legacy, frame-independent)
      3. Verify:
           (a) cavity monotonically grows: debulked_frac_t >= debulked_frac_{t-1}
           (b) alpha is monotonically non-increasing per vertex
           (c) legacy mode shows non-monotonic cavity (proves the regimes differ)

    PASS criterion: accumulate=True is monotonic; accumulate=False is not.
    """
    print("\n[persist] testing persistent-state synth trajectory ...")
    wp.init()
    sys_dict = setup_synthetic_system(N=20)
    n_frames = 40
    traj = make_knife_trajectory(n_frames=n_frames, x_start=0.35, x_end=0.65)
    theta_true = np.array([3.0, 20.0, 0.5])

    # --- Mode A: accumulate=True (Fix 1) ---
    frames_acc = generate_synth_real_trajectory(
        sys_dict, theta_true, traj, k_bulk_true=1.0,
        n_cycles_per_frame=1, accumulate=True,
    )
    deb_acc = np.array([(f["alpha_obs"] < 0.5).mean() for f in frames_acc])
    alpha_acc = np.stack([f["alpha_obs"] for f in frames_acc])  # (T, N)

    # --- Mode B: accumulate=False (legacy) ---
    frames_leg = generate_synth_real_trajectory(
        sys_dict, theta_true, traj, k_bulk_true=1.0,
        n_cycles_per_frame=1, accumulate=False,
    )
    deb_leg = np.array([(f["alpha_obs"] < 0.5).mean() for f in frames_leg])
    alpha_leg = np.stack([f["alpha_obs"] for f in frames_leg])

    print(f"[persist] accumulate=True   debulked frac: "
          f"min={deb_acc.min():.3f}, max={deb_acc.max():.3f}, "
          f"final={deb_acc[-1]:.3f}")
    print(f"[persist] accumulate=False  debulked frac: "
          f"min={deb_leg.min():.3f}, max={deb_leg.max():.3f}, "
          f"final={deb_leg[-1]:.3f}")

    # Check (a): cavity monotonically grows in accumulate mode
    # Allow tiny float noise (1e-3) on the debulked fraction.
    deb_diff_acc = np.diff(deb_acc)
    monotone_cavity = np.all(deb_diff_acc >= -1e-3)
    print(f"[persist] (a) accumulate cavity monotonic (Δdeb ≥ -1e-3): "
          f"{'PASS' if monotone_cavity else 'FAIL'}  "
          f"(min Δdeb = {deb_diff_acc.min():.4f})")

    # Check (b): alpha per-vertex monotonically non-increasing in accumulate mode
    alpha_diff_acc = np.diff(alpha_acc, axis=0)  # (T-1, N)
    # Allow tiny float noise per element
    monotone_alpha = bool(np.all(alpha_diff_acc <= 1e-3))
    max_increase = float(alpha_diff_acc.max())
    print(f"[persist] (b) accumulate alpha non-increasing per vertex "
          f"(max Δα ≤ 1e-3): {'PASS' if monotone_alpha else 'FAIL'}  "
          f"(max per-vertex Δα = {max_increase:.4f})")

    # Check (c): legacy mode shows NON-monotonic cavity (different regime)
    # Knife sweep means each frame independently sees a *moving* knife → cavity
    # appears at different positions → non-monotonic across frames.
    deb_diff_leg = np.diff(deb_leg)
    non_monotone_legacy = bool(np.any(deb_diff_leg < -1e-3))
    print(f"[persist] (c) legacy cavity NON-monotonic (proves regimes differ): "
          f"{'PASS' if non_monotone_legacy else 'FAIL'}  "
          f"(min Δdeb_legacy = {deb_diff_leg.min():.4f})")

    # Show concrete progression in accumulate mode
    print("[persist] accumulate-mode cavity progression (every 5 frames):")
    for t in [0, 5, 10, 15, 20, 25, 30, 35, 39]:
        if t < n_frames:
            print(f"  frame {t:2d}: knife_x={traj[t,0]:.3f}  "
                  f"deb={deb_acc[t]:.3f}  "
                  f"α=[{alpha_acc[t].min():.3f}, {alpha_acc[t].max():.3f}]")

    all_pass = monotone_cavity and monotone_alpha and non_monotone_legacy
    print(f"[persist] {'PASS' if all_pass else 'FAIL'}  "
          f"(monotone_cavity={monotone_cavity}, monotone_alpha={monotone_alpha}, "
          f"legacy_non_monotone={non_monotone_legacy})")
    return all_pass


def outer_grad_test():
    """Phase 4 test: verify outer gradient points toward theta_true.

    theta_true = (3.0, 20.0, 0.5)
    theta_wrong = (5.0, 10.0, 0.5)

    Since alpha_min needs to DECREASE (5→3), gradient should be POSITIVE.
    Since sharpness needs to INCREASE (10→20), gradient should be NEGATIVE.
    Threshold matches, gradient should be near zero.
    """
    print("\n[outer] testing gradient direction ...")
    wp.init()
    sys_dict = setup_synthetic_system(N=20)
    sys_dict["knife_pos"] = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

    # 1. Generate alpha_obs from theta_true
    theta_true = np.array([3.0, 20.0, 0.5])
    params_true = make_params(
        alpha_min=float(theta_true[0]), sharpness=float(theta_true[1]),
        threshold=float(theta_true[2]), requires_grad=False,
    )
    alpha_obs = run_ssa_full(params_true, sys_dict, n_cycles=1).numpy().copy()

    # 2. Compute gradient at theta_wrong
    theta_wrong = np.array([5.0, 10.0, 0.5])
    loss, grad = compute_outer_gradient(theta_wrong, sys_dict, alpha_obs, n_cycles=1)

    print(f"[outer] theta_wrong = (α_min={theta_wrong[0]}, sharp={theta_wrong[1]}, τ={theta_wrong[2]})")
    print(f"[outer] theta_true  = (α_min={theta_true[0]}, sharp={theta_true[1]}, τ={theta_true[2]})")
    print(f"[outer] loss = {loss:.4e}")
    print(f"[outer] grad = (∂L/∂α_min={grad[0]:+.4e}, ∂L/∂sharp={grad[1]:+.4e}, "
          f"∂L/∂τ={grad[2]:+.4e})")
    print(f"[outer] expected directions:")
    print(f"  ∂L/∂α_min > 0  (so Adam decreases α_min from 5 to 3)  → got {grad[0]:+.2e}  "
          f"{'✓' if grad[0] > 0 else '✗'}")
    print(f"  ∂L/∂sharp < 0  (so Adam increases sharp from 10 to 20)  → got {grad[1]:+.2e}  "
          f"{'✓' if grad[1] < 0 else '✗'}")
    print(f"  |∂L/∂τ| small  (τ already correct)  → got {abs(grad[2]):.2e}  "
          f"{'✓' if abs(grad[2]) < max(abs(grad[0]), abs(grad[1])) else '⚠'}")

    pass_dir_a = grad[0] > 0
    pass_dir_s = grad[1] < 0
    if pass_dir_a and pass_dir_s:
        print(f"[outer] PASS  (both gradient directions point toward theta_true)")
        return True
    else:
        print(f"[outer] FAIL  (gradient does NOT point toward theta_true)")
        return False


def plot_convergence(log_path=None, out_path=None):
    """Phase 6: plot online residual calibration convergence.

    Generates 2x2 grid:
      (0,0) alpha_min trajectory vs true
      (0,1) sharpness trajectory vs true
      (1,0) threshold trajectory vs true
      (1,1) L_direct + L_gap log-scale vs frame
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if log_path is None:
        log_path = REPO_ROOT / "tools" / "online_residual_log.npz"
    if out_path is None:
        out_path = REPO_ROOT / "tools" / "online_residual_convergence.png"

    log = np.load(log_path)
    t = log["t"]
    theta_true = log["theta_true"]
    theta_0 = log["theta_0"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "ReSECT Online Residual Calibration — Synth-to-Synth (POC)\n"
        f"θ_0=({theta_0[0]:.1f}, {theta_0[1]:.1f}, {theta_0[2]:.2f})  →  "
        f"θ_true=({theta_true[0]:.1f}, {theta_true[1]:.1f}, {theta_true[2]:.2f})",
        fontsize=11,
    )

    param_names = ["alpha_min (κ)", "sharpness (s)", "threshold (τ)"]
    param_keys = ["alpha_min", "sharpness", "threshold"]
    positions = [(0, 0), (0, 1), (1, 0)]

    for (r, c), name, key, true_v in zip(positions, param_names, param_keys, theta_true):
        ax = axes[r, c]
        ax.plot(t, log[key], "b-", label="estimate", linewidth=1.5)
        ax.axhline(true_v, color="red", linestyle="--", label="ground truth", linewidth=1.2)
        ax.axhline(theta_0[positions.index((r, c))], color="gray", linestyle=":",
                   label="init θ_0", linewidth=1.0)
        ax.set_xlabel("frame")
        ax.set_ylabel(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        final = float(log[key][-1])
        rel = abs(final - true_v) / max(abs(true_v), 1e-6) * 100
        ax.set_title(f"{name}: final={final:.3f}  rel_err={rel:.2f}%", fontsize=10)

    # Loss panel
    ax = axes[1, 1]
    ax.semilogy(t, log["L_direct"], "b-", label="L_direct (cavity gap)", linewidth=1.2)
    ax.semilogy(t, log["L_gap"], "g--", label="L_gap (residual norm)", linewidth=1.0)
    ax.set_xlabel("frame")
    ax.set_ylabel("loss (log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Loss trajectories", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {out_path}")
    return out_path


def integration_test(n_frames=100, lambda_prior=0.0, lambda_traj=0.0, log_out_name=None):
    """Phase 5/7 test: full online loop, synth-to-synth.

    Setup:
      theta_true = (3.0, 20.0, 0.5)
      theta_0    = (5.0, 10.0, 0.5)   (alpha_min 67% off, sharp 50% off)
      knife sweep x=0.35→0.65 (broad cavity)

    Args:
      n_frames     — trajectory length
      lambda_prior — P7 Fix 1 anchor strength (0 = original POC behavior)
      log_out_name — npz file basename for saving log
    """
    print(f"\n[integ] running {n_frames}-frame online calibration "
          f"(λ_prior={lambda_prior}, λ_traj={lambda_traj}) ...")
    wp.init()
    sys_dict = setup_synthetic_system(N=20)

    traj = make_knife_trajectory(n_frames=n_frames, x_start=0.35, x_end=0.65)
    theta_true = np.array([3.0, 20.0, 0.5])
    real_frames = generate_synth_real_trajectory(
        sys_dict, theta_true, traj,
        k_bulk_true=1.0, n_cycles_per_frame=1,
    )

    theta_0 = np.array([5.0, 10.0, 0.5])
    print(f"[integ] theta_true   = {theta_true}")
    print(f"[integ] theta_0      = {theta_0}")

    t_start = time.time()
    result = online_calibration_loop(
        sys_dict, real_frames, theta_0,
        K_inner=30, lr_inner=0.1,
        lr_outer=np.array([0.1, 1.0, 0.05]),
        lambda_prior=lambda_prior,
        lambda_traj=lambda_traj,
        verbose=True, log_every=max(1, n_frames // 10),
    )
    elapsed = time.time() - t_start

    log = result["log"]
    theta_final = result["final_theta"]
    theta_best = result["best_theta"]
    best_step = result["best_step"]
    best_L = result["best_L"]

    err_final = np.abs(theta_final - theta_true)
    rel_final = err_final / np.maximum(np.abs(theta_true), 1e-6)
    err_best = np.abs(theta_best - theta_true)
    rel_best = err_best / np.maximum(np.abs(theta_true), 1e-6)

    print(f"\n[integ] Done in {elapsed:.1f}s ({elapsed/n_frames*1000:.1f} ms/frame)")
    print(f"[integ] theta_true   = {theta_true}")
    print(f"[integ] theta_FINAL  = {theta_final}  (last step)")
    print(f"  rel err            = {(rel_final * 100).round(2)} %")
    print(f"[integ] theta_BEST   = {theta_best}  ★ at step {best_step} (L={best_L:.2e})")
    print(f"  rel err            = {(rel_best * 100).round(2)} %")

    n_pass_final = int((rel_final < 0.30).sum())
    n_pass_best = int((rel_best < 0.30).sum())
    pass_overall = n_pass_best >= 2

    print(f"[integ] Final:  {n_pass_final}/3 params within 30%")
    print(f"[integ] Best:   {n_pass_best}/3 params within 30%  ← Fix1 metric")
    if pass_overall:
        print(f"[integ] PASS (best-so-far has {n_pass_best}/3 within 30%)")
    else:
        print(f"[integ] PARTIAL")

    # Save log
    log_arr = dict(
        t=np.array([e["t"] for e in log]),
        L_direct=np.array([e["L_direct"] for e in log]),
        L_gap=np.array([e["L_gap"] for e in log]),
        alpha_min=np.array([e["alpha_min"] for e in log]),
        sharpness=np.array([e["sharpness"] for e in log]),
        threshold=np.array([e["threshold"] for e in log]),
        grad_a=np.array([e["grad_a"] for e in log]),
        grad_s=np.array([e["grad_s"] for e in log]),
        grad_t=np.array([e["grad_t"] for e in log]),
        grad_prior_a=np.array([e["grad_prior_a"] for e in log]),
        grad_prior_s=np.array([e["grad_prior_s"] for e in log]),
        grad_prior_t=np.array([e["grad_prior_t"] for e in log]),
        L_traj=np.array([e["L_traj"] for e in log]),
        grad_traj_a=np.array([e["grad_traj_a"] for e in log]),
        grad_traj_s=np.array([e["grad_traj_s"] for e in log]),
        grad_traj_t=np.array([e["grad_traj_t"] for e in log]),
        theta_true=theta_true,
        theta_0=theta_0,
        theta_best=theta_best,
        theta_final=theta_final,
        best_step=np.array([best_step]),
        lambda_prior=np.array([lambda_prior]),
        lambda_traj=np.array([lambda_traj]),
    )
    if log_out_name is None:
        log_out_name = "online_residual_log.npz"
    out_path = REPO_ROOT / "tools" / log_out_name
    np.savez(out_path, **log_arr)
    print(f"[integ] log saved to {out_path}")

    return pass_overall, log, result


# =============================================================================
# Cube integration test — Fix 1 cumulative-regime canonical validation
# =============================================================================

def cube_integration_test(
    N_per_dim: int = 12,
    extent: tuple = (0.3, 0.7),
    n_frames: int = 200,
    state_update_mode: str = 'D',
    lambda_prior: float = 0.0,
    lambda_traj: float = 0.0,
    knife_radius: float = 0.1,
    log_out_name: str = None,
):
    """Canonical cumulative-regime POC on 3D cube.

    Setup:
        cube N_per_dim^3 particles in extent³ (default 12³ = 1728 particles in
        [0.3, 0.7]³ → spacing 0.036, ~86 active particles per frame at r=0.1).
        knife sweep along x at center, 200 frames.
        θ_true = (3.0, 20.0, 0.5)   θ_0 = (5.0, 10.0, 0.5)

    Expected (default args):
        rel_err α_min  < 5%
        rel_err sharp  < 5%
        rel_err τ      < 5%      (no clamp pegging — proves Mode D works)

    Args:
        N_per_dim       — cube resolution along each axis (12 = sweet spot)
        extent          — cube extent in each axis
        n_frames        — trajectory length
        state_update_mode — 'D' (default, new) or 'C' (legacy ablation)
        lambda_prior    — P7 anchor strength (0=disabled)
        lambda_traj     — P8 trajectory-consistency weight (0=disabled)
        knife_radius    — knife influence radius
        log_out_name    — npz file basename for saving log
    """
    print(f"\n[cube] cumulative-regime POC on cube {N_per_dim}³ extent {extent}")
    print(f"[cube] mode={state_update_mode}  n_frames={n_frames}  "
          f"λ_prior={lambda_prior}  λ_traj={lambda_traj}")
    wp.init()

    sys_dict = setup_cube_system(N_per_dim=N_per_dim, extent=extent)
    print(f"[cube] N={sys_dict['N']}  E={sys_dict['E']}  knife_r={knife_radius}")
    spacing = (extent[1] - extent[0]) / (N_per_dim - 1)
    active_est = int((4.0 / 3.0) * np.pi * knife_radius**3 / spacing**3)
    print(f"[cube] spacing={spacing:.4f}  est. active per frame ≈ {active_est}  "
          f"({'PASS' if active_est >= 50 else 'WARN: < 50 threshold'})")

    # Knife sweep limited to middle 60% of extent to enter/exit cleanly
    lo, hi = extent; span = hi - lo
    traj = make_cube_knife_trajectory(
        n_frames=n_frames,
        x_start=lo + 0.1 * span,
        x_end=hi - 0.1 * span,
        y=(lo + hi) / 2,
        z=(lo + hi) / 2,
    )

    theta_true = np.array([3.0, 20.0, 0.5])
    theta_0 = np.array([5.0, 10.0, 0.5])
    print(f"[cube] θ_true   = {theta_true}")
    print(f"[cube] θ_0      = {theta_0}")

    real_frames = generate_synth_real_trajectory(
        sys_dict, theta_true, traj, k_bulk_true=1.0,
        n_cycles_per_frame=1, accumulate=True,
    )
    cavity_init = float((real_frames[0]["alpha_obs"] < 0.5).mean())
    cavity_final = float((real_frames[-1]["alpha_obs"] < 0.5).mean())
    print(f"[cube] cavity fraction: t=0 {cavity_init:.3f}  →  t={n_frames-1} {cavity_final:.3f}")

    t_start = time.time()
    result = online_calibration_loop(
        sys_dict, real_frames, theta_0,
        K_inner=30, lr_inner=0.1,
        lr_outer=np.array([0.1, 1.0, 0.05]),
        state_update_mode=state_update_mode,
        lambda_prior=lambda_prior,
        lambda_traj=lambda_traj,
        verbose=True, log_every=max(1, n_frames // 10),
    )
    elapsed = time.time() - t_start

    log = result["log"]
    theta_final = result["final_theta"]
    theta_best = result["best_theta"]

    err_final = np.abs(theta_final - theta_true)
    rel_final = err_final / np.maximum(np.abs(theta_true), 1e-6)
    err_best = np.abs(theta_best - theta_true)
    rel_best = err_best / np.maximum(np.abs(theta_true), 1e-6)

    # Steady-state stats: last half of frames
    half = n_frames // 2
    ss_alpha = np.array([e["alpha_min"] for e in log[-half:]])
    ss_sharp = np.array([e["sharpness"] for e in log[-half:]])
    ss_tau = np.array([e["threshold"] for e in log[-half:]])
    ss = np.array([ss_alpha.mean(), ss_sharp.mean(), ss_tau.mean()])
    rel_ss = np.abs(ss - theta_true) / np.maximum(np.abs(theta_true), 1e-6)
    osc = np.array([ss_alpha.std(), ss_sharp.std(), ss_tau.std()])

    print(f"\n[cube] Done in {elapsed:.1f}s ({elapsed/n_frames*1000:.1f} ms/frame)")
    print(f"[cube] θ_true       = {theta_true}")
    print(f"[cube] θ_FINAL      = {theta_final.round(3)}  rel_err = {(rel_final*100).round(2)} %")
    print(f"[cube] θ_BEST       = {theta_best.round(3)}  rel_err = {(rel_best*100).round(2)} %  (step {result['best_step']})")
    print(f"[cube] θ_SS (last half avg) = {ss.round(3)}  rel_err = {(rel_ss*100).round(2)} %")
    print(f"[cube] θ_SS std = {osc.round(4)}  (oscillation in steady state)")

    # Clamp check: did any param hit a bound?
    tau_at_clamp_upper = theta_final[2] >= 0.949
    tau_at_clamp_lower = theta_final[2] <= 0.051
    if tau_at_clamp_upper or tau_at_clamp_lower:
        print(f"[cube] ⚠️ τ pegged at clamp ({'upper' if tau_at_clamp_upper else 'lower'}) "
              f"— gradient was misdirected. Check state_update_mode.")
    else:
        print(f"[cube] ✓ τ converged interior (no clamp peg) — healthy gradient direction")

    # Pass/fail: rel_ss < 5% on all 3 params
    pass_ss = bool((rel_ss * 100 < 5).all())
    pass_final = bool((rel_final * 100 < 5).all())
    if pass_ss and pass_final:
        print(f"[cube] PASS  (all 3 params < 5% rel_err on both final and steady-state)")
    elif pass_ss:
        print(f"[cube] PARTIAL  (steady-state < 5% but final not — oscillating around true)")
    elif pass_final:
        print(f"[cube] PARTIAL  (final < 5% but steady-state not — late convergence)")
    else:
        print(f"[cube] FAIL  (rel_err > 5% on at least one param)")

    # Save log
    log_arr = dict(
        t=np.array([e["t"] for e in log]),
        L_direct=np.array([e["L_direct"] for e in log]),
        L_gap=np.array([e["L_gap"] for e in log]),
        alpha_min=np.array([e["alpha_min"] for e in log]),
        sharpness=np.array([e["sharpness"] for e in log]),
        threshold=np.array([e["threshold"] for e in log]),
        grad_a=np.array([e["grad_a"] for e in log]),
        grad_s=np.array([e["grad_s"] for e in log]),
        grad_t=np.array([e["grad_t"] for e in log]),
        L_traj=np.array([e["L_traj"] for e in log]),
        theta_true=theta_true,
        theta_0=theta_0,
        theta_best=theta_best,
        theta_final=theta_final,
        theta_ss=ss,
        N_per_dim=np.array([N_per_dim]),
        extent=np.array(extent),
        n_frames=np.array([n_frames]),
        state_update_mode=np.array([state_update_mode]),
    )
    if log_out_name is None:
        log_out_name = f"online_residual_log_cube_{N_per_dim}.npz"
    out_path = REPO_ROOT / "tools" / log_out_name
    np.savez(out_path, **log_arr)
    print(f"[cube] log saved to {out_path}")

    return pass_ss and pass_final, log, result


# =============================================================================
# Real-mesh integration test — surface knife, real cube_tet.msh
# =============================================================================

def mesh_integration_test(
    mesh_path: str = None,
    n_frames: int = 400,
    state_update_mode: str = 'D',
    loss_mode: str = 'direct',
    lambda_prior: float = 0.0,
    lambda_traj: float = 0.0,
    knife_radius: float = 0.025,
    surface_axis: str = 'y',
    theta_true: tuple = (3.0, 20.0, 0.5),
    theta_0: tuple = (5.0, 10.0, 0.5),
    log_out_name: str = None,
):
    """Real-mesh cumulative-regime POC. One sim acts as `real` (θ_true),
    another runs calibration to recover θ from those observations.

    Setup:
        - cube_tet.msh (or any tet mesh) loaded via load_msh_system
        - surface knife: half-exposed at top face (max of surface_axis)
        - slow sweep across the longer non-up axis
        - Mode D state propagation (no residual injection)
        - cumulative damage across frames

    Validates the real-to-sim pipeline on the actual mesh you'll use in
    phantom experiments (or its synthetic twin), not the regular lattice.

    Args:
        mesh_path        — path to .msh (default: resources/cube/cube_tet.msh)
        n_frames         — trajectory length (slower with more frames)
        state_update_mode — 'D' (new default) or 'C' (legacy ablation)
        lambda_prior     — P7 anchor strength
        lambda_traj      — P8 traj consistency weight
        knife_radius     — small surface tool (default 0.025)
        surface_axis     — which axis is up ('x', 'y', 'z')
        theta_true       — (α_min, sharpness, τ) for the 'real' observation
        theta_0          — initial guess for calibration

    Returns:
        (pass: bool, log: list, result: dict)
    """
    if mesh_path is None:
        mesh_path = str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh")

    print(f"\n[mesh-integ] cumulative-regime POC on real mesh")
    print(f"[mesh-integ] mesh={mesh_path}")
    print(f"[mesh-integ] mode={state_update_mode}  n_frames={n_frames}  "
          f"λ_prior={lambda_prior}  λ_traj={lambda_traj}")
    wp.init()

    sys_dict = load_msh_system(mesh_path)
    bbox_lo, bbox_hi = sys_dict["bbox"]
    spacing = float(np.median(np.linalg.norm(
        sys_dict["pos"][sys_dict["edge_a"]] - sys_dict["pos"][sys_dict["edge_b"]],
        axis=1)))
    # Surface knife → half-sphere embedded → (2/3)π r³ active volume
    density = sys_dict["N"] / np.prod(bbox_hi - bbox_lo)
    half_vol = (2.0 / 3.0) * np.pi * knife_radius ** 3
    active_est = int(half_vol * density)
    print(f"[mesh-integ] N={sys_dict['N']}  E={sys_dict['E']}  "
          f"median spacing={spacing:.4f}")
    print(f"[mesh-integ] knife_r={knife_radius}  active per frame ≈ {active_est}  "
          f"({'PASS ≥ 50' if active_est >= 50 else 'WARN < 50 threshold'})")

    # Surface trajectory
    traj = make_surface_knife_trajectory(
        bbox=(bbox_lo, bbox_hi),
        n_frames=n_frames,
        surface_axis=surface_axis,
        margin_fraction=0.15,
    )
    print(f"[mesh-integ] trajectory: surface={surface_axis}={float(bbox_hi[{'x':0,'y':1,'z':2}[surface_axis]]):.3f}  "
          f"sweep [{traj[0]} → {traj[-1]}]")

    theta_true_np = np.asarray(theta_true, dtype=np.float64)
    theta_0_np = np.asarray(theta_0, dtype=np.float64)
    print(f"[mesh-integ] θ_true   = {theta_true_np}")
    print(f"[mesh-integ] θ_0      = {theta_0_np}")

    real_frames = generate_synth_real_trajectory(
        sys_dict, theta_true_np, traj, k_bulk_true=1.0,
        n_cycles_per_frame=1, accumulate=True,
    )
    # IMPORTANT: knife_radius is currently held inside `make_params` as 0.1 by
    # default. We need our small surface-knife radius to propagate. Re-build
    # frames if needed — but simpler: patch make_params inside the inner SSA
    # via state. Actually generate_synth_real_trajectory uses make_params with
    # knife_radius=0.1 baked in. Need to override.
    # For now, run generate manually with the right knife_radius:
    real_frames = []
    state_p = None
    for kp in traj:
        sys_t = dict(sys_dict)
        sys_t["knife_pos"] = np.array([kp], dtype=np.float32)
        params_true = make_params(
            alpha_min=float(theta_true_np[0]),
            sharpness=float(theta_true_np[1]),
            threshold=float(theta_true_np[2]),
            knife_radius=knife_radius,           # ← USE OUR SMALL RADIUS
            requires_grad=False,
        )
        a_wp, dv_wp, ed_wp = run_ssa_full_with_state(
            params_true, sys_t, n_cycles=1, state_in=state_p,
        )
        a_np = a_wp.numpy().copy()
        real_frames.append(dict(
            t=len(real_frames),
            knife_pos=np.asarray(kp, dtype=np.float32).copy(),
            alpha_obs=a_np,
            force_obs=synth_force_proxy(a_np, 1.0),
        ))
        state_p = dict(
            alpha=a_np.astype(np.float32),
            edge_damage=ed_wp.numpy().astype(np.float32),
            particle_damage=dv_wp.numpy().astype(np.float32),
        )
    cavity_init = float((real_frames[0]["alpha_obs"] < 0.5).mean())
    cavity_final = float((real_frames[-1]["alpha_obs"] < 0.5).mean())
    print(f"[mesh-integ] cavity fraction: t=0 {cavity_init:.4f}  →  t={n_frames-1} {cavity_final:.4f}")

    t_start = time.time()
    result = online_calibration_loop(
        sys_dict, real_frames, theta_0_np,
        K_inner=30, lr_inner=0.1,
        lr_outer=np.array([0.1, 1.0, 0.05]),
        knife_radius=knife_radius,    # match the obs-generating knife
        state_update_mode=state_update_mode,
        loss_mode=loss_mode,
        lambda_prior=lambda_prior,
        lambda_traj=lambda_traj,
        verbose=True, log_every=max(1, n_frames // 10),
    )
    elapsed = time.time() - t_start

    log = result["log"]
    theta_final = result["final_theta"]
    theta_best = result["best_theta"]

    err_final = np.abs(theta_final - theta_true_np)
    rel_final = err_final / np.maximum(np.abs(theta_true_np), 1e-6)
    err_best = np.abs(theta_best - theta_true_np)
    rel_best = err_best / np.maximum(np.abs(theta_true_np), 1e-6)

    half = n_frames // 2
    ss_alpha = np.array([e["alpha_min"] for e in log[-half:]])
    ss_sharp = np.array([e["sharpness"] for e in log[-half:]])
    ss_tau = np.array([e["threshold"] for e in log[-half:]])
    ss = np.array([ss_alpha.mean(), ss_sharp.mean(), ss_tau.mean()])
    rel_ss = np.abs(ss - theta_true_np) / np.maximum(np.abs(theta_true_np), 1e-6)
    osc = np.array([ss_alpha.std(), ss_sharp.std(), ss_tau.std()])

    print(f"\n[mesh-integ] Done in {elapsed:.1f}s ({elapsed/n_frames*1000:.1f} ms/frame)")
    print(f"[mesh-integ] θ_true       = {theta_true_np}")
    print(f"[mesh-integ] θ_FINAL      = {theta_final.round(3)}  rel_err = {(rel_final*100).round(2)} %")
    print(f"[mesh-integ] θ_BEST       = {theta_best.round(3)}  rel_err = {(rel_best*100).round(2)} %  (step {result['best_step']})")
    print(f"[mesh-integ] θ_SS (last-half avg) = {ss.round(3)}  rel_err = {(rel_ss*100).round(2)} %  std={osc.round(4)}")

    tau_at_clamp_upper = theta_final[2] >= 0.949
    tau_at_clamp_lower = theta_final[2] <= 0.051
    if tau_at_clamp_upper or tau_at_clamp_lower:
        print(f"[mesh-integ] ⚠️ τ pegged at clamp ({'upper' if tau_at_clamp_upper else 'lower'}) "
              f"— check state_update_mode and knife/mesh sizing.")
    else:
        print(f"[mesh-integ] ✓ τ converged interior (no clamp peg)")

    pass_ss = bool((rel_ss * 100 < 5).all())
    pass_final = bool((rel_final * 100 < 5).all())
    if pass_ss and pass_final:
        print(f"[mesh-integ] PASS  (all 3 params < 5% rel_err)")
    elif pass_ss or pass_final:
        print(f"[mesh-integ] PARTIAL")
    else:
        print(f"[mesh-integ] FAIL")

    log_arr = dict(
        t=np.array([e["t"] for e in log]),
        L_direct=np.array([e["L_direct"] for e in log]),
        L_gap=np.array([e["L_gap"] for e in log]),
        alpha_min=np.array([e["alpha_min"] for e in log]),
        sharpness=np.array([e["sharpness"] for e in log]),
        threshold=np.array([e["threshold"] for e in log]),
        grad_a=np.array([e["grad_a"] for e in log]),
        grad_s=np.array([e["grad_s"] for e in log]),
        grad_t=np.array([e["grad_t"] for e in log]),
        L_traj=np.array([e["L_traj"] for e in log]),
        theta_true=theta_true_np,
        theta_0=theta_0_np,
        theta_best=theta_best,
        theta_final=theta_final,
        theta_ss=ss,
        mesh_path=np.array([mesh_path]),
        n_frames=np.array([n_frames]),
        knife_radius=np.array([knife_radius]),
        state_update_mode=np.array([state_update_mode]),
    )
    if log_out_name is None:
        log_out_name = "online_residual_log_mesh.npz"
    out_path = REPO_ROOT / "tools" / log_out_name
    np.savez(out_path, **log_arr)
    print(f"[mesh-integ] log saved to {out_path}")

    return (pass_ss and pass_final), log, result


def disturbance_recovery_test(
    n_frames=600,
    lambda_traj=0.5,
    disturbances=None,
    reset_adam=False,
):
    """P11: inject perturbations to θ AFTER lock to test stable-equilibrium
    hypothesis. If algorithm has true gradient-driven attraction to θ_true,
    perturbations should be recovered.

    Default disturbance schedule (designed to test isolated + combined hits):
      frame 200: α_min jump   3.0 → 5.0  (after first lock)
      frame 350: sharpness jump  20 → 10  (after first recovery)
      frame 500: all 3 jump simultaneously to (5.0, 10.0, 0.7)
    """
    if disturbances is None:
        disturbances = [
            (200, np.array([+2.0, 0.0, 0.0])),    # α_min +2  (3→5)
            (350, np.array([0.0, -10.0, 0.0])),   # sharp -10 (20→10)
            (500, np.array([+2.0, -10.0, +0.2])), # combined (3,20,0.5)→(5,10,0.7)
        ]

    print("=" * 80)
    print(f"P11 DISTURBANCE RECOVERY TEST  (λ_traj={lambda_traj}, n_frames={n_frames})")
    print(f"  Disturbance schedule:")
    for d_frame, d_delta in disturbances:
        print(f"    frame {d_frame}: Δθ = {d_delta}")
    print("=" * 80)

    wp.init()
    sys_dict = setup_synthetic_system(N=20)
    traj = make_knife_trajectory(n_frames=n_frames, x_start=0.35, x_end=0.65)
    theta_true = np.array([3.0, 20.0, 0.5])
    real_frames = generate_synth_real_trajectory(
        sys_dict, theta_true, traj, k_bulk_true=1.0, n_cycles_per_frame=1,
    )

    theta_0 = np.array([5.0, 10.0, 0.5])
    result = online_calibration_loop(
        sys_dict, real_frames, theta_0,
        K_inner=30, lr_inner=0.1,
        lr_outer=np.array([0.1, 1.0, 0.05]),
        lambda_prior=0.0,
        lambda_traj=lambda_traj,
        disturbances=disturbances,
        reset_adam_on_disturbance=reset_adam,
        verbose=True, log_every=max(1, n_frames // 30),
    )
    log = result["log"]

    # Recovery analysis: for each disturbance, measure time to re-lock
    print("\n" + "=" * 80)
    print("RECOVERY ANALYSIS")
    print("=" * 80)
    for d_frame, d_delta in disturbances:
        # Post-disturbance, find first frame where all |θ - θ_true| / θ_true < 1%
        recovery_frame = None
        max_excursion = 0.0
        for entry in log:
            if entry["t"] <= d_frame:
                continue
            theta_t = np.array([entry["alpha_min"], entry["sharpness"], entry["threshold"]])
            rel_err = np.abs(theta_t - theta_true) / np.maximum(np.abs(theta_true), 1e-6)
            max_excursion = max(max_excursion, rel_err.max())
            if rel_err.max() < 0.01:
                recovery_frame = entry["t"]
                break

        if recovery_frame is not None:
            recovery_time = recovery_frame - d_frame
            print(f"  Disturbance at frame {d_frame}, Δ={d_delta.round(2)}: "
                  f"recovered to <1% by frame {recovery_frame} (recovery time = {recovery_time} frames)")
        else:
            print(f"  Disturbance at frame {d_frame}, Δ={d_delta.round(2)}: "
                  f"NOT recovered to <1% within {n_frames - d_frame} frames "
                  f"(max excursion {max_excursion*100:.1f}%)")

    # Save log
    log_arr = dict(
        t=np.array([e["t"] for e in log]),
        L_direct=np.array([e["L_direct"] for e in log]),
        L_gap=np.array([e["L_gap"] for e in log]),
        alpha_min=np.array([e["alpha_min"] for e in log]),
        sharpness=np.array([e["sharpness"] for e in log]),
        threshold=np.array([e["threshold"] for e in log]),
        theta_true=theta_true,
        theta_0=theta_0,
        disturbance_frames=np.array([d[0] for d in disturbances]),
        disturbance_deltas=np.array([d[1] for d in disturbances]),
    )
    out_path = REPO_ROOT / "tools" / "online_residual_log_disturbance.npz"
    np.savez(out_path, **log_arr)
    print(f"\n[plot] saving disturbance recovery figure ...")

    plot_disturbance_recovery(log_arr)
    return result, log_arr


def plot_disturbance_recovery(log_arr, out_path=None):
    """P11 visualization: param trajectories with disturbance markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        out_path = REPO_ROOT / "tools" / "online_residual_disturbance.png"

    t = log_arr["t"]
    theta_true = log_arr["theta_true"]
    d_frames = log_arr["disturbance_frames"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("P11 Disturbance Recovery Test  (λ_traj=0.5)\n"
                 "Manual θ perturbations injected → test stable-equilibrium hypothesis",
                 fontsize=12)

    for i, (name, key, true_v) in enumerate([
        ("α_min", "alpha_min", theta_true[0]),
        ("sharpness", "sharpness", theta_true[1]),
        ("threshold", "threshold", theta_true[2]),
    ]):
        ax = axes[i // 2, i % 2]
        ax.plot(t, log_arr[key], "b-", linewidth=1.2, label="θ estimate")
        ax.axhline(true_v, color="red", linestyle="--", linewidth=1, label="θ_true")
        for d_f in d_frames:
            ax.axvline(d_f, color="orange", linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(d_f, ax.get_ylim()[1] * 0.95, f"  DIST",
                    rotation=90, fontsize=7, color="orange", va="top")
        ax.set_xlabel("frame")
        ax.set_ylabel(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title(name, fontsize=10)

    # Loss panel
    ax = axes[1, 1]
    ax.semilogy(t, log_arr["L_direct"], "b-", linewidth=1.2, label="L_direct")
    for d_f in d_frames:
        ax.axvline(d_f, color="orange", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("frame")
    ax.set_ylabel("L_direct (log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Loss with disturbance spikes", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {out_path}")
    return out_path


def sweep_obs_noise(
    n_frames=500,
    lambda_traj=0.5,
    noise_levels=(0.0, 0.01, 0.05, 0.1, 0.2),
    theta_0=None,
):
    """P10 sweep: test L_traj robustness under Gaussian perception noise.

    Real-world perception noise (D405 depth + side RGB cavity extraction)
    ~ 1-10% on cavity boundary. This sweep maps noise → final accuracy curve.

    Default theta_0=(5, 10, 0.5) (P5 baseline).
    """
    print("=" * 80)
    print(f"P10 OBS NOISE SWEEP  (λ_traj={lambda_traj}, n_frames={n_frames})")
    print("=" * 80)

    if theta_0 is None:
        theta_0 = np.array([5.0, 10.0, 0.5])
    theta_true = np.array([3.0, 20.0, 0.5])

    wp.init()
    sys_dict = setup_synthetic_system(N=20)
    traj = make_knife_trajectory(n_frames=n_frames, x_start=0.35, x_end=0.65)
    real_frames = generate_synth_real_trajectory(
        sys_dict, theta_true, traj, k_bulk_true=1.0, n_cycles_per_frame=1,
    )

    results = []
    for sigma in noise_levels:
        print(f"\n[σ = {sigma:.2f}]  running 500 frames ...")
        result = online_calibration_loop(
            sys_dict, real_frames, theta_0,
            K_inner=30, lr_inner=0.1,
            lr_outer=np.array([0.1, 1.0, 0.05]),
            lambda_prior=0.0,
            lambda_traj=lambda_traj,
            obs_noise_std=sigma,
            verbose=False,
        )
        log = result["log"]
        final = result["final_theta"]
        best = result["best_theta"]
        rel_f = np.abs(final - theta_true) / np.maximum(np.abs(theta_true), 1e-6) * 100
        rel_b = np.abs(best - theta_true) / np.maximum(np.abs(theta_true), 1e-6) * 100

        # Steady-state stats: last 100 frames
        ss_alpha = np.array([e["alpha_min"] for e in log[-100:]])
        ss_sharp = np.array([e["sharpness"] for e in log[-100:]])
        ss_tau = np.array([e["threshold"] for e in log[-100:]])
        oscillation = np.array([ss_alpha.std(), ss_sharp.std(), ss_tau.std()])

        L_ss_mean = float(np.array([e["L_direct"] for e in log[-100:]]).mean())

        # Save log
        log_arr = dict(
            t=np.array([e["t"] for e in log]),
            L_direct=np.array([e["L_direct"] for e in log]),
            L_gap=np.array([e["L_gap"] for e in log]),
            alpha_min=np.array([e["alpha_min"] for e in log]),
            sharpness=np.array([e["sharpness"] for e in log]),
            threshold=np.array([e["threshold"] for e in log]),
            theta_true=theta_true,
            theta_0=theta_0,
            theta_best=best,
            theta_final=final,
            obs_noise_std=np.array([sigma]),
        )
        np.savez(REPO_ROOT / "tools" / f"online_residual_log_noise_{sigma:.2f}.npz", **log_arr)

        results.append({
            "sigma": sigma,
            "final": final,
            "best": best,
            "rel_final": rel_f,
            "rel_best": rel_b,
            "L_ss_mean": L_ss_mean,
            "oscillation": oscillation,
            "log": log_arr,
        })
        print(f"  σ={sigma:.2f}: final={final.round(3)}  rel_err={rel_f.round(2)}%  "
              f"L_ss={L_ss_mean:.2e}  osc(α,s,τ)={oscillation.round(3)}")

    # Plot
    plot_noise_sweep_comparison(results)

    # Summary table
    print("\n" + "=" * 110)
    print("P10 OBS NOISE SWEEP SUMMARY (last-100-frame steady-state stats):")
    print("=" * 110)
    print(f"{'σ_obs':>6} | {'final err %':>22} | {'oscillation (α,s,τ)':>25} | "
          f"{'L_ss mean':>10} | {'verdict':>20}")
    print("-" * 110)
    for r in results:
        rf = "/".join(f"{x:.2f}" for x in r["rel_final"])
        osc = "/".join(f"{x:.3f}" for x in r["oscillation"])
        verdict = "✅ excellent" if r["rel_final"].max() < 2 else \
                  ("🟢 good" if r["rel_final"].max() < 10 else \
                  ("🟡 degraded" if r["rel_final"].max() < 30 else "❌ FAIL"))
        print(f"{r['sigma']:>6.2f} | {rf:>22} | {osc:>25} | {r['L_ss_mean']:>10.2e} | {verdict:>20}")

    return results


def plot_noise_sweep_comparison(results, out_path=None):
    """P10 noise sweep visualization: overlay trajectories + error vs noise curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        out_path = REPO_ROOT / "tools" / "online_residual_noise_sweep.png"

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    fig.suptitle("P10 Obs Noise Robustness  (λ_traj=0.5, n_frames=500)", fontsize=13)
    theta_true = np.array([3.0, 20.0, 0.5])
    colors = ["C0", "C1", "C2", "C3", "C4"]

    # Top row: param trajectories
    for col, (name, key, true_v) in enumerate([
        ("α_min", "alpha_min", theta_true[0]),
        ("sharpness", "sharpness", theta_true[1]),
        ("threshold", "threshold", theta_true[2]),
    ]):
        ax = fig.add_subplot(gs[0, col])
        for c, r in zip(colors, results):
            ax.plot(r["log"]["t"], r["log"][key], color=c,
                    label=f"σ={r['sigma']:.2f}", linewidth=1.0)
        ax.axhline(true_v, color="red", linestyle="--", linewidth=1, label="true")
        ax.set_xlabel("frame")
        ax.set_ylabel(name)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        ax.set_title(name, fontsize=10)

    # Middle row: zoom on last 100 frames (steady state)
    for col, (name, key, true_v) in enumerate([
        ("α_min (last 100)", "alpha_min", theta_true[0]),
        ("sharpness (last 100)", "sharpness", theta_true[1]),
        ("threshold (last 100)", "threshold", theta_true[2]),
    ]):
        ax = fig.add_subplot(gs[1, col])
        for c, r in zip(colors, results):
            t = r["log"]["t"]
            mask = t >= 400
            ax.plot(t[mask], r["log"][key][mask], color=c,
                    label=f"σ={r['sigma']:.2f}", linewidth=1.0)
        ax.axhline(true_v, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("frame")
        ax.set_ylabel(name)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        ax.set_title(name, fontsize=10)

    # Bottom row: summary curves
    sigmas = np.array([r["sigma"] for r in results])
    final_errs = np.array([r["rel_final"] for r in results])  # (N, 3)
    L_ss = np.array([r["L_ss_mean"] for r in results])
    osc = np.array([r["oscillation"] for r in results])  # (N, 3)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(sigmas, final_errs[:, 0], "o-", label="α_min", color="C0")
    ax.plot(sigmas, final_errs[:, 1], "s-", label="sharpness", color="C1")
    ax.plot(sigmas, final_errs[:, 2], "^-", label="threshold", color="C2")
    ax.set_xlabel("σ_obs")
    ax.set_ylabel("final rel err %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Final accuracy vs noise level", fontsize=10)

    ax = fig.add_subplot(gs[2, 1])
    ax.semilogy(sigmas, L_ss, "o-", color="black")
    ax.set_xlabel("σ_obs")
    ax.set_ylabel("steady-state L_direct (log)")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Loss floor vs noise level", fontsize=10)

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(sigmas, osc[:, 0], "o-", label="α_min", color="C0")
    ax.plot(sigmas, osc[:, 1], "s-", label="sharpness", color="C1")
    ax.plot(sigmas, osc[:, 2], "^-", label="threshold", color="C2")
    ax.set_xlabel("σ_obs")
    ax.set_ylabel("steady-state std of θ")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Param oscillation vs noise level", fontsize=10)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {out_path}")
    return out_path


def plot_cold_start_comparison(results, out_path=None):
    """P9 comparison: overlay param trajectories from multiple cold-start runs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_path is None:
        out_path = REPO_ROOT / "tools" / "online_residual_cold_start.png"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"P9 Cold-Start Robustness  (λ_traj=0.5, n_frames=500)",
        fontsize=12,
    )

    theta_true = np.array([3.0, 20.0, 0.5])
    colors = ["C0", "C1", "C2", "C3", "C4"]

    for i, (name, key, true_v) in enumerate([
        ("α_min", "alpha_min", theta_true[0]),
        ("sharpness", "sharpness", theta_true[1]),
        ("threshold", "threshold", theta_true[2]),
    ]):
        ax = axes[i // 2, i % 2]
        for c, r in zip(colors, results):
            log = r["log"]
            ax.plot(log["t"], log[key], color=c, label=r["label"], linewidth=1.2)
        ax.axhline(true_v, color="red", linestyle="--", linewidth=1, label="true")
        ax.set_xlabel("frame")
        ax.set_ylabel(name)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
        ax.set_title(name)

    # 4th panel: L_direct comparison (log scale)
    ax = axes[1, 1]
    for c, r in zip(colors, results):
        log = r["log"]
        ax.semilogy(log["t"], log["L_direct"], color=c, label=r["label"], linewidth=1.0)
    ax.set_xlabel("frame")
    ax.set_ylabel("L_direct (log)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Loss decay (log scale)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {out_path}")
    return out_path


def cold_start_robustness_test(
    n_frames=500,
    lambda_traj=0.5,
    inits=None,                         # list of (label, theta_0) tuples
):
    """P9: probe L_traj robustness against extreme cold-start inits.

    Each init is run for n_frames with the WINNING lambda_traj from P8.
    Report convergence pattern + final accuracy + best-so-far for each.
    """
    if inits is None:
        # Default set: moderate → extreme → boundary cases
        inits = [
            ("baseline (P5 init)",  np.array([5.0, 10.0, 0.5])),    # 67%, 50%, 0%
            ("EXTREME both dir",    np.array([10.0, 5.0, 0.3])),    # 233%, 75%, 40%
            ("VERY extreme",        np.array([15.0, 3.0, 0.7])),    # 400%, 85%, 40%
            ("flipped (sharp ↑↑)",  np.array([1.0, 50.0, 0.5])),    # 67% under, 150% over
            ("at θ_true",           np.array([3.0, 20.0, 0.5])),    # 0% (sanity: stays)
        ]

    print("=" * 80)
    print(f"P9 COLD-START ROBUSTNESS  (λ_traj={lambda_traj}, n_frames={n_frames})")
    print("=" * 80)

    theta_true = np.array([3.0, 20.0, 0.5])

    wp.init()
    sys_dict = setup_synthetic_system(N=20)
    traj = make_knife_trajectory(n_frames=n_frames, x_start=0.35, x_end=0.65)
    real_frames = generate_synth_real_trajectory(
        sys_dict, theta_true, traj, k_bulk_true=1.0, n_cycles_per_frame=1,
    )

    results = []
    for label, theta_0 in inits:
        rel_init = np.abs(theta_0 - theta_true) / np.maximum(np.abs(theta_true), 1e-6) * 100
        print(f"\n[{label}]  θ_0 = {theta_0}  (init err {rel_init.round(0)}%)")
        result = online_calibration_loop(
            sys_dict, real_frames, theta_0,
            K_inner=30, lr_inner=0.1,
            lr_outer=np.array([0.1, 1.0, 0.05]),
            lambda_prior=0.0,
            lambda_traj=lambda_traj,
            verbose=False,
        )
        # quick stats
        final = result["final_theta"]
        best = result["best_theta"]
        rel_f = np.abs(final - theta_true) / np.maximum(np.abs(theta_true), 1e-6) * 100
        rel_b = np.abs(best - theta_true) / np.maximum(np.abs(theta_true), 1e-6) * 100

        # Frames-to-converge: first t where all 3 rel_err < 10%
        log = result["log"]
        frames_to_10 = -1
        for entry in log:
            r = np.abs(np.array([entry["alpha_min"], entry["sharpness"], entry["threshold"]])
                       - theta_true) / np.maximum(np.abs(theta_true), 1e-6) * 100
            if np.all(r < 10):
                frames_to_10 = entry["t"]
                break

        # Save log for plotting
        log_arr = dict(
            t=np.array([e["t"] for e in log]),
            L_direct=np.array([e["L_direct"] for e in log]),
            L_gap=np.array([e["L_gap"] for e in log]),
            L_traj=np.array([e["L_traj"] for e in log]),
            alpha_min=np.array([e["alpha_min"] for e in log]),
            sharpness=np.array([e["sharpness"] for e in log]),
            threshold=np.array([e["threshold"] for e in log]),
            theta_true=theta_true,
            theta_0=theta_0,
            theta_best=best,
            theta_final=final,
        )
        safe_label = (label.replace(" ", "_").replace("(", "").replace(")", "")
                      .replace(",", "").replace("↑", "U").replace("→", "-"))
        np.savez(REPO_ROOT / "tools" / f"online_residual_log_cold_{safe_label}.npz", **log_arr)

        results.append({
            "label": label,
            "theta_0": theta_0,
            "rel_init": rel_init,
            "final": final,
            "rel_final": rel_f,
            "best": best,
            "rel_best": rel_b,
            "best_step": result["best_step"],
            "best_L": result["best_L"],
            "final_L": result["final_L"],
            "frames_to_10": frames_to_10,
            "log": log_arr,
        })
        print(f"  → final θ = ({final[0]:.3f}, {final[1]:.3f}, {final[2]:.3f})  rel_err={rel_f.round(2)}%")
        print(f"  → best  θ = ({best[0]:.3f}, {best[1]:.3f}, {best[2]:.3f})  rel_err={rel_b.round(2)}%  at step {result['best_step']}")
        print(f"  → frames to <10% rel err: {frames_to_10 if frames_to_10 >= 0 else 'NEVER'}")
        print(f"  → final_L = {result['final_L']:.2e}, best_L = {result['best_L']:.2e}")

    # Plot
    plot_cold_start_comparison(results)

    # Summary table
    print("\n" + "=" * 100)
    print("P9 COLD-START ROBUSTNESS SUMMARY:")
    print("=" * 100)
    print(f"{'label':>25} | {'init err %':>20} | {'final err %':>22} | {'best err %':>22} | "
          f"{'→<10%':>5} | {'final_L':>10}")
    print("-" * 130)
    for r in results:
        ri = "/".join(f"{x:.0f}" for x in r["rel_init"])
        rf = "/".join(f"{x:.2f}" for x in r["rel_final"])
        rb = "/".join(f"{x:.2f}" for x in r["rel_best"])
        ftc = f"{r['frames_to_10']}" if r["frames_to_10"] >= 0 else "NEVER"
        print(f"{r['label']:>25} | {ri:>20} | {rf:>22} | {rb:>22} | "
              f"{ftc:>5} | {r['final_L']:>10.2e}")

    return results


def sweep_lambda_traj(n_frames=500, lambdas=(0.0, 0.1, 0.3, 0.5, 1.0)):
    """P8 sweep: run integration_test at multiple λ_traj values, report best.

    Uses lambda_prior=0 to isolate L_traj contribution.
    """
    print("=" * 70)
    print(f"P8 SWEEP: lambda_traj over {list(lambdas)}, n_frames={n_frames}")
    print("=" * 70)
    results = {}
    for lam in lambdas:
        print(f"\n{'#' * 70}\n#  λ_traj = {lam}\n{'#' * 70}")
        _, _, result = integration_test(
            n_frames=n_frames,
            lambda_prior=0.0,
            lambda_traj=lam,
            log_out_name=f"online_residual_log_traj_{lam:.1f}.npz",
        )
        results[lam] = result

    print("\n" + "=" * 70)
    print("P8 L_TRAJ SWEEP SUMMARY:")
    print("=" * 70)
    theta_true = np.array([3.0, 20.0, 0.5])
    print(f"{'λ_traj':>8} | {'best_step':>9} | {'best_L':>10} | "
          f"{'α_final':>8} | {'s_final':>8} | {'τ_final':>8} | "
          f"{'α_err%':>7} | {'s_err%':>7} | {'τ_err%':>7}")
    print("-" * 120)
    for lam, r in results.items():
        # Report FINAL params (not best) because L_traj should stabilize final
        f = r["final_theta"]
        rel = np.abs(f - theta_true) / np.maximum(np.abs(theta_true), 1e-6) * 100
        print(f"{lam:>8.2f} | {r['best_step']:>9d} | {r['best_L']:>10.2e} | "
              f"{f[0]:>8.3f} | {f[1]:>8.3f} | {f[2]:>8.3f} | "
              f"{rel[0]:>7.2f} | {rel[1]:>7.2f} | {rel[2]:>7.2f}")
    return results


def sweep_lambda_prior(n_frames=500, lambdas=(0.0, 1e-5, 1e-4, 1e-3, 1e-2)):
    """P7 sweep: run integration_test at multiple λ values, report best."""
    print("=" * 70)
    print(f"P7 SWEEP: lambda_prior over {list(lambdas)}, n_frames={n_frames}")
    print("=" * 70)
    results = {}
    for lam in lambdas:
        print(f"\n{'#' * 70}\n#  λ_prior = {lam}\n{'#' * 70}")
        _, _, result = integration_test(
            n_frames=n_frames,
            lambda_prior=lam,
            log_out_name=f"online_residual_log_lambda_{lam:.0e}.npz",
        )
        results[lam] = result

    print("\n" + "=" * 70)
    print("SWEEP SUMMARY:")
    print("=" * 70)
    theta_true = np.array([3.0, 20.0, 0.5])
    print(f"{'λ_prior':>10} | {'best_step':>9} | {'best_L':>10} | "
          f"{'α_best':>7} | {'s_best':>7} | {'τ_best':>7} | "
          f"{'α_err%':>7} | {'s_err%':>7} | {'τ_err%':>7}")
    print("-" * 110)
    for lam, r in results.items():
        b = r["best_theta"]
        rel = np.abs(b - theta_true) / np.maximum(np.abs(theta_true), 1e-6) * 100
        print(f"{lam:>10.0e} | {r['best_step']:>9d} | {r['best_L']:>10.2e} | "
              f"{b[0]:>7.3f} | {b[1]:>7.3f} | {b[2]:>7.3f} | "
              f"{rel[0]:>7.2f} | {rel[1]:>7.2f} | {rel[2]:>7.2f}")
    return results


if __name__ == "__main__":
    import sys as _sys
    if "--smoke" in _sys.argv:
        smoke_test()
    elif "--persist" in _sys.argv:
        persistent_state_test()
    elif "--traj" in _sys.argv:
        trajectory_test()
    elif "--inner" in _sys.argv:
        inner_loop_test()
    elif "--outer" in _sys.argv:
        outer_grad_test()
    elif "--integ" in _sys.argv:
        n = 100
        lam_p = 0.0
        lam_t = 0.0
        for arg in _sys.argv:
            if arg.startswith("--frames="):
                n = int(arg.split("=")[1])
            if arg.startswith("--lambda=") or arg.startswith("--lambda_prior="):
                lam_p = float(arg.split("=")[1])
            if arg.startswith("--lambda_traj="):
                lam_t = float(arg.split("=")[1])
        integration_test(n_frames=n, lambda_prior=lam_p, lambda_traj=lam_t)
    elif "--cube" in _sys.argv:
        n = 200
        N_pd = 12
        mode = 'D'
        lam_p = 0.0
        lam_t = 0.0
        for arg in _sys.argv:
            if arg.startswith("--frames="):
                n = int(arg.split("=")[1])
            elif arg.startswith("--N_per_dim=") or arg.startswith("--N="):
                N_pd = int(arg.split("=")[1])
            elif arg.startswith("--mode="):
                mode = arg.split("=")[1]
            elif arg.startswith("--lambda_prior="):
                lam_p = float(arg.split("=")[1])
            elif arg.startswith("--lambda_traj="):
                lam_t = float(arg.split("=")[1])
        cube_integration_test(
            N_per_dim=N_pd, n_frames=n,
            state_update_mode=mode,
            lambda_prior=lam_p, lambda_traj=lam_t,
        )
    elif "--mesh" in _sys.argv:
        n = 400
        mode = 'D'
        kr = 0.025
        mesh_path = None
        lam_p = 0.0
        lam_t = 0.0
        for arg in _sys.argv:
            if arg.startswith("--frames="):
                n = int(arg.split("=")[1])
            elif arg.startswith("--mode="):
                mode = arg.split("=")[1]
            elif arg.startswith("--knife_radius="):
                kr = float(arg.split("=")[1])
            elif arg.startswith("--mesh_path="):
                mesh_path = arg.split("=")[1]
            elif arg.startswith("--lambda_prior="):
                lam_p = float(arg.split("=")[1])
            elif arg.startswith("--lambda_traj="):
                lam_t = float(arg.split("=")[1])
        mesh_integration_test(
            mesh_path=mesh_path, n_frames=n,
            state_update_mode=mode,
            knife_radius=kr,
            lambda_prior=lam_p, lambda_traj=lam_t,
        )
    elif "--sweep" in _sys.argv:
        n = 500
        for arg in _sys.argv:
            if arg.startswith("--frames="):
                n = int(arg.split("=")[1])
        sweep_lambda_prior(n_frames=n)
    elif "--sweep_traj" in _sys.argv:
        n = 500
        for arg in _sys.argv:
            if arg.startswith("--frames="):
                n = int(arg.split("=")[1])
        sweep_lambda_traj(n_frames=n)
    elif "--cold" in _sys.argv:
        n = 500
        lam_t = 0.5
        for arg in _sys.argv:
            if arg.startswith("--frames="):
                n = int(arg.split("=")[1])
            if arg.startswith("--lambda_traj="):
                lam_t = float(arg.split("=")[1])
        cold_start_robustness_test(n_frames=n, lambda_traj=lam_t)
    elif "--noise" in _sys.argv:
        n = 500
        lam_t = 0.5
        for arg in _sys.argv:
            if arg.startswith("--frames="):
                n = int(arg.split("=")[1])
            if arg.startswith("--lambda_traj="):
                lam_t = float(arg.split("=")[1])
        sweep_obs_noise(n_frames=n, lambda_traj=lam_t)
    elif "--disturb" in _sys.argv:
        n = 600
        lam_t = 0.5
        reset = "--reset_adam" in _sys.argv
        for arg in _sys.argv:
            if arg.startswith("--frames="):
                n = int(arg.split("=")[1])
            if arg.startswith("--lambda_traj="):
                lam_t = float(arg.split("=")[1])
        disturbance_recovery_test(n_frames=n, lambda_traj=lam_t, reset_adam=reset)
    elif "--plot" in _sys.argv:
        plot_convergence()
    else:
        smoke_test()
        trajectory_test()
        inner_loop_test()
        outer_grad_test()
        integration_test(n_frames=50)  # shorter for full-test run
