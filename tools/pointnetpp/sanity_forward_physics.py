"""Forward physics sanity test: NO cutting, NO gravity.

If physics is correct: particles should be EXACTLY STATIC across T frames.
Any motion indicates broken constraint enforcement.

Sweeps physics setting:
   (num_iter=2, substeps=1)  — current "winning" eval regime (32× under-iterated)
   (num_iter=8, substeps=2)  — compromise (16 passes/step)
   (num_iter=16, substeps=4) — match cpp_sim (64 passes/step)

For each setting + each mesh (cube/tumor/torus):
   Render T=30 frames top-down surface view with NO cutting.
   Particles should not visibly move.
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))

import warp as wp  # noqa: E402
wp.init()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.stageA_inclusion_recovery import build_inclusion_truth  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS  # noqa: E402


PHYSICS_REGIMES = [
    dict(name="iter2_sub1", num_iter=2, substeps=1, damping=0.99),
    dict(name="iter8_sub2", num_iter=8, substeps=2, damping=0.95),
    dict(name="iter16_sub4", num_iter=16, substeps=4, damping=0.95),
]


def run_no_cut_forward(mesh_path, regime, T=30):
    """Forward T frames with NO knife cutting, NO gravity.
    Returns (pos_per_t, max_drift_mm).
    """
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / mesh_path))
    N = solver.N
    solver.damping = regime["damping"]
    solver.num_iterations = regime["num_iter"]
    solver.substeps = regime["substeps"]
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)  # no body force
    # 2026-05-30 PHYSICS BUG #2: floor_y=0 default but Warp uses Z-up
    # → floor at z=0 plane. Tumor mesh has z range [-0.012, 0.012] —
    # half buried, collision_kernel pushes those particles violently.
    # Fix: per-mesh floor below bbox.
    pos_rest = mesh_info["vertices"]
    solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))

    # Place knife FAR from mesh (so collision_kernel never triggers)
    far_knife = pos_rest.max(0) + 10.0
    knife_radius = 0.001  # tiny, irrelevant

    # Run forward with stiff=0.8 uniform, tau=0.4 uniform (no cut)
    stiff_pp = np.full(N, 0.8, dtype=np.float32)
    tau_pp = np.full(N, 0.4, dtype=np.float32)
    theta = np.array([3.0, 100.0, 0.3], dtype=np.float32)

    device = "cuda:0"
    theta_t = torch.tensor(theta, dtype=torch.float32, device=device)
    stiff_t = torch.tensor(stiff_pp, dtype=torch.float32, device=device)
    tau_t = torch.tensor(tau_pp, dtype=torch.float32, device=device)

    solver.reset_to_rest(pos_rest)
    state_in = rest_state
    positions = [pos_rest.copy()]
    with torch.no_grad():
        for t in range(T):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(
                theta_t, solver, state_in, far_knife, knife_radius,
                stiff_t, tau_t,
            )
            p = pos_t.cpu().numpy()
            positions.append(p.copy())
            state_in = solver_save_state(solver)

    pos_arr = np.stack(positions, axis=0)  # (T+1, N, 3)
    drift = np.linalg.norm(pos_arr - pos_arr[0], axis=2)  # (T+1, N)
    max_drift_t = drift.max(axis=1) * 1000.0  # mm per frame
    return positions, max_drift_t


def render_drift(positions, mesh_name, regime_name, top_axis, bbox_min, bbox_max, out_path):
    other_axes = [i for i in range(3) if i != top_axis]
    ax_h, ax_v = other_axes[0], other_axes[1]
    T = len(positions)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    sel_frames = [0, T // 3, 2 * T // 3, T - 1]
    for j, t in enumerate(sel_frames):
        ax = axes[j]
        p = positions[t]
        # Color by drift magnitude from t=0
        d = np.linalg.norm(p - positions[0], axis=1) * 1000.0  # mm
        # Show all particles
        sc = ax.scatter(p[:, ax_h], p[:, ax_v], c=d, cmap="viridis", s=8,
                          vmin=0, vmax=max(d.max(), 1e-3), edgecolors="none")
        ax.set_xlim(bbox_min[ax_h] - 0.02 * (bbox_max[ax_h] - bbox_min[ax_h]),
                     bbox_max[ax_h] + 0.02 * (bbox_max[ax_h] - bbox_min[ax_h]))
        ax.set_ylim(bbox_min[ax_v] - 0.02 * (bbox_max[ax_v] - bbox_min[ax_v]),
                     bbox_max[ax_v] + 0.02 * (bbox_max[ax_v] - bbox_min[ax_v]))
        ax.set_xlabel(["x", "y", "z"][ax_h])
        ax.set_ylabel(["x", "y", "z"][ax_v])
        ax.set_title(f"t={t}  ·  max drift={d.max():.3f} mm", fontsize=11)
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.04)
    fig.suptitle(f"{mesh_name}  ·  {regime_name}  ·  No cut, no gravity — should be static",
                  fontsize=13)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=85, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"{'='*78}\nForward Physics Sanity — No cut, No gravity\n{'='*78}", flush=True)
    results = {}
    for cfg in MESH_CONFIGS:
        results[cfg["name"]] = {}
        pos_rest_ref = None
        bbox_min = bbox_max = None
        for regime in PHYSICS_REGIMES:
            print(f"\n[{cfg['name']}] regime={regime['name']} (iter={regime['num_iter']}, sub={regime['substeps']}, damp={regime['damping']})", flush=True)
            t0 = time.time()
            try:
                positions, max_drift = run_no_cut_forward(REPO / cfg["path"], regime, T=30)
            except Exception as e:
                import traceback
                print(f"  FAILED: {e}")
                traceback.print_exc()
                continue
            pos_rest_ref = positions[0]
            bbox_min = pos_rest_ref.min(0)
            bbox_max = pos_rest_ref.max(0)
            results[cfg["name"]][regime["name"]] = max_drift
            print(f"  T=30 drift (mm): "
                   f"t=1 {max_drift[1]:.4f}  "
                   f"t=10 {max_drift[10]:.4f}  "
                   f"t=20 {max_drift[20]:.4f}  "
                   f"t=30 {max_drift[30]:.4f}  "
                   f"({time.time()-t0:.0f}s)", flush=True)
            out_path = REPO / "tools" / "pointnetpp" / f"sanity_{cfg['name']}_{regime['name']}.png"
            render_drift(positions, cfg["name"], regime["name"], cfg["top_axis"],
                          bbox_min, bbox_max, out_path)
            print(f"  saved {out_path}", flush=True)

    # Summary table
    print(f"\n{'='*78}")
    print(f"FORWARD STABILITY: max drift (mm) at t=30 with NO cut + NO gravity")
    print(f"  Ideal: 0 (static). Anything > 0.01mm indicates broken constraint enforcement.")
    print(f"{'='*78}")
    print(f"{'mesh':<12} {'iter2_sub1':>14} {'iter8_sub2':>14} {'iter16_sub4':>14}")
    for cfg in MESH_CONFIGS:
        row = f"{cfg['name']:<12} "
        for r in PHYSICS_REGIMES:
            d = results[cfg["name"]].get(r["name"])
            if d is None:
                row += f"{'FAIL':>14} "
            else:
                row += f"{d[30]:>14.4f} "
        print(row)


if __name__ == "__main__":
    main()
