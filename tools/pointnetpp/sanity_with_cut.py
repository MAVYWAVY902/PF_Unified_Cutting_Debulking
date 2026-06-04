"""With-cut sanity: floor fixed, gravity=0, knife actually cuts.

Compare iter=2 vs iter=16 to see when fluid-like behavior appears
after particles are killed by cutting.

For each mesh:
   Run forward T=30 with realistic knife sweep (T_B trajectory).
   Render side-by-side: iter=2 (current default) | iter=16 (cpp_sim match).
   Look at:
       - Do dead particles (alpha<0.5) fly off? (= fluid)
       - Do alive particles bordering cavity stay connected?
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
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image  # noqa: E402

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state  # noqa: E402
from tools.day6b_full_forward_torch_op import WarpFullForwardOp  # noqa: E402
from tools.stageA_inclusion_recovery import build_inclusion_truth, gen_obs_per_particle_stiff  # noqa: E402
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec  # noqa: E402


def run_with_cut(cfg, regime, T=30, device="cuda:0"):
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]))
    N = solver.N
    solver.damping = regime["damping"]
    solver.num_iterations = regime["num_iter"]
    solver.substeps = regime["substeps"]
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)
    pos_rest = mesh_info["vertices"]
    # Floor below mesh
    solver.floor_y = float(pos_rest[:, 2].min() - 0.1 * (pos_rest[:, 2].max() - pos_rest[:, 2].min()))

    bbox_min = pos_rest.min(0); bbox_max = pos_rest.max(0); bbox_size = bbox_max - bbox_min
    knife_radius = cfg["knife_radius_abs"]
    incl_center = bbox_min + np.array(cfg["incl_center_rel"]) * bbox_size
    incl_radius = cfg["incl_radius_abs"]
    stiff_truth, inclusion_mask, _, _ = build_inclusion_truth(
        pos_rest, stiff_in=cfg["stiff_in"], stiff_out=cfg["stiff_out"],
        center_override=tuple(incl_center), radius_override=float(incl_radius))
    tau_truth = np.where(inclusion_mask, cfg["tau_in"], cfg["tau_out"]).astype(np.float32)

    # Long sweep T_B (80% bbox)
    spec_B = dict(cfg["traj_specs"]["B"])
    spec_B["sweep_range"] = (0.10, 0.90)
    traj = make_traj_from_spec(spec_B, bbox_min, bbox_max, T)

    theta = np.array([3.0, 100.0, 0.3], dtype=np.float32)
    solver.reset_to_rest(pos_rest)
    obs_pos, obs_alpha = gen_obs_per_particle_stiff(
        solver, rest_state, theta, stiff_truth, traj, knife_radius, device,
        tau_truth_np=tau_truth)
    return obs_pos, obs_alpha, traj, pos_rest, bbox_min, bbox_max, incl_center, incl_radius


def render_frame(pos_a, alpha_a, label_a, pos_b, alpha_b, label_b, knife_pos,
                  top_axis, bbox_min, bbox_max, incl_center, incl_radius,
                  t, T, mesh_name, out_path):
    """3D perspective rendering. Two side-by-side 3D scatter panels."""
    fig = plt.figure(figsize=(14, 6.5))
    # Slow rotation for parallax depth cue
    azim = 30.0 + (t / max(T - 1, 1)) * 60.0  # 30°→90°
    elev = 20.0
    for j, (pos, alpha, label) in enumerate([(pos_a, alpha_a, label_a),
                                                 (pos_b, alpha_b, label_b)]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        # Subsample if N very large for render speed
        idx = np.arange(len(pos))
        if len(pos) > 8000:
            rng = np.random.RandomState(0)
            idx = rng.choice(len(pos), 8000, replace=False)
        sc = ax.scatter(pos[idx, 0], pos[idx, 1], pos[idx, 2],
                          c=alpha[idx], cmap="RdBu", s=4, vmin=0, vmax=1,
                          edgecolors="none", alpha=0.6, depthshade=True)
        # Knife as bigger black marker
        ax.scatter([knife_pos[0]], [knife_pos[1]], [knife_pos[2]], marker="X",
                    s=200, color="black", edgecolor="white", linewidth=1.5, zorder=10)
        # Inclusion as wireframe sphere
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi, 8)
        sx = incl_center[0] + incl_radius * np.outer(np.cos(u), np.sin(v))
        sy = incl_center[1] + incl_radius * np.outer(np.sin(u), np.sin(v))
        sz = incl_center[2] + incl_radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, color="green", linewidth=0.6, alpha=0.4)
        # Equal aspect bbox
        size = (bbox_max - bbox_min).max()
        cx = 0.5 * (bbox_min + bbox_max)
        ax.set_xlim(cx[0] - 0.6 * size, cx[0] + 0.6 * size)
        ax.set_ylim(cx[1] - 0.6 * size, cx[1] + 0.6 * size)
        ax.set_zlim(cx[2] - 0.6 * size, cx[2] + 0.6 * size)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        cav = float((alpha < 0.5).mean())
        ax.set_title(f"{label}  ·  cav={cav*100:.2f}%", fontsize=10)
    fig.suptitle(f"{mesh_name}  ·  frame {t+1}/{T}  ·  WITH cutting (floor fixed, g=0)  ·  3D persp",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=85, bbox_inches="tight")
    plt.close(fig)


def main():
    device = "cuda:0"
    regimes = {
        "iter2": dict(num_iter=2, substeps=1, damping=0.99),
        "iter16": dict(num_iter=16, substeps=4, damping=0.95),
    }
    t0_all = time.time()
    for cfg in MESH_CONFIGS:
        print(f"\n{'='*78}\nMESH: {cfg['name']}\n{'='*78}", flush=True)
        results = {}
        for r_name, r in regimes.items():
            print(f"\n[{r_name}] iter={r['num_iter']} sub={r['substeps']} damp={r['damping']}", flush=True)
            t0 = time.time()
            pos_seq, alpha_seq, traj, pos_rest, bbox_min, bbox_max, incl_c, incl_r = \
                run_with_cut(cfg, r, T=30, device=device)
            print(f"  done {time.time()-t0:.0f}s  final cavity {(alpha_seq[-1]<0.5).mean()*100:.2f}%", flush=True)
            # Max drift of ALIVE particles (alpha>0.5) — these should stay solid
            alive_mask = alpha_seq[-1] > 0.5
            drift_alive = np.linalg.norm(pos_seq[-1][alive_mask] - pos_rest[alive_mask], axis=1) * 1000.0
            print(f"  alive-particle max drift = {drift_alive.max():.3f} mm  mean = {drift_alive.mean():.3f} mm", flush=True)
            dead_mask = alpha_seq[-1] < 0.5
            drift_dead = np.linalg.norm(pos_seq[-1][dead_mask] - pos_rest[dead_mask], axis=1) * 1000.0 if dead_mask.sum() > 0 else np.array([0.0])
            print(f"  dead-particle  max drift = {drift_dead.max():.3f} mm  mean = {drift_dead.mean():.3f} mm", flush=True)
            results[r_name] = dict(pos=pos_seq, alpha=alpha_seq, traj=traj,
                                       bbox_min=bbox_min, bbox_max=bbox_max,
                                       incl_c=incl_c, incl_r=incl_r,
                                       drift_alive=drift_alive.max(),
                                       drift_dead=drift_dead.max())
        # Render side-by-side iter2 vs iter16
        out_dir = REPO / "tools" / "pointnetpp" / f"sanity_cut3d_{cfg['name']}"
        out_dir.mkdir(exist_ok=True)
        T = len(results["iter2"]["pos"])
        for t in range(T):
            render_frame(
                results["iter2"]["pos"][t], results["iter2"]["alpha"][t],
                f"iter=2 (current eval)  drift_alive_max={results['iter2']['drift_alive']:.2f}mm",
                results["iter16"]["pos"][t], results["iter16"]["alpha"][t],
                f"iter=16 (cpp_sim match)  drift_alive_max={results['iter16']['drift_alive']:.2f}mm",
                results["iter2"]["traj"][t], cfg["top_axis"],
                results["iter2"]["bbox_min"], results["iter2"]["bbox_max"],
                results["iter2"]["incl_c"], results["iter2"]["incl_r"],
                t, T, cfg["name"], out_dir / f"frame_{t:02d}.png")
        frames = [Image.open(out_dir / f"frame_{t:02d}.png") for t in range(T)]
        gif_path = REPO / "tools" / "pointnetpp" / f"sanity_cut3d_{cfg['name']}.gif"
        frames[0].save(str(gif_path), save_all=True, append_images=frames[1:],
                          duration=200, loop=0)
        print(f"  saved {gif_path}", flush=True)
    print(f"\nTotal: {(time.time()-t0_all)/60:.1f} min")


if __name__ == "__main__":
    main()
