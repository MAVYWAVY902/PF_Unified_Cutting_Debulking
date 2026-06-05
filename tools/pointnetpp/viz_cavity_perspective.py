"""Perspective-view cavity videos matching the 3 convergence GIFs.

Same shallow RASTER trajectory + truth tau as test_group_generalize. Renders a fixed
3D perspective (z-up = cut axis, no rotation) of the cube being debulked: ALIVE
particles (alpha>0.5) shown, colored by their TRUTH tau (soft=blue, hard=red);
REMOVED particles vanish -> the cavity carves through the tau pattern. Soft (low tau)
regions hollow out faster -> the cavity SHAPE itself encodes the tau pattern.

One GIF per pattern: LR_HALVES, DIAG_STRIPES, LATERAL_SPOT.
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD = 0.55; TAU_SOFT = 0.15; TAU_INIT = 0.30; STIFF_UNIFORM = 0.5
OUT = REPO / "tools/pointnetpp/viz_out"; OUT.mkdir(exist_ok=True)
N_FRAMES = 44


def main():
    cfg = MESH_CONFIGS[0]; device = "cuda:0"
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]),
                                                            fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    kr = cfg["knife_radius_abs"]; top_axis = cfg["top_axis"]

    solver.reset_to_rest(pos_rest); far = pos_rest.max(0) + 10.0
    sw = torch.tensor(np.full(N, STIFF_UNIFORM, np.float32), device=device)
    twt = torch.tensor(np.full(N, TAU_INIT, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); st = rest_state
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
    pos_rest = pw.cpu().numpy().copy(); rest_state = st
    bb0, bb1 = pos_rest.min(0), pos_rest.max(0); sp = bb1 - bb0
    lateral = [a for a in range(3) if a != top_axis]
    sweep_axis, lat_axis = lateral[0], lateral[1]
    xn = (pos_rest[:, sweep_axis] - bb0[sweep_axis]) / sp[sweep_axis]
    zn = (pos_rest[:, lat_axis] - bb0[lat_axis]) / sp[lat_axis]

    PATTERNS = {
        "LR_HALVES":    xn < 0.5,
        "DIAG_STRIPES": (np.floor(4.0 * (xn + zn)).astype(int) % 2 == 0),
        "LATERAL_SPOT": ((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2,
    }
    traj = make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.93, 10, 30)
    T = len(traj)
    theta = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=device)
    stiff_t = torch.tensor(np.full(N, STIFF_UNIFORM, np.float32), device=device)

    # plot axis mapping: vertical (up) = top_axis (cut axis)
    PX, PY, PZ = sweep_axis, lat_axis, top_axis
    xl = (pos_rest[:, PX].min(), pos_rest[:, PX].max())
    yl = (pos_rest[:, PY].min(), pos_rest[:, PY].max())
    zl = (pos_rest[:, PZ].min(), pos_rest[:, PZ].max())

    for pname, hard in PATTERNS.items():
        tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
        tau_t = torch.tensor(tau_truth, device=device)
        solver.reset_to_rest(pos_rest); state = rest_state
        POS = []; AL = []
        with torch.no_grad():
            for knife in traj:
                a, p, _ = WarpFullForwardOp.apply(theta, solver, state, knife, kr, stiff_t, tau_t)
                POS.append(p.cpu().numpy()); AL.append(a.cpu().numpy()); state = solver_save_state(solver)
        frames = list(range(0, T, max(1, T // N_FRAMES)))
        print(f"[{pname}] forward done, rendering {len(frames)} frames "
              f"(final removed={(AL[-1] < 0.5).sum()})", flush=True)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

        def update(fi):
            t = frames[fi]
            ax.clear()
            p = POS[t]; a = AL[t]
            alive = a > 0.5
            ax.scatter(p[alive, PX], p[alive, PY], p[alive, PZ],
                       c=tau_truth[alive], cmap="coolwarm", vmin=TAU_SOFT, vmax=TAU_HARD,
                       s=4, alpha=0.85, edgecolors="none")
            k = traj[t]
            ax.scatter([k[PX]], [k[PY]], [k[PZ]], c="black", s=120, marker="v")
            ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=22, azim=-60)
            try:
                ax.set_box_aspect((xl[1]-xl[0], yl[1]-yl[0], zl[1]-zl[0]))
            except Exception:
                pass
            ax.set_title(f"{pname}  —  debulk cavity (truth τ, alive only)\n"
                         f"step {t}/{T}   removed={int((a<0.5).sum())}", fontsize=11)
            return []

        anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
        gif = OUT / f"cavity_{pname}.gif"
        t0 = time.time()
        anim.save(str(gif), writer=PillowWriter(fps=6)); plt.close(fig)
        print(f"  -> saved {gif}  ({time.time()-t0:.0f}s)", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
