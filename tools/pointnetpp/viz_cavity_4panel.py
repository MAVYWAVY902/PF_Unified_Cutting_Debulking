"""Per-pattern 4-panel 3D cavity videos: fake-real GT | sim init | GROUP | per-particle.

For each lateral pattern we (1) calibrate GROUP and per-particle tau on the raster obs,
(2) build 4 tau fields {truth, init=0.30, group, pp}, (3) forward EACH field through a
SLOW raster, recording the cavity (alive particles), (4) render a 1x4 3D perspective
animation (z-up, fixed view). Each panel's alive particles are colored by THAT panel's
tau -> you see both the tau field and the cavity it carves. GROUP should reproduce the
truth cavity+pattern; per-particle stays washed-out/uniform.

Knife slowed: T_per_line doubled + most frames rendered.
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
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.hierarchy import build_hierarchy
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.calibrate_online import calibrate_online
from tools.pointnetpp.surface_projection import column_grid_assignment
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD = 0.55; TAU_SOFT = 0.15; TAU_INIT = 0.30; STIFF_UNIFORM = 0.5
OUT = REPO / "tools/pointnetpp/viz_out"; OUT.mkdir(exist_ok=True)
RENDER_SUB = 8000          # particle subset for rendering speed
N_RENDER_FRAMES = 64


def forward_cavity(solver, rest_state, pos_rest, traj, kr, device, theta, stiff_t, tau_field):
    tau_t = torch.tensor(tau_field, device=device)
    solver.reset_to_rest(pos_rest); state = rest_state
    POS = []; AL = []
    with torch.no_grad():
        for knife in traj:
            a, p, _ = WarpFullForwardOp.apply(theta, solver, state, knife, kr, stiff_t, tau_t)
            POS.append(p.cpu().numpy()); AL.append(a.cpu().numpy()); state = solver_save_state(solver)
    return POS, AL


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

    surf = pos_rest[:, top_axis] > (bb1[top_axis] - 0.10 * sp[top_axis])
    surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=device)
    col_id, _ = column_grid_assignment(pos_rest, surf, top_axis, grid_n=8)

    # calibration trajectories (same as results); render trajectory = SLOWER (2x steps/line)
    calA = make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.93, 10, 30)
    calB = make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.88, 10, 30)
    rtraj = make_raster(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.93, 10, 60)   # slow knife
    Tr = len(rtraj)

    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    eat = torch.tensor(ea, dtype=torch.long, device=device); ebt = torch.tensor(eb, dtype=torch.long, device=device)
    erl = np.linalg.norm(pos_rest[ea] - pos_rest[eb], axis=1)
    ert = torch.tensor(erl, dtype=torch.float32, device=device)
    xr = torch.from_numpy(pos_rest).to(device, torch.float32)
    H = build_hierarchy(xr, M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
    stiff_init = np.full(N, STIFF_UNIFORM, np.float32); tau_init = np.full(N, TAU_INIT, np.float32)
    theta = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=device)
    stiff_t = torch.tensor(np.full(N, STIFF_UNIFORM, np.float32), device=device)
    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=device, knife_radius=kr,
               pos_rest=pos_rest, theta_3d_np=np.array([3., 20., 0.3], np.float32),
               stiff_init_np=stiff_init, tau_init_np=tau_init, tau_bound=(0.05, 0.95),
               surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H,
               group_col_id=col_id)

    rng = np.random.RandomState(0)
    sub = rng.choice(N, min(RENDER_SUB, N), replace=False)
    PX, PY, PZ = sweep_axis, lat_axis, top_axis
    xl = (pos_rest[:, PX].min(), pos_rest[:, PX].max())
    yl = (pos_rest[:, PY].min(), pos_rest[:, PY].max())
    zl = (pos_rest[:, PZ].min(), pos_rest[:, PZ].max())
    frames = list(range(0, Tr, max(1, Tr // N_RENDER_FRAMES)))

    for pname, hard in PATTERNS.items():
        tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
        # obs for calibration
        multi = []
        for tj in (calA, calB):
            solver.reset_to_rest(pos_rest)
            op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                                np.full(N, STIFF_UNIFORM, np.float32), tj, kr, device,
                                                tau_truth_np=tau_truth)
            multi.append((tj, [torch.tensor(p, device=device) for p in op],
                          [torch.tensor(a, device=device) for a in oa]))
        t0 = time.time()
        _, tau_group, _ = calibrate_online("Group+L_pos_partial", ctx, multi, N_passes=4, lr=0.03,
                                           lr_tau=0.5, lr_decay_per_pass=0.6, lambda_hist=0.0,
                                           lambda_outer_cav=1.0, return_best=True, verbose=False)
        _, tau_pp, _ = calibrate_online("L_pos_partial", ctx, multi, N_passes=4, lr=0.03,
                                        lr_tau=0.5, lr_decay_per_pass=0.6, lambda_hist=0.0,
                                        lambda_outer_cav=1.0, return_best=True, verbose=False)
        print(f"[{pname}] calibrated ({time.time()-t0:.0f}s)  "
              f"GROUP contrast={(tau_group[hard].mean()-tau_group[~hard].mean())/0.4*100:.0f}%  "
              f"pp contrast={(tau_pp[hard].mean()-tau_pp[~hard].mean())/0.4*100:.0f}%", flush=True)

        fields = [("fake-real GT", tau_truth),
                  ("sim init (τ=0.30)", np.full(N, TAU_INIT, np.float32)),
                  ("surface-projection GROUP", tau_group),
                  ("per-particle", tau_pp)]
        # forward each field's cavity
        cav = []
        for label, tf in fields:
            POS, AL = forward_cavity(solver, rest_state, pos_rest, rtraj, kr, device, theta, stiff_t, tf)
            cav.append((label, tf, POS, AL))
        print(f"  forwarded 4 cavities, rendering {len(frames)} frames", flush=True)

        fig = plt.figure(figsize=(20, 5.6))
        axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]

        def update(fi):
            t = frames[fi]
            for ax, (label, tf, POS, AL) in zip(axes, cav):
                ax.clear()
                p = POS[t]; a = AL[t]
                m = (a[sub] > 0.5)
                idx = sub[m]
                ax.scatter(p[idx, PX], p[idx, PY], p[idx, PZ],
                           c=tf[idx], cmap="coolwarm", vmin=TAU_SOFT, vmax=TAU_HARD,
                           s=5, alpha=0.85, edgecolors="none")
                k = rtraj[t]
                ax.scatter([k[PX]], [k[PY]], [k[PZ]], c="black", s=90, marker="v")
                ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
                ax.view_init(elev=22, azim=-60)
                try:
                    ax.set_box_aspect((xl[1]-xl[0], yl[1]-yl[0], zl[1]-zl[0]))
                except Exception:
                    pass
                ax.set_title(f"{label}\nremoved={int((a < 0.5).sum())}", fontsize=10)
            fig.suptitle(f"{pname}  —  debulk cavity per τ-field   (step {t}/{Tr})", fontsize=13)
            return []

        anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
        gif = OUT / f"cavity4_{pname}.gif"
        tr = time.time()
        anim.save(str(gif), writer=PillowWriter(fps=7)); plt.close(fig)
        print(f"  -> saved {gif}  ({time.time()-tr:.0f}s)", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
