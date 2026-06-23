"""NO-(b) DEBULK + RECOVERY video, 2 panels:
  LEFT  = DEBULK view (angled top-down): present body = light gray; the SHAVED-OFF material = RED, frozen at
          its original place → you literally see the thin top layer that was removed. On-screen readout of
          removed % and cavity depth (mm).
  RIGHT = RECOVERED τ cutaway (the hard cylinder, evolving random→recovered).
tau_soften OFF (no droop), τ from cavity only (occ, grouping ON), descending raster [3,5,7]mm. clim hard=yellow.
"""
import sys, os, contextlib
from pathlib import Path
import numpy as np
import torch
import pyvista as pv
import imageio, imageio_ffmpeg  # noqa: F401
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
_DN = open(os.devnull, "w")
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.hierarchy import build_hierarchy
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.calibrate_online import calibrate_online
from tools.pointnetpp.surface_projection import column_grid_assignment
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD, TAU_SOFT, STIFF, NUM_ITER = 0.55, 0.15, 0.9, 6
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = NUM_ITER; solver.enable_cutting = True; solver.muller_stiff_remap = False
solver.tau_soften = False
pr0 = mi["vertices"]; tets = np.asarray(mi["tets"]).astype(np.int64); top = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
tht = torch.tensor([3., 20., 0.3], device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
top0 = pr0[:, top].max()

solver.reset_to_rest(pr0); st = rest0; far = pr0.max(0) + 10.
with torch.no_grad(), contextlib.redirect_stdout(_DN):
    for _ in range(140):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, swt(STIFF), swt(0.3)); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
surf = pr[:, top] > (bb1[top] - 0.10 * bs[top]); surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=dev)
col_id, _ = column_grid_assignment(pr, surf, top, grid_n=8)
ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
eat = torch.tensor(ea, dtype=torch.long, device=dev); ebt = torch.tensor(eb, dtype=torch.long, device=dev)
ert = torch.tensor(np.linalg.norm(pr[ea] - pr[eb], axis=1), dtype=torch.float32, device=dev)
H = build_hierarchy(torch.from_numpy(pr).to(dev, torch.float32), M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
hard = ((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
rng = np.random.default_rng(0); tau_init = rng.uniform(0.12, 0.58, N).astype(np.float32)
base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
REACHES_MM = [3.0, 5.0, 7.0]
traj = []
for rmm in REACHES_MM:
    y = bb1[top] - rmm * 1e-3 + kr
    p = [kp.copy().astype(np.float32) for kp in base]
    for kp in p: kp[top] = np.float32(y)
    traj += p

with contextlib.redirect_stdout(_DN):
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                        np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
op = np.asarray(op); oa = np.asarray(oa)
ctx = dict(solver=solver, rest_state=rest_state, N=N, device=dev, knife_radius=kr, pos_rest=pr,
           theta_3d_np=np.array([3., 20., 0.3], np.float32), stiff_init_np=np.full(N, STIFF, np.float32), tau_init_np=tau_init,
           tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt, edges_rest_t=ert, H=H,
           group_col_id=col_id, surf_top_axis=top)
multi = [(traj, [torch.tensor(p, device=dev) for p in op], [torch.tensor(a, device=dev) for a in oa])]
snaps = []
with contextlib.redirect_stdout(_DN):
    _, tau_best, _ = calibrate_online("Group+mesh_nob_debulk", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5, lr_decay_per_pass=0.6,
                     lambda_hist=0.0, lambda_outer_cav=1.0, return_best=True, line_search=False, verbose=False,
                     chamfer_loss=False, lambda_chamfer=0.0, cav_occupancy=True, cav_occ_pointcloud=True,
                     cav_occ_meannorm=True, snapshot_every=12, snapshot_out=snaps)
tau_best = np.asarray(tau_best)
snap_taus = [np.asarray(s[1]) for s in snaps]
_ctr = [(t[hard].mean() - t[~hard].mean()) / (TAU_HARD - TAU_SOFT) for t in snap_taus] or [0]
right_seq = [tau_init] + snap_taus[:int(np.argmax(_ctr)) + 1] + [tau_best]
final_removed = float((oa[-1] < 0.5).mean()) * 100
print(f"calib done; contrast {(tau_best[hard].mean()-tau_best[~hard].mean())/(TAU_HARD-TAU_SOFT)*100:.0f}%; "
      f"removed {final_removed:.1f}%; right_seq={len(right_seq)}", flush=True)


def remap(p):
    return np.stack([p[:, 0], p[:, 2], p[:, 1]], axis=1)


center = remap(pr).mean(0); size = float((remap(pr).max(0) - remap(pr).min(0)).max())
ctypes = np.full(len(tets), pv.CellType.TETRA, np.uint8)
n_anim = 96
out = REPO / "tools/pointnetpp/viz_out/tau_recovery_mesh_NOB_DEBULK_SPOT.mp4"
writer = imageio.get_writer(str(out), fps=10, codec="libx264", quality=8, pixelformat="yuv420p")
pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=[1760, 880], border=False)
cam_L = [center[0] + size * 0.55, center[1] - size * 0.75, center[2] + size * 1.7]   # angled top-down (see the shave)
cam_R = [center[0] + size * 1.4, center[1] - size * 1.5, center[2] + size * 1.0]      # side (see cutaway cylinder)

for i in range(n_anim):
    cur = int(round(i / max(1, n_anim - 1) * (len(op) - 1)))
    rcur = right_seq[int(round(i / max(1, n_anim - 1) * (len(right_seq) - 1)))]
    rct = (rcur[hard].mean() - rcur[~hard].mean()) / (TAU_HARD - TAU_SOFT) * 100
    verts = remap(op[cur]); present = oa[cur] >= 0.5; removed = ~present
    pct = removed.mean() * 100
    depth = (top0 - pr[removed, top].min()) * 1e3 if removed.any() else 0.0
    keep = present[tets].all(axis=1); kt = tets[keep]
    cells = np.hstack([np.full((len(kt), 1), 4, np.int64), kt]).flatten()
    grid = pv.UnstructuredGrid(cells, ctypes[keep], verts)
    kp = traj[cur]; knife = pv.Sphere(radius=kr, center=(kp[0], kp[2], kp[1]))

    # LEFT: debulk view — gray body + RED shaved material (frozen at original place)
    pl.subplot(0, 0)
    pl.add_mesh(grid.extract_surface(), color="#d9d9d9", smooth_shading=True)
    if removed.any():
        pl.add_mesh(pv.PolyData(remap(pr[removed])), color="red", point_size=7, render_points_as_spheres=True)
    pl.add_mesh(knife, color="orange", opacity=0.9)
    pl.add_text(f"DEBULK  —  removed {pct:.1f}%   depth {depth:.0f}mm\nred = shaved-off layer (frozen in place)",
                position="upper_edge", font_size=12, color="black", name="t")
    pl.set_background("white")
    pl.camera.position = cam_L; pl.camera.focal_point = center; pl.camera.up = (0, 0, 1); pl.camera.view_angle = 32

    # RIGHT: recovered τ cutaway cylinder
    pl.subplot(0, 1)
    g = grid.copy(); g["tau"] = rcur
    half = g.clip(normal="y", origin=center, invert=False)
    if half.n_points > 0:
        pl.add_mesh(half, scalars="tau", cmap="viridis", clim=[0.15, 0.55], smooth_shading=True,
                    scalar_bar_args={"title": "τ (yellow=hard)", "color": "black"})
    pl.add_mesh(knife, color="orange", opacity=0.9)
    pl.add_text(f"RECOVERED τ — cutaway cylinder  ({rct:.0f}%)", position="upper_edge", font_size=12, color="black", name="t2")
    pl.set_background("white")
    pl.camera.position = cam_R; pl.camera.focal_point = center; pl.camera.up = (0, 0, 1); pl.camera.view_angle = 32

    writer.append_data(pl.screenshot(return_img=True)); pl.clear()
    if i % 12 == 0 or i == n_anim - 1:
        print(f"  anim {i+1}/{n_anim}  removed {pct:.1f}% depth {depth:.0f}mm", flush=True)

writer.close(); pl.close()
print(f"saved {out}", flush=True)
