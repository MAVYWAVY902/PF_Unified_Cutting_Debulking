"""MESH-BASED version of tau_recovery_persp_firm_SPOT — renders the cube as a SHADED SURFACE MESH
(PyVista), not a point cloud. Same firm physics (num_iter=6, k=0.9, settle 140). Two panels:
  LEFT  = GROUND TRUTH τ on the carving surface
  RIGHT = RECONSTRUCTED τ (evolving during calibration)
Cavity forms as the knife removes particles → surface faces touching removed verts are dropped (a hole).
Knife = red sphere. Output mp4. Coords remapped so gravity-Y points up in the render.
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

TAU_HARD, TAU_SOFT, STIFF, NUM_ITER, REACH_MM, R = 0.55, 0.15, 0.9, 6, 4.0, 3
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = NUM_ITER; solver.enable_cutting = True; solver.muller_stiff_remap = False
pr0 = mi["vertices"]; tets = np.asarray(mi["tets"]).astype(np.int64); top = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
tht = torch.tensor([3., 20., 0.3], device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)


def surface_faces(tets):
    f = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]], axis=0)
    key = np.sort(f, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return f[idx[cnt == 1]]            # boundary triangles (appear in exactly one tet)


faces = surface_faces(tets); n_faces = len(faces)
print(f"N={N}  surface faces={n_faces}", flush=True)

# settle 140 (firm)
solver.tau_soften = False
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

# SPOT inclusion + cut traj
xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
hard = ((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
rng = np.random.default_rng(0); tau_init = rng.uniform(0.12, 0.58, N).astype(np.float32)
y_center = bb1[top] - REACH_MM * 1e-3 + kr
base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
one = [kp.copy().astype(np.float32) for kp in base]
for kp in one: kp[top] = np.float32(y_center)
traj = one * R

# obs + calibration (reconstructed τ snapshots), same as the firm video
solver.tau_soften = True
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
solver.tau_soften = True; solver.num_iterations = NUM_ITER
with contextlib.redirect_stdout(_DN):
    calibrate_online("mesh_spot", ctx, multi, N_passes=4, lr=0.03, lr_tau=0.5, lr_decay_per_pass=0.6,
                     lambda_hist=0.0, lambda_outer_cav=5.0e-4, return_best=True, line_search=False, verbose=False,
                     chamfer_loss=True, lambda_chamfer=1.0, cav_occupancy=True, cav_occ_pointcloud=True,
                     cav_occ_meannorm=True, snapshot_every=17, snapshot_out=snaps)
solver.tau_soften = False
steps = [s[0] for s in snaps]; taus = [s[1] for s in snaps]
print(f"calibration done, {len(taus)} snapshots; final hard {taus[-1][hard].mean():.2f} soft {taus[-1][~hard].mean():.2f}", flush=True)

# ---- PyVista 2-panel mesh render ----
def remap(p):  # sim (x, y=up, z) → pyvista (x, z, y=up)
    return np.stack([p[:, 0], p[:, 2], p[:, 1]], axis=1)


center = remap(pr).mean(0); size = float((remap(pr).max(0) - remap(pr).min(0)).max())
n_anim = len(taus)
out = REPO / "tools/pointnetpp/viz_out/tau_recovery_mesh_firm_SPOT.mp4"
writer = imageio.get_writer(str(out), fps=8, codec="libx264", quality=8, pixelformat="yuv420p")
pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=[1800, 900], border=False)
cam = [center[0] + size * 1.4, center[1] - size * 1.5, center[2] + size * 1.0]

for i in range(n_anim):
    cur = steps[i] % len(traj)
    verts = remap(op[cur])
    rem = oa[cur] < 0.5
    fmask = ~(rem[faces[:, 0]] | rem[faces[:, 1]] | rem[faces[:, 2]])
    af = faces[fmask]
    fpv = np.hstack([np.full((len(af), 1), 3, np.int64), af]).flatten()
    mesh = pv.PolyData(verts, fpv)
    for col, (scal, ttl) in enumerate([(tau_truth, "GROUND TRUTH τ"), (taus[i], "RECONSTRUCTED τ")]):
        pl.subplot(0, col)
        m = mesh.copy(); m["tau"] = scal
        pl.add_mesh(m, scalars="tau", cmap="viridis", clim=[0.10, 0.60], show_edges=False,
                    smooth_shading=True, scalar_bar_args={"title": "τ", "color": "black"})
        ksph = pv.Sphere(radius=kr, center=(op[cur].mean(0)[0], 0, 0))  # placeholder, real knife below
        kp = traj[cur]; pl.add_mesh(pv.Sphere(radius=kr, center=(kp[0], kp[2], kp[1])), color="red", opacity=0.9)
        if col == 1:
            ct = (taus[i][hard].mean() - taus[i][~hard].mean()) / (TAU_HARD - TAU_SOFT) * 100
            ttl = f"{ttl} step {steps[i]}  hard {taus[i][hard].mean():.2f}/soft {taus[i][~hard].mean():.2f} ({ct:.0f}%)"
        pl.add_text(ttl, position="upper_edge", font_size=11, color="black", name="t")
        pl.set_background("white")
        pl.camera.position = cam; pl.camera.focal_point = center; pl.camera.up = (0, 0, 1); pl.camera.view_angle = 32
    img = pl.screenshot(return_img=True)
    writer.append_data(img)
    pl.clear()
    if i % 10 == 0 or i == n_anim - 1:
        print(f"  frame {i+1}/{n_anim}  faces alive {len(af)}/{n_faces}", flush=True)

writer.close(); pl.close()
print(f"saved {out}", flush=True)
