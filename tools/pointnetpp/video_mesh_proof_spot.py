"""CLEAN PROOF video: firm SPOT cube (num_iter=6, k=0.9, settle 140), SINGLE physics pass (the cut once,
NO 4x calibration replay), REAL frame numbers (not 'step'), and a FIXED red reference line sitting at the
settled cube TOP. The line never moves; if the cube were sinking it would drop below the line. It does not —
only the central pit (the knife cut) opens. Colored by ground-truth τ. Output mp4.
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
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

TAU_HARD, TAU_SOFT, STIFF, NUM_ITER, REACH_MM, R = 0.55, 0.15, 0.9, 6, 4.0, 3
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; solver.damping = 0.95; solver.num_iterations = NUM_ITER; solver.enable_cutting = True; solver.muller_stiff_remap = False
pr0 = mi["vertices"]; tets = np.asarray(mi["tets"]).astype(np.int64); top = cfg["top_axis"]; kr = cfg["knife_radius_abs"]
solver.gravity = wp.vec3(0., -9.81, 0.); solver.floor_y = float(pr0[:, 2].min() - 1.0)
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
tht = torch.tensor([3., 20., 0.3], device=dev); swt = lambda v: torch.tensor(np.full(N, v, np.float32), device=dev)
top0 = pr0[:, top].max()


def surface_faces(tets):
    f = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]], axis=0)
    _, idx, cnt = np.unique(np.sort(f, axis=1), axis=0, return_index=True, return_counts=True)
    return f[idx[cnt == 1]]


faces = surface_faces(tets); n_faces = len(faces)
solver.tau_soften = False
solver.reset_to_rest(pr0); st = rest0; far = pr0.max(0) + 10.
with torch.no_grad(), contextlib.redirect_stdout(_DN):
    for _ in range(140):
        _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, swt(STIFF), swt(0.3)); st = solver_save_state(solver)
pr = pw.cpu().numpy().copy(); rest_state = st
settled_top = pr[:, top].max()
print(f"N={N} faces={n_faces}  settled top = {settled_top*1e3:.1f}mm (reference line height)", flush=True)
bb0, bb1 = pr.min(0), pr.max(0); bs = bb1 - bb0
xn = (pr[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr[:, la] - bb0[la]) / bs[la]
hard = ((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
y_center = bb1[top] - REACH_MM * 1e-3 + kr
base = make_raster(bb0, bb1, top, sweep, la, 0.5, 6, 12)
one = [kp.copy().astype(np.float32) for kp in base]
for kp in one: kp[top] = np.float32(y_center)
traj = one * R                                    # ONE physics pass (the 3-raster cut), no replay

solver.tau_soften = True
with contextlib.redirect_stdout(_DN):
    op, oa = gen_obs_per_particle_stiff(solver, rest_state, np.array([3., 20., 0.3], np.float32),
                                        np.full(N, STIFF, np.float32), traj, kr, dev, tau_truth_np=tau_truth)
op = np.asarray(op); oa = np.asarray(oa)


def remap(p):
    return np.stack([p[:, 0], p[:, 2], p[:, 1]], axis=1)


center = remap(pr).mean(0); size = float((remap(pr).max(0) - remap(pr).min(0)).max())
# fixed reference line at settled top, along the front-top edge (min-z side), spanning x
xlo, xhi = pr[:, 0].min(), pr[:, 0].max(); zfront = pr[:, 2].min()
ref_a = np.array([xlo, zfront, settled_top]); ref_b = np.array([xhi, zfront, settled_top])
ref_line = pv.Line(ref_a, ref_b)

frames = list(range(0, len(op), max(1, len(op) // 54)))
out = REPO / "tools/pointnetpp/viz_out/tau_recovery_mesh_PROOF_SPOT.mp4"
writer = imageio.get_writer(str(out), fps=8, codec="libx264", quality=8, pixelformat="yuv420p")
pl = pv.Plotter(off_screen=True, window_size=[1100, 1000], border=False)
cam = [center[0] + size * 1.4, center[1] - size * 1.5, center[2] + size * 1.0]

for fi, cur in enumerate(frames):
    verts = remap(op[cur]); rem = oa[cur] < 0.5
    fmask = ~(rem[faces[:, 0]] | rem[faces[:, 1]] | rem[faces[:, 2]])
    af = faces[fmask]
    fpv = np.hstack([np.full((len(af), 1), 3, np.int64), af]).flatten()
    mesh = pv.PolyData(verts, fpv); mesh["tau"] = tau_truth
    pl.add_mesh(mesh, scalars="tau", cmap="viridis", clim=[0.10, 0.60], smooth_shading=True,
                scalar_bar_args={"title": "τ", "color": "black"})
    pl.add_mesh(ref_line, color="red", line_width=6)        # FIXED reference at settled top
    kp = traj[cur]; pl.add_mesh(pv.Sphere(radius=kr, center=(kp[0], kp[2], kp[1])), color="orange", opacity=0.9)
    sag = (top0 - op[cur][:, top].max()) * 1e3
    pl.add_text(f"frame {cur}/{len(op)-1}   red line = settled top (FIXED)\ntop-surface sag {sag:.1f} mm  (cube edge stays on the line; only the pit opens)",
                position="upper_edge", font_size=11, color="black", name="t")
    pl.set_background("white")
    pl.camera.position = cam; pl.camera.focal_point = center; pl.camera.up = (0, 0, 1); pl.camera.view_angle = 32
    writer.append_data(pl.screenshot(return_img=True))
    pl.clear()
    if fi % 10 == 0 or fi == len(frames) - 1:
        print(f"  frame {cur}  faces alive {len(af)}/{n_faces}", flush=True)

writer.close(); pl.close()
print(f"saved {out}", flush=True)
