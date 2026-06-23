"""GROUND-TRUTH SANITY: show the SPOT hard inclusion is a vertical CYLINDER in the centre.
The surface mesh only shows its top circle (cylinder is interior, doesn't reach the sides). Here we build
the FULL tetrahedral VOLUME, colour by τ truth (clim [0.15,0.55] → hard=bright yellow), and CLIP it in
half so the interior cylinder is visible. 2 panels: LEFT surface (what the video shows) | RIGHT cutaway.
"""
import sys, os, contextlib
from pathlib import Path
import numpy as np
import pyvista as pv
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
_DN = open(os.devnull, "w")
from tools.day2_solver_wrapper import setup_solver_with_rest
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS

TAU_HARD, TAU_SOFT = 0.55, 0.15
cfg = MESH_CONFIGS[0]; dev = "cuda:0"
with contextlib.redirect_stdout(_DN):
    solver, rest0, mi = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
N = solver.N; pr0 = mi["vertices"]; tets = np.asarray(mi["tets"]).astype(np.int64); top = cfg["top_axis"]
lat = [a for a in range(3) if a != top]; sweep, la = lat[0], lat[1]
bb0, bb1 = pr0.min(0), pr0.max(0); bs = bb1 - bb0
xn = (pr0[:, sweep] - bb0[sweep]) / bs[sweep]; zn = (pr0[:, la] - bb0[la]) / bs[la]
hard = ((xn - 0.5) ** 2 + (zn - 0.5) ** 2) < 0.25 ** 2
tau_truth = np.where(hard, TAU_HARD, TAU_SOFT).astype(np.float32)
print(f"N={N} hard(cylinder) particles={int(hard.sum())} = {hard.mean()*100:.1f}% of cube", flush=True)


def remap(p):
    return np.stack([p[:, 0], p[:, 2], p[:, 1]], axis=1)   # sim(x,y_up,z) -> pyvista(x,z,y_up)


verts = remap(pr0)
cells = np.hstack([np.full((len(tets), 1), 4, np.int64), tets]).flatten()
ctypes = np.full(len(tets), pv.CellType.TETRA, np.uint8)
grid = pv.UnstructuredGrid(cells, ctypes, verts)
grid["tau"] = tau_truth
center = verts.mean(0)
half = grid.clip(normal="y", origin=center, invert=False)   # keep FAR half so its cut face (w/ cylinder) faces camera
surf = grid.extract_surface()

pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=[1700, 850], border=False)
size = float((verts.max(0) - verts.min(0)).max())
cam = [center[0] + size * 1.4, center[1] - size * 1.5, center[2] + size * 1.0]
for col, (m, ttl) in enumerate([(surf, "SURFACE (what the video shows)\nonly the cylinder's TOP circle is visible"),
                                 (half, "CUTAWAY (cube sliced in half)\nthe hard τ region is a vertical CYLINDER")]):
    pl.subplot(0, col)
    pl.add_mesh(m, scalars="tau", cmap="viridis", clim=[0.15, 0.55], show_edges=False, smooth_shading=True,
                scalar_bar_args={"title": "τ (yellow=hard 0.55, purple=soft 0.15)", "color": "black"})
    pl.add_text(ttl, position="upper_edge", font_size=12, color="black")
    pl.set_background("white")
    pl.camera.position = cam; pl.camera.focal_point = center; pl.camera.up = (0, 0, 1); pl.camera.view_angle = 32
out = REPO / "tools/pointnetpp/viz_out/gt_spot_cutaway.png"
pl.screenshot(str(out)); pl.close()
print(f"saved {out}", flush=True)
