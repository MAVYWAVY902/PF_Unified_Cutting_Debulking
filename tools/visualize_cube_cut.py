"""Visualize cube cutting simulation — cumulative SURFACE damage progression.

Renders cube_tet.msh (or synthetic lattice) as a sequence of frames showing:
  - Particles colored by alpha (1=healthy green, 0=fully damaged red)
  - Knife sphere (translucent steelblue) at current trajectory position,
    HALF-EMBEDDED in the top surface (surface cutting, not internal)
  - Cube bounding box

Two viewing modes:
  - --side-by-side: left = α_obs (true θ), right = α_sim (init θ).
  - default (single panel): just α_obs progression.

Outputs (in --outdir):
  frame_0000.png, frame_0001.png, ...
  rollout.gif    (if --gif)

Usage:
    python tools/visualize_cube_cut.py                       # default: cube_tet.msh, surface cut
    python tools/visualize_cube_cut.py --frames 400 --gif    # slow surface cut + gif
    python tools/visualize_cube_cut.py --synth --N_per_dim 12 # legacy synthetic cube lattice
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Force offscreen BEFORE pyvista import — must happen here.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import warp as wp  # noqa: E402

wp.init()

from tools.online_residual_synth import (  # noqa: E402
    setup_cube_system,
    make_cube_knife_trajectory,
    make_params,
    run_ssa_full_with_state,
    load_msh_system,
    AdamND,
    compute_outer_gradient,
)


def collect_alpha_trajectory(sys_dict, theta, traj, accumulate=True, knife_radius=0.1):
    """Run forward sim with given theta along the knife trajectory and
    return per-frame alpha snapshots."""
    alpha_list = []
    state_persistent = None
    for kp in traj:
        sys_t = dict(sys_dict)
        sys_t["knife_pos"] = np.array([kp], dtype=np.float32)
        params = make_params(
            alpha_min=float(theta[0]),
            sharpness=float(theta[1]),
            threshold=float(theta[2]),
            knife_radius=knife_radius,
            requires_grad=False,
        )
        alpha_wp, dv_wp, ed_wp = run_ssa_full_with_state(
            params, sys_t, n_cycles=1, state_in=state_persistent,
        )
        alpha_np = alpha_wp.numpy().copy()
        alpha_list.append(alpha_np)
        if accumulate:
            state_persistent = dict(
                alpha=alpha_np.astype(np.float32),
                edge_damage=ed_wp.numpy().astype(np.float32),
                particle_damage=dv_wp.numpy().astype(np.float32),
            )
    return np.array(alpha_list)  # (T, N)


def render_frame_surface(plotter, pos, surface_faces, alpha, knife_pos,
                          knife_radius=0.04, bbox_lo=None, bbox_hi=None,
                          title="", alpha_threshold=0.3):
    """Render the mesh as a reconstructed surface, with α-culled triangles.

    Each triangle whose min-α-over-3-vertices falls below alpha_threshold is
    DROPPED — this is how the C++ viewer carves out the cavity. The result is
    a real 3D hole that grows as damage accumulates.

    Args:
        pos             — (N, 3) vertex positions
        surface_faces   — (F, 3) int32 triangle index array from load_msh_system
        alpha           — (N,) per-vertex α
        alpha_threshold — drop faces with min(α[3 verts]) < threshold
    """
    import pyvista as pv

    # Per-face alpha = min of 3 vertices' alpha. Use min so any "dead" vertex
    # in a triangle causes the whole triangle to be culled.
    face_alpha = alpha[surface_faces].min(axis=1)
    alive_mask = face_alpha > alpha_threshold
    alive_faces = surface_faces[alive_mask]

    if len(alive_faces) > 0:
        n_f = len(alive_faces)
        faces_flat = np.hstack([
            np.full((n_f, 1), 3, dtype=np.int64),
            alive_faces.astype(np.int64),
        ]).ravel()
        surf = pv.PolyData(pos.astype(np.float32), faces_flat)
        surf.point_data["alpha"] = alpha.astype(np.float32)
        plotter.add_mesh(
            surf,
            scalars="alpha",
            cmap="RdYlGn",
            clim=[alpha_threshold, 1.0],
            smooth_shading=True,
            show_edges=False,
            show_scalar_bar=True,
            scalar_bar_args={"title": "α (surface)",
                              "n_labels": 3, "fmt": "%.2f"},
        )

    # Knife sphere
    knife_sphere = pv.Sphere(radius=knife_radius, center=knife_pos,
                              theta_resolution=24, phi_resolution=24)
    plotter.add_mesh(knife_sphere, color="steelblue", opacity=0.55,
                      show_edges=False)
    plotter.add_mesh(
        pv.PolyData(np.asarray(knife_pos, dtype=np.float32).reshape(1, 3)),
        color="navy", point_size=10, render_points_as_spheres=True,
    )

    # Bounding box
    if bbox_lo is not None and bbox_hi is not None:
        bbox = pv.Box(bounds=[float(bbox_lo[0]), float(bbox_hi[0]),
                              float(bbox_lo[1]), float(bbox_hi[1]),
                              float(bbox_lo[2]), float(bbox_hi[2])])
        plotter.add_mesh(bbox, style="wireframe", color="dimgray",
                          line_width=1, opacity=0.5)

    if title:
        plotter.add_text(title, position="upper_edge", font_size=10,
                          color="black")


def render_frame(plotter, pos, edges_a, edges_b, alpha, knife_pos,
                 knife_radius=0.1, bbox_lo=None, bbox_hi=None, title="",
                 show_edges=False):
    """Add particle cloud (+ optional edges) + knife sphere + bbox to a pyvista plotter.
    Returns the actors so caller can update or remove them.
    """
    import pyvista as pv

    actors = []

    # (Optional, off by default) edges between particles where both alive
    if show_edges and edges_a is not None and edges_b is not None:
        a_min = np.minimum(alpha[edges_a], alpha[edges_b])
        alive_mask = a_min > 0.1
        if alive_mask.sum() > 0:
            edge_pts_a = pos[edges_a][alive_mask]
            edge_pts_b = pos[edges_b][alive_mask]
            n_lines = alive_mask.sum()
            lines_pts = np.empty((2 * n_lines, 3), dtype=np.float32)
            lines_pts[0::2] = edge_pts_a
            lines_pts[1::2] = edge_pts_b
            lines_cells = np.empty((n_lines, 3), dtype=np.int64)
            lines_cells[:, 0] = 2
            lines_cells[:, 1] = np.arange(0, 2 * n_lines, 2)
            lines_cells[:, 2] = np.arange(1, 2 * n_lines, 2)
            poly = pv.PolyData(lines_pts, lines=lines_cells.ravel())
            a_edges = plotter.add_mesh(poly, color="lightgray",
                                       line_width=0.5, opacity=0.25)
            actors.append(a_edges)

    # Particle cloud colored by alpha (green = healthy α=1, red = damaged α=0)
    cloud = pv.PolyData(pos.astype(np.float32))
    cloud.point_data["alpha"] = alpha.astype(np.float32)
    a_cloud = plotter.add_mesh(
        cloud,
        scalars="alpha",
        cmap="RdYlGn",     # red at 0 (damaged), yellow at 0.5, green at 1 (healthy)
        clim=[0.0, 1.0],
        point_size=10,
        render_points_as_spheres=True,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "α (1=healthy, 0=damaged)",
            "n_labels": 3, "fmt": "%.1f",
        },
    )
    actors.append(a_cloud)

    # Knife sphere — steelblue so it doesn't clash with healthy=green
    knife_sphere = pv.Sphere(radius=knife_radius, center=knife_pos,
                              theta_resolution=24, phi_resolution=24)
    a_knife = plotter.add_mesh(knife_sphere, color="steelblue", opacity=0.55,
                                show_edges=False)
    actors.append(a_knife)
    a_knife_ctr = plotter.add_mesh(
        pv.PolyData(np.asarray(knife_pos, dtype=np.float32).reshape(1, 3)),
        color="navy", point_size=10, render_points_as_spheres=True,
    )
    actors.append(a_knife_ctr)

    # Bounding box
    if bbox_lo is not None and bbox_hi is not None:
        bbox = pv.Box(bounds=[float(bbox_lo[0]), float(bbox_hi[0]),
                              float(bbox_lo[1]), float(bbox_hi[1]),
                              float(bbox_lo[2]), float(bbox_hi[2])])
        a_bbox = plotter.add_mesh(bbox, style="wireframe", color="dimgray",
                                  line_width=1, opacity=0.5)
        actors.append(a_bbox)

    if title:
        a_title = plotter.add_text(title, position="upper_edge", font_size=12,
                                   color="black")
        actors.append(a_title)

    return actors


def run_calibration_with_capture(sys_dict, real_frames, theta_0,
                                  knife_radius=0.04,
                                  lr_outer=(0.1, 1.0, 0.05),
                                  bounds=((0.5, 50.0), (1.0, 200.0), (0.05, 0.95)),
                                  verbose=False):
    """Mini Mode-D calibration loop that captures per-frame data for viz.

    Returns:
        alpha_sim_traj:  (T, N) per-frame α_sim at θ_t (BEFORE the t-th Adam step)
        theta_traj:      (T, 3) θ_t at each frame
        L_direct_traj:   (T,)
    """
    theta = np.asarray(theta_0, dtype=np.float64).copy()
    opt = AdamND(n_dims=3, lr=np.asarray(lr_outer, dtype=np.float64))
    lo_b = np.asarray([b[0] for b in bounds]); hi_b = np.asarray([b[1] for b in bounds])

    state_p = None
    alpha_sim_traj = []
    theta_traj = []
    L_traj = []

    for t, obs in enumerate(real_frames):
        sys_t = dict(sys_dict)
        sys_t["knife_pos"] = np.array([obs["knife_pos"]], dtype=np.float32)
        state_pre = None if state_p is None else dict(state_p)

        params_curr = make_params(
            alpha_min=float(theta[0]), sharpness=float(theta[1]),
            threshold=float(theta[2]), knife_radius=knife_radius,
            requires_grad=False,
        )
        a_wp, dv_wp, ed_wp = run_ssa_full_with_state(
            params_curr, sys_t, n_cycles=1, state_in=state_pre,
        )
        alpha_sim = a_wp.numpy().copy()
        dv_sim = dv_wp.numpy().copy()
        ed_sim = ed_wp.numpy().copy()

        # Record BEFORE Adam step (so theta_traj[t] is the θ that produced alpha_sim[t])
        alpha_sim_traj.append(alpha_sim)
        theta_traj.append(theta.copy())

        L_d, grad = compute_outer_gradient(
            theta, sys_t, obs["alpha_obs"], n_cycles=1,
            state_in=state_pre, knife_radius=knife_radius,
        )
        L_traj.append(L_d)
        theta = np.clip(opt.step(theta, grad), lo_b, hi_b)

        # Mode D: raw sim state, no residual injection
        state_p = dict(
            alpha=alpha_sim.astype(np.float32),
            edge_damage=ed_sim.astype(np.float32),
            particle_damage=dv_sim.astype(np.float32),
        )

        if verbose and (t % 40 == 0 or t == len(real_frames) - 1):
            print(f"  [calib] t={t:3d}  L={L_d:.2e}  θ=({theta[0]:.3f}, {theta[1]:.3f}, {theta[2]:.3f})")

    return np.array(alpha_sim_traj), np.array(theta_traj), np.array(L_traj)


def render_diff_panel(plotter, pos, alpha_obs, alpha_sim, knife_pos,
                       knife_radius, bbox_lo, bbox_hi, title="",
                       surface_faces=None, mesh_mode='points',
                       alpha_threshold=0.3):
    """Render |α_obs - α_sim| panel. Two modes:

    points: every particle as a dot colored by |Δα|. Shows ALL particle
        diffs including internal ones (cube core not visible to camera).
    surface: render only the reconstructed surface mesh, colored by |Δα|.
        Cavity hole is carved out where MIN(α_obs, α_sim) < threshold (so
        the hole appears in both this panel and the obs/sim panels).

    Colormap: magma_r. White=no diff, dark magenta=large diff.
    """
    import pyvista as pv
    diff = np.abs(alpha_obs - alpha_sim).astype(np.float32)
    max_diff = max(0.05, float(diff.max()) * 1.05)

    if mesh_mode == 'surface' and surface_faces is not None:
        # Cull faces where the cavity already exists (min α drops below threshold
        # in EITHER obs or sim) — gives a unified cavity hole shape so the
        # 3 panels align visually.
        alpha_combined = np.minimum(alpha_obs, alpha_sim)
        face_alpha = alpha_combined[surface_faces].min(axis=1)
        alive_mask = face_alpha > alpha_threshold
        alive_faces = surface_faces[alive_mask]

        if len(alive_faces) > 0:
            n_f = len(alive_faces)
            faces_flat = np.hstack([
                np.full((n_f, 1), 3, dtype=np.int64),
                alive_faces.astype(np.int64),
            ]).ravel()
            surf = pv.PolyData(pos.astype(np.float32), faces_flat)
            surf.point_data["diff"] = diff
            plotter.add_mesh(
                surf, scalars="diff", cmap="magma_r",
                clim=[0.0, max_diff],
                smooth_shading=True, show_edges=False,
                show_scalar_bar=True,
                scalar_bar_args={"title": "|α_obs - α_sim|",
                                  "n_labels": 3, "fmt": "%.3f"},
            )
    else:
        cloud = pv.PolyData(pos.astype(np.float32))
        cloud.point_data["diff"] = diff
        plotter.add_mesh(
            cloud, scalars="diff", cmap="magma_r",
            clim=[0.0, max_diff],
            point_size=10, render_points_as_spheres=True, show_scalar_bar=True,
            scalar_bar_args={"title": "|α_obs - α_sim|",
                              "n_labels": 3, "fmt": "%.2f"},
        )

    # Knife sphere (faint, since this panel is about diff)
    knife_sphere = pv.Sphere(radius=knife_radius, center=knife_pos,
                              theta_resolution=24, phi_resolution=24)
    plotter.add_mesh(knife_sphere, color="steelblue", opacity=0.25)

    # Bounding box
    bbox = pv.Box(bounds=[float(bbox_lo[0]), float(bbox_hi[0]),
                          float(bbox_lo[1]), float(bbox_hi[1]),
                          float(bbox_lo[2]), float(bbox_hi[2])])
    plotter.add_mesh(bbox, style="wireframe", color="dimgray",
                      line_width=1, opacity=0.5)

    if title:
        plotter.add_text(title, position="upper_edge", font_size=10, color="black")


def render_composite_frame(t, frame_idx, total_frames,
                            pos, edges_a, edges_b,
                            alpha_obs_t, alpha_sim_t,
                            knife_pos_t, knife_radius,
                            bbox_lo, bbox_hi,
                            theta_traj, theta_true, L_traj,
                            out_path, window=(1800, 1000),
                            show_diff=False,
                            surface_faces=None,
                            mesh_mode='points',
                            alpha_threshold=0.3):
    """Render one composite frame: pyvista 3D (top) + matplotlib θ-convergence
    curves (bottom).

    If show_diff=True, top row becomes 3 panels: α_obs / α_sim / |α_obs - α_sim|.
    Else, top row is 2 panels: α_obs / α_sim.
    """
    import pyvista as pv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Top: pyvista panels render to in-memory image ---
    n_panels = 3 if show_diff else 2
    pv_w = window[0]; pv_h = int(window[1] * 0.65)
    plotter = pv.Plotter(off_screen=True, window_size=[pv_w, pv_h],
                         shape=(1, n_panels))

    cavity_obs = (alpha_obs_t < 0.5).mean()
    cavity_sim = (alpha_sim_t < 0.5).mean()
    th_now = theta_traj[t]

    def _render_one(alpha_v, ttl):
        if mesh_mode == 'surface' and surface_faces is not None:
            render_frame_surface(plotter, pos, surface_faces, alpha_v, knife_pos_t,
                                 knife_radius=knife_radius,
                                 bbox_lo=bbox_lo, bbox_hi=bbox_hi,
                                 title=ttl, alpha_threshold=alpha_threshold)
        else:
            render_frame(plotter, pos, edges_a, edges_b, alpha_v, knife_pos_t,
                         knife_radius=knife_radius,
                         bbox_lo=bbox_lo, bbox_hi=bbox_hi,
                         title=ttl)

    plotter.subplot(0, 0)
    _render_one(alpha_obs_t,
                f"α_obs (θ_true)    frame {t}/{total_frames-1}    cavity={cavity_obs:.3f}")
    plotter.view_isometric()

    plotter.subplot(0, 1)
    _render_one(alpha_sim_t,
                f"α_sim (θ_t = {th_now[0]:.2f}, {th_now[1]:.2f}, {th_now[2]:.3f})    cavity={cavity_sim:.3f}")
    plotter.view_isometric()

    if show_diff:
        plotter.subplot(0, 2)
        diff = np.abs(alpha_obs_t - alpha_sim_t)
        n_signal = int((diff > 0.05).sum())
        mean_diff = float(diff.mean())
        max_diff = float(diff.max())
        render_diff_panel(plotter, pos, alpha_obs_t, alpha_sim_t, knife_pos_t,
                          knife_radius=knife_radius, bbox_lo=bbox_lo, bbox_hi=bbox_hi,
                          title=f"|α_obs - α_sim|    "
                                f"signal particles (>0.05) = {n_signal}    "
                                f"mean |Δα|={mean_diff:.3f}    max={max_diff:.3f}",
                          surface_faces=surface_faces, mesh_mode=mesh_mode,
                          alpha_threshold=alpha_threshold)
        plotter.view_isometric()

    plotter.link_views()
    pv_img = plotter.screenshot(return_img=True)
    plotter.close()

    # --- Bottom: matplotlib θ trajectories (3 stacked subplots) ---
    fig = plt.figure(figsize=(window[0] / 100.0, window[1] / 100.0), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.45, wspace=0.3,
                          left=0.04, right=0.98, top=0.97, bottom=0.08)

    # Top row spans all 3 columns: pyvista image
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.imshow(pv_img)
    ax_top.axis('off')

    # Bottom row: 3 panels, one per param
    param_names = ['α_min  (knife strength)', 'sharpness  (sigmoid)', 'τ  (threshold)']
    param_colors = ['#1f77b4', '#d62728', '#2ca02c']
    ylims = [(0.0, 6.0), (0.0, 25.0), (0.0, 1.0)]
    for col, (name, c, ylim, true_v) in enumerate(zip(param_names, param_colors, ylims, theta_true)):
        ax = fig.add_subplot(gs[1, col])
        # Full trajectory in light, up-to-now in solid
        ax.plot(np.arange(total_frames), theta_traj[:, col],
                color=c, linewidth=0.7, alpha=0.25, label=None)
        ax.plot(np.arange(t + 1), theta_traj[:t + 1, col],
                color=c, linewidth=1.5, label=f'θ_t = {theta_traj[t, col]:.3f}')
        ax.axhline(true_v, color='red', linestyle='--', linewidth=1.0,
                    label=f'true = {true_v}')
        ax.axvline(t, color='black', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.set_xlim(0, total_frames - 1)
        ax.set_ylim(*ylim)
        ax.set_xlabel('frame')
        ax.set_title(name, fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        # Rel err annotation
        rel = abs(theta_traj[t, col] - true_v) / max(abs(true_v), 1e-6) * 100
        ax.text(0.98, 0.05, f'rel_err = {rel:.2f}%', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    fig.savefig(str(out_path), dpi=100, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Geometry source
    ap.add_argument("--mesh", default=str(REPO_ROOT / "resources" / "cube" / "cube_tet.msh"),
                    help="tet .msh file (default: resources/cube/cube_tet.msh)")
    ap.add_argument("--synth", action="store_true",
                    help="bypass --mesh and use synthetic cube lattice")
    ap.add_argument("--N_per_dim", type=int, default=12,
                    help="synth cube resolution per axis (only with --synth)")
    ap.add_argument("--extent", type=float, nargs=2, default=[0.3, 0.7],
                    help="synth cube extent along each axis (only with --synth)")
    # Trajectory
    ap.add_argument("--frames", type=int, default=400,
                    help="trajectory length (default 400 = slower motion)")
    ap.add_argument("--stride", type=int, default=8,
                    help="render every Nth frame (default 8)")
    ap.add_argument("--theta_true", type=float, nargs=3, default=[3.0, 20.0, 0.5])
    ap.add_argument("--theta_init", type=float, nargs=3, default=[5.0, 10.0, 0.5])
    # Knife
    ap.add_argument("--knife_radius", type=float, default=0.025,
                    help="knife sphere radius (default 0.025 — small surface tool)")
    ap.add_argument("--surface_axis", choices=["x", "y", "z"], default="y",
                    help="which axis is 'up' (knife on max face of this axis). default y")
    # Output
    ap.add_argument("--side-by-side", action="store_true",
                    help="show α_obs (true θ) AND α_sim (init θ) side by side")
    ap.add_argument("--with-convergence", action="store_true",
                    help="run live calibration + plot θ convergence curves below the 3D viz")
    ap.add_argument("--with-diff", action="store_true",
                    help="(only with --with-convergence) add a 3rd panel showing |α_obs - α_sim|")
    ap.add_argument("--surface", action="store_true",
                    help="render reconstructed surface mesh with α-culled triangles "
                         "(real 3D cavity hole) instead of particle points")
    ap.add_argument("--surface-alpha-threshold", type=float, default=0.3,
                    help="surface mode: cull face if min(α on 3 verts) < this (default 0.3)")
    ap.add_argument("--outdir", default="tools/viz_cube_frames",
                    help="output directory for PNG frames")
    ap.add_argument("--gif", action="store_true", help="also produce rollout.gif")
    ap.add_argument("--window", type=int, nargs=2, default=[1100, 800])
    args = ap.parse_args()

    import pyvista as pv
    pv.OFF_SCREEN = True
    pv.global_theme.background = "white"

    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[viz] outdir: {outdir}")

    # 1. Setup geometry — default to real mesh; fallback to synth lattice if requested
    if args.synth:
        sys_dict = setup_cube_system(N_per_dim=args.N_per_dim, extent=tuple(args.extent))
        bbox_lo = np.array([args.extent[0]] * 3)
        bbox_hi = np.array([args.extent[1]] * 3)
        print(f"[viz] synth cube {args.N_per_dim}³: N={sys_dict['N']} E={sys_dict['E']}")
    else:
        sys_dict = load_msh_system(args.mesh)
        bbox_lo, bbox_hi = sys_dict["bbox"]
    pos = sys_dict["pos"]
    edges_a = sys_dict["edge_a"]
    edges_b = sys_dict["edge_b"]
    surface_faces = sys_dict.get("surface_faces")  # may be None for synth lattice
    if args.surface and surface_faces is None:
        # Synth lattice: no surface_faces. We could derive from cube faces, but for
        # now warn and fall back.
        print("[viz] WARNING: --surface requested but no surface_faces in sys_dict "
              "(synth lattice). Falling back to points mode.")
        args.surface = False
    if args.surface:
        print(f"[viz] surface mode: F={len(surface_faces)} triangles, "
              f"cull threshold α<{args.surface_alpha_threshold}")

    # 2. Knife trajectory — knife center ON THE TOP SURFACE so half-sphere
    #    is exposed (above) and half is embedded (below). Sweep along x.
    axis_map = {"x": 0, "y": 1, "z": 2}
    up = axis_map[args.surface_axis]
    surface_pos = float(bbox_hi[up])
    # Pick the two NON-up axes; sweep along the longer one, fix the other to center
    other_axes = [i for i in range(3) if i != up]
    spans = [(bbox_hi[i] - bbox_lo[i], i) for i in other_axes]
    spans.sort(reverse=True)
    sweep_axis = spans[0][1]
    other_axis = spans[1][1]
    span = float(bbox_hi[sweep_axis] - bbox_lo[sweep_axis])
    x_start = float(bbox_lo[sweep_axis] + 0.15 * span)
    x_end = float(bbox_hi[sweep_axis] - 0.15 * span)
    other_center = float((bbox_hi[other_axis] + bbox_lo[other_axis]) / 2.0)

    sweep_vals = np.linspace(x_start, x_end, args.frames, dtype=np.float32)
    traj = np.zeros((args.frames, 3), dtype=np.float32)
    traj[:, sweep_axis] = sweep_vals
    traj[:, up] = surface_pos
    traj[:, other_axis] = other_center
    speed_per_frame = (x_end - x_start) / (args.frames - 1)
    print(f"[viz] knife: r={args.knife_radius}  surface={args.surface_axis}={surface_pos:.3f} "
          f"(half-exposed)  sweep along axis {sweep_axis} "
          f"[{x_start:.3f} → {x_end:.3f}]  speed {speed_per_frame:.5f}/frame")

    # 3. Collect alpha trajectories
    theta_true = np.array(args.theta_true)
    print(f"[viz] computing α_obs trajectory (θ_true={theta_true})…")
    alpha_obs_traj = collect_alpha_trajectory(sys_dict, theta_true, traj, accumulate=True,
                                                knife_radius=args.knife_radius)
    print(f"[viz]   α_obs shape: {alpha_obs_traj.shape}  "
          f"final cavity: {(alpha_obs_traj[-1] < 0.5).mean():.3f}")

    if args.side_by_side and not args.with_convergence:
        theta_init = np.array(args.theta_init)
        print(f"[viz] computing α_sim trajectory (θ_init={theta_init}, accumulate=True)…")
        alpha_sim_traj = collect_alpha_trajectory(sys_dict, theta_init, traj, accumulate=True,
                                                    knife_radius=args.knife_radius)
        print(f"[viz]   α_sim shape: {alpha_sim_traj.shape}  "
              f"final cavity: {(alpha_sim_traj[-1] < 0.5).mean():.3f}")

    # 3b. Convergence mode: run live calibration, capture per-frame data.
    if args.with_convergence:
        theta_init = np.array(args.theta_init)
        print(f"[viz] running LIVE calibration (θ_init={theta_init}, Mode D)…")
        # Wrap obs as dicts matching what calibration loop expects
        real_frames = [
            dict(t=i, knife_pos=np.asarray(traj[i], dtype=np.float32),
                 alpha_obs=alpha_obs_traj[i])
            for i in range(args.frames)
        ]
        alpha_sim_traj, theta_traj, L_traj = run_calibration_with_capture(
            sys_dict, real_frames, theta_init,
            knife_radius=args.knife_radius, verbose=True,
        )
        rel_final = np.abs(theta_traj[-1] - theta_true) / np.maximum(np.abs(theta_true), 1e-6) * 100
        print(f"[viz]   final θ = {theta_traj[-1].round(3)}  rel_err = {rel_final.round(2)} %")
        print(f"[viz]   final α_sim cavity = {(alpha_sim_traj[-1] < 0.5).mean():.3f}")

    # 4. Determine which frames to render
    frame_indices = list(range(0, args.frames, args.stride))
    if frame_indices[-1] != args.frames - 1:
        frame_indices.append(args.frames - 1)
    print(f"[viz] rendering {len(frame_indices)} frames (stride={args.stride})")

    # 5. Render loop
    w, h = args.window
    if args.side_by_side and args.window == [1100, 800]:
        w = 1800  # widen for two panels

    written = []
    for f_idx, t in enumerate(frame_indices):
        if args.with_convergence:
            path = outdir / f"frame_{f_idx:04d}.png"
            # Bigger window if --with-diff (3 panels need more width)
            comp_w = args.window[0] if args.window[0] >= 1600 else (2400 if args.with_diff else 1800)
            comp_h = args.window[1] if args.window[1] >= 1000 else 1100
            render_composite_frame(
                t=t, frame_idx=f_idx, total_frames=args.frames,
                pos=pos, edges_a=edges_a, edges_b=edges_b,
                alpha_obs_t=alpha_obs_traj[t], alpha_sim_t=alpha_sim_traj[t],
                knife_pos_t=traj[t], knife_radius=args.knife_radius,
                bbox_lo=bbox_lo, bbox_hi=bbox_hi,
                theta_traj=theta_traj, theta_true=theta_true, L_traj=L_traj,
                out_path=path, window=(comp_w, comp_h),
                show_diff=args.with_diff,
                surface_faces=surface_faces,
                mesh_mode=('surface' if args.surface else 'points'),
                alpha_threshold=args.surface_alpha_threshold,
            )
            written.append(path)
            if f_idx % 5 == 0 or f_idx == len(frame_indices) - 1:
                print(f"[viz]   wrote {path.name} (frame index t={t})")
            continue
        def _draw(alpha_v, ttl):
            if args.surface and surface_faces is not None:
                render_frame_surface(plotter, pos, surface_faces, alpha_v, traj[t],
                                     knife_radius=args.knife_radius,
                                     bbox_lo=bbox_lo, bbox_hi=bbox_hi,
                                     title=ttl,
                                     alpha_threshold=args.surface_alpha_threshold)
            else:
                render_frame(plotter, pos, edges_a, edges_b, alpha_v, traj[t],
                             knife_radius=args.knife_radius,
                             bbox_lo=bbox_lo, bbox_hi=bbox_hi, title=ttl)

        if args.side_by_side:
            plotter = pv.Plotter(off_screen=True, window_size=[w, h], shape=(1, 2))
            cavity_obs = (alpha_obs_traj[t] < 0.5).mean()
            cavity_sim = (alpha_sim_traj[t] < 0.5).mean()
            plotter.subplot(0, 0)
            _draw(alpha_obs_traj[t],
                  f"α_obs (θ_true)  frame {t}/{args.frames-1}  cavity={cavity_obs:.3f}")
            plotter.view_isometric()
            plotter.subplot(0, 1)
            _draw(alpha_sim_traj[t],
                  f"α_sim (θ_init)  frame {t}/{args.frames-1}  cavity={cavity_sim:.3f}")
            plotter.view_isometric()
            plotter.link_views()
        else:
            plotter = pv.Plotter(off_screen=True, window_size=[w, h])
            cavity = (alpha_obs_traj[t] < 0.5).mean()
            _draw(alpha_obs_traj[t],
                  f"frame {t}/{args.frames-1}   "
                  f"knife_x={traj[t,0]:.3f}   cavity={cavity:.3f}")
            plotter.view_isometric()

        path = outdir / f"frame_{f_idx:04d}.png"
        plotter.screenshot(str(path))
        plotter.close()
        written.append(path)
        if f_idx % 5 == 0 or f_idx == len(frame_indices) - 1:
            print(f"[viz]   wrote {path.name} (frame index t={t})")

    print(f"[viz] done. {len(written)} PNGs in {outdir}")

    # 6. Optional GIF
    if args.gif:
        try:
            import imageio.v2 as imageio
        except ImportError:
            try:
                import imageio
            except ImportError:
                print("[viz] imageio not available — skipping GIF.")
                return
        gif_path = outdir / "rollout.gif"
        print(f"[viz] assembling GIF → {gif_path}")
        with imageio.get_writer(str(gif_path), mode="I", duration=0.1, loop=0) as writer:
            for path in written:
                writer.append_data(imageio.imread(str(path)))
        print(f"[viz] GIF size: {gif_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
