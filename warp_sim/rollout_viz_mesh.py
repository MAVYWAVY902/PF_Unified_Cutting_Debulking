"""Mesh-based rollout visualization matching the C++ OpenGL viewer style.

Renders tumor/nerve as triangle surface mesh (not particles). Faces whose
vertices are "removed" (alpha < threshold) are culled, producing natural
cavity growth as the knife debulks — identical mechanism to cpp_sim/render/viewer.h
_upload_surface_mesh().

Outputs:
  1. Per-frame PNG images in rollout_frames_mesh/
  2. MP4 video via ffmpeg
  3. trajectory.npz (knife_pos, rewards, coverage)

Usage (run AFTER tmux training finishes to avoid GPU contention):
    python warp_sim/rollout_viz_mesh.py \\
        --model checkpoints/ppo_debulk_final.zip \\
        --config warp_sim/config/debulk_ppo.yaml \\
        --frames 256 \\
        --outdir rollout_frames_mesh
"""
import sys
import os
import argparse
from collections import defaultdict
import numpy as np
import yaml

# IMPORTANT: offscreen BEFORE any pyvista/vtk import that may create a window
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("DISPLAY", "")  # force truly offscreen

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pf_shape_matching_warp"))

import warp as wp
wp.init()

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    pv.global_theme.window_size = [1280, 720]
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("[WARN] pyvista not available; falling back to matplotlib Poly3DCollection (slower, less pretty).")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from warp_sim.envs.debulk_env import DebulkEnv


# ============================================================
# Surface-face extraction (mirrors cpp_sim/core/tet_mesh.h surface_faces)
# ============================================================

def extract_surface_faces(tets: np.ndarray) -> np.ndarray:
    """Return (F, 3) int array of triangle faces lying on the tet-mesh boundary.

    A face is a boundary face iff it belongs to exactly one tet. Same logic
    as cpp_sim/core/tet_mesh.h and our `_build_initial_surface_voxels` method.
    """
    face_count = defaultdict(int)
    face_first = {}  # key → unsorted vertex tuple (preserves orientation)
    for tet in tets:
        a, b, c, d = int(tet[0]), int(tet[1]), int(tet[2]), int(tet[3])
        # 4 tet faces, outward winding by right-hand rule of the opposite vertex
        candidate_faces = [
            (a, c, b),   # opposite d
            (a, b, d),   # opposite c
            (a, d, c),   # opposite b
            (b, c, d),   # opposite a
        ]
        for face in candidate_faces:
            key = tuple(sorted(face))
            face_count[key] += 1
            if key not in face_first:
                face_first[key] = face
    boundary = [face_first[k] for k, cnt in face_count.items() if cnt == 1]
    return np.asarray(boundary, dtype=np.int32)


def classify_face_tissue(faces: np.ndarray,
                         nerve_mask: np.ndarray,
                         fixed_mask: np.ndarray) -> np.ndarray:
    """Return per-face tissue code: 0=tumor, 1=nerve, 2=fixed/base.
    Majority vote across the 3 vertices of each face.
    """
    codes = np.zeros(len(faces), dtype=np.int32)
    for i, f in enumerate(faces):
        is_nerve = nerve_mask[f].astype(bool)
        is_fixed = fixed_mask[f].astype(bool)
        if is_nerve.sum() >= 2:
            codes[i] = 1
        elif is_fixed.sum() >= 2:
            codes[i] = 2
        else:
            codes[i] = 0
    return codes


# ============================================================
# Face-alive filtering (cavity growth)
# ============================================================

def filter_alive_faces(faces: np.ndarray,
                       alpha: np.ndarray,
                       threshold: float = 0.5) -> np.ndarray:
    """Keep only faces whose 3 vertices are all alive (alpha >= threshold).

    Matches cpp_sim/render/viewer.h:560-567 — "skip triangles whose any
    vertex is removed" producing the visible cavity as debulking proceeds.
    """
    alive = alpha >= threshold
    keep = alive[faces[:, 0]] & alive[faces[:, 1]] & alive[faces[:, 2]]
    return faces[keep]


# ============================================================
# PyVista frame renderer (primary, C++-viewer look)
# ============================================================

# Colors match cpp_sim/render/viewer.h:635 (tumor default 0.85, 0.35, 0.3)
COLOR_TUMOR = (0.88, 0.45, 0.38)
COLOR_NERVE = (0.95, 0.15, 0.15)
COLOR_BASE = (0.55, 0.52, 0.50)
COLOR_KNIFE = (0.25, 0.85, 0.35)
COLOR_TRAJ = (0.15, 0.95, 0.45)
COLOR_BG = (0.12, 0.12, 0.15)
LIGHT_DIR = (0.3, 1.0, 0.5)  # matches C++ viewer uLightDir


def render_frame_pyvista(positions: np.ndarray,
                         faces_tumor: np.ndarray,
                         faces_nerve: np.ndarray,
                         faces_base: np.ndarray,
                         knife_pos: np.ndarray,
                         knife_radius: float,
                         knife_traj: np.ndarray,
                         bbox: tuple,
                         step: int,
                         total_steps: int,
                         reward_cum: float,
                         coverage_frac: float,
                         nerve_damage_frac: float,
                         out_path: str,
                         orbit_angle: float = 0.0):
    """Render one frame using pyvista offscreen. Saves PNG to out_path."""
    pl = pv.Plotter(off_screen=True, window_size=(1280, 720))
    pl.background_color = COLOR_BG
    pl.enable_anti_aliasing()

    # Helper: pv.PolyData from (verts, faces)
    def _make_poly(verts, faces_idx):
        if len(faces_idx) == 0:
            return None
        n_tri = len(faces_idx)
        # pv expects faces array as [3, i0, i1, i2, 3, i0', i1', i2', ...]
        faces_flat = np.hstack(
            [np.full((n_tri, 1), 3, dtype=np.int64), faces_idx.astype(np.int64)]
        ).ravel()
        return pv.PolyData(verts, faces_flat)

    # Add tumor, nerve, base meshes
    tumor = _make_poly(positions, faces_tumor)
    nerve = _make_poly(positions, faces_nerve)
    base = _make_poly(positions, faces_base)

    if tumor is not None:
        pl.add_mesh(tumor, color=COLOR_TUMOR, smooth_shading=True,
                    specular=0.35, specular_power=15, ambient=0.25)
    if nerve is not None:
        pl.add_mesh(nerve, color=COLOR_NERVE, smooth_shading=True,
                    specular=0.4, specular_power=20, ambient=0.3)
    if base is not None:
        pl.add_mesh(base, color=COLOR_BASE, smooth_shading=True,
                    specular=0.1, ambient=0.2)

    # Knife as sphere
    knife_sphere = pv.Sphere(radius=knife_radius, center=knife_pos,
                             theta_resolution=28, phi_resolution=24)
    pl.add_mesh(knife_sphere, color=COLOR_KNIFE, opacity=0.45,
                smooth_shading=True, specular=0.6, specular_power=30)

    # Knife trajectory polyline
    if len(knife_traj) >= 2:
        traj = pv.lines_from_points(knife_traj)
        pl.add_mesh(traj, color=COLOR_TRAJ, line_width=3.0)

    # Lighting matching C++ viewer (directional)
    pl.remove_all_lights()
    light = pv.Light(position=(LIGHT_DIR[0] * 10, LIGHT_DIR[1] * 10, LIGHT_DIR[2] * 10),
                     focal_point=(0.5, 0.5, 0.5), color='white',
                     intensity=0.9, light_type='scene light')
    pl.add_light(light)
    ambient_light = pv.Light(color='white', intensity=0.35,
                             light_type='headlight')
    pl.add_light(ambient_light)

    # Orbiting camera
    cx, cy, cz = 0.5 * (bbox[0] + bbox[1]), 0.5 * (bbox[2] + bbox[3]), 0.5 * (bbox[4] + bbox[5])
    r = 2.2 * max(bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4])
    cam_x = cx + r * np.cos(orbit_angle) * 0.9
    cam_y = cy + r * 0.4
    cam_z = cz + r * np.sin(orbit_angle) * 0.9
    pl.camera_position = [(cam_x, cam_y, cam_z), (cx, cy, cz), (0.0, 1.0, 0.0)]

    # Overlay text — step, reward, coverage, nerve damage
    title = f"Step {step:3d} / {total_steps}"
    stats = (
        f"Cumulative reward: {reward_cum:+8.2f}\n"
        f"Tumor coverage:    {coverage_frac*100:5.1f} %\n"
        f"Nerve damage:      {nerve_damage_frac*100:5.1f} %\n"
        f"Knife: ({knife_pos[0]:.3f}, {knife_pos[1]:.3f}, {knife_pos[2]:.3f})"
    )
    pl.add_text(title, position='upper_left', font_size=14, color='white')
    pl.add_text(stats, position='lower_left', font_size=11, color='white',
                font='courier')

    pl.screenshot(out_path, transparent_background=False)
    pl.close()


# ============================================================
# Matplotlib fallback (slower, only if pyvista missing)
# ============================================================

def render_frame_matplotlib(positions, faces_tumor, faces_nerve, faces_base,
                            knife_pos, knife_radius, knife_traj, bbox,
                            step, total_steps, reward_cum, coverage_frac,
                            nerve_damage_frac, out_path, orbit_angle=0.0):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12.8, 7.2), facecolor='#1e1e24')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1e1e24')

    def _add_tris(faces_idx, color, alpha=1.0):
        if len(faces_idx) == 0:
            return
        tris = positions[faces_idx]  # (F, 3, 3)
        coll = Poly3DCollection(tris, facecolors=color, edgecolors='none',
                                alpha=alpha, linewidth=0.0)
        ax.add_collection3d(coll)

    _add_tris(faces_tumor, COLOR_TUMOR, 0.95)
    _add_tris(faces_nerve, COLOR_NERVE, 1.0)
    _add_tris(faces_base, COLOR_BASE, 0.8)

    # Knife sphere
    u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:8j]
    sx = knife_pos[0] + knife_radius * np.cos(u) * np.sin(v)
    sy = knife_pos[1] + knife_radius * np.cos(v)
    sz = knife_pos[2] + knife_radius * np.sin(u) * np.sin(v)
    ax.plot_surface(sx, sy, sz, color=COLOR_KNIFE, alpha=0.4, linewidth=0)

    # Trajectory
    if len(knife_traj) >= 2:
        ax.plot(knife_traj[:, 0], knife_traj[:, 1], knife_traj[:, 2],
                color=COLOR_TRAJ, linewidth=2.0)

    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_zlim(bbox[4], bbox[5])
    ax.set_box_aspect([bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]])
    ax.view_init(elev=20, azim=np.degrees(orbit_angle))
    ax.set_axis_off()

    fig.text(0.02, 0.95, f"Step {step}/{total_steps}",
             color='white', fontsize=14, family='monospace')
    fig.text(0.02, 0.06,
             f"Cum reward: {reward_cum:+.2f}\n"
             f"Coverage  : {coverage_frac*100:.1f}%\n"
             f"Nerve dmg : {nerve_damage_frac*100:.1f}%\n"
             f"Knife     : ({knife_pos[0]:.3f},{knife_pos[1]:.3f},{knife_pos[2]:.3f})",
             color='white', fontsize=10, family='monospace')
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=100)
    plt.close(fig)


# ============================================================
# Rollout driver
# ============================================================

def rollout_and_render(model_path: str,
                       cfg: dict,
                       num_steps: int,
                       out_dir: str,
                       alpha_threshold: float = 0.5,
                       deterministic: bool = True,
                       record_only: bool = False,
                       vecnormalize_path: str = None):
    os.makedirs(out_dir, exist_ok=True)

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    print(f"[setup] Building env + loading model: {model_path}")

    # Always wrap in DummyVecEnv so the rest of the code has a single interface.
    # If a vecnormalize.pkl is supplied, load it (training=False so stats don't drift).
    def _make_env():
        return DebulkEnv(cfg)
    vec_env = DummyVecEnv([_make_env])

    if vecnormalize_path and os.path.isfile(vecnormalize_path):
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False         # freeze running stats during rollout
        vec_env.norm_reward = False      # expose RAW rewards for metrics
        print(f"[setup] Loaded VecNormalize: {vecnormalize_path} "
              f"(training=False, norm_reward=False for rollout metrics)")
    elif vecnormalize_path:
        print(f"[setup] WARN: vecnormalize_path '{vecnormalize_path}' not found — "
              f"running without normalization. Obs distribution MISMATCH likely.")

    # Inner DebulkEnv (for direct access to solver / nerve_mask / rest_pos etc.)
    inner_env = vec_env.venv.envs[0] if isinstance(vec_env, VecNormalize) else vec_env.envs[0]
    env = vec_env  # alias for the rest of the function
    model = PPO.load(model_path, env=vec_env)

    # --- One-time: extract mesh topology for rendering ---
    tets = inner_env._tets
    N = inner_env.solver.N
    nerve_mask_np = inner_env._nerve_mask_np
    fixed_mask_np = inner_env._fixed_np

    print(f"[setup] Extracting surface faces from {len(tets)} tets ...")
    surface_faces = extract_surface_faces(tets)
    tissue_code = classify_face_tissue(surface_faces, nerve_mask_np, fixed_mask_np)
    print(f"[setup] Surface faces: {len(surface_faces)} "
          f"(tumor={np.sum(tissue_code==0)}, nerve={np.sum(tissue_code==1)}, "
          f"base={np.sum(tissue_code==2)})")

    faces_tumor_static = surface_faces[tissue_code == 0]
    faces_nerve_static = surface_faces[tissue_code == 1]
    faces_base_static = surface_faces[tissue_code == 2]

    # --- Rollout (VecEnv interface: 4-tuple, shape (1, ...)) ---
    obs = env.reset()
    wp.synchronize()

    knife_positions, rewards, all_pos, all_alpha = [], [], [], []
    total_tumor = int((fixed_mask_np == 0).sum() - (nerve_mask_np == 1).sum())
    total_nerve = int((nerve_mask_np == 1).sum())

    print(f"[rollout] Running policy for {num_steps} steps ...")
    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r_arr, done_arr, info = env.step(action)    # VecEnv 4-tuple
        wp.synchronize()

        r = float(r_arr[0])
        done = bool(done_arr[0])

        # Physics state read from the INNER DebulkEnv (VecNormalize wraps obs/reward only)
        knife_positions.append(inner_env.solver.knife_pos.numpy().reshape(3).copy())
        rewards.append(r)
        all_pos.append(inner_env.solver.p["pos"].numpy().reshape(-1, 3).copy())
        all_alpha.append(inner_env.solver.p["alpha"].numpy().copy())

        if step % 20 == 0 or step == num_steps - 1:
            alive_tumor = int(((all_alpha[-1] >= alpha_threshold) &
                               (fixed_mask_np == 0) & (nerve_mask_np == 0)).sum())
            cov = 1.0 - alive_tumor / max(1, total_tumor)
            nerve_dmg = 1.0 - (
                int(((all_alpha[-1] >= alpha_threshold) & (nerve_mask_np == 1)).sum())
                / max(1, total_nerve)
            )
            print(f"  step {step:3d}: reward={r:+7.2f}  cov={cov*100:5.1f}%  "
                  f"nerve_dmg={nerve_dmg*100:5.1f}%")

        if done:
            # VecEnv auto-resets on done — stop collecting here so positions
            # don't jump to next episode.
            print(f"[rollout] Episode done at step {step} (VecEnv auto-reset)")
            break

    knife_positions = np.asarray(knife_positions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    T = len(all_pos)

    # Bounding box for camera
    rest = inner_env.rest_pos.numpy().reshape(-1, 3)
    pad = 0.05
    bbox = (rest[:, 0].min() - pad, rest[:, 0].max() + pad,
            rest[:, 1].min() - pad, rest[:, 1].max() + pad,
            rest[:, 2].min() - pad, rest[:, 2].max() + pad)

    # Per-frame coverage / nerve damage
    coverage_hist = np.zeros(T, dtype=np.float32)
    nerve_damage_hist = np.zeros(T, dtype=np.float32)
    for t, a in enumerate(all_alpha):
        alive = a >= alpha_threshold
        alive_tumor = int((alive & (fixed_mask_np == 0) & (nerve_mask_np == 0)).sum())
        coverage_hist[t] = 1.0 - alive_tumor / max(1, total_tumor)
        alive_nerve = int((alive & (nerve_mask_np == 1)).sum())
        nerve_damage_hist[t] = 1.0 - alive_nerve / max(1, total_nerve)

    knife_radius = cfg.get("knife_radius", 0.1)

    # --- Save FULL trajectory (positions + alpha per frame + topology) ---
    # Large but enables offline interactive replay without re-running env.
    full_pos = np.stack(all_pos, axis=0).astype(np.float32)        # (T, N, 3)
    full_alpha = np.stack(all_alpha, axis=0).astype(np.float32)    # (T, N)
    np.savez_compressed(
        os.path.join(out_dir, "trajectory.npz"),
        # dynamic per-frame
        positions=full_pos,
        alpha=full_alpha,
        knife_positions=knife_positions,
        rewards=rewards,
        coverage=coverage_hist,
        nerve_damage=nerve_damage_hist,
        # static topology & masks (needed for interactive replay)
        surface_faces=surface_faces.astype(np.int32),
        tissue_code=tissue_code.astype(np.int32),
        nerve_mask=nerve_mask_np.astype(np.int32),
        fixed_mask=fixed_mask_np.astype(np.int32),
        rest_pos=rest.astype(np.float32),
        bbox=np.asarray(bbox, dtype=np.float32),
        knife_radius=np.float32(knife_radius),
        alpha_threshold=np.float32(alpha_threshold),
    )
    npz_size_mb = os.path.getsize(os.path.join(out_dir, "trajectory.npz")) / 1e6
    print(f"[save] trajectory.npz ({npz_size_mb:.1f} MB) — use with rollout_interactive.py")

    if record_only:
        print(f"[record-only] Skipping offscreen PNG frames. "
              f"Run: python warp_sim/rollout_interactive.py --replay {out_dir}/trajectory.npz")
        _print_summary(rewards, coverage_hist, nerve_damage_hist, knife_positions, T)
        return

    # --- Render frames (offscreen PNG for MP4 video) ---
    render = render_frame_pyvista if HAS_PYVISTA else render_frame_matplotlib
    print(f"[render] Using {'pyvista' if HAS_PYVISTA else 'matplotlib'} for {T} frames")

    for t in range(T):
        alpha = all_alpha[t]
        ft = filter_alive_faces(faces_tumor_static, alpha, alpha_threshold)
        fn = filter_alive_faces(faces_nerve_static, alpha, alpha_threshold)
        fb = filter_alive_faces(faces_base_static, alpha, alpha_threshold)
        orbit = 2 * np.pi * (t / max(1, T)) * 0.5  # half turn per episode

        render(
            positions=all_pos[t],
            faces_tumor=ft, faces_nerve=fn, faces_base=fb,
            knife_pos=knife_positions[t],
            knife_radius=knife_radius,
            knife_traj=knife_positions[:t + 1],
            bbox=bbox,
            step=t, total_steps=T,
            reward_cum=float(rewards[:t + 1].sum()),
            coverage_frac=float(coverage_hist[t]),
            nerve_damage_frac=float(nerve_damage_hist[t]),
            out_path=os.path.join(out_dir, f"frame_{t:04d}.png"),
            orbit_angle=orbit,
        )
        if t % 20 == 0 or t == T - 1:
            print(f"  rendered frame {t+1}/{T}")

    # ffmpeg → mp4
    video_path = os.path.join(out_dir, "rollout.mp4")
    try:
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-framerate", "20",
            "-i", os.path.join(out_dir, "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", video_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[video] {video_path}")
    except Exception as e:
        print(f"[video] ffmpeg failed ({e}); assemble manually:\n"
              f"        ffmpeg -y -framerate 20 -i {out_dir}/frame_%04d.png "
              f"-c:v libx264 -pix_fmt yuv420p {video_path}")

    _print_summary(rewards, coverage_hist, nerve_damage_hist, knife_positions, T)


def _print_summary(rewards, coverage_hist, nerve_damage_hist, knife_positions, T):
    print("\n==== Rollout Summary ====")
    print(f"  Total steps:        {T}")
    print(f"  Cumulative reward:  {rewards.sum():+.2f}")
    print(f"  Final tumor cov:    {coverage_hist[-1]*100:.1f}%")
    print(f"  Final nerve damage: {nerve_damage_hist[-1]*100:.1f}%")
    print(f"  Knife path length:  {np.linalg.norm(np.diff(knife_positions, axis=0), axis=1).sum():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ppo_debulk_final.zip",
                        help="Path to PPO checkpoint (.zip)")
    parser.add_argument("--config", default="warp_sim/config/debulk_ppo.yaml")
    parser.add_argument("--frames", type=int, default=256,
                        help="Number of env steps to roll out")
    parser.add_argument("--outdir", default="rollout_frames_mesh")
    parser.add_argument("--alpha-threshold", type=float, default=0.5,
                        help="Alpha below this = face culled (cavity effect)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy instead of deterministic")
    parser.add_argument("--record-only", action="store_true",
                        help="Only record trajectory.npz (skip PNG + MP4). "
                             "Use rollout_interactive.py to view afterwards.")
    parser.add_argument("--vecnormalize", default=None,
                        help="Path to vecnormalize .pkl from training. "
                             "If omitted, auto-discovers sibling pkl next to --model.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["episode_length"] = args.frames

    # Auto-discover vecnormalize.pkl next to the model file if not given
    vecnorm_path = args.vecnormalize
    if vecnorm_path is None:
        # e.g., checkpoints/ppo_debulk_150000_steps.zip
        #   →   checkpoints/ppo_debulk_vecnormalize_150000_steps.pkl
        #   OR: ppo_debulk_final.zip → vecnormalize_final.pkl
        candidates = []
        model_dir = os.path.dirname(args.model) or "."
        model_base = os.path.splitext(os.path.basename(args.model))[0]
        # SB3 CheckpointCallback naming: <prefix>_vecnormalize_<steps>.pkl
        if "_steps" in model_base:
            cb_name = model_base.replace("ppo_debulk_", "ppo_debulk_vecnormalize_") + ".pkl"
            candidates.append(os.path.join(model_dir, cb_name))
        # Final-save naming convention (train_ppo.py)
        if "final" in model_base:
            candidates.append("vecnormalize_final.pkl")
        # Legacy guess
        candidates.append(os.path.join(model_dir, "vecnormalize.pkl"))
        for c in candidates:
            if os.path.isfile(c):
                vecnorm_path = c
                break

    rollout_and_render(
        model_path=args.model,
        cfg=cfg,
        num_steps=args.frames,
        out_dir=args.outdir,
        alpha_threshold=args.alpha_threshold,
        deterministic=not args.stochastic,
        record_only=args.record_only,
        vecnormalize_path=vecnorm_path,
    )


if __name__ == "__main__":
    main()
