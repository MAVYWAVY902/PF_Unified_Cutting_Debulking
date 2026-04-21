"""Interactive rollout viewer — matches cpp_sim's GLFW viewer mouse behavior.

Loads a trajectory.npz previously saved by rollout_viz_mesh.py (--record-only
is fastest) and opens a PyVista interactive window with:

  Mouse:
    Left-drag   = orbit camera
    Middle-drag = pan
    Right-drag / Scroll = zoom
    'r' (vtk)   = reset camera to fit scene

  Keyboard (added here):
    Space       = play / pause
    Right arrow = step +1 frame
    Left arrow  = step -1 frame
    '0'         = jump to frame 0
    'e'         = jump to end
    '+' / '-'   = faster / slower playback
    'w'         = toggle wireframe on tumor mesh
    'q'         = quit (vtk default)

  Slider: drag to scrub frames.

This viewer does NOT need CUDA / warp / stable_baselines3 — it's a pure
visualization replay. Safe to run while tmux training is active.

Usage:
    python warp_sim/rollout_interactive.py --replay rollout_frames_mesh/trajectory.npz
"""
import argparse
import os
import sys
import time
import numpy as np

import pyvista as pv


# Match cpp_sim/render/viewer.h look
COLOR_TUMOR = (0.88, 0.45, 0.38)
COLOR_NERVE = (0.95, 0.15, 0.15)
COLOR_BASE = (0.55, 0.52, 0.50)
COLOR_KNIFE = (0.25, 0.85, 0.35)
COLOR_TRAJ = (0.15, 0.95, 0.45)
COLOR_BG = (0.12, 0.12, 0.15)


def _faces_to_pv_flat(faces_idx: np.ndarray) -> np.ndarray:
    """pv.PolyData wants faces flat as [3, i0, i1, i2, 3, i0', ...]."""
    if len(faces_idx) == 0:
        # Return one degenerate triangle so PolyData isn't empty (avoids crashes on hide/show)
        return np.array([3, 0, 0, 0], dtype=np.int64)
    n_tri = len(faces_idx)
    return np.hstack(
        [np.full((n_tri, 1), 3, dtype=np.int64), faces_idx.astype(np.int64)]
    ).ravel()


def _alive_faces(faces_static: np.ndarray,
                 alpha: np.ndarray,
                 threshold: float) -> np.ndarray:
    """Keep only faces whose 3 vertices are still alive (cavity effect)."""
    alive = alpha >= threshold
    keep = alive[faces_static[:, 0]] & alive[faces_static[:, 1]] & alive[faces_static[:, 2]]
    return faces_static[keep]


class RolloutPlayer:
    """Holds state + updates meshes in the pyvista plotter."""

    def __init__(self, npz_path: str):
        print(f"[load] {npz_path}")
        data = np.load(npz_path)

        # Per-frame dynamic data
        self.positions = data["positions"]             # (T, N, 3)
        self.alpha = data["alpha"]                     # (T, N)
        self.knife_positions = data["knife_positions"] # (T, 3)
        self.rewards = data["rewards"]                 # (T,)
        self.coverage = data["coverage"]               # (T,)
        self.nerve_damage = data["nerve_damage"]       # (T,)

        # Static topology
        self.surface_faces = data["surface_faces"]     # (F, 3)
        self.tissue_code = data["tissue_code"]         # (F,) 0=tumor,1=nerve,2=base
        self.rest_pos = data["rest_pos"]               # (N, 3)
        self.bbox = data["bbox"]                       # (6,)
        self.knife_radius = float(data["knife_radius"])
        self.alpha_threshold = float(data["alpha_threshold"])

        self.T = len(self.positions)
        self.N = self.positions.shape[1]
        self.faces_tumor = self.surface_faces[self.tissue_code == 0]
        self.faces_nerve = self.surface_faces[self.tissue_code == 1]
        self.faces_base = self.surface_faces[self.tissue_code == 2]

        print(f"[load] T={self.T} frames, N={self.N} particles, "
              f"F_tumor={len(self.faces_tumor)}, F_nerve={len(self.faces_nerve)}, "
              f"F_base={len(self.faces_base)}")

        # Playback state
        self.frame = 0
        self.playing = False
        self.speed = 1.0  # frames per tick (tick ≈ 50ms)
        self._speed_accum = 0.0

        # Plotter + actors
        self.plotter: pv.Plotter = None
        self.tumor_mesh: pv.PolyData = None
        self.nerve_mesh: pv.PolyData = None
        self.base_mesh: pv.PolyData = None
        self.knife_mesh: pv.PolyData = None
        self.traj_mesh: pv.PolyData = None
        self.tumor_wireframe_on = False

    # ---------- Build initial scene ----------
    def build_scene(self) -> pv.Plotter:
        p = pv.Plotter(window_size=(1280, 720),
                       title="P(F) Rollout — interactive")
        p.background_color = COLOR_BG
        p.enable_anti_aliasing()

        pos0 = self.positions[0]
        alpha0 = self.alpha[0]

        ft0 = _alive_faces(self.faces_tumor, alpha0, self.alpha_threshold)
        fn0 = _alive_faces(self.faces_nerve, alpha0, self.alpha_threshold)
        fb0 = _alive_faces(self.faces_base, alpha0, self.alpha_threshold)

        self.tumor_mesh = pv.PolyData(pos0, _faces_to_pv_flat(ft0))
        self.nerve_mesh = pv.PolyData(pos0, _faces_to_pv_flat(fn0))
        self.base_mesh = pv.PolyData(pos0, _faces_to_pv_flat(fb0))

        p.add_mesh(self.tumor_mesh, color=COLOR_TUMOR, smooth_shading=True,
                   specular=0.35, specular_power=15, ambient=0.25,
                   name="tumor")
        p.add_mesh(self.nerve_mesh, color=COLOR_NERVE, smooth_shading=True,
                   specular=0.4, specular_power=20, ambient=0.3,
                   name="nerve")
        p.add_mesh(self.base_mesh, color=COLOR_BASE, smooth_shading=True,
                   specular=0.1, ambient=0.2,
                   name="base")

        # Knife sphere
        self.knife_mesh = pv.Sphere(radius=self.knife_radius,
                                    center=self.knife_positions[0],
                                    theta_resolution=28, phi_resolution=24)
        p.add_mesh(self.knife_mesh, color=COLOR_KNIFE, opacity=0.45,
                   smooth_shading=True, specular=0.6, specular_power=30,
                   name="knife")

        # Trajectory polyline (full path, always shown up to current frame)
        self.traj_mesh = pv.lines_from_points(self.knife_positions[:2])
        p.add_mesh(self.traj_mesh, color=COLOR_TRAJ, line_width=3.0,
                   name="traj")

        # Lighting mimicking cpp_sim viewer
        p.remove_all_lights()
        p.add_light(pv.Light(position=(3, 10, 5), focal_point=(0.5, 0.5, 0.5),
                             color='white', intensity=0.9))
        p.add_light(pv.Light(color='white', intensity=0.35,
                             light_type='headlight'))

        # Fit camera
        cx = 0.5 * (self.bbox[0] + self.bbox[1])
        cy = 0.5 * (self.bbox[2] + self.bbox[3])
        cz = 0.5 * (self.bbox[4] + self.bbox[5])
        r = 2.2 * max(self.bbox[1] - self.bbox[0],
                      self.bbox[3] - self.bbox[2],
                      self.bbox[5] - self.bbox[4])
        p.camera_position = [(cx + r * 0.9, cy + r * 0.4, cz + r * 0.9),
                             (cx, cy, cz), (0.0, 1.0, 0.0)]

        # HUD text actors (mutable)
        self._title_actor = p.add_text(self._title_str(), position='upper_left',
                                       font_size=14, color='white', name='title')
        self._hud_actor = p.add_text(self._hud_str(), position='lower_left',
                                     font_size=11, color='white',
                                     font='courier', name='hud')
        self._help_actor = p.add_text(
            "[Space] play/pause   [←/→] step   [0/e] start/end   "
            "[+/-] speed   [w] wireframe   [q] quit",
            position='lower_right', font_size=10, color=(0.7, 0.7, 0.7),
            name='help',
        )

        # Frame slider
        p.add_slider_widget(
            callback=self._on_slider,
            rng=[0, self.T - 1],
            value=0,
            title="Frame",
            pointa=(0.25, 0.92), pointb=(0.75, 0.92),
            style="modern",
            fmt="%.0f",
        )

        # Key bindings
        p.add_key_event("space", self._toggle_play)
        p.add_key_event("Right", lambda: self._step(+1))
        p.add_key_event("Left", lambda: self._step(-1))
        p.add_key_event("0", lambda: self._seek(0))
        p.add_key_event("e", lambda: self._seek(self.T - 1))
        p.add_key_event("plus", self._faster)
        p.add_key_event("equal", self._faster)  # shift+= is usually + but vtk names it equal
        p.add_key_event("minus", self._slower)
        p.add_key_event("w", self._toggle_wireframe)

        self.plotter = p
        return p

    # ---------- Per-frame update ----------
    def _update_meshes(self):
        t = self.frame
        pos = self.positions[t]
        alpha = self.alpha[t]
        ft = _alive_faces(self.faces_tumor, alpha, self.alpha_threshold)
        fn = _alive_faces(self.faces_nerve, alpha, self.alpha_threshold)
        fb = _alive_faces(self.faces_base, alpha, self.alpha_threshold)

        # Update points in place (much faster than rebuilding PolyData)
        self.tumor_mesh.points = pos
        self.nerve_mesh.points = pos
        self.base_mesh.points = pos

        # Update faces (cavity effect requires rebuilding the connectivity)
        self.tumor_mesh.faces = _faces_to_pv_flat(ft)
        self.nerve_mesh.faces = _faces_to_pv_flat(fn)
        self.base_mesh.faces = _faces_to_pv_flat(fb)

        # Knife sphere — rebuild (cheap, <1ms)
        new_knife = pv.Sphere(radius=self.knife_radius,
                              center=self.knife_positions[t],
                              theta_resolution=28, phi_resolution=24)
        self.knife_mesh.copy_from(new_knife)

        # Trajectory polyline up to current frame
        kt = self.knife_positions[: max(2, t + 1)]
        new_traj = pv.lines_from_points(kt)
        self.traj_mesh.copy_from(new_traj)

        # Update HUD text
        self._title_actor.SetText(2, self._title_str())   # 2 = upper_left region
        self._hud_actor.SetText(0, self._hud_str())       # 0 = lower_left region

    def _title_str(self) -> str:
        return f"Step {self.frame:4d} / {self.T - 1}  |  speed x{self.speed:.1f}"

    def _hud_str(self) -> str:
        t = self.frame
        return (
            f"Reward (step):     {self.rewards[t]:+8.2f}\n"
            f"Reward (cum):      {self.rewards[:t+1].sum():+8.2f}\n"
            f"Tumor coverage:    {self.coverage[t]*100:5.1f} %\n"
            f"Nerve damage:      {self.nerve_damage[t]*100:5.1f} %\n"
            f"Knife:             ({self.knife_positions[t,0]:.3f},"
            f"{self.knife_positions[t,1]:.3f},{self.knife_positions[t,2]:.3f})\n"
            f"Status:            {'▶ PLAY' if self.playing else '⏸ PAUSE'}"
        )

    # ---------- Controls ----------
    def _toggle_play(self):
        self.playing = not self.playing

    def _step(self, d: int):
        self.playing = False
        new_f = max(0, min(self.T - 1, self.frame + d))
        self._seek(new_f)

    def _seek(self, t: int):
        self.frame = int(max(0, min(self.T - 1, t)))
        self._update_meshes()

    def _on_slider(self, value: float):
        self._seek(int(round(value)))
        self.playing = False

    def _faster(self):
        self.speed = min(8.0, self.speed * 1.5)

    def _slower(self):
        self.speed = max(0.1, self.speed / 1.5)

    def _toggle_wireframe(self):
        self.tumor_wireframe_on = not self.tumor_wireframe_on
        actor = self.plotter.renderer.actors.get("tumor")
        if actor is not None:
            prop = actor.GetProperty()
            if self.tumor_wireframe_on:
                prop.SetRepresentationToWireframe()
            else:
                prop.SetRepresentationToSurface()

    # ---------- Main loop ----------
    def run(self):
        p = self.plotter
        p.show(auto_close=False, interactive_update=True)

        try:
            last = time.time()
            while True:
                # Check if window closed
                if not p.iren.initialized:
                    break
                try:
                    if p.render_window is None or bool(p.render_window.GetNeverRendered()) is False:
                        pass
                except Exception:
                    break

                # Auto-advance if playing
                now = time.time()
                dt = now - last
                last = now
                if self.playing:
                    # Advance self.speed frames per second, normalized
                    self._speed_accum += self.speed * dt * 20.0  # 20 fps base
                    while self._speed_accum >= 1.0:
                        self._speed_accum -= 1.0
                        if self.frame < self.T - 1:
                            self.frame += 1
                            self._update_meshes()
                        else:
                            self.playing = False
                            break

                p.update()
                time.sleep(1.0 / 60)
        except KeyboardInterrupt:
            print("\n[viewer] interrupted")
        finally:
            try:
                p.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True,
                    help="Path to trajectory.npz from rollout_viz_mesh.py")
    args = ap.parse_args()

    if not os.path.isfile(args.replay):
        print(f"[error] file not found: {args.replay}")
        sys.exit(1)

    player = RolloutPlayer(args.replay)
    player.build_scene()
    player.run()


if __name__ == "__main__":
    main()
