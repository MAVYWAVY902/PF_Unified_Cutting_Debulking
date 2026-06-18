"""Cheap diagnostic for floating-particle artifact: is it (i) clusters span the cut
(coupling persists) or (ii) decoupled but gravity too weak vs shape-matching stiffness?

Part 1 (geometry, no sim): cluster size + spatial diameter; remove a middle slab of
thickness T (in edge-lengths) -> count BRIDGING clusters (>=1 alive member above slab
AND >=1 below). bridging>0 => coupling (case i). Sweep T to find decouple threshold.

Part 2 (forward): at a T where bridging~0, force slab removed (alpha=0 held) + gravity,
run 150 steps, measure ROOF centroid drop. drops => decoupled+gravity OK (floating in
GIFs = cuts too thin). barely drops => case (ii) shape-matching-to-rest holds it.
"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS


def main():
    cfg = MESH_CONFIGS[0]; device = "cuda:0"
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]), fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos = mesh_info["vertices"].copy()
    top_axis = cfg["top_axis"]
    bb0, bb1 = pos.min(0), pos.max(0); sp = bb1 - bb0
    fr = (pos[:, top_axis] - bb0[top_axis]) / sp[top_axis]
    fixed = solver.p["fixed"].numpy().astype(bool)

    # mean edge length (for thickness units)
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    el = np.linalg.norm(pos[ea] - pos[eb], axis=1); mean_el = float(el.mean())

    # cluster arrays
    starts = solver.c["starts"].numpy(); sizes = solver.c["sizes"].numpy(); pids = solver.c["particle_ids"].numpy()
    csz = sizes[sizes > 0]
    # cluster spatial diameter (max pairwise extent along top_axis, per cluster) — sample
    diam_top = []
    for cid in range(0, solver.num_clusters, 50):  # sample every 50th
        s0 = starts[cid]; n = sizes[cid]
        if n < 2: continue
        m = pids[s0:s0 + n]
        diam_top.append(pos[m, top_axis].max() - pos[m, top_axis].min())
    diam_top = np.array(diam_top)

    print(f"N={N}  mean_edge_len={mean_el:.4f}  cube_span(top)={sp[top_axis]:.4f}")
    print(f"cluster size: min={csz.min()} max={csz.max()} avg={csz.mean():.1f}")
    print(f"cluster TOP-axis diameter: avg={diam_top.mean():.4f} ({diam_top.mean()/mean_el:.1f} edge-len, "
          f"{100*diam_top.mean()/sp[top_axis]:.1f}% of cube)  max={diam_top.max():.4f}", flush=True)

    # ---- Part 1: bridging clusters vs slab thickness ----
    print("\n=== Part 1: BRIDGING clusters vs removed-slab thickness (coupling test) ===")
    band_center = 0.55  # middle (above the bottom-20% anchor)
    print(f"  slab centered at fr={band_center} (top-axis); thickness in edge-lengths")
    print(f"  {'T(edge)':>8} {'T(frac)':>8} {'#removed':>9} {'#roof':>7} {'#base':>7} {'bridging_clusters':>18}")
    for T_el in [1, 2, 3, 4, 6, 8]:
        T_frac = T_el * mean_el / sp[top_axis]
        lo = band_center - T_frac / 2; hi = band_center + T_frac / 2
        removed = (fr >= lo) & (fr < hi)
        roof = fr >= hi; base = fr < lo
        alive = ~removed
        # bridging cluster: has >=1 alive ROOF member AND >=1 alive BASE member
        bridging = 0
        for cid in range(solver.num_clusters):
            s0 = starts[cid]; n = sizes[cid]
            if n < 2: continue
            m = pids[s0:s0 + n]
            ma = m[alive[m]]
            if len(ma) < 2: continue
            if roof[ma].any() and base[ma].any():
                bridging += 1
        print(f"  {T_el:>8} {T_frac:>8.3f} {int(removed.sum()):>9} {int(roof.sum()):>7} "
              f"{int(base.sum()):>7} {bridging:>18}", flush=True)

    # ---- Part 2: forward — decoupled roof under gravity ----
    print("\n=== Part 2: forward sag test (gravity vs shape-matching, decoupled slab) ===")
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos[:, 2].min() - 5.0)
    far = pos.max(0) + 100.0
    sw = torch.tensor(np.full(N, 0.5, np.float32), device=device)
    twt = torch.tensor(np.full(N, 0.30, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device)

    for T_el in [4, 8]:
        T_frac = T_el * mean_el / sp[top_axis]
        lo = band_center - T_frac / 2; hi = band_center + T_frac / 2
        removed = (fr >= lo) & (fr < hi); roof = (fr >= hi) & (~fixed)
        solver.reset_to_rest(pos)
        # force slab removed: high damage + alpha 0, held each step
        dmg = np.zeros(N, np.float32); dmg[removed] = 10.0
        roof_top0 = pos[roof, top_axis].mean()
        st = rest_state
        for step in range(150):
            # hold slab damage high so persistence keeps alpha=0
            d = solver.p["damage"].numpy(); d[removed] = 10.0
            solver.p["damage"].assign(d)
            a = solver.p["alpha"].numpy(); a[removed] = 0.0
            solver.p["alpha"].assign(a)
            _, pw, aw = WarpFullForwardOp.apply(tht, solver, st, far, 0.0, sw, twt)
            st = solver_save_state(solver)
        pf = pw.cpu().numpy()
        roof_drop = roof_top0 - pf[roof, top_axis].mean()
        # free-fall reference over 150 steps (dt unknown; report drop in mm + as % of slab gap)
        gap = T_frac * sp[top_axis]
        print(f"  T={T_el}edge (gap={gap*1000:.1f}mm, removed={int(removed.sum())}): "
              f"roof centroid drop = {roof_drop*1000:.2f} mm  ({100*roof_drop/max(gap,1e-9):.0f}% of gap)  "
              f"-> {'FALLS (decoupled+gravity OK)' if roof_drop > 0.3*gap else 'HELD (case ii: stiffness>>gravity OR still coupled)'}", flush=True)


if __name__ == "__main__":
    main()
