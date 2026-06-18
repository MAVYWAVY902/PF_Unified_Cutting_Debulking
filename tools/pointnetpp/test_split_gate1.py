"""Gate-1: does warp between-frame cluster split make a cut roof FALL (not float),
WITHOUT imploding the cube (unlike rings=1)? Thin mid-band removed + gravity, 150 steps.
Compare NO-split (current warp, ~floats) vs SPLIT. Report roof drop + base integrity.
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
from tools.pointnetpp.warp_cluster_split import split_clusters_inplace

cfg = MESH_CONFIGS[0]; device = "cuda:0"; top_axis = cfg["top_axis"]
T_EL = 2; BAND_C = 0.55; STEPS = 150; SPLIT_EVERY = 5


def run(do_split):
    solver, rest_state, mi = setup_solver_with_rest(str(REPO / cfg["path"]), rings=2, fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos = mi["vertices"].copy()
    bb0, bb1 = pos.min(0), pos.max(0); sp = bb1 - bb0
    fr = (pos[:, top_axis] - bb0[top_axis]) / sp[top_axis]
    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    mean_el = float(np.linalg.norm(pos[ea] - pos[eb], axis=1).mean())
    Tf = T_EL * mean_el / sp[top_axis]
    removed = (fr >= BAND_C - Tf / 2) & (fr < BAND_C + Tf / 2)
    roof = (fr >= BAND_C + Tf / 2) & (~solver.p["fixed"].numpy().astype(bool))
    base = (fr < BAND_C - Tf / 2)
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos[:, 2].min() - 5.0)
    far = pos.max(0) + 100.0
    sw = torch.tensor(np.full(N, 0.5, np.float32), device=device)
    twt = torch.tensor(np.full(N, 0.30, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device)
    solver.reset_to_rest(pos)
    # bake band removal INTO the state dict (apply() reloads alpha/damage from state,
    # so writing solver.p directly gets wiped — must modify the passed state)
    st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rest_state.items()}
    st["alpha"][removed] = 0.0; st["particle_damage"][removed] = 10.0   # bake band into INITIAL state
    roof_top0 = pos[roof, top_axis].mean(); base_c0 = pos[base, top_axis].mean()
    Cs = []
    with torch.no_grad():
        for step in range(STEPS):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, 0.0, sw, twt)
            if do_split and step % SPLIT_EVERY == 0:
                info = split_clusters_inplace(solver, drop_small=False)
                Cs.append(info["C_after"])
            st = solver_save_state(solver)           # save AFTER split → consistent cluster_quat size
            st["alpha"][removed] = 0.0               # re-bake band removal for NEXT apply
            st["particle_damage"][removed] = 10.0
    pf = pw.cpu().numpy()
    roof_drop = (roof_top0 - pf[roof, top_axis].mean()) * 1000
    base_drop = (base_c0 - pf[base, top_axis].mean()) * 1000
    base_maxdisp = np.linalg.norm(pf[base] - pos[base], axis=1).max() * 1000
    nan = bool(np.isnan(pf).any())
    return roof_drop, base_drop, base_maxdisp, nan, (Cs[-1] if Cs else solver.num_clusters)


print(f"=== Gate-1: thin band (~{T_EL}edge) removed + gravity, {STEPS} steps ===", flush=True)
print(f"  {'mode':10} | {'roof drop':>10} | {'base drop':>10} | {'base maxdisp':>12} | {'#clusters':>9} | NaN")
for do_split, name in [(False, "NO-split"), (True, "SPLIT")]:
    rd, bd, bm, nan, C = run(do_split)
    print(f"  {name:10} | {rd:8.1f}mm | {bd:8.2f}mm | {bm:10.1f}mm | {C:>9} | {nan}", flush=True)
print("\n  want: SPLIT roof drop >> NO-split (顶盖掉), base drop≈0 + base maxdisp small (cube不塌), no NaN")
