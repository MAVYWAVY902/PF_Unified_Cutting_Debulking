import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path('/home/yunxin/pf-shape-matching')
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'pf_shape_matching_warp'))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster

cfg = MESH_CONFIGS[0]; ta = cfg['top_axis']; kr = cfg['knife_radius_abs']; dev = 'cuda:0'
s, rs, mi = setup_solver_with_rest(str(REPO / cfg['path']), rings=2, fix_threshold_fraction=0.20)
N = s.N; s.damping = 0.95; s.num_iterations = 2
p0 = mi['vertices'].copy(); bb0, bb1 = p0.min(0), p0.max(0)
lat = [a for a in range(3) if a != ta]
sw = torch.tensor(np.full(N, 0.5, np.float32), device=dev)
tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=dev)
print(f'knife_radius={kr}', flush=True)
for tau_test in [0.35, 0.20]:
    twt = torch.tensor(np.full(N, tau_test, np.float32), device=dev)
    traj = make_raster(bb0, bb1, ta, lat[0], lat[1], 0.55, 8, 40)
    s.reset_to_rest(p0); st = rs
    with torch.no_grad():
        for kp in traj:
            s.gravity = wp.vec3(0., 0., 0.)
            aw, pw, _ = WarpFullForwardOp.apply(tht, s, st, np.asarray(kp, np.float32), kr, sw, twt)
            st = solver_save_state(s)
    removed = int((aw.cpu().numpy() < 0.5).sum())
    print(f'  TAU={tau_test}: removed={removed} / {N}  ({100*removed/N:.1f}%)', flush=True)
