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
from tools.pointnetpp.warp_cluster_split import split_clusters_inplace

cfg = MESH_CONFIGS[0]; ta = cfg['top_axis']; dev = 'cuda:0'


def run(SPLIT):
    s, rs, mi = setup_solver_with_rest(str(REPO / cfg['path']), rings=2, fix_threshold_fraction=0.20)
    N = s.N; s.damping = 0.95; s.num_iterations = 2
    p0 = mi['vertices'].copy(); bb0, bb1 = p0.min(0), p0.max(0); sp = bb1 - bb0
    fr = (p0[:, ta] - bb0[ta]) / sp[ta]
    ea = s.e['a'].numpy().astype(int); eb = s.e['b'].numpy().astype(int)
    mel = float(np.linalg.norm(p0[ea] - p0[eb], axis=1).mean()); Tf = 3 * mel / sp[ta]
    band = (fr >= 0.55 - Tf / 2) & (fr < 0.55 + Tf / 2)
    roof = (fr >= 0.55 + Tf / 2) & (~s.p['fixed'].numpy().astype(bool))
    s.gravity = wp.vec3(0., -9.81, 0.); s.floor_y = float(p0[:, 2].min() - 8)
    sw = torch.tensor(np.full(N, 0.5, np.float32), device=dev)
    twt = torch.tensor(np.full(N, 0.3, np.float32), device=dev)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=dev); far = p0.max(0) + 100
    s.reset_to_rest(p0); st = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in rs.items()}
    r0 = p0[roof, ta].mean(); nC = s.num_clusters
    with torch.no_grad():
        for f in range(155):
            st['alpha'][band] = 0.; st['particle_damage'][band] = 10.
            aw, pw, _ = WarpFullForwardOp.apply(tht, s, st, far, 0., sw, twt)
            if SPLIT and f % 5 == 0:
                nC = split_clusters_inplace(s, drop_small=False)['C_after']
            st = solver_save_state(s)
    pf = pw.cpu().numpy()
    return (r0 - pf[roof, ta].mean()) * 1000, nC


d0, c0 = run(False); d1, c1 = run(True)
print(f'OFF roof drop={d0:.0f}mm clusters={c0}', flush=True)
print(f'ON  roof drop={d1:.0f}mm clusters={c1}', flush=True)
