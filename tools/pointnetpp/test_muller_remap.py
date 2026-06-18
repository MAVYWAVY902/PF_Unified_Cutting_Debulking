import sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/home/yunxin/pf-shape-matching")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()
from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS
from tools.pointnetpp.test_group_generalize import make_raster
cfg = MESH_CONFIGS[0]; dev="cuda:0"; TAU=0.30; THETA=np.array([3.,20.,0.3],np.float32)
solver,rest_state,mi=setup_solver_with_rest(str(REPO/cfg["path"]),fix_threshold_fraction=0.20)
N=solver.N; solver.damping=0.95; solver.substeps=1; solver.num_iterations=2
pr0=mi["vertices"]; solver.gravity=wp.vec3(0.,-9.81,0.); solver.floor_y=float(pr0[:,2].min()-1.0)
kr=cfg["knife_radius_abs"]; top=cfg["top_axis"]
tht=torch.tensor(THETA,device=dev); swt=lambda v: torch.tensor(np.full(N,v,np.float32),device=dev)
solver.reset_to_rest(pr0); FAR=pr0.max(0)+10.; st=rest_state
with torch.no_grad():
    for _ in range(60):
        _,pw,_=WarpFullForwardOp.apply(tht,solver,st,FAR,kr,swt(0.5),swt(TAU)); st=solver_save_state(solver)
prs=pw.cpu().numpy().copy(); REST={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in st.items()}
bb0,bb1=prs.min(0),prs.max(0); sp=bb1-bb0; center=0.5*(bb0+bb1); diag=float(np.linalg.norm(sp))
lat=[a for a in range(3) if a!=top]; sweep,la=lat[0],lat[1]
traj=make_raster(bb0,bb1,top,sweep,la,0.93,6,30)
print(f"N={N} diag={diag:.3f}m traj={len(traj)} (cut)")
def run(ni, remap, stiff=0.5):
    solver.num_iterations=ni; solver.muller_stiff_remap=remap; solver.tw_gate=0.0; solver.collision_in_loop=True
    solver.reset_to_rest(prs); st={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in REST.items()}
    sw,twt=swt(stiff),swt(TAU); onset=None; peak=0.0
    with torch.no_grad():
        for step,kp in enumerate(traj):
            _,pw,_=WarpFullForwardOp.apply(tht,solver,st,kp.astype(np.float32),kr,sw,twt)
            st=solver_save_state(solver); p=st["pos"]
            if not np.isfinite(p).all(): onset=onset or step; peak=float("inf"); break
            peak=max(peak,float(np.abs(p-center).max()))
            if onset is None and peak>2*diag: onset=step
    return onset,peak
print(f"  {'num_iter':>8s} {'remap':>6s} {'onset':>8s} {'peak|pos|':>12s}")
for ni in (2, 6, 8):
    for remap in (False, True):
        o,pk=run(ni,remap); on="NEVER" if o is None else f"f{o}"
        pks=f"{pk:.3f}m" if pk<1e4 else f"{pk:.1e}m"
        print(f"  {ni:8d} {str(remap):>6s} {on:>8s} {pks:>12s}",flush=True)
solver.muller_stiff_remap=False; solver.num_iterations=2
