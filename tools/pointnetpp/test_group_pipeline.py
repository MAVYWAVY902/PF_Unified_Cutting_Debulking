"""Step 2 verification: surface-projection GROUP parameterization in the FULL
phantom-realistic online pipeline (gravity + anchor + warm-up + T=300 + L_hist +
L_cav + lr decay). Compares Group vs Hier vs per-particle (Liang) on the same task.
Goal: Group should lift the 14% ceiling toward ~50%+ on the real pipeline."""
import sys, time
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "pf_shape_matching_warp"))
import warp as wp; wp.init()

from tools.day2_solver_wrapper import setup_solver_with_rest, solver_save_state
from tools.day6b_full_forward_torch_op import WarpFullForwardOp
from tools.stageA_inclusion_recovery import gen_obs_per_particle_stiff
from tools.pointnetpp.hierarchy import build_hierarchy
from tools.pointnetpp.multi_mesh_test import MESH_CONFIGS, make_traj_from_spec
from tools.pointnetpp.calibrate_online import calibrate_online
from tools.pointnetpp.surface_projection import column_grid_assignment

TAU_LEFT = 0.15; TAU_RIGHT = 0.55; TAU_INIT = 0.30; STIFF_UNIFORM = 0.5
T_TRAJ = 300


def main():
    cfg = MESH_CONFIGS[0]; device = "cuda:0"
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]),
                                                             fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    kr = cfg["knife_radius_abs"]; top_axis = cfg["top_axis"]
    # warm-up
    solver.reset_to_rest(pos_rest); far = pos_rest.max(0) + 10.0
    sw = torch.tensor(np.full(N, STIFF_UNIFORM, np.float32), device=device)
    twt = torch.tensor(np.full(N, TAU_INIT, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); st = rest_state
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
    pos_rest = pw.cpu().numpy().copy(); mesh_info["vertices"] = pos_rest; rest_state = st
    bb0, bb1 = pos_rest.min(0), pos_rest.max(0); bs = bb1 - bb0
    x_mid = 0.5 * (pos_rest[:, 0].min() + pos_rest[:, 0].max()); lr_mask = pos_rest[:, 0] < x_mid
    tau_truth = np.where(lr_mask, TAU_LEFT, TAU_RIGHT).astype(np.float32)
    stiff_uniform = np.full(N, STIFF_UNIFORM, np.float32)
    surf = pos_rest[:, top_axis] > (bb1[top_axis] - 0.10 * bs[top_axis])
    surf_idx = torch.tensor(np.where(surf)[0], dtype=torch.long, device=device)

    # surface-projection grouping (Step 1 module)
    col_id, ginfo = column_grid_assignment(pos_rest, surf, top_axis, grid_n=8)
    print(f"Group: K={ginfo['K']}, surf/group mean={ginfo['surf_per_group_mean']:.1f}, "
          f"touch={ginfo['groups_touching_surface']}/{ginfo['K']}", flush=True)

    # 2 calib trajectories
    def mk(spec_off):
        s = dict(cfg["traj_specs"]["A"]); s["sweep_range"] = (0.10, 0.90)
        f = dict(s["fixed"])
        for ax in [i for i in range(3) if i != s["sweep_axis"]]:
            if ax != top_axis:
                f[ax] = max(0.20, min(0.80, f[ax] + spec_off)); break
        s["fixed"] = f
        return make_traj_from_spec(s, bb0, bb1, T_TRAJ)
    traj_A = mk(0.0); traj_B = mk(0.20)
    theta_truth = np.array([3., 20., 0.3], np.float32)
    print("Gen obs ...", flush=True)
    solver.reset_to_rest(pos_rest)
    opA, oaA = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, stiff_uniform, traj_A, kr, device, tau_truth_np=tau_truth)
    solver.reset_to_rest(pos_rest)
    opB, oaB = gen_obs_per_particle_stiff(solver, rest_state, theta_truth, stiff_uniform, traj_B, kr, device, tau_truth_np=tau_truth)
    oaA_t = [torch.tensor(a, device=device) for a in oaA]; opA_t = [torch.tensor(p, device=device) for p in opA]
    oaB_t = [torch.tensor(a, device=device) for a in oaB]; opB_t = [torch.tensor(p, device=device) for p in opB]
    multi = [(traj_A, opA_t, oaA_t), (traj_B, opB_t, oaB_t)]

    ea = solver.e["a"].numpy().astype(np.int64); eb = solver.e["b"].numpy().astype(np.int64)
    eat = torch.tensor(ea, dtype=torch.long, device=device); ebt = torch.tensor(eb, dtype=torch.long, device=device)
    erl = np.linalg.norm(pos_rest[ea] - pos_rest[eb], axis=1)
    ert = torch.tensor(erl, dtype=torch.float32, device=device)
    xr = torch.from_numpy(pos_rest).to(device, torch.float32)
    H = build_hierarchy(xr, M1=min(512, max(32, N // 20)), M2=min(64, max(8, N // 200)), k_neighbors=3)
    stiff_init = np.full(N, STIFF_UNIFORM, np.float32); tau_init = np.full(N, TAU_INIT, np.float32)
    ctx = dict(solver=solver, rest_state=rest_state, N=N, device=device, knife_radius=kr,
               pos_rest=pos_rest, theta_3d_np=theta_truth, stiff_init_np=stiff_init, tau_init_np=tau_init,
               tau_bound=(0.05, 0.95), surface_idx=surf_idx, edges_a_t=eat, edges_b_t=ebt,
               edges_rest_t=ert, H=H, group_col_id=col_id)

    LR = 0.03; LR_TAU = 0.5; NP = 4; LH = 1.0; LC = 1.0; LD = 0.5
    def run(cond, label):
        torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        _, tau, _ = calibrate_online(cond, ctx, multi, N_passes=NP, lr=LR, lr_tau=LR_TAU,
                                       lr_decay_per_pass=LD, lambda_hist=LH, lambda_outer_cav=LC, log_every=600)
        gap = tau[~lr_mask].mean() - tau[lr_mask].mean()
        print(f"{label}: L={tau[lr_mask].mean():.3f} R={tau[~lr_mask].mean():.3f}  "
              f"gap={gap:.3f} ({gap/0.40*100:.0f}%)  mem={torch.cuda.max_memory_allocated()/1e6:.0f}MB  {time.time()-t0:.0f}s", flush=True)
        return gap
    print("\n=== Step 2: Group vs Hier vs per-particle on FULL pipeline ===")
    gG = run("Group+L_pos_partial", "GROUP    ")
    gH = run("Hier+L_pos_partial",  "Hier     ")
    gL = run("L_pos_partial",       "Liang(pp)")
    print(f"\nSUMMARY (truth gap 0.40):  GROUP={gG/0.40*100:.0f}%  Hier={gH/0.40*100:.0f}%  Liang={gL/0.40*100:.0f}%")
    print(f"(reference: per-particle ceiling was ~14%)")


if __name__ == "__main__":
    main()
