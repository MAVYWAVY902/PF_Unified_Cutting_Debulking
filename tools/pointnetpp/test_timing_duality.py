"""Timing-duality verification: WHY does dynamic exposure capture so little depth tau?

Claim: a deep particle's tau signal lives in its "sensitive zone" (damage climbing
toward tau, where alpha responds to tau). For a buried particle this approach happens
WHILE IT IS STILL BURIED (unobserved). By the time it is EXPOSED (becomes topmost-
alive), its damage is already at/past tau -> you only see the collapse tail.

Test: run the deep buried-inclusion cut with truth tau, recording per-frame per-particle
DAMAGE and ALPHA. For each inclusion particle compute:
   t_sensitive = first frame in the sensitive zone |damage - tau| < band
   t_exposed   = first frame it is the topmost-alive (observed) in its column
   capture     = (# frames in sensitive zone AND observed) / (# frames in sensitive zone)
Compare inclusion vs original-surface particles.

PREDICTION (if claim true): for inclusion particles, t_sensitive << t_exposed (sensitive
before exposed), capture rate ~0, and alpha at first exposure already low (dead-on-arrival).
For original-surface particles the opposite (observed from the start -> capture ~1).
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
from tools.pointnetpp.test_dynamic_surface_depth import make_layered_debulk

TAU_INCL = 0.55; TAU_BG = 0.30; STIFF_UNIFORM = 0.5
EXPOSE_BAND = 0.10      # topmost-alive band (fraction of depth span)
SENS_BAND = 0.10        # sensitive zone half-width in damage units
GN = 8


def main():
    cfg = MESH_CONFIGS[0]; device = "cuda:0"
    solver, rest_state, mesh_info = setup_solver_with_rest(str(REPO / cfg["path"]),
                                                            fix_threshold_fraction=0.20)
    N = solver.N; solver.damping = 0.95; solver.num_iterations = 2
    pos_rest = mesh_info["vertices"]
    solver.gravity = wp.vec3(0.0, -9.81, 0.0); solver.floor_y = float(pos_rest[:, 2].min() - 1.0)
    kr = cfg["knife_radius_abs"]; top_axis = cfg["top_axis"]

    solver.reset_to_rest(pos_rest); far = pos_rest.max(0) + 10.0
    sw = torch.tensor(np.full(N, STIFF_UNIFORM, np.float32), device=device)
    twt = torch.tensor(np.full(N, 0.30, np.float32), device=device)
    tht = torch.tensor(np.array([3., 20., 0.3], np.float32), device=device); st = rest_state
    with torch.no_grad():
        for _ in range(60):
            _, pw, _ = WarpFullForwardOp.apply(tht, solver, st, far, kr, sw, twt); st = solver_save_state(solver)
    pos_rest = pw.cpu().numpy().copy(); rest_state = st
    bb0, bb1 = pos_rest.min(0), pos_rest.max(0); sp = bb1 - bb0
    lateral = [a for a in range(3) if a != top_axis]
    sweep_axis, lat_axis = lateral[0], lateral[1]

    xc = bb0[sweep_axis] + 0.5 * sp[sweep_axis]; zc = bb0[lat_axis] + 0.5 * sp[lat_axis]
    y_lo = bb0[top_axis] + 0.35 * sp[top_axis]; y_hi = bb0[top_axis] + 0.62 * sp[top_axis]
    incl = ((np.abs(pos_rest[:, sweep_axis] - xc) < 0.22 * sp[sweep_axis]) &
            (np.abs(pos_rest[:, lat_axis] - zc) < 0.22 * sp[lat_axis]) &
            (pos_rest[:, top_axis] > y_lo) & (pos_rest[:, top_axis] < y_hi))
    surf = pos_rest[:, top_axis] > (bb1[top_axis] - 0.10 * sp[top_axis])
    tau_truth = np.where(incl, TAU_INCL, TAU_BG).astype(np.float32)
    print(f"N={N} inclusion={int(incl.sum())} surface={int(surf.sum())}", flush=True)

    traj = make_layered_debulk(bb0, bb1, top_axis, sweep_axis, lat_axis, 0.92, 0.50, 8, 60)
    T = len(traj)

    # forward with truth tau, record damage (dv) + alpha per frame
    theta = torch.tensor([3., 20., 0.3], dtype=torch.float32, device=device)
    stiff_t = torch.tensor(np.full(N, STIFF_UNIFORM, np.float32), device=device)
    tau_t = torch.tensor(tau_truth, device=device)
    solver.reset_to_rest(pos_rest); state = rest_state
    DV = np.zeros((T, N), np.float32); AL = np.zeros((T, N), np.float32)
    with torch.no_grad():
        for t, knife in enumerate(traj):
            alpha_t, pos_t, dv_t = WarpFullForwardOp.apply(theta, solver, state, knife, kr, stiff_t, tau_t)
            DV[t] = dv_t.cpu().numpy(); AL[t] = alpha_t.cpu().numpy()
            state = solver_save_state(solver)
    print(f"forward done T={T}; max damage={DV.max():.2f}", flush=True)

    # dynamic observed mask per frame (topmost-alive band per lateral column)
    depth = pos_rest[:, top_axis]
    lo = pos_rest.min(0); hi = pos_rest.max(0); span = hi - lo + 1e-9
    b0 = np.clip(((pos_rest[:, lateral[0]] - lo[lateral[0]]) / span[lateral[0]] * GN).astype(int), 0, GN - 1)
    b1 = np.clip(((pos_rest[:, lateral[1]] - lo[lateral[1]]) / span[lateral[1]] * GN).astype(int), 0, GN - 1)
    lat_id = b0 * GN + b1; n_lat = GN * GN
    band = EXPOSE_BAND * span[top_axis]

    OBS = np.zeros((T, N), bool)
    for t in range(T):
        alive = AL[t] > 0.5
        d_alive = np.where(alive, depth, -1e9)
        col_top = np.full(n_lat, -1e9, np.float32)
        np.maximum.at(col_top, lat_id, d_alive)
        OBS[t] = alive & (depth >= col_top[lat_id] - band) & (col_top[lat_id] > -1e8)

    # sensitive zone: |damage - tau| < SENS_BAND  (alpha responsive to tau here)
    SENS = np.abs(DV - tau_truth[None, :]) < SENS_BAND

    def stats(mask_set, name):
        idx = np.where(mask_set)[0]
        ever_sens = SENS[:, idx].any(0)
        cut = idx[ever_sens]               # particles that actually enter sensitive zone
        if len(cut) == 0:
            print(f"  {name}: none enter sensitive zone"); return
        lags = []; caps = []; alpha_arrival = []; sens_before_exp = 0
        for i in cut:
            ts = np.where(SENS[:, i])[0]
            to = np.where(OBS[:, i])[0]
            t_sens0 = ts[0]
            n_sens = len(ts)
            n_sens_obs = int((SENS[:, i] & OBS[:, i]).sum())
            caps.append(n_sens_obs / n_sens)
            if len(to) > 0:
                t_exp0 = to[0]
                lags.append(t_exp0 - t_sens0)
                alpha_arrival.append(AL[t_exp0, i])
                if t_sens0 < t_exp0:
                    sens_before_exp += 1
            else:
                lags.append(np.nan); alpha_arrival.append(np.nan)
        lags = np.array(lags); caps = np.array(caps); aa = np.array(alpha_arrival)
        never_obs = int(np.isnan(lags).sum())
        print(f"  {name}: cut(enter sensitive)={len(cut)}", flush=True)
        print(f"    capture rate (frames sensitive&observed / sensitive): "
              f"median={np.nanmedian(caps):.3f} mean={np.nanmean(caps):.3f}", flush=True)
        print(f"    t_exposed - t_sensitive: median={np.nanmedian(lags):.0f} frames "
              f"(>0 = exposed AFTER sensitive); sensitive-before-exposed = {sens_before_exp}/{len(cut)}", flush=True)
        print(f"    alpha at first exposure: median={np.nanmedian(aa):.3f} "
              f"(dead-on-arrival alpha<0.5: {int((aa<0.5).sum())}/{int((~np.isnan(aa)).sum())})", flush=True)
        print(f"    never observed at all: {never_obs}/{len(cut)}", flush=True)

    print("\n=== TIMING DUALITY: inclusion (deep) vs original surface ===")
    stats(incl, "INCLUSION (deep)")
    stats(surf, "SURFACE (shallow)")
    print("\nPREDICTION: inclusion -> low capture, exposed AFTER sensitive, low arrival alpha;")
    print("            surface   -> high capture, sensitive while already observed.")


if __name__ == "__main__":
    main()
