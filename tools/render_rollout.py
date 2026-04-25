#!/usr/bin/env python3
"""Thin wrapper over rollout_viz_capsule.py to render a checkpoint by step number.

Why: `rollout_viz_capsule.py` takes raw paths. After running `eval_checkpoint_sweep.py`
we want to say "render the best checkpoint" or "render step 250000" without
hand-assembling filenames.

Usage:
  # render by specific step
  python tools/render_rollout.py --run-name cap_armC_torus_seed0 --step 250000

  # render the final checkpoint
  python tools/render_rollout.py --run-name cap_armC_torus_seed0 --step final

  # auto-pick the best checkpoint from sweep summary (highest interior_mean
  # among those with cap_mean <= 0.05 — safe picks only)
  python tools/render_rollout.py --run-name cap_armC_torus_seed0 --best

Output: rollout_frames_{run_name}_step{N}/{*.png, rollout.mp4, summary.json}
"""
import argparse
import csv
import os
import subprocess
import sys


def load_summary(summary_csv):
    rows = []
    with open(summary_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "step": int(row["step"]),
                "is_final": row.get("is_final", "False") in ("True", "1", "true"),
                "interior_mean": float(row["interior_mean"]),
                "cap_mean": float(row["cap_mean"]),
                "catastrophic_rate": float(row["catastrophic_rate"]),
            })
    return rows


def pick_best(rows, cap_budget=0.05):
    """Highest interior_mean among rows with cap_mean <= cap_budget.
    Fallback: if none meet the budget, highest interior overall."""
    safe = [r for r in rows if r["cap_mean"] <= cap_budget]
    pool = safe if safe else rows
    return max(pool, key=lambda r: r["interior_mean"]) if pool else None


def resolve_checkpoint(run_name, step_arg, ckpt_dir):
    """Return (ckpt_path, vec_path) for either an int step or the literal 'final'."""
    if step_arg == "final":
        ck = os.path.join(ckpt_dir, f"{run_name}_final.zip")
        vn = os.path.join(ckpt_dir, f"{run_name}_final_vecnormalize.pkl")
    else:
        step = int(step_arg)
        ck = os.path.join(ckpt_dir, f"{run_name}_{step}_steps.zip")
        vn = os.path.join(ckpt_dir, f"{run_name}_vecnormalize_{step}_steps.pkl")
    if not os.path.exists(ck):
        raise FileNotFoundError(f"checkpoint not found: {ck}")
    if not os.path.exists(vn):
        raise FileNotFoundError(f"vecnormalize not found: {vn}")
    return ck, vn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True, help="e.g. cap_armC_torus_seed0")
    ap.add_argument("--cfg", default=None,
                    help="config yaml (default: warp_sim/config/debulk_cap_armC.yaml)")
    ap.add_argument("--step", default=None,
                    help="training step number, or literal 'final'")
    ap.add_argument("--best", action="store_true",
                    help="Auto-pick best checkpoint from sweep summary CSV")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="default: checkpoints/{run_name}")
    ap.add_argument("--sweep-dir", default=None,
                    help="default: results/sweep/{run_name}")
    ap.add_argument("--out-dir", default=None,
                    help="default: rollout_frames_{run_name}_step{N}")
    ap.add_argument("--frames", type=int, default=1024,
                    help="Number of rollout frames (default: episode_length)")
    ap.add_argument("--cap-budget", type=float, default=0.05,
                    help="For --best: max capsule breach to consider safe")
    ap.add_argument("--render-every", type=int, default=1)
    args = ap.parse_args()

    cfg_path = args.cfg or "warp_sim/config/debulk_cap_armC.yaml"
    ckpt_dir = args.checkpoint_dir or f"checkpoints/{args.run_name}"
    sweep_dir = args.sweep_dir or f"results/sweep/{args.run_name}"

    if args.best:
        summary_csv = os.path.join(sweep_dir, f"{args.run_name}_summary.csv")
        if not os.path.exists(summary_csv):
            print(f"[error] summary CSV not found: {summary_csv}\n"
                  f"        run eval_checkpoint_sweep.py first.", file=sys.stderr)
            return 1
        rows = load_summary(summary_csv)
        best = pick_best(rows, cap_budget=args.cap_budget)
        if best is None:
            print("[error] no valid rows in summary", file=sys.stderr)
            return 1
        step_arg = "final" if best["is_final"] else str(best["step"])
        print(f"[best] step={best['step']}{' (final)' if best['is_final'] else ''}, "
              f"interior={best['interior_mean']*100:.2f}%, "
              f"cap={best['cap_mean']*100:.2f}%, "
              f"catastrophic={best['catastrophic_rate']*100:.1f}%")
    elif args.step is not None:
        step_arg = args.step
    else:
        print("[error] need --step or --best", file=sys.stderr)
        return 1

    ck, vn = resolve_checkpoint(args.run_name, step_arg, ckpt_dir)
    step_tag = step_arg if step_arg == "final" else f"step{int(step_arg):06d}"
    out_dir = args.out_dir or f"rollout_frames_{args.run_name}_{step_tag}"

    print(f"[render] checkpoint: {ck}")
    print(f"[render] vecnorm:    {vn}")
    print(f"[render] out_dir:    {out_dir}")
    print(f"[render] frames:     {args.frames}")

    cmd = [
        sys.executable, "warp_sim/rollout_viz_capsule.py",
        "--model", ck,
        "--vecnorm", vn,
        "--cfg", cfg_path,
        "--frames", str(args.frames),
        "--out-dir", out_dir,
        "--render-every", str(args.render_every),
    ]
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)

    # Stitch MP4 via imageio if rollout_viz_capsule didn't (ffmpeg missing)
    mp4 = os.path.join(out_dir, "rollout.mp4")
    if not os.path.exists(mp4):
        print("[mp4] ffmpeg path failed; stitching via imageio fallback ...")
        import glob
        pngs = sorted(glob.glob(os.path.join(out_dir, "frame_*.png")))
        if pngs:
            import imageio.v2 as iio
            with iio.get_writer(mp4, fps=20, codec="libx264", quality=7) as w:
                for p in pngs:
                    w.append_data(iio.imread(p))
            print(f"[mp4] wrote {mp4} ({os.path.getsize(mp4)/1e6:.1f} MB, {len(pngs)} frames)")
        else:
            print(f"[mp4] no frames found in {out_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
