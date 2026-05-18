#!/usr/bin/env python3
"""
RealSense D415 hardware test — Phase A through D.

Run BEFORE making phantom to verify camera works for our use case (capturing
phantom deformation + post-cut cavity geometry).

Phases:
  A — Connectivity check (USB type, firmware, FPS)
  B — Python pipeline (save N frames + intrinsics to disk)
  C — Accuracy on known flat object (per-pixel temporal σ)
  D — Surface material test (valid pixel rate on translucent / wet / matte)

Setup before running:
  pip install pyrealsense2 numpy
  pip install opencv-python   # optional, for saving color PNGs
  pip install open3d           # optional, for point cloud viz

Usage:
  python test_realsense_d415_hardware.py --phase a
  python test_realsense_d415_hardware.py --phase b --output ./d415_capture --frames 100
  python test_realsense_d415_hardware.py --phase c --frames 50 --distance 280
  python test_realsense_d415_hardware.py --phase d --frames 50 --surface_label vaseline
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("[ERROR] pyrealsense2 not installed. Run:")
    print("  pip install pyrealsense2")
    sys.exit(1)


# ============================================================================
# Common helpers
# ============================================================================

def setup_pipeline(width_depth=1280, height_depth=720, fps=30,
                   width_color=1920, height_color=1080):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width_depth, height_depth,
                        rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width_color, height_color,
                        rs.format.bgr8, fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    # Warmup: skip first 30 frames for auto-exposure to settle
    for _ in range(30):
        pipeline.wait_for_frames()
    return pipeline, profile, align


def get_device_info(profile):
    dev = profile.get_device()
    info = {
        'name': dev.get_info(rs.camera_info.name),
        'serial': dev.get_info(rs.camera_info.serial_number),
        'firmware': dev.get_info(rs.camera_info.firmware_version),
    }
    try:
        info['usb_type'] = dev.get_info(rs.camera_info.usb_type_descriptor)
    except RuntimeError:
        info['usb_type'] = 'unknown'
    return info


def mark(ok, msg_pass="PASS", msg_fail="FAIL"):
    return f"[OK]   {msg_pass}" if ok else f"[FAIL] {msg_fail}"


def banner(title):
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ============================================================================
# Phase A — Connectivity
# ============================================================================

def phase_a():
    banner("PHASE A - Connectivity check")

    # 1. Device enumeration
    ctx = rs.context()
    devices = list(ctx.query_devices())
    print(f"  Devices found: {len(devices)}")
    if len(devices) == 0:
        print("  [FAIL] No RealSense device detected.")
        print("    - Confirm USB 3.0 Micro-B cable is plugged in")
        print("    - Try a blue USB-A port (USB 3.0)")
        print("    - Try a different USB port / cable")
        return False

    # 2. Start pipeline + get info
    try:
        pipeline, profile, align = setup_pipeline()
    except Exception as e:
        print(f"  [FAIL] Pipeline start failed: {e}")
        return False

    info = get_device_info(profile)
    print(f"  Device:   {info['name']}")
    print(f"  Serial:   {info['serial']}")
    print(f"  Firmware: {info['firmware']}")
    print(f"  USB type: {info['usb_type']}")

    usb_ok = '3' in info['usb_type']
    print(f"  {mark(usb_ok, 'USB 3.x OK', 'USB 2.x - FIX THIS FIRST (port or cable wrong)')}")

    # 3. FPS measurement
    print("\n  Measuring effective FPS (3s)...")
    t0 = time.time()
    n_depth = n_color = 0
    while time.time() - t0 < 3.0:
        frames = pipeline.wait_for_frames()
        if frames.get_depth_frame(): n_depth += 1
        if frames.get_color_frame(): n_color += 1
    dt = time.time() - t0
    fps_d, fps_c = n_depth / dt, n_color / dt
    pipeline.stop()

    fps_ok = fps_d > 25 and fps_c > 25
    print(f"  Depth FPS: {fps_d:.1f}")
    print(f"  Color FPS: {fps_c:.1f}")
    print(f"  {mark(fps_ok, '~30 fps OK', 'Low FPS - bandwidth issue')}")

    all_ok = usb_ok and fps_ok
    print()
    print(f"  Phase A: {mark(all_ok, 'PASSED - ready for Phase B', 'FAILED')}")
    return all_ok


# ============================================================================
# Phase B — Save N frames
# ============================================================================

def phase_b(output_dir, n_frames):
    banner(f"PHASE B - Capture {n_frames} frames to {output_dir}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline, profile, align = setup_pipeline()

    # Save intrinsics + depth scale
    depth_intr = profile.get_stream(rs.stream.depth) \
        .as_video_stream_profile().get_intrinsics()
    color_intr = profile.get_stream(rs.stream.color) \
        .as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    meta = {
        'depth': {
            'fx': depth_intr.fx, 'fy': depth_intr.fy,
            'cx': depth_intr.ppx, 'cy': depth_intr.ppy,
            'width': depth_intr.width, 'height': depth_intr.height,
            'coeffs': list(depth_intr.coeffs),
            'model': str(depth_intr.model),
        },
        'color': {
            'fx': color_intr.fx, 'fy': color_intr.fy,
            'cx': color_intr.ppx, 'cy': color_intr.ppy,
            'width': color_intr.width, 'height': color_intr.height,
            'coeffs': list(color_intr.coeffs),
            'model': str(color_intr.model),
        },
        'depth_scale_m_per_unit': depth_scale,
    }
    with open(out / 'intrinsics.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved intrinsics -> {out/'intrinsics.json'}")
    print(f"  Depth scale: {depth_scale*1000:.5f} mm per raw unit")

    # Try OpenCV for color PNG
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False
        print("  [WARN] cv2 not installed - saving color as .npy instead of PNG")

    print(f"  Capturing {n_frames} frames...")
    n_dropped = 0
    t0 = time.time()
    for i in range(n_frames):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            n_dropped += 1; continue
        aligned = align.process(frames)
        df = aligned.get_depth_frame()
        cf = aligned.get_color_frame()
        if not df or not cf:
            n_dropped += 1; continue
        depth = np.asanyarray(df.get_data())   # uint16, raw units
        color = np.asanyarray(cf.get_data())   # uint8 BGR
        np.save(out / f'depth_{i:04d}.npy', depth)
        if has_cv2:
            cv2.imwrite(str(out / f'color_{i:04d}.png'), color)
        else:
            np.save(out / f'color_{i:04d}.npy', color)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_frames} ({(i+1)/n_frames*100:.0f}%)")
    dt = time.time() - t0
    pipeline.stop()

    kept = n_frames - n_dropped
    print(f"  Captured {kept}/{n_frames} in {dt:.1f}s ({kept/dt:.1f} fps)")
    print(f"  Dropped: {n_dropped}")

    drop_ok = n_dropped < 0.05 * n_frames
    print(f"  {mark(drop_ok, 'Drop rate OK', f'High drop rate ({n_dropped}/{n_frames})')}")
    print(f"  Phase B: {mark(drop_ok, 'PASSED', 'FAILED')}")
    return drop_ok


# ============================================================================
# Phase C — Accuracy test on known matte object
# ============================================================================

def phase_c(n_frames, roi=None, distance_mm=None):
    banner("PHASE C - Accuracy on known matte object")
    print("  SETUP:")
    print("    1. Place a flat MATTE object (paper-wrapped box, matte cube)")
    print("       at 25-30cm from camera")
    print("    2. Camera looks straight down at it")
    print("    3. Object fills ~30% of frame center")
    print()
    input("  Press ENTER when ready...")

    pipeline, profile, align = setup_pipeline()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    print(f"  Capturing {n_frames} frames...")
    stack = []
    for i in range(n_frames):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        df = aligned.get_depth_frame()
        stack.append(np.asanyarray(df.get_data()).astype(np.float32))
    pipeline.stop()
    stack = np.stack(stack, axis=0)   # (N, H, W) in raw units
    H, W = stack.shape[1:]

    # ROI
    if roi is None:
        cx, cy = W // 2, H // 2
        x, y, w, h = cx - 100, cy - 100, 200, 200
        print(f"  Using default center ROI: ({x}, {y}, {w}x{h})")
    else:
        x, y, w, h = roi
        print(f"  Using user ROI: ({x}, {y}, {w}x{h})")

    roi_stack = stack[:, y:y+h, x:x+w]

    # Valid pixel stats
    valid_per_pix_frame = roi_stack > 0
    valid_rate = valid_per_pix_frame.mean()
    always_valid = valid_per_pix_frame.all(axis=0)   # (h, w)
    n_always = int(always_valid.sum())
    total_pix = h * w
    print(f"\n  ROI total pixels: {total_pix}")
    print(f"  Always-valid: {n_always} ({100*n_always/total_pix:.1f}%)")
    print(f"  Overall valid rate: {valid_rate*100:.1f}%")

    if n_always < 100:
        print("  [FAIL] Too few always-valid pixels. Move object closer / matte side.")
        return False

    # Temporal stats on always-valid pixels
    masked = roi_stack[:, always_valid]                # (N, n_pix)
    mean_mm = masked.mean(axis=0) * depth_scale * 1000  # (n_pix,)
    std_mm = masked.std(axis=0) * depth_scale * 1000

    print(f"\n  Mean depth across ROI:")
    print(f"    median: {np.median(mean_mm):.1f} mm")
    print(f"    spatial spread (std): {mean_mm.std():.2f} mm")

    print(f"\n  Per-pixel temporal sigma over {n_frames} frames:")
    print(f"    median: {np.median(std_mm):.3f} mm")
    print(f"    mean:   {std_mm.mean():.3f} mm")
    print(f"    p90:    {np.percentile(std_mm, 90):.3f} mm")
    print(f"    max:    {std_mm.max():.3f} mm")

    # Pass criteria
    median_sigma = float(np.median(std_mm))
    sigma_ok = median_sigma < 1.0
    valid_ok = valid_rate > 0.95

    print()
    print(f"  {mark(sigma_ok, f'sigma {median_sigma:.2f}mm < 1mm', f'sigma {median_sigma:.2f}mm too noisy')}")
    print(f"  {mark(valid_ok, f'valid {valid_rate*100:.1f}% > 95%', f'valid {valid_rate*100:.1f}% too low')}")

    if distance_mm is not None:
        measured = float(np.median(mean_mm))
        err = abs(measured - distance_mm)
        err_pct = err / distance_mm * 100
        accuracy_ok = err_pct < 2.0
        print(f"  Expected {distance_mm}mm vs measured {measured:.1f}mm: "
              f"err {err:.1f}mm ({err_pct:.2f}%) {mark(accuracy_ok)}")

    all_ok = sigma_ok and valid_ok
    print(f"\n  Phase C: {mark(all_ok, 'PASSED', 'FAILED')}")
    return all_ok


# ============================================================================
# Phase D — Surface material test
# ============================================================================

def phase_d(n_frames, surface_label, log_path):
    banner(f"PHASE D - Surface material: '{surface_label}'")
    print(f"  SETUP:")
    print(f"    1. Place '{surface_label}' sample at 25-30cm")
    print(f"    2. Sample should cover most of center FOV")
    print(f"    3. This tests if IR active stereo works on this material")
    print()
    print("  Suggested surrogate materials (no phantom yet):")
    print("    matte_paper, glossy_paper, vaseline, jello, milk_glass,")
    print("    transparent_jar, water_in_cup, wet_sponge, talcum_dusted")
    print()
    input("  Press ENTER when ready...")

    pipeline, profile, align = setup_pipeline()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    print(f"  Capturing {n_frames} frames...")
    stack = []
    for i in range(n_frames):
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        df = aligned.get_depth_frame()
        stack.append(np.asanyarray(df.get_data()).astype(np.float32))
    pipeline.stop()
    stack = np.stack(stack, axis=0)
    H, W = stack.shape[1:]

    # Center 400x400 ROI (broader for material test)
    cx, cy = W // 2, H // 2
    x, y, w, h = cx - 200, cy - 200, 400, 400
    roi_stack = stack[:, y:y+h, x:x+w]

    # Per-frame valid rate (how much of surface is detected each frame)
    valid_per_frame = (roi_stack > 0).mean(axis=(1, 2))   # (N,)
    valid_mean = float(valid_per_frame.mean())
    valid_std = float(valid_per_frame.std())

    # Always-valid pixels
    always_valid = (roi_stack > 0).all(axis=0)
    n_always = int(always_valid.sum())
    always_pct = 100 * n_always / (h * w)

    print(f"\n  Per-frame valid rate: {valid_mean*100:.1f}% (std {valid_std*100:.2f}%)")
    print(f"  Always-valid: {n_always}/{h*w} ({always_pct:.1f}%)")

    # Sigma on always-valid (skip if too few)
    if n_always > 100:
        masked = roi_stack[:, always_valid]
        sigma_mm = masked.std(axis=0) * depth_scale * 1000
        median_sigma = float(np.median(sigma_mm))
        print(f"  Median sigma on always-valid: {median_sigma:.3f} mm")
    else:
        median_sigma = float('nan')
        print(f"  [WARN] Too few always-valid pixels to compute sigma")

    valid_ok = valid_mean > 0.80
    sigma_ok = (median_sigma < 1.5) if not np.isnan(median_sigma) else False
    print()
    print(f"  {mark(valid_ok, f'valid {valid_mean*100:.1f}% > 80%', f'valid {valid_mean*100:.1f}% - IR cannot see surface')}")
    print(f"  {mark(sigma_ok, f'sigma {median_sigma:.2f}mm < 1.5mm', f'sigma {median_sigma:.2f}mm or no data')}")

    # Append to CSV log
    if log_path:
        log = Path(log_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log.exists() or log.stat().st_size == 0
        with open(log, 'a') as f:
            if write_header:
                f.write("surface,n_frames,valid_rate,always_valid_pct,"
                       "median_sigma_mm,verdict\n")
            verdict = 'PASS' if (valid_ok and sigma_ok) else 'FAIL'
            f.write(f"{surface_label},{n_frames},{valid_mean:.4f},"
                   f"{always_pct:.2f},{median_sigma:.4f},{verdict}\n")
        print(f"  Logged -> {log}")

    all_ok = valid_ok and sigma_ok
    print(f"\n  Phase D ({surface_label}): {mark(all_ok, 'PASSED', 'NEEDS SURFACE TREATMENT')}")
    if not all_ok:
        print("    Mitigation options if real phantom shows similar behavior:")
        print("    - Dust phantom surface with talcum / cornstarch")
        print("    - Spray-paint matte white / colored")
        print("    - Add coffee grounds / dye into agar")
    return all_ok


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="RealSense D415 hardware test for phantom deformation capture")
    ap.add_argument('--phase', choices=['a', 'b', 'c', 'd'], required=True)
    ap.add_argument('--output', default='./d415_capture',
                    help='Output dir for Phase B captures')
    ap.add_argument('--frames', type=int, default=100,
                    help='Number of frames (B/C/D); default 100')
    ap.add_argument('--roi', type=str, default=None,
                    help='ROI as x,y,w,h pixels for Phase C; default center 200x200')
    ap.add_argument('--distance', type=float, default=None,
                    help='Expected GT distance in mm for Phase C accuracy check')
    ap.add_argument('--surface_label', type=str, default='unknown',
                    help='Material label for Phase D (e.g. vaseline, jello)')
    ap.add_argument('--log', type=str, default='./d415_surface_log.csv',
                    help='Append Phase D results to this CSV')
    args = ap.parse_args()

    roi = None
    if args.roi:
        parts = args.roi.split(',')
        assert len(parts) == 4, "ROI must be x,y,w,h"
        roi = tuple(int(p) for p in parts)

    if args.phase == 'a':
        ok = phase_a()
    elif args.phase == 'b':
        ok = phase_b(args.output, args.frames)
    elif args.phase == 'c':
        ok = phase_c(args.frames, roi=roi, distance_mm=args.distance)
    elif args.phase == 'd':
        ok = phase_d(args.frames, args.surface_label, args.log)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
