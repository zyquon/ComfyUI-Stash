"""Parity check: vr_remap_torch (grid_sample) vs vr_remap (cv2) -- the oracle.

Runs on CPU torch, so it verifies CORRECTNESS, not speed (this box has no GPU;
measure speedup on a GPU host). bicubic grid_sample is not bit-identical to
cv2 INTER_CUBIC, so the bar is "within tolerance", not exact. The same code runs
on CUDA in ComfyUI -- only the device changes.

    python3 tests/test_vr_remap_torch.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vr_remap as vr  # noqa: E402  (numpy/cv2 oracle)
import vr_remap_torch as vrt  # noqa: E402  (torch port under test)

MAE_TOL = 2.0  # mean abs error in 0..255; bicubic vs INTER_CUBIC differ slightly
YAW, PITCH = 18.0, -12.0
PATCH_FOV, PATCH_SIZE = 80, 384


def _mae(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def _rand_eye(h, w, c=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)


def main():
    print(f'torch device for this run: {vrt.describe_device()}')
    eye = _rand_eye(900, 900)
    patch_rgba = np.dstack([_rand_eye(PATCH_SIZE, PATCH_SIZE, seed=1),
                            np.full((PATCH_SIZE, PATCH_SIZE, 1), 255, np.uint8)])

    # --- equirect e2p ---
    ref = vr.e2p(eye, PATCH_FOV, YAW, PITCH, PATCH_SIZE, PATCH_SIZE)
    got = vrt.e2p(eye, PATCH_FOV, YAW, PITCH, PATCH_SIZE, PATCH_SIZE)
    e_mae = _mae(ref, got)
    print(f'equirect e2p     MAE: {e_mae:.3f}/255  shapes {ref.shape} vs {got.shape}')
    assert ref.shape == got.shape and e_mae < MAE_TOL, f'e2p parity off: {e_mae}'

    # --- equirect p2e_window (returns window + offset) ---
    r_win, rx, ry = vr.p2e_window(patch_rgba, PATCH_FOV, YAW, PITCH, 900, 900)
    g_win, gx, gy = vrt.p2e_window(patch_rgba, PATCH_FOV, YAW, PITCH, 900, 900)
    assert (rx, ry) == (gx, gy), f'window offset differs: {(rx, ry)} vs {(gx, gy)}'
    p_mae = _mae(r_win, g_win)
    print(f'equirect p2e_win MAE: {p_mae:.3f}/255  window {g_win.shape} @ ({gx},{gy})')
    assert r_win.shape == g_win.shape and p_mae < MAE_TOL, f'p2e parity off: {p_mae}'

    # --- fisheye e2p ---
    ref = vr.fisheye_e2p(eye, PATCH_FOV, YAW, PITCH, PATCH_SIZE, PATCH_SIZE)
    got = vrt.fisheye_e2p(eye, PATCH_FOV, YAW, PITCH, PATCH_SIZE, PATCH_SIZE)
    fe_mae = _mae(ref, got)
    print(f'fisheye  e2p     MAE: {fe_mae:.3f}/255')
    assert ref.shape == got.shape and fe_mae < MAE_TOL, f'fisheye e2p parity off: {fe_mae}'

    # --- fisheye p2e_window ---
    r_win, rx, ry = vr.fisheye_p2e_window(patch_rgba, PATCH_FOV, YAW, PITCH, 900, 900)
    g_win, gx, gy = vrt.fisheye_p2e_window(patch_rgba, PATCH_FOV, YAW, PITCH, 900, 900)
    assert (rx, ry) == (gx, gy), f'fisheye window offset differs: {(rx, ry)} vs {(gx, gy)}'
    fp_mae = _mae(r_win, g_win)
    print(f'fisheye  p2e_win MAE: {fp_mae:.3f}/255  window {g_win.shape} @ ({gx},{gy})')
    assert r_win.shape == g_win.shape and fp_mae < MAE_TOL, f'fisheye p2e parity off: {fp_mae}'

    print('PASS')


if __name__ == '__main__':
    main()
