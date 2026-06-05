"""Microbenchmark: rectify (e2p) vs unrectify (p2e) cost vs frame resolution.

Confirms the asymmetry: e2p arrays are sized by patch_size (constant), p2e arrays
are sized by the full eye (scales with resolution^2). Run in an env with cv2+numpy:

    python3 tests/bench_vr_remap.py

No insightface needed -- this times the projection geometry only, which is the
part you control and the part that dominates the unrectify cost.
"""
import os
import sys
import time

import numpy as np

# tests/ live below the package root; put it on the path so the modules import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vr_remap as vr  # noqa: E402

# (label, eye edge) for the pre-cropped mono cases. Mono => eye = whole frame.
CASES = [('4kvr', 2048), ('6kvr', 2880), ('8kvr', 4096)]
PATCH_FOV = 80
PATCH_SIZE = 768
YAW, PITCH = 15.0, -10.0
FACES = 2        # entries per frame; both nodes scale linearly in this
REPEATS = 5      # median of N timings


def _median_ms(fn, n=REPEATS):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2]


def main():
    print(f'patch_size={PATCH_SIZE} patch_fov={PATCH_FOV} faces/frame={FACES} '
          f'median of {REPEATS}\n')
    print(f'{"case":<6} {"eye":>11} {"rectify":>9} {"p2e full":>10} '
          f'{"p2e window":>11} {"speedup":>8} {"window px":>14}')
    for label, edge in CASES:
        eye = (np.random.rand(edge, edge, 3) * 255).astype(np.uint8)
        rgba = np.dstack([np.full((PATCH_SIZE, PATCH_SIZE, 3), 200, np.uint8),
                          np.full((PATCH_SIZE, PATCH_SIZE, 1), 255, np.uint8)])

        rect = _median_ms(lambda: vr.e2p(eye, PATCH_FOV, YAW, PITCH,
                                         PATCH_SIZE, PATCH_SIZE))
        full = _median_ms(lambda: vr.p2e(rgba, PATCH_FOV, YAW, PITCH, edge, edge))
        win = _median_ms(lambda: vr.p2e_window(rgba, PATCH_FOV, YAW, PITCH, edge, edge))

        # Equivalence: windowed output must match the full p2e inside its window.
        ref = vr.p2e(rgba, PATCH_FOV, YAW, PITCH, edge, edge)
        w, x0, y0 = vr.p2e_window(rgba, PATCH_FOV, YAW, PITCH, edge, edge)
        sub = ref[y0:y0 + w.shape[0], x0:x0 + w.shape[1]]
        mae = np.abs(sub.astype(np.float32) - w.astype(np.float32)).mean()
        assert mae < 0.5, f'{label}: windowed != full, MAE={mae}'
        win_frac = (w.shape[0] * w.shape[1]) / (edge * edge) * 100

        print(f'{label:<6} {edge}x{edge:<6} {rect:>7.1f}ms {full:>8.1f}ms '
              f'{win:>9.1f}ms {full / win:>6.1f}x {w.shape[1]}x{w.shape[0]:>6} '
              f'({win_frac:.1f}%)')


if __name__ == '__main__':
    main()
