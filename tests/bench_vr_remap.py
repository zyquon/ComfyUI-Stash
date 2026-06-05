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

try:
    import torch  # noqa: E402
    import vr_remap_torch as vrt  # noqa: E402
except ImportError:
    torch = vrt = None

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
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()  # GPU is async; wait before stopping the clock
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2]


def main():
    print(f'patch_size={PATCH_SIZE} patch_fov={PATCH_FOV} faces/frame={FACES} '
          f'median of {REPEATS}')
    if vrt is not None:
        print(f'torch backend device: {vrt.describe_device()}')
    else:
        print('torch backend: unavailable (numpy/cv2 only)')
    print()
    head = f'{"case":<6} {"eye":>11} {"cv2 full":>10} {"cv2 window":>11}'
    if vrt is not None:
        head += f' {"torch window":>13} {"gpu vs cv2win":>14}'
    head += f' {"window px":>14}'
    print(head)

    for label, edge in CASES:
        rgba = np.dstack([np.full((PATCH_SIZE, PATCH_SIZE, 3), 200, np.uint8),
                          np.full((PATCH_SIZE, PATCH_SIZE, 1), 255, np.uint8)])

        full = _median_ms(lambda: vr.p2e(rgba, PATCH_FOV, YAW, PITCH, edge, edge))
        win = _median_ms(lambda: vr.p2e_window(rgba, PATCH_FOV, YAW, PITCH, edge, edge))

        # Equivalence: windowed output must match the full p2e inside its window.
        ref = vr.p2e(rgba, PATCH_FOV, YAW, PITCH, edge, edge)
        w, x0, y0 = vr.p2e_window(rgba, PATCH_FOV, YAW, PITCH, edge, edge)
        sub = ref[y0:y0 + w.shape[0], x0:x0 + w.shape[1]]
        mae = np.abs(sub.astype(np.float32) - w.astype(np.float32)).mean()
        assert mae < 0.5, f'{label}: windowed != full, MAE={mae}'
        win_frac = (w.shape[0] * w.shape[1]) / (edge * edge) * 100

        row = (f'{label:<6} {edge}x{edge:<6} {full:>8.1f}ms {win:>9.1f}ms')
        if vrt is not None:
            twin = _median_ms(lambda: vrt.p2e_window(rgba, PATCH_FOV, YAW, PITCH, edge, edge))
            # torch path must agree with the cv2 window within bicubic tolerance.
            tw, _, _ = vrt.p2e_window(rgba, PATCH_FOV, YAW, PITCH, edge, edge)
            tmae = np.abs(tw.astype(np.float32) - w.astype(np.float32)).mean()
            assert tmae < 2.0, f'{label}: torch vs cv2 window MAE={tmae}'
            row += f' {twin:>11.1f}ms {win / twin:>12.1f}x'
        row += f' {w.shape[1]}x{w.shape[0]:>6} ({win_frac:.1f}%)'
        print(row)


if __name__ == '__main__':
    main()
