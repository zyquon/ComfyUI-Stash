"""Offline smoke test for the VR face nodes' wrapper plumbing.

Run in an env with numpy + cv2 + torch (the one where vrproj.py was validated):

    python3 tests/test_vr_face.py

It does NOT need ComfyUI or insightface: detection is monkeypatched to a fixed
center. It exercises VRFaceRectify -> (identity passthrough) -> VRFaceUnrectify and
checks the rectify/un-rectify round-trip reproduces the patched region. This proves
the tensor<->numpy, RGBA-alpha, SBS split/merge, and compositing glue are correct
on top of the already-validated geometry.
"""
import os
import sys

import numpy as np
import torch

# tests/ live below the package root; put it on the path so the modules import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vr_face  # noqa: E402
from vr_face import VRFaceRectify, VRFaceUnrectify  # noqa: E402


def _synthetic_sbs(eye_w=512, eye_h=512):
    """One SBS frame: a smooth gradient + a bright box where the 'face' is."""
    rng = np.random.default_rng(0)
    eye = (rng.integers(40, 80, size=(eye_h, eye_w, 3))).astype(np.uint8)
    yy, xx = np.mgrid[0:eye_h, 0:eye_w]
    eye[..., 0] = (xx / eye_w * 200).astype(np.uint8)
    eye[..., 1] = (yy / eye_h * 200).astype(np.uint8)
    # 'face' box near center of each eye
    cx, cy = eye_w // 2, eye_h // 2
    eye[cy - 40:cy + 40, cx - 40:cx + 40] = (230, 60, 60)
    frame = np.concatenate([eye, eye.copy()], axis=1)  # SBS
    return torch.from_numpy(frame.astype(np.float32) / 255.0)[None,], (cx, cy)


def main():
    image, (cx, cy) = _synthetic_sbs()

    # Force detection to the known face center in each eye.
    vr_face._detect_faces = lambda eye_rgb: [(cx, cy)]

    rect = VRFaceRectify()
    unrect = VRFaceUnrectify()

    # --- SBS layout: split into L/R, one patch per eye ---
    patches, rmap, count = rect.run(image, input_layout='sbs', projection='equirect',
                                    patch_fov=80, patch_size=256)
    print(f'sbs rectify -> count={count}, patches.shape={tuple(patches.shape)}')
    assert count == 2, f'sbs: expected 2 patches (L+R eye), got {count}'

    (out,) = unrect.run(patches, rmap)  # identity "swap": patches pass through
    assert out.shape == image.shape, f'{out.shape} != {image.shape}'
    orig = image[0, cy - 30:cy + 30, cx - 30:cx + 30, :].numpy()
    got = out[0, cy - 30:cy + 30, cx - 30:cx + 30, :].numpy()
    err = np.abs(orig - got).mean() * 255.0
    print(f'sbs face-region round-trip MAE: {err:.3f}/255')
    assert err < 6.0, f'sbs round-trip error too high: {err}'

    # --- mono layout (default): whole frame is one eye, one patch ---
    mono, (mcx, mcy) = _synthetic_sbs(eye_w=512, eye_h=512)
    mono = mono[:, :, :512, :].contiguous()  # take one eye -> a mono frame
    vr_face._detect_faces = lambda eye_rgb: [(mcx, mcy)]
    mp, mmap, mcount = rect.run(mono, input_layout='mono', projection='equirect',
                                patch_fov=80, patch_size=256)
    print(f'mono rectify -> count={mcount}, patches.shape={tuple(mp.shape)}')
    assert mcount == 1, f'mono: expected 1 patch, got {mcount}'
    (mout,) = unrect.run(mp, mmap)
    assert mout.shape == mono.shape, f'{mout.shape} != {mono.shape}'
    morig = mono[0, mcy - 30:mcy + 30, mcx - 30:mcx + 30, :].numpy()
    mgot = mout[0, mcy - 30:mcy + 30, mcx - 30:mcx + 30, :].numpy()
    merr = np.abs(morig - mgot).mean() * 255.0
    print(f'mono face-region round-trip MAE: {merr:.3f}/255')
    assert merr < 6.0, f'mono round-trip error too high: {merr}'

    # --- tb layout: top/bottom halves, one patch per half ---
    vr_face._detect_faces = lambda eye_rgb: [(cx, cy)]
    tb = torch.cat([image[:, :, :512, :], image[:, :, :512, :]], dim=1).contiguous()  # stack one eye over itself
    tp, tmap, tcount = rect.run(tb, input_layout='tb', projection='equirect',
                                patch_fov=80, patch_size=256)
    print(f'tb rectify -> count={tcount}, patches.shape={tuple(tp.shape)}')
    assert tcount == 2, f'tb: expected 2 patches (top+bottom), got {tcount}'
    (tout,) = unrect.run(tp, tmap)
    assert tout.shape == tb.shape, f'{tout.shape} != {tb.shape}'

    print('PASS')


if __name__ == '__main__':
    main()
