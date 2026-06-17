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


def _face_at(cx, cy, half=40, gender=-1):
    """A _detect_faces record (dict) centered at (cx, cy), for monkeypatching."""
    return {'cx': cx, 'cy': cy,
            'bbox': (cx - half, cy - half, cx + half, cy + half),
            'gender': gender}


def main():
    image, (cx, cy) = _synthetic_sbs()

    # Force detection to the known face center in each eye.
    vr_face._detect_faces = lambda eye_rgb: [_face_at(cx, cy)]

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
    vr_face._detect_faces = lambda eye_rgb: [_face_at(mcx, mcy)]
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
    vr_face._detect_faces = lambda eye_rgb: [_face_at(cx, cy)]
    tb = torch.cat([image[:, :, :512, :], image[:, :, :512, :]], dim=1).contiguous()  # stack one eye over itself
    tp, tmap, tcount = rect.run(tb, input_layout='tb', projection='equirect',
                                patch_fov=80, patch_size=256)
    print(f'tb rectify -> count={tcount}, patches.shape={tuple(tp.shape)}')
    assert tcount == 2, f'tb: expected 2 patches (top+bottom), got {tcount}'
    (tout,) = unrect.run(tp, tmap)
    assert tout.shape == tb.shape, f'{tout.shape} != {tb.shape}'

    # --- regression: an RGBA patch batch must not crash cv2.remap ---
    # A swapper (or source frame) can hand us 4-channel patches; appending our own
    # alpha then used to make 5 channels and trip cv2.remap's channels()<=4 assert.
    # Force the numpy/cv2 backend (vrt=None) since that is the path that asserted;
    # the torch backend tolerates the extra channel and would hide the regression.
    rgba_patches = torch.cat([mp, torch.ones_like(mp[..., :1])], dim=-1)
    assert rgba_patches.shape[-1] == 4, 'test setup: patches should be RGBA here'
    saved_vrt = vr_face.vrt
    vr_face.vrt = None
    try:
        (rout,) = unrect.run(rgba_patches, mmap)
    finally:
        vr_face.vrt = saved_vrt
    assert rout.shape == mono.shape, f'{rout.shape} != {mono.shape}'
    print('rgba patch round-trip (cv2 backend): no channel crash')

    _test_selection()

    print('PASS')


def _test_selection():
    """Frame-level face selection: order/index/gender pick which faces get patches."""
    # Three faces in one mono eye: small-left-female, big-mid-male, mid-right-female.
    small = _face_at(60, 250, half=20, gender=0)    # area 1600
    big = _face_at(256, 250, half=80, gender=1)     # area 25600 (largest)
    right = _face_at(450, 250, half=40, gender=0)   # area 6400 (rightmost)
    faces = [small, big, right]

    sel = vr_face._select_faces  # (faces, order, index, gender)

    # Default: dominant face only (largest), index 0 -> the big one.
    got = sel(faces, 'large-small', '0', 'no')
    assert got == [big], 'default large-small/0 should pick the single largest face'

    # 'all' emits every detected face (the old fan-out, now opt-in).
    assert len(sel(faces, 'large-small', 'all', 'no')) == 3, "'all' should keep every face"

    # Ranking is frame-level: rightmost via order, 2nd-largest via index.
    assert sel(faces, 'right-left', '0', 'no') == [right], 'right-left/0 = rightmost face'
    assert sel(faces, 'large-small', '1', 'no') == [right], 'large-small/1 = 2nd largest'

    # Gender filter (insightface: 0=female, 1=male) drops non-matching faces.
    # large-small order -> females in ranked order: right (6400) before small (1600).
    assert sel(faces, 'large-small', 'all', 'female') == [right, small], 'female keeps only females'
    assert sel(faces, 'large-small', '0', 'female') == [], 'largest is male -> female filter empties'

    # Out-of-range index is dropped, not an error; empty/garbage index -> [0].
    assert sel(faces, 'large-small', '9', 'no') == [], 'out-of-range index selects nothing'
    assert sel(faces, 'large-small', '', 'no') == [big], 'empty index defaults to 0'

    # A wired OPTIONS dict overrides the widget args.
    opts = {'input_faces_order': 'right-left', 'input_faces_index': '0', 'detect_gender_input': 'no'}
    resolved = vr_face._resolve_selection(opts, 'large-small', '0', 'no')
    assert resolved == ('right-left', '0', 'no'), 'OPTIONS should override widgets'
    assert vr_face._resolve_selection(None, 'top-bottom', '2', 'male') == ('top-bottom', '2', 'male'), \
        'no OPTIONS -> widgets pass through'

    # End-to-end through rect.run: 3 faces but default selection -> 1 patch.
    mono, _ = _synthetic_sbs(eye_w=512, eye_h=512)
    mono = mono[:, :, :512, :].contiguous()
    vr_face._detect_faces = lambda eye_rgb: [small, big, right]
    rect = VRFaceRectify()
    _, _, c_default = rect.run(mono, input_layout='mono', patch_size=128)
    assert c_default == 1, f'default selection should emit 1 patch, got {c_default}'
    _, _, c_all = rect.run(mono, input_layout='mono', patch_size=128, input_faces_index='all')
    assert c_all == 3, f"'all' should emit 3 patches, got {c_all}"
    print('selection: order/index/gender/all + OPTIONS override all correct')


if __name__ == '__main__':
    main()
