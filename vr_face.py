"""ComfyUI nodes to undistort VR faces for face-swapping, then re-distort them.

VRFaceRectify pulls each detected face out of an equirect/fisheye VR180-SBS frame
into a flat rectilinear patch (what a face swapper like ReActor expects). Run the
swapper on the patch batch, then VRFaceUnrectify projects each swapped patch back
into its exact original frame/eye/position and composites it.

The two nodes are bracketed around the swapper:

    IMAGE(VR frames) --> VRFaceRectify --> (patches IMAGE, VR_RECTIFY_MAP)
                                                 |               |
                                            patches IMAGE        |
                                                 v               |
                                              ReActor            |
                                                 v               |
                                       swapped patches IMAGE      |
                                                 |               |
                         VRFaceUnrectify <-------+---------------+
                                                 v
                                        IMAGE(VR frames, faces swapped)

VR_RECTIFY_MAP is an opaque object (like this package's STASH type) carrying the
base frames plus one geometry entry per patch, so Unrectify needs no other wiring.
"""
import numpy as np
import torch

try:
    from . import vr_remap as vr
except ImportError:  # allow standalone import (tests, offline geometry checks)
    import vr_remap as vr

try:
    from . import vr_remap_torch as vrt  # GPU/torch port of the hot projections
except ImportError:
    try:
        import vr_remap_torch as vrt
    except ImportError:
        vrt = None  # torch absent -> fall back to the numpy/cv2 geometry

NODE_CATEGORY = 'Stash/VR'


def _torch_device():
    """ComfyUI's chosen torch device, or None to let vr_remap_torch auto-pick."""
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return None


def _log_backend(node_name, device):
    """Log the geometry backend + actual compute device, so a run makes it certain
    whether GPU acceleration is live (prints e.g. 'cuda:0 (NVIDIA ...)')."""
    if vrt is not None:
        print(f'{node_name}: geometry backend: torch, device: {vrt.describe_device(device)}')
    else:
        print(f'{node_name}: geometry backend: numpy/cv2 (torch not importable)')


def _rectify_patch(eye, projection, patch_fov, yaw, pitch, patch_size,
                   fisheye_fov, cx, cy, radius, device):
    """e2p/fisheye_e2p on the torch backend if available, else numpy/cv2."""
    if vrt is not None:
        if projection == 'fisheye':
            return vrt.fisheye_e2p(eye, patch_fov, yaw, pitch, patch_size, patch_size,
                                   fisheye_fov, cx, cy, radius, device=device)
        return vrt.e2p(eye, patch_fov, yaw, pitch, patch_size, patch_size, device=device)
    if projection == 'fisheye':
        return vr.fisheye_e2p(eye, patch_fov, yaw, pitch, patch_size, patch_size,
                              fisheye_fov, cx, cy, radius)
    return vr.e2p(eye, patch_fov, yaw, pitch, patch_size, patch_size)


def _unrectify_window(rgba, e, eye_w, device):
    """Windowed p2e/fisheye_p2e on the torch backend if available, else numpy/cv2."""
    if vrt is not None:
        if e['projection'] == 'fisheye':
            return vrt.fisheye_p2e_window(rgba, e['patch_fov'], e['yaw'], e['pitch'],
                                          eye_w, e['eye_h'], e['fisheye_fov'],
                                          e['cx'], e['cy'], e['radius'], device=device)
        return vrt.p2e_window(rgba, e['patch_fov'], e['yaw'], e['pitch'],
                              eye_w, e['eye_h'], device=device)
    if e['projection'] == 'fisheye':
        return vr.fisheye_p2e_window(rgba, e['patch_fov'], e['yaw'], e['pitch'],
                                     eye_w, e['eye_h'], e['fisheye_fov'],
                                     e['cx'], e['cy'], e['radius'])
    return vr.p2e_window(rgba, e['patch_fov'], e['yaw'], e['pitch'], eye_w, e['eye_h'])


# ---- ComfyUI IMAGE <-> numpy uint8 RGB ----
def _img_to_np(t):
    """One ComfyUI image (H,W,C) float 0..1 -> uint8 numpy RGB (H,W,3).

    Drops any alpha/extra channels: ComfyUI IMAGE is usually RGB but can arrive
    RGBA, and the downstream cv2.remap rejects >4 channels once we append our own
    alpha for compositing.
    """
    return (t.detach().cpu().numpy()[..., :3] * 255.0).clip(0, 255).astype(np.uint8)


def _np_to_img(a):
    """uint8 numpy (H,W,3) RGB -> ComfyUI image tensor (1,H,W,3) float 0..1."""
    return torch.from_numpy(a.astype(np.float32) / 255.0)[None,]


# ---- face detection (lazy: only imported when a rectify actually runs) ----
_FACE_APP = None


def _detect_faces(eye_rgb):
    """Return a list of face records for faces in one RGB uint8 eye image.

    Each record is a dict: ``{'cx', 'cy', 'bbox', 'gender'}`` -- the bbox center
    (the rectify geometry only needs a rough center; ReActor re-detects precisely
    on the patch), the bbox ``(x1, y1, x2, y2)`` for ordering, and the gender
    (0=female, 1=male, -1=unknown), matching insightface/ReActor conventions.

    Uses insightface (ReActor's own detector, already present in the env).
    Isolated here so an alternate detector (e.g. SAM3-video) can swap in cleanly.
    """
    global _FACE_APP
    import cv2
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l',
                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _FACE_APP = app
    bgr = cv2.cvtColor(eye_rgb, cv2.COLOR_RGB2BGR)
    faces = []
    for face in _FACE_APP.get(bgr):
        x1, y1, x2, y2 = (float(v) for v in face.bbox)
        faces.append({
            'cx': (x1 + x2) / 2.0, 'cy': (y1 + y2) / 2.0,
            'bbox': (x1, y1, x2, y2),
            'gender': int(getattr(face, 'gender', -1)),
        })
    return faces


# ---- face selection (frame-level, mirrors ReActor's own options) ----
def _bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


# Ordering keys match ReActor's sort_by_order: left/top use the bbox edge, size
# uses bbox area. (sort_key, reverse) per mode.
_FACE_ORDER_KEYS = {
    'left-right':  (lambda f: f['bbox'][0], False),
    'right-left':  (lambda f: f['bbox'][0], True),
    'top-bottom':  (lambda f: f['bbox'][1], False),
    'bottom-top':  (lambda f: f['bbox'][1], True),
    'small-large': (lambda f: _bbox_area(f['bbox']), False),
    'large-small': (lambda f: _bbox_area(f['bbox']), True),
}


def _parse_faces_index(faces_index):
    """ReActor-compatible index string -> list of ints (empty/invalid -> [0]).

    Also accepts the literal 'all' (case-insensitive) -> the string 'all', a small
    superset of ReActor for the 'swap everyone in frame' case.
    """
    if isinstance(faces_index, str) and faces_index.strip().lower() == 'all':
        return 'all'
    idx = [int(x) for x in str(faces_index).strip(',').split(',') if x.strip().isdigit()]
    return idx or [0]


def _select_faces(faces, input_faces_order, input_faces_index, detect_gender_input):
    """Pick which detected faces to rectify, at the frame/eye level.

    Mirrors ReActor's selection (sort_by_order + faces_index + gender) so rectify
    isolates the target character BEFORE slicing, instead of emitting one patch per
    face and letting ReActor re-select inside each single-face patch (where its
    ranking options are meaningless). Default '0'/'large-small' -> the dominant
    face only, the common one-character-per-frame case.
    """
    if not faces:
        return []
    key, reverse = _FACE_ORDER_KEYS.get(input_faces_order, _FACE_ORDER_KEYS['large-small'])
    ordered = sorted(faces, key=key, reverse=reverse)

    index = _parse_faces_index(input_faces_index)
    selected = ordered if index == 'all' else [ordered[i] for i in index if 0 <= i < len(ordered)]

    # Gender filter: 'no' keeps all; else keep faces matching (0=female, 1=male).
    if detect_gender_input in ('female', 'male'):
        want = 0 if detect_gender_input == 'female' else 1
        selected = [f for f in selected if f['gender'] == want]
    return selected


def _resolve_selection(options, input_faces_order, input_faces_index, detect_gender_input):
    """A wired OPTIONS dict (e.g. from ReActor) wins; else the node's widgets.

    Mirrors ReActorPlusOpt's precedence -- a connected OPTIONS overrides the inline
    widgets -- but reads only the target-side keys that are meaningful frame-level.
    """
    if options:
        return (
            options.get('input_faces_order', input_faces_order),
            options.get('input_faces_index', input_faces_index),
            options.get('detect_gender_input', detect_gender_input),
        )
    return input_faces_order, input_faces_index, detect_gender_input


class VRFaceRectify:
    """
    Undistort VR faces for swapping. For each VR180-SBS frame, detects faces in
    each eye, selects the target face(s) (ReActor-style order/index/gender), and
    rectifies each selected face to a flat patch a face swapper can handle.
    Outputs the patch batch (feed to ReActor) and a VR_RECTIFY_MAP for un-rectify.

    Selection happens HERE, at the frame level, where ranking ("largest",
    "rightmost") is meaningful -- ReActor only ever sees one-face patches, so its
    own order/index options can't isolate a character once we've sliced. Defaults
    ('0' / 'large-small') emit just the dominant face per eye: the usual
    one-character-per-job case. A wired OPTIONS input overrides the widgets.
    """
    DESCRIPTION = __doc__
    CATEGORY = NODE_CATEGORY
    NAME = 'VR Face Rectify'

    RETURN_NAMES = ('patches', 'rectify_map', 'count')
    RETURN_TYPES = ('IMAGE', 'VR_RECTIFY_MAP', 'INT')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'image': ('IMAGE', {'tooltip': 'VR180-SBS frame(s); a batch is allowed'}),
            },
            'optional': {
                'input_layout': (['mono', 'sbs', 'tb'], {
                    'tooltip': 'mono = whole frame is one eye (per-eye pipelines); '
                               'sbs = left|right halves; tb = top|bottom halves',
                }),
                'projection': (['equirect', 'fisheye'], {
                    'tooltip': 'Source VR projection of each eye',
                }),
                'patch_fov': ('INT', {
                    'default': 80, 'min': 20, 'max': 150,
                    'tooltip': 'Field of view (deg) of the flat patch; ~75-80 suits ReActor',
                }),
                'patch_size': ('INT', {
                    'default': 768, 'min': 128, 'max': 2048, 'step': 64,
                    'tooltip': 'Square patch edge in pixels',
                }),
                'fisheye_fov': ('FLOAT', {
                    'default': 135.0, 'min': 90.0, 'max': 220.0, 'step': 1.0,
                    'tooltip': 'Full fisheye FOV (deg); only used for fisheye projection',
                }),
                'input_faces_order': (
                    ['left-right', 'right-left', 'top-bottom', 'bottom-top',
                     'small-large', 'large-small'],
                    {'default': 'large-small',
                     'tooltip': 'How to rank faces in each eye before picking by index '
                                '(matches ReActor). Ignored if an OPTIONS input is wired.'},
                ),
                'input_faces_index': ('STRING', {
                    'default': '0',
                    'tooltip': 'Which ranked face(s) to swap: "0" (default = dominant face), '
                               '"0,1", or "all". Selection is frame-level -> one patch per '
                               'selected face. Ignored if an OPTIONS input is wired.',
                }),
                'detect_gender_input': (['no', 'female', 'male'], {
                    'default': 'no',
                    'tooltip': 'Only rectify faces of this gender (insightface). '
                               '"no" = any. Ignored if an OPTIONS input is wired.',
                }),
                'options': ('OPTIONS', {
                    'tooltip': 'Optional ReActor-compatible OPTIONS dict; when wired it '
                               'overrides the three selection widgets above.',
                }),
            },
        }

    FUNCTION = 'run'

    def run(self, image, input_layout='mono', projection='equirect',
            patch_fov=80, patch_size=768, fisheye_fov=135.0,
            input_faces_order='large-small', input_faces_index='0',
            detect_gender_input='no', options=None):
        device = _torch_device()
        _log_backend('VR Face Rectify', device)
        order, index, gender = _resolve_selection(
            options, input_faces_order, input_faces_index, detect_gender_input)
        frames = [_img_to_np(image[b]) for b in range(image.shape[0])]

        patches = []
        entries = []
        for frame_idx, frame in enumerate(frames):
            if input_layout == 'sbs':
                left, right = vr.split_sbs(frame)
                eyes = (('L', left), ('R', right))
            elif input_layout == 'tb':
                h = frame.shape[0]
                eyes = (('T', frame[:h // 2]), ('B', frame[h // 2:]))
            else:  # mono: the whole frame is a single eye
                eyes = (('M', frame),)
            for eye_name, eye in eyes:
                eye_h, eye_w = eye.shape[:2]
                cx = cy = radius = None
                if projection == 'fisheye':
                    cx, cy, radius = vr.detect_fisheye_circle(eye)

                selected = _select_faces(_detect_faces(eye), order, index, gender)
                for face in selected:
                    fx, fy = face['cx'], face['cy']
                    yaw, pitch = vr.pixel_to_yaw_pitch(
                        fx, fy, eye_w, eye_h, projection, fisheye_fov, cx, cy, radius)
                    patch = _rectify_patch(eye, projection, patch_fov, yaw, pitch,
                                           patch_size, fisheye_fov, cx, cy, radius, device)
                    entries.append({
                        'patch_index': len(patches),
                        'frame_idx': frame_idx, 'eye': eye_name,
                        'projection': projection, 'fisheye_fov': fisheye_fov,
                        'cx': cx, 'cy': cy, 'radius': radius,
                        'yaw': yaw, 'pitch': pitch,
                        'patch_fov': patch_fov, 'patch_size': patch_size,
                        'eye_w': eye_w, 'eye_h': eye_h,
                    })
                    patches.append(_np_to_img(patch))

        count = len(patches)
        print(f'VR Face Rectify: patches: {count}')
        if patches:
            out = torch.cat(patches, dim=0)
        else:
            # No faces anywhere: emit one black patch so the IMAGE wire stays valid;
            # empty entries means Unrectify passes the frames through unchanged.
            out = torch.zeros((1, patch_size, patch_size, 3), dtype=torch.float32)

        rectify_map = {'frames': frames, 'count': count, 'entries': entries}
        return (out, rectify_map, count)


class VRFaceUnrectify:
    """
    Re-impose VR distortion after swapping. Takes the swapped patch batch and the
    VR_RECTIFY_MAP from VR Face Rectify, projects each swapped patch back into its
    exact original frame/eye/position, and composites onto the original frames.
    """
    DESCRIPTION = __doc__
    CATEGORY = NODE_CATEGORY
    NAME = 'VR Face Unrectify'

    RETURN_NAMES = ('image',)
    RETURN_TYPES = ('IMAGE',)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            'required': {
                'patches': ('IMAGE', {'tooltip': 'Swapped patch batch from ReActor'}),
                'rectify_map': ('VR_RECTIFY_MAP', {'tooltip': 'From VR Face Rectify'}),
            },
        }

    FUNCTION = 'run'

    def run(self, patches, rectify_map):
        device = _torch_device()
        _log_backend('VR Face Unrectify', device)
        frames = [f.copy() for f in rectify_map['frames']]
        entries = rectify_map['entries']

        if len(entries) > patches.shape[0]:
            raise ValueError(
                f'VR Face Unrectify: {len(entries)} patches expected from the map but '
                f'only {patches.shape[0]} arrived; the swapper must preserve batch order/count')

        for e in entries:
            frame = frames[e['frame_idx']]
            eye = e['eye']
            eye_w, eye_h = e['eye_w'], e['eye_h']
            # eye_img is a VIEW into frame for every layout, so compositing into it
            # in place updates the frame directly -- no full-eye write-back needed.
            if eye == 'M':
                eye_img = frame
            elif eye in ('L', 'R'):
                left, right = vr.split_sbs(frame)
                eye_img = left if eye == 'L' else right
            else:  # 'T' / 'B' top|bottom halves
                eye_img = frame[:eye_h] if eye == 'T' else frame[eye_h:]

            swapped = _img_to_np(patches[e['patch_index']])
            alpha = np.full(swapped.shape[:2] + (1,), 255, dtype=np.uint8)
            rgba = np.concatenate([swapped, alpha], axis=2)

            # Windowed un-rectify: reproject only the eye sub-region the patch
            # covers, so cost tracks the face footprint, not the frame resolution.
            win, x0, y0 = _unrectify_window(rgba, e, eye_w, device)

            if win is None:  # patch projects nowhere in the eye -> nothing to do
                continue
            y1, x1 = y0 + win.shape[0], x0 + win.shape[1]
            eye_img[y0:y1, x0:x1] = vr.alpha_over(eye_img[y0:y1, x0:x1], win)

        print(f'VR Face Unrectify: composited patches: {len(entries)}')
        out = torch.cat([_np_to_img(f) for f in frames], dim=0)
        return (out,)
