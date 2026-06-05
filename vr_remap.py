"""VR projection math for the face-swap distortion fix (self-contained: numpy+cv2).

Ported verbatim from the validated vr-research/insert-character/equirect.py
(round-trip proven loss-free at 0.19/255). Supports equirect (DOME/SBS, linear in
angle) and equidistant fisheye (r = f*theta). The whole point: rectify a face
region to a flat rectilinear patch so a face swapper sees a NORMAL face, then
un-rectify to re-impose the exact local distortion.

This module has ZERO ComfyUI imports on purpose: it is unit-testable offline and
the node wrappers in vr_face.py are thin adapters over it.
"""
import numpy as np
import cv2


def split_sbs(img):
    w = img.shape[1]
    return img[:, :w // 2], img[:, w // 2:]


def merge_sbs(left, right):
    return np.concatenate([left, right], axis=1)


def alpha_over(dst_rgb, src_rgba):
    a = src_rgba[..., 3:4].astype(np.float32) / 255.0
    out = dst_rgb.astype(np.float32) * (1 - a) + src_rgba[..., :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _rot_yaw_pitch(yaw_deg, pitch_deg):
    y, p = np.deg2rad(yaw_deg), np.deg2rad(pitch_deg)
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return Ry @ Rx


def detect_fisheye_circle(eye, thresh=10):
    g = eye if eye.ndim == 2 else eye.max(axis=2)
    ys, xs = np.where(g > thresh)
    cx = (xs.min() + xs.max()) / 2.0
    cy = (ys.min() + ys.max()) / 2.0
    radius = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0
    return cx, cy, radius


# ---- equirect ----
def e2p(eye, fov_deg, yaw_deg, pitch_deg, out_w, out_h, h_fov=180.0, v_fov=180.0,
        interp=cv2.INTER_CUBIC):
    eye_h, eye_w = eye.shape[:2]
    f = (out_w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    j, i = np.meshgrid(np.arange(out_w), np.arange(out_h))
    cam = np.stack([(j - out_w / 2.0 + 0.5), -(i - out_h / 2.0 + 0.5),
                    np.full(j.shape, f)], axis=-1)
    cam /= np.linalg.norm(cam, axis=-1, keepdims=True)
    dirs = cam @ _rot_yaw_pitch(yaw_deg, pitch_deg).T
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y, -1, 1))
    half_h, half_v = np.deg2rad(h_fov) / 2, np.deg2rad(v_fov) / 2
    map_x = ((lon + half_h) / (2 * half_h) * (eye_w - 1)).astype(np.float32)
    map_y = ((half_v - lat) / (2 * half_v) * (eye_h - 1)).astype(np.float32)
    return cv2.remap(eye, map_x, map_y, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def p2e(patch, fov_deg, yaw_deg, pitch_deg, eye_w, eye_h, h_fov=180.0, v_fov=180.0,
        interp=cv2.INTER_CUBIC):
    p_h, p_w = patch.shape[:2]
    f = (p_w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    jx, iy = np.meshgrid(np.arange(eye_w), np.arange(eye_h))
    half_h, half_v = np.deg2rad(h_fov) / 2, np.deg2rad(v_fov) / 2
    lon = jx / (eye_w - 1) * (2 * half_h) - half_h
    lat = half_v - iy / (eye_h - 1) * (2 * half_v)
    cl = np.cos(lat)
    world = np.stack([cl * np.sin(lon), np.sin(lat), cl * np.cos(lon)], axis=-1)
    cam = world @ _rot_yaw_pitch(yaw_deg, pitch_deg)
    cz = cam[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        px = cam[..., 0] / cz * f + p_w / 2.0 - 0.5
        py = -cam[..., 1] / cz * f + p_h / 2.0 - 0.5
    inside = (cz > 1e-6) & (px >= 0) & (px <= p_w - 1) & (py >= 0) & (py <= p_h - 1)
    out = cv2.remap(patch, np.where(inside, px, -1).astype(np.float32),
                    np.where(inside, py, -1).astype(np.float32), interp,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    out[~inside] = 0
    return out


# ---- fisheye (equidistant) ----
def fisheye_e2p(eye, fov_deg, yaw_deg, pitch_deg, out_w, out_h, fisheye_fov_deg=135.0,
                cx=None, cy=None, radius=None, interp=cv2.INTER_CUBIC):
    if cx is None:
        cx, cy, radius = detect_fisheye_circle(eye)
    theta_max = np.deg2rad(fisheye_fov_deg) / 2.0
    f = (out_w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    j, i = np.meshgrid(np.arange(out_w), np.arange(out_h))
    cam = np.stack([(j - out_w / 2.0 + 0.5), -(i - out_h / 2.0 + 0.5),
                    np.full(j.shape, f)], axis=-1)
    cam /= np.linalg.norm(cam, axis=-1, keepdims=True)
    dirs = cam @ _rot_yaw_pitch(yaw_deg, pitch_deg).T
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta = np.arccos(np.clip(z, -1, 1))
    phi = np.arctan2(y, x)
    r = (theta / theta_max) * radius
    map_x = (cx + r * np.cos(phi)).astype(np.float32)
    map_y = (cy - r * np.sin(phi)).astype(np.float32)
    return cv2.remap(eye, map_x, map_y, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def fisheye_p2e(patch, fov_deg, yaw_deg, pitch_deg, eye_w, eye_h, fisheye_fov_deg=135.0,
                cx=None, cy=None, radius=None, interp=cv2.INTER_CUBIC):
    if cx is None:
        cx, cy, radius = eye_w / 2.0, eye_h / 2.0, min(eye_w, eye_h) / 2.0
    theta_max = np.deg2rad(fisheye_fov_deg) / 2.0
    p_h, p_w = patch.shape[:2]
    f = (p_w / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    jx, iy = np.meshgrid(np.arange(eye_w), np.arange(eye_h))
    nx = (jx - cx) / radius
    ny = -(iy - cy) / radius
    rr = np.sqrt(nx * nx + ny * ny)
    theta = rr * theta_max
    phi = np.arctan2(ny, nx)
    st = np.sin(theta)
    world = np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)
    cam = world @ _rot_yaw_pitch(yaw_deg, pitch_deg)
    cz = cam[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        px = cam[..., 0] / cz * f + p_w / 2.0 - 0.5
        py = -cam[..., 1] / cz * f + p_h / 2.0 - 0.5
    inside = (rr <= 1.0) & (cz > 1e-6) & (px >= 0) & (px <= p_w - 1) & (py >= 0) & (py <= p_h - 1)
    out = cv2.remap(patch, np.where(inside, px, -1).astype(np.float32),
                    np.where(inside, py, -1).astype(np.float32), interp,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    out[~inside] = 0
    return out


def pixel_to_yaw_pitch(px, py, eye_w, eye_h, projection='equirect',
                       fisheye_fov_deg=135.0, cx=None, cy=None, radius=None):
    """Face-detection bbox center (pixels) -> (yaw, pitch) to center a rectify patch."""
    if projection == 'fisheye':
        if cx is None:
            cx, cy, radius = eye_w / 2.0, eye_h / 2.0, min(eye_w, eye_h) / 2.0
        theta_max = np.deg2rad(fisheye_fov_deg) / 2.0
        nx = (px - cx) / radius
        ny = -(py - cy) / radius
        r = np.hypot(nx, ny)
        theta = r * theta_max
        phi = np.arctan2(ny, nx)
        d = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    else:
        lon = (px / (eye_w - 1) - 0.5) * np.pi      # 180 deg span
        lat = (0.5 - py / (eye_h - 1)) * np.pi
        d = np.array([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)])
    yaw = np.degrees(np.arctan2(d[0], d[2]))
    pitch = -np.degrees(np.arcsin(np.clip(d[1], -1, 1)))
    return float(yaw), float(pitch)
