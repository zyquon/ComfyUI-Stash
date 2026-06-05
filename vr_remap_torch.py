"""GPU/torch port of the hot vr_remap projections (numpy in/out, torch inside).

Prototype for moving the VR rectify/un-rectify geometry onto the GPU. Each
function mirrors its vr_remap.py counterpart's signature and still takes/returns
numpy uint8 at the boundary, but the coordinate math + the cubic resample run as
torch ops, so they execute on CUDA inside ComfyUI and on CPU torch in offline
tests -- one implementation, device chosen at call time.

The single heavy primitive, cv2.remap(INTER_CUBIC), becomes
F.grid_sample(mode='bicubic', align_corners=True, padding_mode='zeros'):
  - align_corners=True matches our (size-1) pixel normalization;
  - bicubic ~ INTER_CUBIC (NOT bit-identical -> parity is "within tolerance");
  - zeros padding matches BORDER_CONSTANT 0.

The cheap, constant-cost forward maps + footprint window are reused verbatim from
vr_remap (patch-sized, not worth porting); only the resolution-scaling inverse
work runs on device. This keeps the validated window bbox identical to the CPU path.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

try:
    from . import vr_remap as vr
except ImportError:  # standalone import (tests, offline checks)
    import vr_remap as vr


def _device(device=None):
    """Resolve the compute device: explicit arg, else CUDA if present, else CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def describe_device(device=None):
    """Human-readable device tag for logs, e.g. 'cuda:0 (NVIDIA RTX 4090)' or 'cpu'."""
    dev = _device(device)
    if dev.type == 'cuda':
        idx = 0 if dev.index is None else dev.index
        return f'cuda:{idx} ({torch.cuda.get_device_name(idx)})'
    return dev.type


def _rot(yaw_deg, pitch_deg, device, dtype):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = torch.tensor([[math.cos(y), 0, math.sin(y)], [0, 1, 0],
                       [-math.sin(y), 0, math.cos(y)]], device=device, dtype=dtype)
    rx = torch.tensor([[1, 0, 0], [0, math.cos(p), -math.sin(p)],
                       [0, math.sin(p), math.cos(p)]], device=device, dtype=dtype)
    return ry @ rx


def _grid(w, h, x0, y0, device, dtype):
    """Pixel-index grids (j=col, i=row) for an output window, like np.meshgrid('xy')."""
    xs = torch.arange(x0, x0 + w, device=device, dtype=dtype)
    ys = torch.arange(y0, y0 + h, device=device, dtype=dtype)
    return xs.view(1, -1).expand(h, w), ys.view(-1, 1).expand(h, w)


def _sample(src_np, map_x, map_y, in_w, in_h, device, dtype):
    """cv2.remap analog via grid_sample. src_np uint8 (H,W,C) -> float (Hout,Wout,C)."""
    src = (torch.from_numpy(np.ascontiguousarray(src_np)).to(device=device, dtype=dtype)
           / 255.0).permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    gx = map_x / (in_w - 1) * 2.0 - 1.0
    gy = map_y / (in_h - 1) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # (1,Hout,Wout,2)
    out = F.grid_sample(src, grid, mode='bicubic', padding_mode='zeros',
                        align_corners=True)
    return out.squeeze(0).permute(1, 2, 0)  # (Hout,Wout,C) float


def _to_uint8(t):
    return t.clamp(0, 1).mul(255.0).round().to(torch.uint8).cpu().numpy()


# ---- equirect ----
def e2p(eye, fov_deg, yaw_deg, pitch_deg, out_w, out_h, h_fov=180.0, v_fov=180.0,
        device=None):
    dev, dtype = _device(device), torch.float32
    eye_h, eye_w = eye.shape[:2]
    f = (out_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    j, i = _grid(out_w, out_h, 0, 0, dev, dtype)
    cam = torch.stack([(j - out_w / 2.0 + 0.5), -(i - out_h / 2.0 + 0.5),
                       torch.full_like(j, f)], dim=-1)
    cam = cam / cam.norm(dim=-1, keepdim=True)
    dirs = cam @ _rot(yaw_deg, pitch_deg, dev, dtype).T
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = torch.atan2(x, z)
    lat = torch.asin(y.clamp(-1, 1))
    half_h, half_v = math.radians(h_fov) / 2, math.radians(v_fov) / 2
    map_x = (lon + half_h) / (2 * half_h) * (eye_w - 1)
    map_y = (half_v - lat) / (2 * half_v) * (eye_h - 1)
    return _to_uint8(_sample(eye, map_x, map_y, eye_w, eye_h, dev, dtype))


def p2e_window(patch, fov_deg, yaw_deg, pitch_deg, eye_w, eye_h, h_fov=180.0,
               v_fov=180.0, device=None):
    map_x, map_y = vr._e2p_maps(eye_w, eye_h, fov_deg, yaw_deg, pitch_deg,
                                patch.shape[1], patch.shape[0], h_fov, v_fov)
    win = vr._footprint_window(map_x, map_y, eye_w, eye_h)
    if win is None:
        return None, 0, 0
    x0, y0, x1, y1 = win
    dev, dtype = _device(device), torch.float32
    p_h, p_w = patch.shape[:2]
    f = (p_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    jx, iy = _grid(x1 - x0, y1 - y0, x0, y0, dev, dtype)
    half_h, half_v = math.radians(h_fov) / 2, math.radians(v_fov) / 2
    lon = jx / (eye_w - 1) * (2 * half_h) - half_h
    lat = half_v - iy / (eye_h - 1) * (2 * half_v)
    cl = torch.cos(lat)
    world = torch.stack([cl * torch.sin(lon), torch.sin(lat), cl * torch.cos(lon)], dim=-1)
    cam = world @ _rot(yaw_deg, pitch_deg, dev, dtype)
    cz = cam[..., 2]
    px = cam[..., 0] / cz * f + p_w / 2.0 - 0.5
    py = -cam[..., 1] / cz * f + p_h / 2.0 - 0.5
    inside = (cz > 1e-6) & (px >= 0) & (px <= p_w - 1) & (py >= 0) & (py <= p_h - 1)
    px = torch.where(inside, px, torch.zeros_like(px))
    py = torch.where(inside, py, torch.zeros_like(py))
    out = _sample(patch, px, py, p_w, p_h, dev, dtype)
    out[~inside] = 0
    return _to_uint8(out), x0, y0


# ---- fisheye (equidistant) ----
def fisheye_e2p(eye, fov_deg, yaw_deg, pitch_deg, out_w, out_h, fisheye_fov_deg=135.0,
                cx=None, cy=None, radius=None, device=None):
    if cx is None:
        cx, cy, radius = vr.detect_fisheye_circle(eye)
    dev, dtype = _device(device), torch.float32
    eye_h, eye_w = eye.shape[:2]
    theta_max = math.radians(fisheye_fov_deg) / 2.0
    f = (out_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    j, i = _grid(out_w, out_h, 0, 0, dev, dtype)
    cam = torch.stack([(j - out_w / 2.0 + 0.5), -(i - out_h / 2.0 + 0.5),
                       torch.full_like(j, f)], dim=-1)
    cam = cam / cam.norm(dim=-1, keepdim=True)
    dirs = cam @ _rot(yaw_deg, pitch_deg, dev, dtype).T
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta = torch.arccos(z.clamp(-1, 1))
    phi = torch.atan2(y, x)
    r = (theta / theta_max) * radius
    map_x = cx + r * torch.cos(phi)
    map_y = cy - r * torch.sin(phi)
    return _to_uint8(_sample(eye, map_x, map_y, eye_w, eye_h, dev, dtype))


def fisheye_p2e_window(patch, fov_deg, yaw_deg, pitch_deg, eye_w, eye_h,
                       fisheye_fov_deg=135.0, cx=None, cy=None, radius=None, device=None):
    if cx is None:
        cx, cy, radius = eye_w / 2.0, eye_h / 2.0, min(eye_w, eye_h) / 2.0
    map_x, map_y = vr._fisheye_e2p_maps(fov_deg, yaw_deg, pitch_deg, patch.shape[1],
                                        patch.shape[0], fisheye_fov_deg, cx, cy, radius)
    win = vr._footprint_window(map_x, map_y, eye_w, eye_h)
    if win is None:
        return None, 0, 0
    x0, y0, x1, y1 = win
    dev, dtype = _device(device), torch.float32
    theta_max = math.radians(fisheye_fov_deg) / 2.0
    p_h, p_w = patch.shape[:2]
    f = (p_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    jx, iy = _grid(x1 - x0, y1 - y0, x0, y0, dev, dtype)
    nx = (jx - cx) / radius
    ny = -(iy - cy) / radius
    rr = torch.sqrt(nx * nx + ny * ny)
    theta = rr * theta_max
    phi = torch.atan2(ny, nx)
    st = torch.sin(theta)
    world = torch.stack([st * torch.cos(phi), st * torch.sin(phi), torch.cos(theta)], dim=-1)
    cam = world @ _rot(yaw_deg, pitch_deg, dev, dtype)
    cz = cam[..., 2]
    px = cam[..., 0] / cz * f + p_w / 2.0 - 0.5
    py = -cam[..., 1] / cz * f + p_h / 2.0 - 0.5
    inside = (rr <= 1.0) & (cz > 1e-6) & (px >= 0) & (px <= p_w - 1) & (py >= 0) & (py <= p_h - 1)
    px = torch.where(inside, px, torch.zeros_like(px))
    py = torch.where(inside, py, torch.zeros_like(py))
    out = _sample(patch, px, py, p_w, p_h, dev, dtype)
    out[~inside] = 0
    return _to_uint8(out), x0, y0
