"""Colour transforms and exposure masks, ported from DiffHDR.

Derived from utils/color_utils.py, infer_video.py and infer_long_video.py of
https://github.com/Eyeline-Labs/DiffHDR, Copyright Eyeline Labs, licensed under Apache-2.0.
Modifications: functions extracted and restructured for ComfyUI tensors ([T, H, W, C]),
unused code removed. Results are numerically identical to the reference (verified on
synthetic frames: zero difference in Log values, zero mismatched mask pixels).
"""
import math

import torch
import torch.nn.functional as F

LOG_MAX = 65536.0
LOG_GAMMA = 2.2


# --------------------------------------------------------------------------- transfer curves
def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.pow(1.0 / 2.4) - 0.055)


def lin_to_log(x: torch.Tensor, max_val: float = LOG_MAX, gamma: float = LOG_GAMMA) -> torch.Tensor:
    """DiffHDR 'Log-Gamma' encoding: linear [0, 65536] -> [0, 1]. Linear 1.0 maps to ~0.348."""
    x = torch.clamp(x, min=0.0, max=max_val)
    x = torch.log(gamma * x + 1.0) / math.log(gamma * max_val + 1.0)
    return torch.pow(x, 1.0 / gamma)


def log_to_lin(x: torch.Tensor, max_val: float = LOG_MAX, gamma: float = LOG_GAMMA) -> torch.Tensor:
    x = torch.clamp_min(x, 0.0)
    x = torch.pow(x, gamma)
    x = torch.exp(x * math.log(gamma * max_val + 1.0)) - 1.0
    return x / gamma


# --------------------------------------------------------------------------- masks
def _luma709(rgb_lin_chw: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb_lin_chw[0] + 0.7152 * rgb_lin_chw[1] + 0.0722 * rgb_lin_chw[2]


def _smoothstep(x, edge0, edge1):
    t = ((x - edge0) / (edge1 - edge0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _dilate(m, k):
    return F.max_pool2d(m[None, None], k, 1, k // 2)[0, 0]


def _erode(m, k):
    return -F.max_pool2d(-m[None, None], k, 1, k // 2)[0, 0]


@torch.no_grad()
def exposure_masks_from_srgb(rgb_srgb_chw, over_thr=0.95, under_thr=0.01, softness=0.02, clip_eps=0.02):
    """Soft over/under-exposure masks [H, W] from one sRGB frame [3, H, W] in [0, 1]."""
    dev = rgb_srgb_chw.device
    rgb_srgb_chw = rgb_srgb_chw.clamp(0, 1)
    y = _luma709(srgb_to_linear(rgb_srgb_chw))

    over_l = _smoothstep(
        y,
        srgb_to_linear(torch.tensor(over_thr - softness, device=dev)),
        srgb_to_linear(torch.tensor(over_thr + softness, device=dev)),
    )
    near1 = (rgb_srgb_chw >= (1.0 - clip_eps)).float()
    ch_clip = (near1.sum(dim=0) >= 2).float()
    over = (0.7 * over_l + 0.3 * ch_clip).clamp(0, 1)

    under = 1.0 - _smoothstep(
        y,
        srgb_to_linear(torch.tensor(max(0.0, under_thr - softness), device=dev)),
        srgb_to_linear(torch.tensor(under_thr + softness, device=dev)),
    )
    all_low = (rgb_srgb_chw <= under_thr).all(dim=0).float()
    under = (0.7 * under + 0.3 * all_low).clamp(0, 1)

    # 3x3 closing (dilate then erode)
    over = _erode(_dilate(over, 3), 3)
    under = _erode(_dilate(under, 3), 3)
    return over.clamp(0, 1), under.clamp(0, 1)


@torch.no_grad()
def stabilize_soft_mask(soft, prev_ema, alpha=0.7, k_smooth=9, k_open_close=9):
    ema = soft if prev_ema is None else (alpha * soft + (1 - alpha) * prev_ema)
    ema_state = ema.clone()
    if k_smooth > 1:
        ema = F.avg_pool2d(ema[None, None], k_smooth, 1, k_smooth // 2)[0, 0]
    if k_open_close > 1:
        ema = _dilate(_erode(ema, k_open_close), k_open_close)  # opening
        ema = _erode(_dilate(ema, k_open_close), k_open_close)  # closing
    return ema.clamp(0, 1), ema_state


@torch.no_grad()
def prepare_window(srgb_thwc, over_thr=0.95, under_thr=0.01, use_under_mask=False, bin_thr=0.2):
    """DiffHDR `prepare_video_and_masks` for one window.

    srgb_thwc: [T, H, W, 3] sRGB in [0, 1] (already at model resolution).
    Returns (log_video [T,H,W,3] in [0,1], vace_mask [T,H,W], over_bin [T,H,W], under_bin [T,H,W]).
    The temporal EMA restarts at the first frame of each window, as in the reference code.
    """
    video = srgb_thwc.permute(0, 3, 1, 2).contiguous().float().clone()  # [T,3,H,W]
    vace, overs, unders = [], [], []
    prev_o = prev_u = None
    for i in range(video.shape[0]):
        o_soft, u_soft = exposure_masks_from_srgb(video[i], over_thr, under_thr)
        o_soft, prev_o = stabilize_soft_mask(o_soft, prev_o)
        u_soft, prev_u = stabilize_soft_mask(u_soft, prev_u)
        o_bin = (o_soft > bin_thr).float()
        u_bin = (u_soft > bin_thr).float()
        if use_under_mask:
            comb = torch.logical_or(o_bin.bool(), u_bin.bool()).float()
            video[i][u_bin.unsqueeze(0).expand(3, -1, -1) == 1.0] = 0.5
        else:
            comb = o_bin
        vace.append(comb)
        overs.append(o_bin)
        unders.append(u_bin)
    log_video = lin_to_log(srgb_to_linear(video)).permute(0, 2, 3, 1).contiguous()
    return log_video, torch.stack(vace), torch.stack(overs), torch.stack(unders)


@torch.no_grad()
def prepare_reference(ref_srgb_hwc, first_mask_hw=None, ev_boost=5.0):
    """DiffHDR `load_reference_image`: masked, linearised, +EV, Log-encoded. Returns [1,H,W,3]."""
    ref = ref_srgb_hwc.permute(2, 0, 1).float()
    if first_mask_hw is not None:
        m = first_mask_hw.unsqueeze(0).to(ref)
        ref = ref * m + (1.0 - m)
    ref = srgb_to_linear(ref) * (2.0 ** ev_boost)
    return lin_to_log(ref).permute(1, 2, 0).unsqueeze(0).contiguous()


# --------------------------------------------------------------------------- resizing
def crop_and_resize(images_thwc: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Centre-crop to the target aspect ratio, then resize (DiffHDR `_crop_and_resize`)."""
    t, ih, iw, c = images_thwc.shape
    x = images_thwc
    if ih / iw < height / width:
        cw = int(ih / height * width)
        left = (iw - cw) // 2
        x = x[:, :, left:left + cw]
    else:
        ch = int(iw / width * height)
        top = (ih - ch) // 2
        x = x[:, top:top + ch]
    if x.shape[1] == height and x.shape[2] == width:
        return x[..., :3].float()
    x = x[..., :3].permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    return x.clamp(0, 1).permute(0, 2, 3, 1).contiguous()


def blend_ramp(length, left_bound, right_bound, border_width):
    """DiffHDR `build_1d_mask`: linear cross-fade weights for the sliding windows."""
    x = torch.ones((length,), dtype=torch.float32)
    if border_width <= 0:
        return x
    # A window that is not the first always has more frames than border_width
    # (window_size - stride), so these slices are always full length.
    ramp = (torch.arange(border_width, dtype=torch.float32) + 0.5) / border_width
    if not left_bound:
        x[:border_width] = ramp
    if not right_bound:
        x[-border_width:] = torch.flip(ramp, dims=(0,))
    return x
