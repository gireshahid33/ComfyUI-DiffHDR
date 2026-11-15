"""ComfyUI nodes for DiffHDR (LDR video -> linear HDR) on Wan2.1-VACE-14B.

Reference: https://github.com/Eyeline-Labs/DiffHDR (Apache-2.0). The DiffHDR pipeline is
standard Wan VACE conditioning plus (1) a Log-Gamma colour transform, (2) exposure masks,
(3) a LoRA on the VACE blocks only. These nodes reproduce those pieces on top of ComfyUI's
own Wan model loading and sampling.
"""
import logging
import os

import numpy as np
import torch

import comfy.latent_formats
import comfy.lora
import comfy.model_management
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import node_helpers
import nodes

from . import diffhdr_color as dc

log = logging.getLogger("DiffHDR")
CATEGORY = "DiffHDR"
TRANSFERS = ["sRGB (display-referred LDR)", "linear Rec.709 (e.g. EXR plate)"]
DECODE_MODES = ["fast (fp32, 768px tiles)", "exact (fp32, full frame, slower)"]


def _to_srgb(images, transfer):
    if transfer.startswith("linear"):
        return dc.linear_to_srgb(images.float())
    return images.float().clamp(0, 1)


# =========================================================================== loaders
class DiffHDRLoraLoader:
    """Loads the DiffHDR LoRA (PEFT keys `vace_blocks.N.*.lora_A.default.weight`) onto a Wan VACE model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01,
                                   "tooltip": "DiffHDR uses 1.0 (no rank scaling)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model, lora_name, strength):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        remapped = {}
        for k, v in sd.items():
            k2 = k.replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")
            if not k2.startswith("diffusion_model."):
                k2 = "diffusion_model." + k2
            remapped[k2] = v
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        loaded = comfy.lora.load_lora(remapped, key_map)
        expected = sum(1 for k in remapped if ".lora_B." in k)
        if len(loaded) == 0:
            raise RuntimeError(
                "DiffHDR LoRA matched no layers. The model must be Wan2.1-VACE-14B "
                "(e.g. wan2.1_vace_14B_fp16.safetensors), not a plain Wan T2V/I2V model.")
        if len(loaded) != expected:
            log.warning("DiffHDR LoRA: patched %d of %d layers", len(loaded), expected)
        else:
            log.info("DiffHDR LoRA: patched %d VACE layers", len(loaded))
        m = model.clone()
        m.add_patches(loaded, strength)
        return (m,)


class DiffHDRVAELoader:
    """Loads a VAE at a fixed precision. DiffHDR runs the Wan VAE in fp32: Log-space rounding errors
    are amplified when converted back to linear (bf16 measured ~0.9% mean / 4.9% p99 error in
    highlights vs ~0.3% / 2.7% for the tiled fp32 path used by the sampler)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae_name": (folder_paths.get_filename_list("vae"),),
            "precision": (["fp32", "bf16", "fp16", "auto"], {"default": "fp32"}),
        }}

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, vae_name, precision):
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, "auto": None}[precision]
        try:  # newer ComfyUI passes file metadata to the VAE
            sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
            vae = comfy.sd.VAE(sd=sd, metadata=metadata, dtype=dtype)
        except TypeError:
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path), dtype=dtype)
        if hasattr(vae, "throw_exception_if_invalid"):
            vae.throw_exception_if_invalid()
        return (vae,)


# =========================================================================== core
def _prepare_on_gpu(frames, transfer, width, height, over_thr, under_thr, use_under):
    """Resize + masks + Log on the GPU (the 9x9 morphology takes ~50 s per window on the CPU)."""
    dev = comfy.model_management.get_torch_device()
    srgb = dc.crop_and_resize(_to_srgb(frames.to(dev), transfer), width, height)
    log_video, vace_mask, over, under = dc.prepare_window(srgb, over_thr, under_thr, use_under)
    return srgb.cpu(), log_video.cpu(), vace_mask.cpu()


def _free_vram_for_vae():
    """The fp32 Wan VAE runs close to the 16 GB limit; anything else resident (the 14B DiT, the
    text encoder) pushes it into Windows shared memory, which is 10-20x slower. Unload everything
    first; the DiT reloads from RAM in seconds for the next window."""
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()


def _vram_note():
    """Free VRAM on the device (includes other apps such as Nuke) and this process's peak use."""
    dev = comfy.model_management.get_torch_device()
    if dev.type != "cuda":
        return ""
    free, total = torch.cuda.mem_get_info(dev)
    peak = torch.cuda.max_memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    return f"[GPU free {free / 2**30:.1f}/{total / 2**30:.1f} GB, peak since last {peak / 2**30:.1f} GB]"


# VAE tiling (fp32). Untiled, a 720p decode needs >16 GB and spills to shared memory (189 s for
# 33 frames). The reference infer_video.py also runs its VAE tiled. Measured vs untiled fp32:
#   768 px / 192 overlap  -> 0.26% mean, 2.7% p99 error in linear (default, lowest peak memory)
#   full-frame tile       -> bit-identical, ~90 s decode
ENC_TILE = dict(tile_x=768, tile_y=768, overlap=192, tile_t=256, overlap_t=8)  # pixels / frames


def _vace_window(model, positive, negative, vae, log_video, vace_mask, ref_log,
                 width, height, steps, seed, cfg, sampler_name, scheduler, strength, decode_mode="fast"):
    """One DiffHDR window: identical to ComfyUI WanVaceToVideo + KSampler + TrimVideoLatent + VAEDecode."""
    import time
    t0 = time.perf_counter()
    _free_vram_for_vae()
    v0 = _vram_note()
    length = log_video.shape[0]
    latent_length = ((length - 1) // 4) + 1

    m = vace_mask.unsqueeze(-1).float()
    cv = log_video.float() - 0.5
    inactive = vae.encode_tiled((cv * (1 - m)) + 0.5, **ENC_TILE)
    reactive = vae.encode_tiled((cv * m) + 0.5, **ENC_TILE)
    control_latent = torch.cat((inactive, reactive), dim=1)

    ref_lat = None
    if ref_log is not None:
        ref_lat = vae.encode_tiled(ref_log[:, :, :, :3], **ENC_TILE)
        ref_lat = torch.cat([ref_lat, comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_lat))], dim=1)
        control_latent = torch.cat((ref_lat, control_latent), dim=2)

    s = 8
    hm, wm = height // s, width // s
    mask = m.view(length, hm, s, wm, s).permute(2, 4, 0, 1, 3).reshape(s * s, length, hm, wm)
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, hm, wm), mode="nearest-exact").squeeze(0)

    trim = 0
    if ref_lat is not None:
        mask = torch.cat((torch.zeros_like(mask[:, :ref_lat.shape[2]]), mask), dim=1)
        latent_length += ref_lat.shape[2]
        trim = ref_lat.shape[2]
    mask = mask.unsqueeze(0)

    vals = {"vace_frames": [control_latent], "vace_mask": [mask], "vace_strength": [strength]}
    pos = node_helpers.conditioning_set_values(positive, vals, append=True)
    neg = node_helpers.conditioning_set_values(negative, vals, append=True)

    latent = torch.zeros([1, 16, latent_length, height // 8, width // 8],
                         device=comfy.model_management.intermediate_device())
    t1 = time.perf_counter()
    v1 = _vram_note()
    out = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, pos, neg,
                                {"samples": latent}, denoise=1.0)[0]
    t2 = time.perf_counter()
    samples = out["samples"][:, :, trim:]
    _free_vram_for_vae()
    v2 = _vram_note()
    if decode_mode.startswith("exact"):
        tile = max(width, height) // 8
        images = vae.decode_tiled(samples, tile_x=tile, tile_y=tile, overlap=tile // 4, tile_t=64, overlap_t=1)
    else:
        images = vae.decode_tiled(samples, tile_x=96, tile_y=96, overlap=24, tile_t=64, overlap_t=1)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    t3 = time.perf_counter()
    log.info("DiffHDR window timing: encode %.0fs %s | sample %.0fs %s | decode %.0fs %s %s",
             t1 - t0, v0, t2 - t1, v1, t3 - t2, v2, _vram_note())
    return images[:length].float().cpu().clamp_min(0.0)  # Log space, [L, H, W, 3]


class DiffHDRVideoSampler:
    """LDR video -> linear HDR video, any length.

    Runs DiffHDR's sliding window (infer_long_video.py): windows of `window_size` frames every
    `window_stride` frames, linear cross-fades in the overlaps, and (optionally) each window
    conditioned on the previous window's output frame for temporal consistency.
    Clips up to `window_size` frames run as a single window (infer_video.py)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Wan2.1-VACE-14B + DiffHDR LoRA, through ModelSamplingSD3 shift 5."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE", {"tooltip": "Wan 2.1 VAE, ideally loaded in fp32."}),
                "images": ("IMAGE", {"tooltip": "LDR frames. Centre-cropped and resized to width x height."}),
                "input_transfer": (TRANSFERS, {"default": TRANSFERS[0]}),
                "width": ("INT", {"default": 1280, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 720, "min": 256, "max": 2048, "step": 16}),
                "window_size": ("INT", {"default": 33, "min": 5, "max": 129, "step": 4,
                                        "tooltip": "DiffHDR was trained on 33 frames. Must be 4n+1."}),
                "window_stride": ("INT", {"default": 16, "min": 1, "max": 128}),
                "use_prev_window_reference": ("BOOLEAN", {"default": True}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100,
                                  "tooltip": "Authors' demos use 10; the script default is 50."}),
                "seed": ("INT", {"default": 10, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                                  "tooltip": "DiffHDR uses 1.0 (negative prompt unused, one pass per step)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "vace_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "over_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.005,
                                             "tooltip": "sRGB luma above which pixels are treated as clipped and regenerated."}),
                "under_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.2, "step": 0.005}),
                "use_under_exposure_mask": ("BOOLEAN", {"default": False,
                                                        "tooltip": "Also regenerate crushed shadows."}),
                "decode": (DECODE_MODES, {"default": DECODE_MODES[0],
                                          "tooltip": "fast: fp32 in 768 px tiles (~20 s/window, ~0.3% error). "
                                                     "exact: fp32 full frame (~90 s/window on 16 GB)."}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Optional sRGB image that guides what fills the blown-out areas."}),
                "reference_ev": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 12.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("hdr_linear", "exposure_mask", "ldr_resized")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, positive, negative, vae, images, input_transfer, width, height, window_size,
            window_stride, use_prev_window_reference, steps, seed, cfg, sampler_name, scheduler,
            vace_strength, over_threshold, under_threshold, use_under_exposure_mask,
            decode=DECODE_MODES[0], reference_image=None, reference_ev=5.0):
        if (window_size - 1) % 4 != 0:
            raise ValueError(f"window_size must be 4n+1 (e.g. 33), got {window_size}")
        if window_stride >= window_size:
            raise ValueError("window_stride must be smaller than window_size")
        width, height = (width // 16) * 16, (height // 16) * 16
        total = images.shape[0]
        border = window_size - window_stride

        user_ref = None
        if reference_image is not None:
            user_ref = dc.crop_and_resize(_to_srgb(reference_image[:1], TRANSFERS[0]), width, height)[0]

        starts = []
        for t in range(0, total, window_stride):
            if t - window_stride >= 0 and t - window_stride + window_size >= total:
                continue
            starts.append(t)

        value = torch.zeros((total, height, width, 3), dtype=torch.float32)
        weight = torch.zeros((total,), dtype=torch.float32)
        out_mask = torch.zeros((total, height, width), dtype=torch.float32)
        ldr = torch.zeros((total, height, width, 3), dtype=torch.float32)
        prev_ref = None
        pbar = comfy.utils.ProgressBar(len(starts))

        for wi, t in enumerate(starts):
            comfy.model_management.throw_exception_if_processing_interrupted()
            t_end = min(t + window_size, total)
            n_real = t_end - t
            idx = list(range(t, t_end)) + [t_end - 1] * (window_size - n_real)  # pad by repeating last frame
            srgb, log_video, vace_mask = _prepare_on_gpu(images[idx], input_transfer, width, height,
                                                         over_threshold, under_threshold, use_under_exposure_mask)

            if use_prev_window_reference and prev_ref is not None:
                ref = prev_ref
            elif user_ref is not None:
                ref = dc.prepare_reference(user_ref, vace_mask[0], reference_ev)
            else:
                ref = None

            log.info("DiffHDR window %d/%d: frames %d-%d%s%s", wi + 1, len(starts), t, t_end - 1,
                     f" (padded to {window_size})" if n_real < window_size else "",
                     ", with reference" if ref is not None else "")
            out_log = _vace_window(model, positive, negative, vae, log_video, vace_mask, ref, width, height,
                                   steps, seed, cfg, sampler_name, scheduler, vace_strength, decode)[:n_real]

            if use_prev_window_reference:
                ri = window_stride if n_real > window_stride else n_real - 1
                prev_ref = out_log[ri:ri + 1].clone()

            w = dc.blend_ramp(n_real, t == 0, t_end == total, border)
            value[t:t_end] += dc.log_to_lin(out_log) * w.view(-1, 1, 1, 1)
            weight[t:t_end] += w
            out_mask[t:t_end] = vace_mask[:n_real]
            ldr[t:t_end] = srgb[:n_real]
            pbar.update(1)

        hdr = value / weight.clamp(min=1e-8).view(-1, 1, 1, 1)
        return (hdr, out_mask, ldr)


# =========================================================================== single-window helpers
class DiffHDRPrepare:
    """Builds DiffHDR's VACE inputs for the native WanVaceToVideo node (one window, <= 33 frames).
    Feed the decoded result through `DiffHDR Log to Linear`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "input_transfer": (TRANSFERS, {"default": TRANSFERS[0]}),
            "width": ("INT", {"default": 1280, "min": 256, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 720, "min": 256, "max": 2048, "step": 16}),
            "over_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 1.0, "step": 0.005}),
            "under_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.2, "step": 0.005}),
            "use_under_exposure_mask": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "IMAGE", "INT")
    RETURN_NAMES = ("control_video_log", "control_mask", "over_mask", "under_mask", "ldr_resized", "length")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, input_transfer, width, height, over_threshold, under_threshold, use_under_exposure_mask):
        dev = comfy.model_management.get_torch_device()
        srgb = dc.crop_and_resize(_to_srgb(images.to(dev), input_transfer), width, height)
        n = srgb.shape[0]
        length = ((n + 2) // 4) * 4 + 1  # next 4k+1 >= n; pad by repeating the last frame
        if length > n:
            srgb = torch.cat([srgb, srgb[-1:].expand(length - n, -1, -1, -1)], dim=0)
        log_video, vace, over, under = dc.prepare_window(srgb, over_threshold, under_threshold, use_under_exposure_mask)
        return tuple(x.cpu() for x in (log_video, vace, over, under, srgb)) + (length,)


class DiffHDRLogToLinear:
    """DiffHDR Log-Gamma (decoded VAE output) -> scene-linear Rec.709."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)},
                "optional": {"frame_count": ("INT", {"default": 0, "min": 0, "max": 100000,
                                                     "tooltip": "Trim padded frames (0 = keep all)."})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("hdr_linear",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images, frame_count=0):
        x = images.float()
        if frame_count > 0:
            x = x[:frame_count]
        return (dc.log_to_lin(x),)


class DiffHDRLinearToLog:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, images):
        return (dc.lin_to_log(images.float()),)


class DiffHDRPreview:
    """Linear HDR -> viewable sRGB at a chosen exposure (for previews only)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "hdr_linear": ("IMAGE",),
            "exposure": ("FLOAT", {"default": 0.0, "min": -16.0, "max": 16.0, "step": 0.25,
                                   "tooltip": "Stops. Try -3 to see detail rebuilt in the highlights."}),
            "tonemap": (["clip", "reinhard"], {"default": "clip"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, hdr_linear, exposure, tonemap):
        x = hdr_linear.float() * (2.0 ** exposure)
        if tonemap == "reinhard":
            x = x / (1.0 + x)
        return (dc.linear_to_srgb(x),)


# =========================================================================== EXR output
class DiffHDRSaveEXR:
    """Writes linear Rec.709 EXR frames (RGB, scanline) for Nuke etc."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "folder": ("STRING", {"default": "DiffHDR/shot",
                                  "tooltip": "Absolute path, or relative to ComfyUI's output folder."}),
            "filename_prefix": ("STRING", {"default": "diffhdr"}),
            "start_frame": ("INT", {"default": 1001, "min": 0, "max": 10000000}),
            "bit_depth": (["half", "float"], {"default": "half"}),
            "compression": (["ZIP", "PIZ", "ZIPS", "DWAA", "NONE"], {"default": "ZIP"}),
            "overwrite": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("first_file",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def save(self, images, folder, filename_prefix, start_frame, bit_depth, compression, overwrite):
        try:
            import OpenEXR
        except ImportError as e:
            raise RuntimeError("DiffHDR Save EXR needs the OpenEXR Python package. Install it into ComfyUI's "
                               "Python: python -m pip install \"OpenEXR>=3.2\"") from e
        if not hasattr(OpenEXR, "File"):
            raise RuntimeError(f"OpenEXR {getattr(OpenEXR, '__version__', '?')} is too old; DiffHDR needs >= 3.2 "
                               "(python -m pip install -U \"OpenEXR>=3.2\")")

        folder = folder.strip().strip('"')
        out_dir = folder if os.path.isabs(folder) else os.path.join(folder_paths.get_output_directory(), folder)
        os.makedirs(out_dir, exist_ok=True)
        comp = {"ZIP": OpenEXR.ZIP_COMPRESSION, "PIZ": OpenEXR.PIZ_COMPRESSION, "ZIPS": OpenEXR.ZIPS_COMPRESSION,
                "DWAA": OpenEXR.DWAA_COMPRESSION, "NONE": OpenEXR.NO_COMPRESSION}[compression]
        header = {"compression": comp, "type": OpenEXR.scanlineimage}
        first = None
        for i in range(images.shape[0]):
            path = os.path.join(out_dir, f"{filename_prefix}.{start_frame + i:04d}.exr")
            if os.path.exists(path) and not overwrite:
                raise FileExistsError(path)
            rgb = images[i, :, :, :3].float().cpu().numpy()
            rgb = np.clip(rgb, 0.0, 65504.0).astype(np.float16) if bit_depth == "half" else rgb.astype(np.float32)
            with OpenEXR.File(header, {"RGB": np.ascontiguousarray(rgb)}) as f:
                f.write(path)
            first = first or path
        log.info("DiffHDR: wrote %d EXR frames to %s", images.shape[0], out_dir)
        return {"ui": {"text": [f"{images.shape[0]} frames -> {out_dir}"]}, "result": (first or "",)}


NODE_CLASS_MAPPINGS = {
    "DiffHDRLoraLoader": DiffHDRLoraLoader,
    "DiffHDRVAELoader": DiffHDRVAELoader,
    "DiffHDRVideoSampler": DiffHDRVideoSampler,
    "DiffHDRPrepare": DiffHDRPrepare,
    "DiffHDRLogToLinear": DiffHDRLogToLinear,
    "DiffHDRLinearToLog": DiffHDRLinearToLog,
    "DiffHDRPreview": DiffHDRPreview,
    "DiffHDRSaveEXR": DiffHDRSaveEXR,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffHDRLoraLoader": "DiffHDR LoRA Loader",
    "DiffHDRVAELoader": "DiffHDR VAE Loader (fp32)",
    "DiffHDRVideoSampler": "DiffHDR Video Sampler (LDR to HDR)",
    "DiffHDRPrepare": "DiffHDR Prepare VACE Inputs",
    "DiffHDRLogToLinear": "DiffHDR Log to Linear",
    "DiffHDRLinearToLog": "DiffHDR Linear to Log",
    "DiffHDRPreview": "DiffHDR HDR Preview (exposure)",
    "DiffHDRSaveEXR": "DiffHDR Save EXR Sequence",
}
