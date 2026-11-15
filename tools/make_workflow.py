"""Regenerates the example workflows (UI + API format).

Usage: python tools/make_workflow.py <out_ui.json> <out_api.json> <video file name> [frame_load_cap=33] [lora name]
"""
import json, sys

out_ui, out_api, video_name = sys.argv[1], sys.argv[2], sys.argv[3]
frame_cap = int(sys.argv[4]) if len(sys.argv) > 4 else 33
LORA = sys.argv[5] if len(sys.argv) > 5 else "DiffHDR.safetensors"

HF_WAN = "https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files"
MODELS = {  # shown by ComfyUI's "missing models" dialog, which can download them
    "unet": [{"name": "wan2.1_vace_14B_fp16.safetensors", "directory": "diffusion_models",
              "url": f"{HF_WAN}/diffusion_models/wan2.1_vace_14B_fp16.safetensors"}],
    "lora": [{"name": "DiffHDR.safetensors", "directory": "loras",
              "url": "https://huggingface.co/Eyeline-Labs/DiffHDR/resolve/main/DiffHDR.safetensors"}],
    "clip": [{"name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "directory": "text_encoders",
              "url": f"{HF_WAN}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"}],
    "vae": [{"name": "wan_2.1_vae.safetensors", "directory": "vae", "url": f"{HF_WAN}/vae/wan_2.1_vae.safetensors"}],
}

NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
       "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
       "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

NOTE = """## DiffHDR: LDR video → linear HDR EXR

**What it does:** rebuilds blown-out highlights (and optionally crushed shadows) of an 8-bit sRGB clip and writes **scene-linear Rec.709 EXRs** (Eyeline Labs DiffHDR, Wan2.1-VACE-14B + LoRA).

**Models** (ComfyUI offers to download missing ones): `wan2.1_vace_14B_fp16` → diffusion_models, `umt5_xxl_fp8_e4m3fn_scaled` → text_encoders, `wan_2.1_vae` → vae, `DiffHDR.safetensors` → loras (huggingface.co/Eyeline-Labs/DiffHDR). Needs ComfyUI-VideoHelperSuite for video loading.

**Run it**
1. Pick your clip in **Load Video**. `frame_load_cap` is **33** = one window (a few minutes on a 16 GB card); set it to **0** for the whole clip.
2. Queue. EXRs go to `ComfyUI/output/DiffHDR/shot/diffhdr.1001.exr…` (change *folder* to an absolute path to write straight into your shot).
3. The preview shows the result at **−3 EV** so you can see the rebuilt highlight detail; the mask preview shows what was regenerated (white).

**Settings that matter**
- Output is **1280×720** (the model's training size). In Nuke: Read as *linear / Utility - Linear - Rec.709*, upscale, and **Keymix it over your plate with the exposure mask**, so only the clipped areas are replaced.
- `steps` 10 = authors' demos; 20–30 is cleaner but slower.
- `over_threshold` 0.95: lower it to regenerate more of the bright areas.
- `use_under_exposure_mask`: also rebuild crushed blacks.
- Long clips: 33-frame windows every 16 frames, cross-faded, each window guided by the previous one (`use_prev_window_reference`).
- 16 GB VRAM: the model is cast to **fp8_e4m3fn** (≈17 GB, partly offloaded to RAM). `default` (fp16, 35 GB) is a little more accurate but much slower.
- **Speed (RTX 5070 Ti, 16 GB):** about 5 min per 33-frame window (sampling ~3 min, VAE ~1 min). **Close other GPU-heavy apps first** (Nuke, Resolve, games): with 7–8 GB taken by another app the same job ran 3–10× slower. The console prints free VRAM for each window.
- Long clips: highlight brightness can drift a little between windows (it cross-fades smoothly, with no pops), so grade it in comp if needed.
- Keep the VAE at **fp32** (Log-space rounding is amplified in linear). `decode` fast = 768 px tiles (~20 s, ~0.3% error); exact = full frame (~90 s).
"""

nodes, links = [], []
def node(nid, ntype, pos, size, widgets, inputs=(), outputs=(), title=None, mode=0, models=None):
    n = {"id": nid, "type": ntype, "pos": list(pos), "size": list(size), "flags": {}, "order": nid, "mode": mode,
         "inputs": [{"name": nm, "type": tp, "link": None} for nm, tp in inputs],
         "outputs": [{"name": nm, "type": tp, "links": [], "slot_index": i} for i, (nm, tp) in enumerate(outputs)],
         "properties": {"Node name for S&R": ntype}, "widgets_values": widgets}
    if title: n["title"] = title
    if models: n["properties"]["models"] = models
    nodes.append(n); return n
def link(a, a_slot, b, b_input):
    lid = len(links) + 1
    na = next(n for n in nodes if n["id"] == a); nb = next(n for n in nodes if n["id"] == b)
    tp = na["outputs"][a_slot]["type"]
    slot = next(i for i, x in enumerate(nb["inputs"]) if x["name"] == b_input)
    na["outputs"][a_slot]["links"].append(lid); nb["inputs"][slot]["link"] = lid
    links.append([lid, a, a_slot, b, slot, tp])

M = [("MODEL", "MODEL")]
node(1, "UNETLoader", (40, 60), (360, 82), ["wan2.1_vace_14B_fp16.safetensors", "fp8_e4m3fn"], outputs=M,
     title="Wan2.1 VACE 14B", models=MODELS["unet"])
node(2, "DiffHDRLoraLoader", (40, 190), (360, 82), [LORA, 1.0], [("model", "MODEL")], M, models=MODELS["lora"])
node(3, "ModelSamplingSD3", (40, 320), (360, 58), [5.0], [("model", "MODEL")], M, title="Shift 5 (DiffHDR)")
node(4, "CLIPLoader", (40, 430), (360, 106), ["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default"],
     outputs=[("CLIP", "CLIP")], models=MODELS["clip"])
node(5, "CLIPTextEncode", (440, 60), (380, 120), [""], [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")],
     title="Prompt (leave empty, or 'over-exposed: <what is in the highlights>')")
node(6, "CLIPTextEncode", (440, 220), (380, 120), [NEG], [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")],
     title="Negative (unused at cfg 1)")
node(7, "DiffHDRVAELoader", (40, 590), (360, 82), ["wan_2.1_vae.safetensors", "fp32"], outputs=[("VAE", "VAE")],
     models=MODELS["vae"])
node(8, "VHS_LoadVideo", (440, 390), (380, 560),
     {"video": video_name, "force_rate": 0, "custom_width": 0, "custom_height": 0, "frame_load_cap": frame_cap,
      "skip_first_frames": 0, "select_every_nth": 1, "format": "AnimateDiff",
      "videopreview": {"hidden": False, "paused": False, "params": {}}},
     [("meta_batch", "VHS_BatchManager"), ("vae", "VAE")],
     [("IMAGE", "IMAGE"), ("frame_count", "INT"), ("audio", "AUDIO"), ("video_info", "VHS_VIDEOINFO")], title="Load LDR clip")
sampler_widgets = [
    "sRGB (display-referred LDR)", 1280, 720, 33, 16, True, 10, 10, "fixed", 1.0, "euler", "simple",
    1.0, 0.95, 0.01, False, "fast (fp32, 768px tiles)", 5.0]
node(9, "DiffHDRVideoSampler", (860, 60), (400, 560), sampler_widgets,
     [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("vae", "VAE"),
      ("images", "IMAGE"), ("reference_image", "IMAGE")],
     [("hdr_linear", "IMAGE"), ("exposure_mask", "MASK"), ("ldr_resized", "IMAGE")])
node(10, "DiffHDRSaveEXR", (1300, 60), (380, 200), ["DiffHDR/shot", "diffhdr", 1001, "half", "ZIP", True],
     [("images", "IMAGE")], [("first_file", "STRING")])
node(11, "DiffHDRPreview", (1300, 300), (380, 82), [-3.0, "clip"], [("hdr_linear", "IMAGE")], [("IMAGE", "IMAGE")],
     title="HDR preview at -3 EV")
vc = lambda prefix: {"frame_rate": 30, "loop_count": 0, "filename_prefix": prefix, "format": "video/h264-mp4",
                     "pix_fmt": "yuv420p", "crf": 19, "save_metadata": False, "trim_to_audio": False,
                     "pingpong": False, "save_output": False, "videopreview": {"hidden": False, "paused": False, "params": {}}}
VCIN = [("images", "IMAGE"), ("audio", "AUDIO"), ("meta_batch", "VHS_BatchManager"), ("vae", "VAE")]
node(12, "VHS_VideoCombine", (1300, 420), (380, 420), vc("DiffHDR/preview_minus3EV"), VCIN, [("Filenames", "VHS_FILENAMES")],
     title="Preview: HDR at -3 EV")
node(13, "MaskToImage", (1720, 60), (260, 30), [], [("mask", "MASK")], [("IMAGE", "IMAGE")])
node(14, "VHS_VideoCombine", (1720, 140), (380, 420), vc("DiffHDR/preview_mask"), VCIN, [("Filenames", "VHS_FILENAMES")],
     title="Preview: regenerated areas")
node(15, "MarkdownNote", (40, 720), (380, 640), [NOTE], title="How to use")

link(1, 0, 2, "model"); link(2, 0, 3, "model"); link(3, 0, 9, "model")
link(4, 0, 5, "clip"); link(4, 0, 6, "clip")
link(5, 0, 9, "positive"); link(6, 0, 9, "negative"); link(7, 0, 9, "vae"); link(8, 0, 9, "images")
link(9, 0, 10, "images"); link(9, 0, 11, "hdr_linear"); link(11, 0, 12, "images")
link(9, 1, 13, "mask"); link(13, 0, 14, "images")

ui = {"last_node_id": max(n["id"] for n in nodes), "last_link_id": len(links), "nodes": nodes, "links": links,
      "groups": [], "config": {}, "extra": {"ds": {"scale": 0.7, "offset": [40, 40]}}, "version": 0.4}
json.dump(ui, open(out_ui, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- API format (for scripted runs)
sw = dict(zip(["input_transfer", "width", "height", "window_size", "window_stride", "use_prev_window_reference",
               "steps", "seed", "_ctl", "cfg", "sampler_name", "scheduler", "vace_strength", "over_threshold",
               "under_threshold", "use_under_exposure_mask", "decode", "reference_ev"], sampler_widgets)); sw.pop("_ctl")
api = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.1_vace_14B_fp16.safetensors", "weight_dtype": "fp8_e4m3fn"}},
    "2": {"class_type": "DiffHDRLoraLoader", "inputs": {"model": ["1", 0], "lora_name": LORA, "strength": 1.0}},
    "3": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": 5.0}},
    "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
    "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": ""}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 0], "text": NEG}},
    "7": {"class_type": "DiffHDRVAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors", "precision": "fp32"}},
    "8": {"class_type": "VHS_LoadVideo", "inputs": {"video": video_name, "force_rate": 0, "custom_width": 0, "custom_height": 0,
                                                    "frame_load_cap": frame_cap, "skip_first_frames": 0, "select_every_nth": 1, "format": "AnimateDiff"}},
    "9": {"class_type": "DiffHDRVideoSampler", "inputs": {"model": ["3", 0], "positive": ["5", 0], "negative": ["6", 0],
                                                          "vae": ["7", 0], "images": ["8", 0], **sw}},
    "10": {"class_type": "DiffHDRSaveEXR", "inputs": {"images": ["9", 0], "folder": "DiffHDR/shot", "filename_prefix": "diffhdr",
                                                      "start_frame": 1001, "bit_depth": "half", "compression": "ZIP", "overwrite": True}},
    "11": {"class_type": "DiffHDRPreview", "inputs": {"hdr_linear": ["9", 0], "exposure": -3.0, "tonemap": "clip"}},
    "12": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["11", 0], "frame_rate": 30, "loop_count": 0, "filename_prefix": "DiffHDR/preview_minus3EV",
                                                        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": False,
                                                        "trim_to_audio": False, "pingpong": False, "save_output": True}},
    "13": {"class_type": "MaskToImage", "inputs": {"mask": ["9", 1]}},
    "14": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["13", 0], "frame_rate": 30, "loop_count": 0, "filename_prefix": "DiffHDR/preview_mask",
                                                        "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 19, "save_metadata": False,
                                                        "trim_to_audio": False, "pingpong": False, "save_output": True}},
}
json.dump(api, open(out_api, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok", len(nodes), "nodes", len(links), "links")
