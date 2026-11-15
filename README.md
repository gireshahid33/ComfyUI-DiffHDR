# ComfyUI-DiffHDR

**Turn ordinary 8-bit video into scene-linear HDR inside ComfyUI.**
These nodes run [DiffHDR](https://github.com/Eyeline-Labs/DiffHDR) (Eyeline Labs, ECCV 2026). It
regenerates blown-out highlights (and, optionally, crushed shadows) of an LDR clip with Wan2.1-VACE-14B,
then writes **linear Rec.709 EXR** sequences ready for Nuke, Resolve or any comp/grading tool.

- Any clip length: 33-frame sliding windows with cross-fades, each window guided by the previous one,
  as in the authors' long-video script.
- Built on ComfyUI's own Wan model loading and sampling, so fp8 weights, offloading and your attention
  backend all work.
- The colour and mask maths match the official code exactly (verified against the reference implementation).
- Runs on a 16 GB GPU.

> This is a community port. DiffHDR itself (the method, code and weights) is by Zhengming Yu, Li Ma,
> Mingming He, Paul Debevec and co-authors at Texas A&M University, Eyeline Labs and Netflix.
> Please cite their paper (below).

---

## Install

**ComfyUI-Manager:** *Custom Nodes Manager → Install via Git URL* →
`https://github.com/gireshahid33/ComfyUI-DiffHDR`

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gireshahid33/ComfyUI-DiffHDR
python -m pip install -r ComfyUI-DiffHDR/requirements.txt
```
(Portable/embedded installs: use `python_embeded\python.exe -m pip install -r ...`.)

The example workflow also uses [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
to load and preview video. ComfyUI-Manager's *Install Missing Custom Nodes* will offer to install it.

### Models

| File | Folder | Size | Download |
|---|---|---|---|
| `wan2.1_vace_14B_fp16.safetensors` | `models/diffusion_models` | 34.7 GB | [Comfy-Org/Wan-2.1](https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/diffusion_models/wan2.1_vace_14B_fp16.safetensors) |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `models/text_encoders` | 6.7 GB | [Comfy-Org/Wan-2.1](https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors) |
| `wan_2.1_vae.safetensors` | `models/vae` | 254 MB | [Comfy-Org/Wan-2.1](https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/vae/wan_2.1_vae.safetensors) |
| `DiffHDR.safetensors` (LoRA) | `models/loras` | 61 MB | [Eyeline-Labs/DiffHDR](https://huggingface.co/Eyeline-Labs/DiffHDR/resolve/main/DiffHDR.safetensors) |

The example workflow carries these links, so ComfyUI's missing-models dialog can download them for you.
Any Wan2.1 **VACE 14B** checkpoint works. Plain Wan T2V/I2V models won't, because the LoRA only patches the VACE blocks.

## Use

1. Load `example_workflows/DiffHDR_video_LDR_to_HDR.json`. With this pack installed, it also appears in
   ComfyUI's template browser under *ComfyUI-DiffHDR*.
2. Pick your clip in **Load LDR clip**. `frame_load_cap` starts at **33** (one window) for a quick test.
   Set it to **0** for the whole clip.
3. Queue. EXRs are written to `ComfyUI/output/DiffHDR/shot/diffhdr.1001.exr`, `.1002`, and so on. Set *folder*
   to an absolute path to write straight into a shot directory.
4. Check the two previews: the HDR at **−3 EV** shows the rebuilt highlight detail, and the mask shows
   what was regenerated (white).

### In Nuke (or any compositor)
- Read the EXRs as **linear** (`Utility - Linear - Rec.709` under ACES).
- The output is **1280×720**, the model's training size. DiffHDR regenerates the *whole* frame, so the best
  result is to upscale it and **Keymix it over your original plate using the exposure mask**. That
  replaces only the clipped areas and keeps the plate's full resolution and detail everywhere else.

## Nodes

| Node | What it does |
|---|---|
| **DiffHDR Video Sampler (LDR to HDR)** | The full pipeline for any clip length. Outputs linear HDR frames, the regenerated-area mask, and the resized LDR. |
| **DiffHDR LoRA Loader** | Loads the official LoRA as published. Its PEFT key names (`vace_blocks.N…lora_A.default.weight`) aren't recognised by ComfyUI's standard LoRA loader. |
| **DiffHDR VAE Loader (fp32)** | Wan VAE at a fixed precision. DiffHDR uses fp32, because Log-space rounding is amplified in linear. |
| **DiffHDR Save EXR Sequence** | RGB EXR frames, half or float, ZIP/PIZ/DWAA, with a start frame number (default 1001). |
| **DiffHDR HDR Preview (exposure)** | Linear to sRGB at a chosen exposure, for previews. |
| **DiffHDR Prepare VACE Inputs**, **Log to Linear**, **Linear to Log** | For building a single-window graph yourself with the native `WanVaceToVideo`, `KSampler`, `TrimVideoLatent` and `VAEDecode` nodes. |

### Sampler settings
| Setting | Default | Notes |
|---|---|---|
| `width` × `height` | 1280 × 720 | The input is centre-cropped and resized to this, as in the original code. |
| `window_size` / `window_stride` | 33 / 16 | The model was trained on 33 frames. |
| `use_prev_window_reference` | on | Each window is conditioned on the previous window's output, for temporal consistency. |
| `steps` | 10 | The authors' demos use 10. 20–30 is cleaner but slower. |
| `cfg` | 1.0 | As in DiffHDR. At 1.0 the negative prompt is ignored and there is one model pass per step. |
| `over_threshold` | 0.95 | sRGB level above which pixels count as clipped. Lower it to regenerate more of the bright areas. |
| `use_under_exposure_mask` | off | Also regenerate crushed shadows. |
| `input_transfer` | sRGB | Choose *linear Rec.709* for linear EXR plates. They are clamped to 0–1 and converted to sRGB first. |
| `decode` | fast | *fast*: fp32 in 768 px tiles (about 20 s per window). *exact*: one full-frame tile, bit-identical to untiled fp32 (about 90 s on 16 GB). |
| `reference_image` (optional) | — | An sRGB image that guides what fills the blown-out areas (the paper's image-conditioned mode). |

Positive prompt: leave it empty, or describe the highlight content, e.g.
`over-exposed: a bright ocean landscape seen through the window`.

## Performance

Measured on an RTX 5070 Ti (16 GB) with 64 GB RAM, the fp8-cast model, 1280×720 and 10 steps:
**about 5 minutes per 33-frame window** (sampling about 3 min, VAE about 1 min).

- **Close other GPU-heavy applications** (Nuke, Resolve, games) while it runs. When another app held
  7–8 GB of VRAM, the same job ran 3–10× slower because the VAE had to page memory over PCIe. The console
  logs free VRAM for each window.
- The DiT is unloaded before every VAE step, and the VAE runs in fp32 tiles. An untiled fp32 decode at
  720p needs more than 16 GB.
- The 768 px tiles add 0.26% mean (2.7% p99) error in linear, measured against untiled fp32.
  bf16 would be 0.9% (4.9% p99), which is why the VAE stays in fp32.
- `weight_dtype` `default` (fp16, 35 GB) is slightly more accurate than `fp8_e4m3fn`, but much slower on 16 GB cards.

## How it works

DiffHDR is standard Wan2.1-VACE inpainting plus three extra steps:
1. **Log-Gamma encoding.** Frames go sRGB → linear → `(log(2.2·x+1) / log(2.2·65536+1))^(1/2.2)`.
   Linear 1.0 maps to 0.348, which leaves the upper code range for highlights up to 65536.
2. **Exposure masks** come from the sRGB frames: smoothed luma and channel-clipping tests, stabilised over time with
   an EMA and morphology, then binarised. Masked areas are regenerated, and the rest is given to the model as context.
3. **A rank-32 LoRA on the 8 VACE blocks.**
The decoded result is converted from Log back to linear. `diffhdr_color.py` is a direct port of
the reference functions.

## Limitations
- Output is 1280×720 (the training resolution). Larger sizes are untested.
- Highlight brightness can drift slightly between windows on long clips. It cross-fades smoothly,
  with no pops, but may need a grade.
- The HDRI panorama model (`DiffHDR_Pano.safetensors`) isn't supported by these nodes yet.
- Tested on ComfyUI 0.45 with PyTorch 2.11 on Windows. Wan VACE support and recent `VAE.decode_tiled`
  are required, so use a 2026 ComfyUI build.

## Credits & licence

- Method, original code and LoRA weights: **DiffHDR**, Eyeline Labs / Texas A&M University / Netflix,
  Apache-2.0. [GitHub](https://github.com/Eyeline-Labs/DiffHDR) ·
  [Hugging Face](https://huggingface.co/Eyeline-Labs/DiffHDR) · [arXiv 2604.06161](https://arxiv.org/abs/2604.06161)
- Base model: [Wan2.1-VACE-14B](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) (Apache-2.0); ComfyUI
  repackaging by [Comfy-Org](https://huggingface.co/Comfy-Org/Wan-2.1).
- ComfyUI nodes: VFX SHOWDOWN (www.vfxshowdown.com). Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).

```bibtex
@inproceedings{yu2026diffhdr,
  title={DiffHDR: Re-Exposing LDR Videos with Video Diffusion Models},
  author={Yu, Zhengming and Ma, Li and He, Mingming and Isikdogan, Leo and Xu, Yuancheng and Smirnov, Dmitriy and Salamanca, Pablo and Mi, Dao and Delgado, Pablo and Yu, Ning and Philip, Julien and Li, Xin and Wang, Wenping and Debevec, Paul},
  booktitle={European Conference on Computer Vision},
  year={2026},
  organization={Springer}
}
```
