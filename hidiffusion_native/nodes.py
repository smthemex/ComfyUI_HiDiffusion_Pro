"""ComfyUI nodes for the native HiDiffusion chain.

Intended graph (every other node is stock ComfyUI):

    CheckpointLoaderSimple -> MODEL / CLIP / VAE
    CLIPTextEncode x2      -> positive / negative CONDITIONING
    EmptyLatentImage       -> LATENT (e.g. 2048x2048)
    HiDiffusionApply(MODEL) -> patched MODEL
    KSampler -> VAEDecode -> SaveImage / PreviewImage

HiDiffusion is training-free: there are NO extra weights and therefore no
loader node. RAUNet + MSW-MSA are installed as ModelPatcher transformer
patches, so LoRA / ControlNet / FreeU / VAE / samplers / memory management
all remain native. No diffusers / transformers / cv2 / omegaconf imports.
"""
from __future__ import annotations

import uuid

from .core import (ARCH_CHOICES, LAYOUTS, detect_arch, validate_layout)
from . import patches


class HiDiffusionApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_arch": (ARCH_CHOICES, {"default": "auto",
                    "tooltip": "auto detects a stock SDXL (9 input blocks) or "
                               "SD1.5/SD2.x (12 input blocks) UNet at runtime."}),
                "apply_raunet": ("BOOLEAN", {"default": True,
                    "tooltip": "Resolution-Aware UNet: early denoising steps run "
                               "part of the UNet at half resolution to suppress "
                               "object duplication at 2K/4K output."}),
                "apply_window_attn": ("BOOLEAN", {"default": False,
                    "tooltip": "MSW-MSA window self-attention on 10 (SDXL) / 5 "
                               "(SD1.5) transformer blocks. Upstream draws a "
                               "random shift every step; usually only useful at "
                               "very high resolution, so it is OFF by default "
                               "(matching the old plugin's default)."}),
                "aggressive_raunet": ("BOOLEAN", {"default": True,
                    "tooltip": "Upstream 'aggressive RAUNet' (SDXL, latent<512): "
                               "pool in [16%, T1) of steps and use stride-4 "
                               "downsampling in the first 16%. Leaving this ON "
                               "reproduces the original project's behaviour."}),
                "t1_ratio": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0,
                    "step": 0.01, "tooltip": "-1 = upstream auto preset, chosen "
                    "from the latent size. Override with 0..1."}),
                "t2_ratio": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1.0,
                    "step": 0.01, "tooltip": "-1 = upstream auto preset."}),
                "controlnet_t2i": ("BOOLEAN", {"default": False,
                    "tooltip": "Use upstream's text-to-image+ControlNet preset "
                               "(T1=0.5) instead of the plain SDXL one (T1=0.4)."}),
                "window_shift": (["random", "fixed"], {"default": "random",
                    "tooltip": "random reproduces upstream torch.rand(1); fixed "
                    "pins one of the 4 shift candidates for reproducible output "
                    "and numerical parity tests."}),
                "window_shift_index": ("INT", {"default": 0, "min": 0, "max": 3,
                    "tooltip": "0..3, only used when window_shift=fixed."}),
            },
            "optional": {
                "aggressive_start_ratio": ("FLOAT", {"default": -1.0,
                    "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "-1 = upstream 8/50 = 0.16."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Hidiffusion_Pro/Native"
    DESCRIPTION = (
        "Apply HiDiffusion (RAUNet + optional MSW-MSA) to a NATIVE ComfyUI "
        "MODEL. Insert between CheckpointLoaderSimple and KSampler; everything "
        "downstream stays native. No diffusers/transformers required. "
        "Notes: (1) HiDiffusion targets 2K/4K output - at 1024px the difference "
        "is small; (2) when using ControlNet together with RAUNet, the control "
        "residuals may not match the pooled feature size - disable RAUNet in "
        "that case; (3) turning both switches OFF yields a bit-identical stock "
        "MODEL, useful as a parity baseline.")

    def apply(self, model, model_arch, apply_raunet, apply_window_attn,
              aggressive_raunet, t1_ratio, t2_ratio, controlnet_t2i,
              window_shift, window_shift_index,
              aggressive_start_ratio=-1.0):
        m = model.clone()
        unet = m.model.diffusion_model
        arch = detect_arch(unet) if model_arch == "auto" else model_arch
        if arch not in LAYOUTS:
            raise RuntimeError("HiDiffusion: unknown model_arch %r" % arch)
        layout = validate_layout(unet, arch)

        opts = {
            "state_id": uuid.uuid4().hex,
            "apply_raunet": bool(apply_raunet),
            "apply_window_attn": bool(apply_window_attn),
            "aggressive": bool(aggressive_raunet),
            "t1_ratio": None if t1_ratio < 0 else float(t1_ratio),
            "t2_ratio": None if t2_ratio < 0 else float(t2_ratio),
            "controlnet_t2i": bool(controlnet_t2i),
            "aggressive_start_ratio": (None if aggressive_start_ratio < 0
                                       else float(aggressive_start_ratio)),
            "shift_mode": window_shift,
            "shift_index": int(window_shift_index),
        }
        patches.install(m, arch, layout, opts)
        print("[HiDiffusion] arch=%s  raunet=%s  msw_msa=%s  aggressive=%s  "
              "t1=%s  t2=%s  shift=%s  (%d window attn1 layers)"
              % (arch, opts["apply_raunet"], opts["apply_window_attn"],
                 opts["aggressive"], "auto" if opts["t1_ratio"] is None else opts["t1_ratio"],
                 "auto" if opts["t2_ratio"] is None else opts["t2_ratio"],
                 window_shift,
                 len(layout["window_attn1"]) if opts["apply_window_attn"] else 0))
        return (m,)


# ---------------------------------------------------------------------------
# Deprecated node IDs: keep old graphs loadable, but fail with a clear
# migration message instead of silently importing diffusers/transformers.
# Top level of this file intentionally has zero heavy imports.
# ---------------------------------------------------------------------------

_LEGACY_MSG = (
    "HiDiffusion_Pro: this node belongs to the removed diffusers wrapper. "
    "Replace it with the native chain: CheckpointLoaderSimple -> "
    "HiDiffusionApply -> KSampler -> VAEDecode. See the plugin README.")


class _DeprecatedNode:
    CATEGORY = "Hidiffusion_Pro/Deprecated"
    DESCRIPTION = _LEGACY_MSG

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("HIDIF_MODEL",)
    RETURN_NAMES = ("deprecated",)
    OUTPUT_NODE = False

    def deprecated(self, **kwargs):
        raise RuntimeError(_LEGACY_MSG)


class HI_Diffusers_Model_Loader(_DeprecatedNode):
    FUNCTION = "deprecated"


class Hi_Sampler(_DeprecatedNode):
    FUNCTION = "deprecated"
    RETURN_TYPES = ("IMAGE",)


NODE_CLASS_MAPPINGS = {
    "HiDiffusionApply": HiDiffusionApply,
    "HI_Diffusers_Model_Loader": HI_Diffusers_Model_Loader,
    "Hi_Sampler": Hi_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HiDiffusionApply": "HiDiffusion Apply to MODEL (Native)",
    "HI_Diffusers_Model_Loader": "[DEPRECATED] HI_Diffusers_Model_Loader",
    "Hi_Sampler": "[DEPRECATED] Hi_Sampler",
}
