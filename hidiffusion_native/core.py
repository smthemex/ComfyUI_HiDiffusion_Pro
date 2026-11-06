"""HiDiffusion core algorithms - a verbatim, dependency-free port.

Ported from D:/Downloads/HiDiffusion-main/hidiffusion/hidiffusion.py
(RAUNet resolution schedule + MSW-MSA window self-attention), plus the
ComfyUI UNet block layout measured on the user's real checkpoints with
``_hd_probe.py``.

This module is intentionally pure ``torch``: it can be imported and
numerically compared without ComfyUI, diffusers or transformers installed.
None of the upstream quirks were "fixed" - see README / methodology docs.
"""
from __future__ import annotations

import math
import re
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Upstream constants (verbatim from hidiffusion.py)
# ---------------------------------------------------------------------------

SWITCHING_THRESHOLD_RATIO_DICT = {
    'sd15_1024': {'T1_ratio': 0.4, 'T2_ratio': 0.0},
    'sd15_2048': {'T1_ratio': 0.7, 'T2_ratio': 0.3},
    'sdxl_2048': {'T1_ratio': 0.4, 'T2_ratio': 0.0},
    'sdxl_4096': {'T1_ratio': 0.7, 'T2_ratio': 0.3},
    'sdxl_turbo_1024': {'T1_ratio': 0.5, 'T2_ratio': 0.0},
}
TEXT_TO_IMG_CONTROLNET_RATIO_DICT = {
    'sdxl_2048': {'T1_ratio': 0.5, 'T2_ratio': 0.0},
}

AGGRESSIVE_STEP = 8                 # upstream ``aggressive_step``
AGGRESSIVE_DENOM = 50               # upstream hard-coded ``8 / 50``
IS_AGGRESSIVE_RAUNET = True         # upstream global, only set in sdxl latent<512
INPAINTING_IS_AGGRESSIVE = False
PLAYGROUND_IS_AGGRESSIVE = False

# ---------------------------------------------------------------------------
# 2. ComfyUI UNet layout (measured, not guessed)
#
# diffusers module key                       -> ComfyUI (section, block_id, depth)
#   sdxl  down_blocks.1.attentions.{0,1}     -> input_blocks.4 / .5  (640, depth 2)
#         up_blocks.1.attentions.{0,1,2}     -> output_blocks.3/.4/.5 (640, depth 2)
#         down_blocks.1.downsamplers.0.conv  -> input_blocks.6  (Downsample)
#         up_blocks.0.upsamplers.0.conv      -> output_blocks.2 (Upsample, upstream no-op)
#   sd15  down_blocks.0.attentions.{0,1}     -> input_blocks.1 / .2  (320, depth 1)
#         up_blocks.3.attentions.{0,1,2}     -> output_blocks.9/.10/.11 (320, depth 1)
#         down_blocks.0.downsamplers.0.conv  -> input_blocks.3  (Downsample)
#         down_blocks.1 (i==0 output)        -> input_blocks.4  (avg_pool, T2)
#         up_blocks.2 (i==2 before cat)      -> output_blocks.8 (bicubic, T2)
#
# IMPORTANT: the bicubic "up" patch is attached to the block *before* its
# skip-connection concat, which is exactly where diffusers' ``if i == 1``
# interpolation lands (up_blocks.N resnet #1 finished, #2 not started yet).
# ---------------------------------------------------------------------------

SDXL_WINDOW_KEYS = [
    ("input", 4, 0), ("input", 4, 1),
    ("input", 5, 0), ("input", 5, 1),
    ("output", 3, 0), ("output", 3, 1),
    ("output", 4, 0), ("output", 4, 1),
    ("output", 5, 0), ("output", 5, 1),
]
SD15_WINDOW_KEYS = [
    ("input", 1, 0), ("input", 2, 0),
    ("output", 9, 0), ("output", 10, 0), ("output", 11, 0),
]

LAYOUTS = {
    "sdxl": {
        "pool_input_block": ("input", 4),       # T1: avg_pool after down_blocks.1 i==0
        "pool_output_block": ("output", 5),     # T1: bicubic before up_blocks.1 i==2 cat
        "stride_input_block": ("input", 6),     # T2: downsampler conv stride 2 -> 4
        "window_attn1": SDXL_WINDOW_KEYS,
        "window_inner_dim": 640,
        "n_input_blocks": 9,
        "attn_dims": {640, 1280},
    },
    "sdxl_turbo": {
        "pool_input_block": ("input", 4),
        "pool_output_block": ("output", 5),
        "stride_input_block": None,             # turbo preset has no T2 downsampler
        "window_attn1": SDXL_WINDOW_KEYS,
        "window_inner_dim": 640,
        "n_input_blocks": 9,
        "attn_dims": {640, 1280},
    },
    "sd15": {
        "pool_input_block": ("input", 4),       # T2: avg_pool after down_blocks.1 i==0
        "pool_output_block": ("output", 8),     # T2: bicubic before up_blocks.2 i==2 cat
        "stride_input_block": ("input", 3),     # T1: downsampler conv stride 2 -> 4
        "window_attn1": SD15_WINDOW_KEYS,
        "window_inner_dim": 320,
        "n_input_blocks": 12,
        "attn_dims": {320, 640, 1280},
    },
}

ARCH_CHOICES = ["auto", "sdxl", "sd15", "sdxl_turbo"]


# ---------------------------------------------------------------------------
# 3. UNet structure enumeration / validation (runtime, shape-checked)
# ---------------------------------------------------------------------------

def enumerate_attn1(diffusion_model):
    """Return ``[(key3, inner_dim), ...]`` in ComfyUI registration/execution order.

    ``key3`` is exactly the key looked up in ``patches_replace["attn1"]``:
    ``(section, block_number, transformer_block_index)``.
    """
    out = []
    for name, module in diffusion_model.named_modules():
        if not name.endswith(".attn1"):
            continue
        m = re.match(
            r"^(input_blocks|middle_block|output_blocks)\.(\d+)\..*?"
            r"transformer_blocks\.(\d+)\.attn1$", name)
        if m is None:
            continue
        section = {"input_blocks": "input",
                   "middle_block": "middle",
                   "output_blocks": "output"}[m.group(1)]
        block_number = 0 if section == "middle" else int(m.group(2))
        depth = int(m.group(3))
        out.append(((section, block_number, depth), module.to_q.out_features))
    return out


def detect_arch(diffusion_model):
    n_in = len(diffusion_model.input_blocks)
    dims = {d for _, d in enumerate_attn1(diffusion_model)}
    if n_in == 9 and {640, 1280} <= dims:
        return "sdxl"
    if n_in == 12 and {320, 640, 1280} <= dims:
        return "sd15"
    raise RuntimeError(
        "HiDiffusion: unrecognised UNet (input_blocks=%d, attn dims=%s). "
        "Supported: stock SDXL (9 input blocks) and SD1.5/SD2.x (12 input blocks). "
        "Pass model_arch explicitly only if you know the structure matches."
        % (n_in, sorted(dims)))


def validate_layout(diffusion_model, arch):
    """Fail loudly instead of silently running with wrong injection points."""
    layout = LAYOUTS[arch]
    n_in = len(diffusion_model.input_blocks)
    if n_in != layout["n_input_blocks"]:
        raise RuntimeError(
            "HiDiffusion: %s expects %d input_blocks, got %d. Refusing to patch."
            % (arch, layout["n_input_blocks"], n_in))

    desc = dict(enumerate_attn1(diffusion_model))
    for key3 in layout["window_attn1"]:
        if key3 not in desc:
            raise RuntimeError("HiDiffusion: missing attn1 layer %r for %s" % (key3, arch))
        if desc[key3] != layout["window_inner_dim"]:
            raise RuntimeError(
                "HiDiffusion: attn1 %r inner_dim=%d, expected %d. Refusing to patch "
                "with wrong window-attention layer." % (key3, desc[key3],
                                                        layout["window_inner_dim"]))
    for label in ("pool_input_block", "pool_output_block"):
        section, idx = layout[label]
        blocks = (diffusion_model.input_blocks if section == "input"
                  else diffusion_model.output_blocks)
        if idx >= len(blocks):
            raise RuntimeError("HiDiffusion: %s=%r out of range" % (label, (section, idx)))
    if layout["stride_input_block"] is not None:
        section, idx = layout["stride_input_block"]
        blk = diffusion_model.input_blocks[idx]
        if not any(type(m).__name__ == "Downsample" for m in blk):
            raise RuntimeError(
                "HiDiffusion: input_blocks.%d has no Downsample module; the T2 "
                "stride patch cannot be applied." % idx)
    return layout


# ---------------------------------------------------------------------------
# 4. RAUNet step schedule
#
# Upstream counts UNet calls (``self.timestep += 1``) against
# ``pipeline._num_timesteps``. ComfyUI gives us the full sigma table in
# ``transformer_options["sample_sigmas"]`` and the current sigma in
# ``transformer_options["sigmas"]``; argmin gives the exact step index with
# no mutable state, so chunked CFG / batching / repeated runs are all safe.
# ---------------------------------------------------------------------------

def compute_step(transformer_options):
    """Return ``(step_index, total_steps)`` or ``None`` when unavailable."""
    ss = transformer_options.get("sample_sigmas")
    cur = transformer_options.get("sigmas")
    if ss is None or cur is None:
        return None
    try:
        s = torch.as_tensor(cur).flatten()[0]
        ss = torch.as_tensor(ss).to(device=s.device, dtype=torch.float32)
        idx = int((ss - s.to(torch.float32)).abs().argmin())
        return idx, int(ss.shape[0]) - 1
    except Exception:
        return None


def build_schedule(arch, ori_h, ori_w, total_steps, *,
                   aggressive=True,
                   t1_ratio=None, t2_ratio=None,
                   controlnet_t2i=False,
                   aggressive_start_ratio=None):
    """Reproduce upstream's T1/T2 timing, including its "aggressive" quirk.

    Upstream (sdxl, latent < 512):
      * cross_attn_down/up blocks use T1_ratio with aggressive window
        ``[int(8/50 * steps), int(steps * T1_ratio))`` -> avg_pool + bicubic
      * downsampler conv uses T2_ratio but aggressive OVERRIDES it with
        ``T1 = int(8/50 * steps)`` -> stride 4 for the first 16% of steps,
        *even when T2_ratio == 0*. This is easy to miss; do not "fix" it.
    Upstream (sd15/2.x): aggressive is always False; T1 drives the
      downsampler stride and T2 drives avg_pool + bicubic.
    """
    total_steps = max(int(total_steps), 1)
    if arch == "sdxl":
        if ori_h < 512 or ori_w < 512:
            preset = (TEXT_TO_IMG_CONTROLNET_RATIO_DICT['sdxl_2048']
                      if controlnet_t2i else SWITCHING_THRESHOLD_RATIO_DICT['sdxl_2048'])
            aggr = bool(aggressive)
            preset_name = "controlnet_t2i" if controlnet_t2i else "sdxl_2048"
        else:
            # upstream never sets aggressive_raunet in this branch -> class default False
            preset = SWITCHING_THRESHOLD_RATIO_DICT['sdxl_4096']
            aggr = False
            preset_name = "sdxl_4096"
    elif arch == "sdxl_turbo":
        preset = SWITCHING_THRESHOLD_RATIO_DICT['sdxl_turbo_1024']
        aggr = False
        preset_name = "sdxl_turbo_1024"
    else:
        preset = (SWITCHING_THRESHOLD_RATIO_DICT['sd15_1024']
                  if (ori_h < 256 or ori_w < 256)
                  else SWITCHING_THRESHOLD_RATIO_DICT['sd15_2048'])
        aggr = False
        preset_name = "sd15_1024" if (ori_h < 256 or ori_w < 256) else "sd15_2048"

    r1 = preset['T1_ratio'] if t1_ratio is None else float(t1_ratio)
    r2 = preset['T2_ratio'] if t2_ratio is None else float(t2_ratio)
    ar = (AGGRESSIVE_STEP / AGGRESSIVE_DENOM if aggressive_start_ratio is None
          else float(aggressive_start_ratio))
    a_start = int(ar * total_steps)

    if arch in ("sdxl", "sdxl_turbo"):
        if aggr:
            pool_start, pool_end = a_start, int(total_steps * r1)
            # downstream downsampler ignores r2 under aggressive mode
            stride_end = a_start
        else:
            pool_start, pool_end = 0, int(total_steps * r1)
            stride_end = int(total_steps * r2)
    else:
        # sd15/2.x: T1 -> downsampler stride, T2 -> avg_pool/bicubic
        stride_end = int(total_steps * r1)
        pool_start, pool_end = 0, int(total_steps * r2)

    return {
        "arch": arch,
        "preset": preset_name,
        "total_steps": total_steps,
        "ori_h": int(ori_h), "ori_w": int(ori_w),
        "t1_ratio": r1, "t2_ratio": r2,
        "aggressive": aggr,
        "aggressive_start": a_start,
        "pool_start": pool_start, "pool_end": pool_end,
        "stride_end": 0 if LAYOUTS[arch]["stride_input_block"] is None else stride_end,
    }


def pool_active(sched, step):
    return sched["pool_start"] <= step < sched["pool_end"]


def stride_active(sched, step):
    return step < sched["stride_end"]


# ---------------------------------------------------------------------------
# 5. MSW-MSA: window_partition / window_reverse
#
# Verbatim port of the two nested functions inside upstream's
# ``make_diffusers_transformer_block.forward``. Kept as free functions so the
# parity harness can call them without diffusers installed.
# ---------------------------------------------------------------------------

def _bicubic(x_bchw, size):
    try:
        return F.interpolate(x_bchw, size=size, mode="bicubic")
    except RuntimeError:
        # Some CUDA/torch builds reject bicubic on fp16; upstream fp16 pipelines
        # normally accept it, so this is only a compatibility fallback.
        return F.interpolate(x_bchw.float(), size=size, mode="bicubic").to(x_bchw.dtype)


def window_partition(x, window_size, shift_size, H, W):
    """``x:[B,N,C] -> [B*4, ws0*ws1, C]`` (2x2 windows, optional roll)."""
    B, N, C = x.shape
    x = x.view(B, H, W, C)
    if H % 2 != 0 or W % 2 != 0:
        x = _bicubic(x.permute(0, 3, 1, 2).contiguous(),
                     (window_size[0] * 2, window_size[1] * 2)) \
            .permute(0, 2, 3, 1).contiguous()
    if isinstance(shift_size, (list, tuple)):
        if shift_size[0] > 0:
            x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
    else:
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    x = x.view(B, 2, window_size[0], 2, window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
               .view(-1, window_size[0], window_size[1], C)
    return windows.view(-1, window_size[0] * window_size[1], C)


def window_reverse(windows, window_size, H, W, shift_size):
    """``[B*4, ws0*ws1, C] -> [B,H*W,C]``."""
    _B, _N, C = windows.shape
    windows = windows.view(-1, window_size[0], window_size[1], C)
    B = int(windows.shape[0] / 4)                       # 2x2 windows
    x = windows.view(B, 2, 2, window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous() \
         .view(B, window_size[0] * 2, window_size[1] * 2, -1)
    if isinstance(shift_size, (list, tuple)):
        if shift_size[0] > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
    else:
        if shift_size > 0:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
    if H % 2 != 0 or W % 2 != 0:
        x = _bicubic(x.permute(0, 3, 1, 2).contiguous(), (H, W)) \
            .permute(0, 2, 3, 1).contiguous()
    return x.view(B, H * W, C)


def choose_shift(window_size, mode="random", fixed_index=0):
    """Upstream draws ``torch.rand(1)`` once per transformer block per UNet call.

    ``mode="fixed"`` is provided for deterministic parity checks / reproducible
    output; it selects one of the four upstream shift candidates.
    """
    if mode == "fixed":
        idx = int(fixed_index) % 4
    else:
        r = float(torch.rand(1).item())
        if r <= 0.25:
            idx = 0
        elif r <= 0.5:
            idx = 1
        elif r <= 0.75:
            idx = 2
        else:
            idx = 3
    k = (window_size[0] // 4) * idx
    k2 = (window_size[1] // 4) * idx
    return (k, k2)


def msw_msa_geometry(ori_h, ori_w, N):
    """Upstream's window geometry from UNet input latent size + token count."""
    downsample_ratio = round(((ori_h * ori_w) / N) ** 0.5)
    if downsample_ratio < 1:
        downsample_ratio = 1
    H = math.ceil(ori_h / downsample_ratio)
    W = math.ceil(ori_w / downsample_ratio)
    window_size = (math.ceil(H / 2), math.ceil(W / 2))
    return H, W, window_size


def msw_msa(q, k, v, ori_h, ori_w, dim_head=64, *,
            shift_mode="random", fixed_index=0, shift_size=None):
    """Windowed self-attention in the *projected* q/k/v domain.

    Upstream partitions ``norm_hidden_states`` before ``attn1`` (which does
    to_q/k/v + SDPA + to_out) and reverses after to_out. Since projections are
    per-token linear, partitioning q/k/v instead is numerically identical;
    ComfyUI's ``patches_replace["attn1"]`` hands us exactly these q/k/v and
    applies ``to_out`` to whatever we return.
    """
    B, N, C = q.shape
    H, W, window_size = msw_msa_geometry(ori_h, ori_w, N)
    if H * W != N or min(H, W) < 2:
        return None                       # caller falls back to global attention
    if shift_size is None:
        shift_size = choose_shift(window_size, shift_mode, fixed_index)

    heads = C // dim_head
    if heads * dim_head != C:
        return None

    def split(t):
        return t.view(t.shape[0], -1, heads, dim_head).transpose(1, 2)

    qw = window_partition(q, window_size, shift_size, H, W)
    kw = window_partition(k, window_size, shift_size, H, W)
    vw = window_partition(v, window_size, shift_size, H, W)
    out = F.scaled_dot_product_attention(split(qw), split(kw), split(vw))
    out = out.transpose(1, 2).reshape(qw.shape[0], -1, heads * dim_head)
    out = window_reverse(out, window_size, H, W, shift_size)
    return out.to(q.dtype)
