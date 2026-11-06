"""ComfyUI ModelPatcher injection points for HiDiffusion.

Everything here is attached to ``model_options["transformer_options"]`` of a
cloned ModelPatcher. ComfyUI source is NOT modified, and the UNet module
instance itself is untouched (no class swapping, no forward hooks on weights).

Injection points used (all stock ComfyUI):
  * patches["emb_patch"]                       -> compute the RAUNet schedule once
  * patches["input_block_patch"]              -> avg_pool down + remember size
  * patches["output_block_patch"]             -> bicubic restore before skip cat
  * patches["forward_timestep_embed_patch"]   -> resolution-aware Downsample conv
  * patches_replace["attn1"][key3]            -> MSW-MSA window self-attention

Why this works at all: ComfyUI's UNetModel already calls every Upsample with
``output_shape = hs[-1].shape`` (openaimodel.py). That is *exactly* what the
upstream obtains by hacking ``diffusion_model.num_upsamplers += 12`` in
diffusers, so the RAUNet resolution changes are absorbed automatically.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from comfy.ldm.modules.diffusionmodules.openaimodel import Downsample
from comfy.ldm.modules.attention import optimized_attention

from .core import (LAYOUTS, compute_step, build_schedule, pool_active,
                   stride_active, msw_msa, _bicubic)

# Keys we write into the per-forward transformer_options dict. It is freshly
# built for every UNet call, so there is no cross-run mutable state.
K_OWNER = "_hd_owner"
K_SCHED = "_hd_sched"
K_STEP = "_hd_step"
K_POOL = "_hd_pool_active"
K_STRIDE = "_hd_stride_active"
K_SIZE = "_hd_upsample_size"
K_WARNED = "_hd_schedule_warned"


def _mark(fn, state_id):
    fn.__hd_owner__ = state_id
    return fn


def _strip_fns(lst, state_id=None):
    """Remove previously installed HiDiffusion patches (re-apply = replace)."""
    out = []
    for x in lst:
        owner = getattr(x, "__hd_owner__", None)
        if owner is None:
            out.append(x)
        elif state_id is not None and owner != state_id:
            out.append(x)
    return out


def _strip_tuple_patches(lst):
    return [(c, h) for (c, h) in lst
            if getattr(h, "__hd_owner__", None) is None]


# ---------------------------------------------------------------------------
# schedule (emb_patch runs before any input_block / forward_timestep_embed)
# ---------------------------------------------------------------------------

def make_emb_patch(state_id, arch, opts):
    def emb_patch(emb, model_channels, transformer_options):
        to = transformer_options
        original_shape = to.get("original_shape")
        step_info = compute_step(to)
        if original_shape is None or step_info is None:
            to[K_SCHED] = None
            to[K_STEP] = 0
            to[K_POOL] = False
            to[K_STRIDE] = False
        else:
            step, total = step_info
            ori_h, ori_w = int(original_shape[2]), int(original_shape[3])
            sched = build_schedule(
                arch, ori_h, ori_w, total,
                aggressive=opts["aggressive"],
                t1_ratio=opts["t1_ratio"],
                t2_ratio=opts["t2_ratio"],
                controlnet_t2i=opts["controlnet_t2i"],
                aggressive_start_ratio=opts["aggressive_start_ratio"])
            to[K_SCHED] = sched
            to[K_STEP] = step
            to[K_POOL] = pool_active(sched, step)
            to[K_STRIDE] = stride_active(sched, step)
        to[K_OWNER] = state_id
        return emb
    return _mark(emb_patch, state_id)


# ---------------------------------------------------------------------------
# RAUNet T1: avg_pool after the first (resnet, attention) pair, bicubic back
# ---------------------------------------------------------------------------

def make_input_block_patch(state_id, layout, opts):
    def input_block_patch(h, transformer_options):
        to = transformer_options
        if not opts["apply_raunet"]:
            return h
        if to.get("block") == layout["pool_input_block"] and to.get(K_POOL):
            # upstream records the size BEFORE pooling, then avg_pool2d(2, ceil_mode=True)
            to[K_SIZE] = (h.shape[2], h.shape[3])
            h = F.avg_pool2d(h, kernel_size=(2, 2), ceil_mode=True)
        return h
    return _mark(input_block_patch, state_id)


def make_output_block_patch(state_id, layout, opts):
    def output_block_patch(h, hsp, transformer_options):
        to = transformer_options
        if not opts["apply_raunet"]:
            return (h, hsp)
        if to.get("block") == layout["pool_output_block"] and to.get(K_POOL):
            size = to.get(K_SIZE)
            if size is not None:
                h = _bicubic(h, size)
        return (h, hsp)
    return _mark(output_block_patch, state_id)


# ---------------------------------------------------------------------------
# RAUNet T2 / SD1.5 T1: resolution-aware downsampler conv
#   upstream temporarily sets stride=(4,4), padding=(2,2), dilation=(2,2)
#   on the SAME conv weights, then restores. We do exactly that via the layer's
#   own forward(), which also keeps comfy's cast-weights / fp8 path intact.
# ---------------------------------------------------------------------------

def make_stride_handler(state_id, layout, opts):
    def handler(layer, x, emb, context, transformer_options, output_shape,
                time_context, num_video_frames, image_only_indicator):
        to = transformer_options
        target = layout["stride_input_block"]
        if (not opts["apply_raunet"] or target is None
                or to.get("block") != target or not to.get(K_STRIDE)):
            return layer(x)
        op = getattr(layer, "op", None)
        if op is None or not hasattr(op, "stride") or not hasattr(op, "weight"):
            return layer(x)                # non-conv downsampler: upstream n/a
        original = (op.stride, op.padding, op.dilation)
        try:
            op.stride = (4, 4)
            op.padding = (2, 2)
            op.dilation = (2, 2)
            return layer(x)
        finally:
            op.stride, op.padding, op.dilation = original
    return _mark(handler, state_id)


# ---------------------------------------------------------------------------
# MSW-MSA: replace attn1 on the 10 (SDXL) / 5 (SD1.5) selected blocks
# ---------------------------------------------------------------------------

def make_msw_msa_patch(state_id, opts):
    def attn1_patch(q, k, v, extra_options):
        original_shape = extra_options.get("original_shape")
        dim_head = int(extra_options.get("dim_head", 64))
        heads = q.shape[-1] // dim_head
        if original_shape is None or heads * dim_head != q.shape[-1]:
            return optimized_attention(q, k, v, max(heads, 1))
        ori_h, ori_w = int(original_shape[2]), int(original_shape[3])
        out = msw_msa(q, k, v, ori_h, ori_w, dim_head,
                      shift_mode=opts["shift_mode"],
                      fixed_index=opts["shift_index"])
        if out is None:
            # Non-square / odd feature size that upstream could not partition:
            # keep stock global attention instead of crashing the whole sample.
            return optimized_attention(q, k, v, heads)
        return out
    return _mark(attn1_patch, state_id)


# ---------------------------------------------------------------------------
# install everything on a cloned ModelPatcher
# ---------------------------------------------------------------------------

def install(model_patcher, arch, layout, opts):
    """Attach all patches in-place on a cloned ModelPatcher's model_options."""
    state_id = opts["state_id"]
    to = model_patcher.model_options["transformer_options"]

    patches = dict(to.get("patches", {}))

    emb = _strip_fns(list(patches.get("emb_patch", [])))
    emb.append(make_emb_patch(state_id, arch, opts))
    patches["emb_patch"] = emb

    ibp = _strip_fns(list(patches.get("input_block_patch", [])))
    ibp.append(make_input_block_patch(state_id, layout, opts))
    patches["input_block_patch"] = ibp

    obp = _strip_fns(list(patches.get("output_block_patch", [])))
    obp.append(make_output_block_patch(state_id, layout, opts))
    patches["output_block_patch"] = obp

    ftep = _strip_tuple_patches(list(patches.get("forward_timestep_embed_patch", [])))
    if layout["stride_input_block"] is not None:
        ftep.append((Downsample, make_stride_handler(state_id, layout, opts)))
    patches["forward_timestep_embed_patch"] = ftep

    to["patches"] = patches

    # IMPORTANT: do NOT call set_model_patch_replace() - it replaces the whole
    # transformer_options dict and silently drops keys attached above.
    if opts["apply_window_attn"]:
        pr = dict(to.get("patches_replace", {}))
        a1 = dict(pr.get("attn1", {}))
        # drop any keys a previous HiDiffusion apply (different arch) may own
        for other in LAYOUTS.values():
            for key3 in other["window_attn1"]:
                if getattr(a1.get(key3), "__hd_owner__", None) is not None:
                    a1.pop(key3, None)
        fn = make_msw_msa_patch(state_id, opts)
        for key3 in layout["window_attn1"]:
            a1[key3] = fn
        pr["attn1"] = a1
        to["patches_replace"] = pr
    else:
        # ensure window attention is fully off (stock attn1 everywhere)
        pr = dict(to.get("patches_replace", {}))
        a1 = dict(pr.get("attn1", {}))
        for key3 in layout["window_attn1"]:
            if getattr(a1.get(key3), "__hd_owner__", None) is not None:
                a1.pop(key3, None)
        pr["attn1"] = a1
        to["patches_replace"] = pr

    return model_patcher
