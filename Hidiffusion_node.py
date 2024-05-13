# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import os
from PIL import Image
import numpy as np
import sys
from diffusers import (StableDiffusionXLPipeline, DiffusionPipeline, DDIMScheduler, ControlNetModel,
                       KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler,
                       AutoPipelineForInpainting, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
                       EulerDiscreteScheduler, HeunDiscreteScheduler,
                       AutoPipelineForText2Image, StableDiffusionXLControlNetImg2ImgPipeline, KDPM2DiscreteScheduler,
                       EulerAncestralDiscreteScheduler, UniPCMultistepScheduler,
                       StableDiffusionXLControlNetPipeline)
from .hidiffusion.hidiffusion import apply_hidiffusion
import cv2
import folder_paths

dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)

paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "model_index.json" in files:
                paths.append(os.path.relpath(root, start=search_path))
            elif "config.json" in files:
                if "controlnet-canny-sdxl-1.0" in subdir:
                    paths.append(os.path.relpath(root, start=search_path))

if paths!=[]:
    paths = [] + [x for x in paths if x]
else:
    paths = ["no model in default diffusers directory",]

scheduler_list = [
    "Euler",
    "Euler a",
    "DDIM",
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "DPM++ 2M SDE",
    "DPM++ 2M SDE Karras",
    "DPM++ SDE",
    "DPM++ SDE Karras",
    "DPM2",
    "DPM2 Karras",
    "DPM2 a",
    "DPM2 a Karras",
    "DPM++ 3M SDE",
    "DPM++ 3M SDE",
    "Heun",
    "LMS",
    "LMS Karras",
    "UniPC",
    "UniPC_Bh2"
]


def get_sheduler(name):
    scheduler = False
    if name == "Euler":
        scheduler = EulerDiscreteScheduler()
    elif name == "Euler a":
        scheduler = EulerAncestralDiscreteScheduler()
    elif name == "DDIM":
        scheduler = DDIMScheduler()
    elif name == "DPM++ 2M":
        scheduler = DPMSolverMultistepScheduler()
    elif name == "DPM++ 2M Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True)
    elif name == "DPM++ 2M SDE":
        scheduler = DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ 2M SDE Karras":
        scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif name == "DPM++ SDE":
        scheduler = DPMSolverSinglestepScheduler()
    elif name == "DPM++ SDE Karras":
        scheduler = DPMSolverSinglestepScheduler(use_karras_sigmas=True)
    elif name == "DPM2":
        scheduler = KDPM2DiscreteScheduler()
    elif name == "DPM2 Karras":
        scheduler = KDPM2DiscreteScheduler(use_karras_sigmas=True)
    elif name == "DPM2 a":
        scheduler = KDPM2AncestralDiscreteScheduler()
    elif name == "DPM2 a Karras":
        scheduler = KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True)
    elif name == "Heun":
        scheduler = HeunDiscreteScheduler()
    elif name == "LMS":
        scheduler = LMSDiscreteScheduler()
    elif name == "LMS Karras":
        scheduler = LMSDiscreteScheduler(use_karras_sigmas=True)
    elif name == "UniPC":
        scheduler = UniPCMultistepScheduler()
    elif name == "UniPC_Bh2":
        scheduler = UniPCMultistepScheduler(solver_type="bh2")
    return scheduler


def get_local_path(file_path, model_path):
    path = os.path.join(file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform.startswith('win32'):
        model_path = model_path.replace('\\', "/")
    return model_path


class Hidiffusion_Text2Image:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                                      "default": "Standing tall amidst the ruins, a stone golem awakens, vines and "
                                                 "flowers sprouting from the crevices in its body"}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "blurry, ugly, duplicate, poorly drawn face, deformed, "
                                                          "mosaic, artifacts, bad limbs"}),
                "model_local_path": (paths,),
                "repo_id": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "scheduler": (scheduler_list,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "height": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number"}),
                "width": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "text2image"
    CATEGORY = "Hidiffusion_Pro"

    def text2image(self, prompt, negative_prompt, model_local_path, repo_id, scheduler, seed, steps, cfg, eta, height,
                   width):
        if model_local_path == ["no model in default diffusers directory",] and repo_id == "":
            raise "you need fill repo_id or download model in diffusers dir "

        model_path = get_local_path(file_path, model_local_path)
        if repo_id == "":
            repo_id = model_path
        model_type = repo_id.split("/")[-1]
        scheduler_used = get_sheduler(scheduler)
        model_list = ["stable-diffusion-2-1-base", "stable-diffusion-v1-5", "Ghibli-Diffusion"]
        if model_type == "stable-diffusion-xl-base-1.0":
            scheduler = scheduler_used.from_pretrained(repo_id, subfolder="scheduler")
            pipe = StableDiffusionXLPipeline.from_pretrained(repo_id, scheduler=scheduler,
                                                             torch_dtype=torch.float16, variant="fp16").to("cuda")
        elif model_type == "sdxl-turbo":
            pipe = AutoPipelineForText2Image.from_pretrained(repo_id, torch_dtype=torch.float16,
                                                             variant="fp16").to('cuda')
        elif model_type == "playground-v2-1024px-aesthetic":
            scheduler = scheduler_used.from_pretrained(repo_id, subfolder="scheduler")
            pipe = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler,
                                                     torch_dtype=torch.float16, use_safetensors=True,
                                                     add_watermarker=False, variant="fp16").to("cuda")
        elif model_type in model_list:
            scheduler = scheduler_used.from_pretrained(repo_id, subfolder="scheduler")
            pipe = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler,
                                                     torch_dtype=torch.float16, variant="fp16").to("cuda")
        else:
            raise "Unsupported model_path or repo_id"

        # Optional. enable_xformers_memory_efficient_attention can save memory usage and increase inference
        # speed. enable_model_cpu_offload and enable_vae_tiling can save memory usage.
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()
        # Apply hidiffusion with a single line of code.
        apply_hidiffusion(pipe)
        if steps > 0 and model_type == "sdxl-turbo":
            image = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg, height=height,
                         width=width, eta=eta, seed=seed,
                         negative_prompt=negative_prompt).images[0]
        else:
            image = pipe(prompt, guidance_scale=cfg, height=height, width=width, eta=eta, num_inference_steps=steps,
                         seed=seed,
                         negative_prompt=negative_prompt).images[0]

        output_image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
        return (output_image,)


class Hidiffusion_Controlnet_Image:

    def __init__(self):
        pass

    def tensor_to_image(self, tensor):
        tensor = tensor.cpu()
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        image = Image.fromarray(image_np, mode='RGB')
        return image

    def get_canny_image(self, img, a, b):  # get canny image
        img = np.array(img)
        img = cv2.Canny(img, a, b)
        img = img[:, :, None]
        img = np.concatenate([img, img, img], axis=2)
        canny_image = Image.fromarray(img)
        return canny_image

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True,
                                      "default": "Lara Croft with brown hair, and is wearing a tank top, a brown "
                                                 "backpack."
                                                 "The room is dark and has an old-fashioned decor with a patterned "
                                                 "floor and a wall"
                                                 "featuring a design with arches and a dark area on the right side, "
                                                 "muted color,"
                                                 " high detail, 8k high definition award winning"}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "underexposed, poorly drawn hands, duplicate hands, "
                                                          "overexposed,"
                                                          "bad art, beginner, amateur, abstract, disfigured, "
                                                          "deformed, close up, weird colors, watermark"}),
                "model_local_path": (paths,),
                "repo_id": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "scheduler": (scheduler_list,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "controlnet_local_path": (paths,),
                "controlnet_repo_id": ("STRING", {"default": "diffusers/controlnet-canny-sdxl-1.0"}),
                "cfg": ("FLOAT", {"default": 12.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "canny_minval": ("INT", {"default": 100, "min": 0, "max": 255}),
                "canny_maxval": ("INT", {"default": 200, "min": 0, "max": 255}),
                "controlnet_type": (["text2img", "img2img", ],),
                "controlnet_conditioning_scale": (
                    "FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01}),
                "controlnet_strength": (
                    "FLOAT", {"default": 0.99, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01}),
                "height": ("INT", {"default": 1536, "min": 64, "max": 8192, "step": 4, "display": "number"}),
                "width": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 4, "display": "number"}),
            },
            "optional": {
                "mask_image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "controlnet_image"
    CATEGORY = "Hidiffusion_Pro"

    def controlnet_image(self, image, prompt, negative_prompt, model_local_path, repo_id, scheduler,
                         controlnet_local_path, controlnet_repo_id, seed, steps, cfg,
                         eta, canny_minval, canny_maxval, controlnet_type,
                         controlnet_conditioning_scale, controlnet_strength, height, width,
                         mask_image):

        scheduler_used = get_sheduler(scheduler)
        if model_local_path == ["no model in default diffusers directory",] and repo_id == "":
            raise "you need fill repo_id or download model in diffusers dir "
        model_path = get_local_path(file_path, model_local_path)
        controlnet_local_path = get_local_path(file_path, controlnet_local_path)
        if repo_id == "":
            repo_id = model_path
        if controlnet_repo_id == "":
            controlnet_repo_id = controlnet_local_path

        # print(model_path,controlnet_local_path)

        ori_image = self.tensor_to_image(image)
        mask = self.tensor_to_image(mask_image)
        scheduler = scheduler_used.from_pretrained(repo_id, subfolder="scheduler")
        model_type = controlnet_repo_id.split("/")[-1]

        if model_type == "stable-diffusion-xl-1.0-inpainting-0.1":
            pipe = AutoPipelineForInpainting.from_pretrained(
                controlnet_repo_id, torch_dtype=torch.float16, variant="fp16",
                scheduler=scheduler
            )

            apply_hidiffusion(pipe)
            pipe.enable_xformers_memory_efficient_attention()
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_tiling()

            output_img = pipe(prompt=prompt, image=ori_image, mask_image=mask, height=height, width=width,
                              strength=controlnet_strength, num_inference_steps=steps, seed=seed,
                              guidance_scale=cfg, negative_prompt=negative_prompt, eta=eta).images[0]
        elif model_type == "controlnet-canny-sdxl-1.0":
            controlnet = ControlNetModel.from_pretrained(controlnet_repo_id, torch_dtype=torch.float16,
                                                         variant="fp16").to(
                "cuda")
            if controlnet_type == "img2img":
                # get canny image
                canny_image = self.get_canny_image(ori_image, canny_minval, canny_maxval)
                pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(repo_id, controlnet=controlnet,
                                                                                  scheduler=scheduler,
                                                                                  torch_dtype=torch.float16, ).to(
                    "cuda")
                apply_hidiffusion(pipe)
                pipe.enable_xformers_memory_efficient_attention()
                pipe.enable_model_cpu_offload()
                pipe.enable_vae_tiling()

                output_img = pipe(prompt=prompt,
                                  image=ori_image,
                                  control_image=canny_image,
                                  height=height,
                                  width=width,
                                  seed=seed,
                                  strength=controlnet_strength,
                                  num_inference_steps=steps,
                                  controlnet_conditioning_scale=controlnet_conditioning_scale,
                                  guidance_scale=cfg,
                                  negative_prompt=negative_prompt,
                                  eta=eta
                                  ).images[0]

            elif controlnet_type == "text2img":
                canny_image = self.get_canny_image(ori_image, canny_minval, canny_maxval)
                pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    repo_id, controlnet=controlnet, torch_dtype=torch.float16,
                    scheduler=scheduler
                )

                apply_hidiffusion(pipe)
                pipe.enable_xformers_memory_efficient_attention()
                pipe.enable_model_cpu_offload()
                pipe.enable_vae_tiling()

                output_img = pipe(
                    prompt=prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image,
                    seed=seed,
                    height=height, width=width, guidance_scale=cfg, negative_prompt=negative_prompt,
                    num_inference_steps=steps, eta=eta
                ).images[0]
            else:
                raise "Unsupported model"
        else:
            raise "Unsupported model_path or repo_id"

        output_image = torch.from_numpy(np.array(output_img).astype(np.float32) / 255.0).unsqueeze(0)
        return (output_image,)


NODE_CLASS_MAPPINGS = {
    "Hidiffusion_Text2Image": Hidiffusion_Text2Image,
    "Hidiffusion_Controlnet_Image": Hidiffusion_Controlnet_Image
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hidiffusion_Text2Image": "Hidiffusion_Text2Image",
    "Hidiffusion_Controlnet_Image": "Hidiffusion_Controlnet_Image"
}
