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
                       EulerDiscreteScheduler, HeunDiscreteScheduler, UNet2DConditionModel,
                       AutoPipelineForText2Image, StableDiffusionXLControlNetImg2ImgPipeline, KDPM2DiscreteScheduler,
                       EulerAncestralDiscreteScheduler, UniPCMultistepScheduler, AutoencoderKL,
                       StableDiffusionXLControlNetPipeline, DDPMScheduler, TCDScheduler, LCMScheduler)

from .hidiffusion.hidiffusion import apply_hidiffusion
import folder_paths
from safetensors.torch import load_file


dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)

paths = []
paths_a = []
paths_unet = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "model_index.json" in files:
                paths.append(os.path.relpath(root, start=search_path))
            if "config.json" in files:
                paths_a.append(os.path.relpath(root, start=search_path))
                paths_a = ([
                               z for z in paths_a if "controlnet-canny-sdxl-1.0" in z]
                           + [p for p in paths_a if "MistoLine" in p]
                           + [o for o in paths_a if "lcm-sdxl" in o] + [Q for Q in paths_a if "controlnet-openpose-sdxl-1.0" in Q])

if paths != [] or paths_a != []:
    paths = [] + [x for x in paths if x] + [y for y in paths_a if y]
else:
    paths = ["no model in default diffusers directory", ]

scheduler_list = [
    "Euler",
    "Euler a",
    "DDIM",
    "DDPM",
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
    "Heun",
    "LCM",
    "LMS",
    "LMS Karras",
    "UniPC",
    "TCD"
]

sdxl_lightning_list = [
    "sdxl_lightning_1step_unet_x0.safetensors",
    "sdxl_lightning_2step_unet.safetensors",
    "sdxl_lightning_4step_unet.safetensors",
    "sdxl_lightning_8step_unet.safetensors"
    "Hyper-SDXL-1step-Unet.safetensors",
    "lcm-sdxl-base-1.0.safetensors",
]

normal_model_list = ["stable-diffusion-2-1-base", "stable-diffusion-v1-5", "playground-v2.5-1024px-aesthetic",
                     "Ghibli-Diffusion", "playground-v2-1024px-aesthetic"]


def get_sheduler(name):
    scheduler = False
    if name == "Euler":
        scheduler = EulerDiscreteScheduler()
    elif name == "Euler a":
        scheduler = EulerAncestralDiscreteScheduler()
    elif name == "DDIM":
        scheduler = DDIMScheduler()
    elif name == "DDPM":
        scheduler = DDPMScheduler()
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
    elif name == "LCM":
        scheduler = LCMScheduler()
    elif name == "LMS":
        scheduler = LMSDiscreteScheduler()
    elif name == "LMS Karras":
        scheduler = LMSDiscreteScheduler(use_karras_sigmas=True)
    elif name == "UniPC":
        scheduler = UniPCMultistepScheduler()
    elif name == "TCD":
        scheduler = TCDScheduler()
    return scheduler


def tensor_to_image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


def get_local_path(file_path, model_path):
    path = os.path.join(file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform.startswith('win32'):
        model_path = model_path.replace('\\', "/")
    return model_path


def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform.startswith('win32'):
        instance_path = instance_path.replace('\\', "/")
    return instance_path


class Diffusers_Or_Repo_Choice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_model_path": (paths,),
                "repo_id": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("repo_id",)
    FUNCTION = "repo_choice"
    CATEGORY = "Hidiffusion_Pro"

    def repo_choice(self, local_model_path, repo_id):
        if repo_id == "":
            if local_model_path == ["no model in default diffusers directory", ]:
                raise "you need fill repo_id or download model in diffusers directory "
            elif local_model_path != ["no model in default diffusers directory", ]:
                model_path = get_local_path(file_path, local_model_path)
                repo_id = get_instance_path(model_path)
        elif repo_id != "" and repo_id.find("/") == -1:
            raise "Incorrect repo_id format"
        elif repo_id != "" and repo_id.find("\\") != -1:
            repo_id = get_instance_path(repo_id)
        return (repo_id,)


class Hi_Text2Img:
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
                "repo_id": ("STRING", {"forceInput": True}),
                "scheduler": (scheduler_list,),
                "unet_model": ([] + folder_paths.get_filename_list("unet"),),
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

    def text2image(self, prompt, negative_prompt, repo_id, scheduler, unet_model, seed,
                   steps, cfg, eta, height, width):
        model_type = repo_id.rsplit("/")[-1]
        scheduler_used = get_sheduler(scheduler)
        if model_type == "stable-diffusion-xl-base-1.0":
            if unet_model in sdxl_lightning_list:
                light_path = os.path.join(file_path, "models", "unet", unet_model)
                ckpt = get_instance_path(light_path)
                # UNet2DConditionModel.load_config(repo_id, subfolder="unet")
                unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet").to("cuda", torch.float16)
                unet.load_state_dict(load_file(ckpt, device="cuda"), strict=False, )
                pipe = StableDiffusionXLPipeline.from_pretrained(repo_id, unet=unet, torch_dtype=torch.float16,
                                                                 variant="fp16").to("cuda")
                # Ensure sampler uses "trailing" timesteps.
                if unet_model == "lcm-sdxl-base-1.0.safetensors" or unet_model == "Hyper-SDXL-1step-Unet.safetensors":
                    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
                else:
                    pipe.scheduler = scheduler_used.from_config(pipe.scheduler.config, timestep_spacing="trailing")
            else:
                scheduler = scheduler_used.from_pretrained(repo_id, subfolder="scheduler")
                pipe = StableDiffusionXLPipeline.from_pretrained(repo_id, scheduler=scheduler,
                                                                 torch_dtype=torch.float16, variant="fp16").to("cuda")
        elif model_type == "sdxl-turbo":
            pipe = AutoPipelineForText2Image.from_pretrained(repo_id, torch_dtype=torch.float16,
                                                             variant="fp16").to('cuda')
        elif model_type in normal_model_list:
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
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=cfg, height=height, width=width, eta=eta,
                     seed=seed, negative_prompt=negative_prompt).images[0]
        output_image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
        return (output_image,)


class Hi_Control2Img:
    def __init__(self):
        pass

    def inpainting_gener(self, image, repo_id, controlnet_repo_id, prompt, negative_prompt, scheduler,
                         unet_model, model_type, seed, steps, cfg, eta, controlnet_strength, height, width, mask_image):
        scheduler_used = get_sheduler(scheduler)
        scheduler = scheduler_used.from_pretrained(repo_id, subfolder="scheduler")
        if unet_model in sdxl_lightning_list:
            light_path = os.path.join(file_path, "models", "unet", unet_model)
            ckpt = get_instance_path(light_path)
            # UNet2DConditionModel.load_config(repo_id, subfolder="unet")
            unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet").to("cuda", torch.float16)
            unet.load_state_dict(load_file(ckpt, device="cuda"), strict=False, )
            pipe = StableDiffusionXLPipeline.from_pretrained(repo_id, unet=unet, torch_dtype=torch.float16,
                                                             variant="fp16").to("cuda")
            # Ensure sampler uses "trailing" timesteps.
            if unet_model == "lcm-sdxl-base-1.0.safetensors" or unet_model == "Hyper-SDXL-1step-Unet.safetensors":
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
            else:
                pipe.scheduler = scheduler_used.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        else:
            pipe = AutoPipelineForInpainting.from_pretrained(
                controlnet_repo_id, torch_dtype=torch.float16, variant="fp16",
                scheduler=scheduler
            )
        apply_hidiffusion(pipe)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()

        output_img = pipe(prompt=prompt, image=image, mask_image=mask_image, height=height, width=width,
                          strength=controlnet_strength, num_inference_steps=steps, seed=seed,
                          guidance_scale=cfg, negative_prompt=negative_prompt, eta=eta).images[0]
        return output_img

    def control_img2img(self, image, prompt, negative_prompt, repo_id, controlnet_repo_id, scheduler,
                        unet_model, model_type, seed, steps, cfg, eta, controlnet_conditioning_scale,
                        controlnet_strength, height, width, control_image):
        scheduler_used = get_sheduler(scheduler)
        controlnet = ControlNetModel.from_pretrained(controlnet_repo_id, torch_dtype=torch.float16,
                                                     variant="fp16").to("cuda")
        if unet_model in sdxl_lightning_list:
            light_path = os.path.join(file_path, "models", "unet", unet_model)
            ckpt = get_instance_path(light_path)

            # UNet2DConditionModel.load_config(repo_id, subfolder="unet")
            unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet").to("cuda", torch.float16)
            unet.load_state_dict(load_file(ckpt, device="cuda"), strict=False, )
            pipe = StableDiffusionXLPipeline.from_pretrained(repo_id, unet=unet, torch_dtype=torch.float16,
                                                             variant="fp16").to("cuda")
            # Ensure sampler uses "trailing" timesteps.
            if unet_model == "lcm-sdxl-base-1.0.safetensors" or unet_model == "Hyper-SDXL-1step-Unet.safetensors":
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
            else:
                pipe.scheduler = scheduler_used.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        else:
            scheduler = scheduler_used.from_pretrained(repo_id, subfolder="scheduler")
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(repo_id, controlnet=controlnet,
                                                                              scheduler=scheduler,
                                                                              torch_dtype=torch.float16, ).to("cuda")
        apply_hidiffusion(pipe)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()

        output_img = pipe(prompt=prompt,
                          image=image,
                          control_image=control_image,
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
        return output_img

    def control_txt2img(self, prompt, negative_prompt, repo_id, controlnet_repo_id, scheduler,
                        unet_model, model_type, seed, steps, cfg, eta, controlnet_conditioning_scale,
                        controlnet_strength, height, width, control_image):
        scheduler_used = get_sheduler(scheduler)
        controlnet = ControlNetModel.from_pretrained(controlnet_repo_id, torch_dtype=torch.float16,
                                                     variant="fp16").to("cuda")
        if unet_model in sdxl_lightning_list:
            light_path = os.path.join(file_path, "models", "unet", unet_model)
            ckpt = get_instance_path(light_path)
            # UNet2DConditionModel.load_config(repo_id, subfolder="unet")
            unet = UNet2DConditionModel.from_config(repo_id, subfolder="unet").to("cuda", torch.float16)
            unet.load_state_dict(load_file(ckpt, device="cuda"), strict=False, )
            pipe = StableDiffusionXLPipeline.from_pretrained(repo_id, unet=unet, torch_dtype=torch.float16,
                                                             variant="fp16").to("cuda")
            # Ensure sampler uses "trailing" timesteps.
            if unet_model == "lcm-sdxl-base-1.0.safetensors" or unet_model == "Hyper-SDXL-1step-Unet.safetensors":
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
            else:
                pipe.scheduler = scheduler_used.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        else:
            scheduler = scheduler_used.from_pretrained(repo_id, subfolder="scheduler")
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                repo_id, controlnet=controlnet, torch_dtype=torch.float16,
                scheduler=scheduler
            )
        apply_hidiffusion(pipe)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()

        output_img = pipe(
            prompt=prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=control_image,
            height=height, width=width, guidance_scale=cfg, negative_prompt=negative_prompt, seed=seed,
            strength=controlnet_strength,
            num_inference_steps=steps, eta=eta
        ).images[0]
        return output_img

    def openpose_gener(self, prompt, negative_prompt, repo_id, controlnet_repo_id, scheduler,
                       unet_model, model_type, seed, steps, cfg, eta, controlnet_conditioning_scale,
                       controlnet_strength, height, width, mask_image):
        scheduler_used = get_sheduler(scheduler)
        controlnet = ControlNetModel.from_pretrained(controlnet_repo_id, torch_dtype=torch.float16,
                                                     variant="fp16").to("cuda")
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        if unet_model in sdxl_lightning_list:
            light_path = os.path.join(file_path, "models", "unet", unet_model)
            ckpt = get_instance_path(light_path)
            # UNet2DConditionModel.load_config(repo_id, subfolder="unet")
            unet = UNet2DConditionModel.from_config(repo_id, subfolder="unet").to("cuda", torch.float16)
            unet.load_state_dict(load_file(ckpt, device="cuda"), strict=False, )
            pipe = StableDiffusionXLPipeline.from_pretrained(repo_id, unet=unet, torch_dtype=torch.float16,
                                                             variant="fp16").to("cuda")
            # Ensure sampler uses "trailing" timesteps.
            if unet_model == "lcm-sdxl-base-1.0.safetensors" or unet_model == "Hyper-SDXL-1step-Unet.safetensors":
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
            else:
                pipe.scheduler = scheduler_used.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        else:
            scheduler = scheduler_used.from_pretrained(repo_id, subfolder="scheduler")
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                repo_id, controlnet=controlnet, vae=vae, torch_dtype=torch.float16,
                scheduler=scheduler
            )

        apply_hidiffusion(pipe)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_tiling()

        output_img = pipe(
            prompt=prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=mask_image,
            height=height, width=width, guidance_scale=cfg, negative_prompt=negative_prompt, seed=seed,
            strength=controlnet_strength,
            num_inference_steps=steps, eta=eta
        ).images[0]
        return output_img

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "repo_id": ("STRING", {"forceInput": True}),
                "controlnet_repo_id": ("STRING", {"forceInput": True}),
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

                "scheduler": (scheduler_list,),
                "unet_model": ([] + folder_paths.get_filename_list("unet"),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 12.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "controlnet_choice": (["text2img", "img2img", ],),
                "controlnet_conditioning_scale": (
                    "FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01}),
                "controlnet_strength": (
                    "FLOAT", {"default": 0.99, "min": 0.1, "max": 1.0, "step": 0.1, "round": 0.01}),
                "height": ("INT", {"default": 1536, "min": 64, "max": 8192, "step": 4, "display": "number"}),
                "width": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 4, "display": "number"})
            },
            "optional": {
                "control_image": ("IMAGE",),
                "mask_image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "xl_controlnet_image"
    CATEGORY = "Hidiffusion_Pro"

    def xl_controlnet_image(self, image, repo_id, controlnet_repo_id, prompt, negative_prompt, scheduler,
                            unet_model, seed, steps, cfg,
                            eta, controlnet_choice,
                            controlnet_conditioning_scale, controlnet_strength, height, width, control_image,
                            mask_image, ):

        image = tensor_to_image(image)
        control_image = np.transpose(control_image, (
        0, 3, 1, 2))  # 将comfy输入的（batch_size,channels,rows,cols）改成条件需求的（3通道，16输出，3，1 ，pading=1）
        mask_image = tensor_to_image(mask_image)
        model_type = repo_id.rsplit("/")[-1]
        control_model_type = controlnet_repo_id.rsplit("/")[-1]
        controlnet_suport = ["controlnet-canny-sdxl-1.0", "MistoLine"]

        if control_model_type == "stable-diffusion-xl-1.0-inpainting-0.1":
            output_img = self.inpainting_gener(image, repo_id, controlnet_repo_id, prompt, negative_prompt, scheduler,
                                               unet_model, model_type, seed, steps, cfg, eta, controlnet_strength,
                                               height, width, mask_image)
        elif control_model_type == "controlnet-openpose-sdxl-1.0":
            output_img = self.openpose_gener(prompt, negative_prompt, repo_id, controlnet_repo_id, scheduler,
                       unet_model, model_type, seed, steps, cfg, eta, controlnet_conditioning_scale,
                       controlnet_strength, height, width, mask_image)
        elif control_model_type in controlnet_suport:
            if controlnet_choice == "img2img":
                output_img = self.control_img2img(image, prompt, negative_prompt, repo_id, controlnet_repo_id,
                                                  scheduler,
                                                  unet_model, model_type, seed, steps, cfg, eta,
                                                  controlnet_conditioning_scale,
                                                  controlnet_strength, height, width, control_image)
            else:
                output_img = self.control_txt2img(prompt, negative_prompt, repo_id, controlnet_repo_id, scheduler,
                                                  unet_model, model_type, seed, steps, cfg, eta,
                                                  controlnet_conditioning_scale,
                                                  controlnet_strength, height, width, control_image)
        else:
            raise "Unsupported model_path or repo_id"
        output_img = torch.from_numpy(np.array(output_img).astype(np.float32) / 255.0).unsqueeze(0)
        return (output_img,)


NODE_CLASS_MAPPINGS = {
    "Diffusers_Or_Repo_Choice": Diffusers_Or_Repo_Choice,
    "Hi_Text2Img": Hi_Text2Img,
    "Hi_Control2Img": Hi_Control2Img
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Diffusers_Or_Repo_Choice": "Diffusers_Or_Repo_Choice",
    "Hi_Text2Img": "Hi_Text2Img",
    "Hi_Control2Img": "Hi_Control2Img"
}
