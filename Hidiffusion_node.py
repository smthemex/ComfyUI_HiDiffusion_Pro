# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2
import torch
import os
from PIL import Image
import numpy as np
import sys
from diffusers import (StableDiffusionXLPipeline, DiffusionPipeline, DDIMScheduler, ControlNetModel,
                       KDPM2AncestralDiscreteScheduler, LMSDiscreteScheduler,
                       AutoPipelineForInpainting, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler,
                       EulerDiscreteScheduler, HeunDiscreteScheduler, UNet2DConditionModel,
                       StableDiffusionXLImg2ImgPipeline,AutoPipelineForImage2Image,
                       AutoPipelineForText2Image, StableDiffusionXLControlNetImg2ImgPipeline, KDPM2DiscreteScheduler,
                       EulerAncestralDiscreteScheduler, UniPCMultistepScheduler, AutoencoderKL,
                       StableDiffusionXLControlNetPipeline, DDPMScheduler, TCDScheduler, LCMScheduler,
                       StableDiffusionPipeline, StableDiffusionControlNetPipeline)
from .hidiffusion.hidiffusion import apply_hidiffusion
import folder_paths
from safetensors.torch import load_file, load
import yaml
import diffusers
import random

dif_version = str(diffusers.__version__)
dif_version_int = int(dif_version.split(".")[1])
if dif_version_int >= 28:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
else:
    from diffusers.models.unet_2d_condition import UNet2DConditionModel
from comfy.utils import common_upscale
from .guided_filter import FastGuidedFilter
dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)

paths = []
paths_a = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "model_index.json" in files:
                paths.append(os.path.relpath(root, start=search_path))
            if "config.json" in files:
                paths_a.append(os.path.relpath(root, start=search_path))
                paths_a = ([z for z in paths_a if "controlnet-canny-sdxl-1.0" in z]
                           + [p for p in paths_a if "MistoLine" in p]
                           + [o for o in paths_a if "lcm-sdxl" in o]
                           + [Q for Q in paths_a if "controlnet-openpose-sdxl-1.0" in Q]
                           + [Z for Z in paths_a if "controlnet-scribble-sdxl-1.0" in Z]
                           +[x for x in paths_a if "controlnet-tile-sdxl-1.0" in x])

if paths != [] or paths_a != []:
    paths = ["none"] + [x for x in paths if x] + [y for y in paths_a if y]
else:
    paths = ["none", ]

scheduler_list = ["DDIM",
    "Euler",
    "Euler a",
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
]

fs = open(os.path.join(dir_path, "model.yaml"), encoding="UTF-8")
datas = yaml.load(fs, Loader=yaml.FullLoader)

normal_model_list = datas["surport_model"]
sdxl_lightning_list = datas["lightning_unet"]
controlnet_suport = datas["surport_controlnet"]
xl_model_support = datas["sdxl_model"]
lightning_lora=datas["lightning_lora"]
lightning_xl_lora=datas["lightning_xl_lora"]

lcm_unet = ["dmd2_sdxl_4step_unet_fp16.bin", "dmd2_sdxl_1step_unet_fp16.bin", "lcm-sdxl-base-1.0.safetensors",
            "Hyper-SDXL-1step-Unet.safetensors"]

def tensor_to_image(tensor):
    #tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def nomarl_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_image(samples)
    return img_pil

def resize_image_control(control_image, resolution):
    HH, WW, _ = control_image.shape
    crop_h = random.randint(0, HH - resolution[1])
    crop_w = random.randint(0, WW - resolution[0])
    crop_image = control_image[crop_h:crop_h+resolution[1], crop_w:crop_w+resolution[0], :]
    return crop_image, crop_w, crop_h

def apply_gaussian_blur(image_np, ksize=5, sigmaX=1.0):
    if ksize % 2 == 0:
        ksize += 1  # ksize must be odd
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=sigmaX)
    return blurred_image

def apply_guided_filter(image_np, radius, eps, scale):
    filter = FastGuidedFilter(image_np, radius, eps, scale)
    return filter.filter(image_np)

def input_size_adaptation_output(img_tensor,base_in, width, height):
    #basein=1024
    if width == height:
        img_pil = nomarl_upscale(img_tensor, base_in, base_in)  # 2pil
    else:
        if min(1,width/ height)==1: #高
            r=height/base_in
            img_pil = nomarl_upscale(img_tensor, round(width/r), base_in)  # 2pil
        else: #宽
            r=width/base_in
            img_pil = nomarl_upscale(img_tensor, base_in, round(height/r))  # 2pil
    return img_pil

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
    return scheduler


def get_local_path(file_path, model_path):
    path = os.path.join(file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform == 'win32':
        model_path = model_path.replace('\\', "/")
    return model_path


def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


def instance_path(path, repo):
    if repo == "":
        if path == "none":
            repo = "none"
        else:
            model_path = get_local_path(file_path, path)
            repo = get_instance_path(model_path)
    return repo


class HI_Diffusers_Model_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_model_path": (paths,),
                "repo_id": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "unet_model": (["none"] + folder_paths.get_filename_list("unet"),),
                "controlnet_local_model": (paths,),
                "controlnet_repo_id": ("STRING", {"default": "diffusers/controlnet-canny-sdxl-1.0"}),
                "function_choice": (["txt2img", "img2img", ],),
                "scheduler": (scheduler_list,),
                "lora": (["none"] + folder_paths.get_filename_list("loras"),),
                "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "trigger_words": ("STRING", {"default": "best quality"}),
                "apply_window_attn":("BOOLEAN", {"default": False},)
            }
        }

    RETURN_TYPES = ("MODEL", "STRING",)
    RETURN_NAMES = ("model", "model_info",)
    FUNCTION = "loader_models"
    CATEGORY = "Hidiffusion_Pro"

    def loader_models(self, local_model_path, repo_id, unet_model, controlnet_local_model, controlnet_repo_id,
                      function_choice,scheduler,lora,lora_scale,trigger_words,apply_window_attn):
        repo_id = instance_path(local_model_path, repo_id)
        controlnet_repo_id = instance_path(controlnet_local_model, controlnet_repo_id)
        scheduler_used = get_sheduler(scheduler)
        if repo_id == "none":
            raise "need repo_id or model path"
        model_type = repo_id.rsplit("/")[-1]
        if controlnet_repo_id=="none":
            control_model_type="none"
        else:
            control_model_type = controlnet_repo_id.rsplit("/")[-1]
        if control_model_type == "none":
            if model_type in xl_model_support:
                if unet_model in sdxl_lightning_list:
                    light_path = os.path.join(file_path, "models", "unet", unet_model)
                    ckpt = get_instance_path(light_path)
                    if unet_model.rsplit('.', 1)[-1] == "bin":
                        unet = UNet2DConditionModel.from_config(repo_id, subfolder="unet").to("cuda", torch.float16)
                        unet.load_state_dict(torch.load(ckpt))
                    else:
                        unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet").to("cuda", torch.float16)
                        unet.load_state_dict(load_file(ckpt, device="cuda"), strict=False, )
                    if function_choice == "img2img":
                        model = StableDiffusionXLImg2ImgPipeline.from_pretrained(repo_id, unet=unet, torch_dtype=torch.float16,
                                                                          variant="fp16").to("cuda")
                    else:
                        model = StableDiffusionXLPipeline.from_pretrained(repo_id, unet=unet, torch_dtype=torch.float16,
                                                                          variant="fp16").to("cuda")
                else:
                    if function_choice == "img2img":
                        model = StableDiffusionXLImg2ImgPipeline.from_pretrained(repo_id,
                                                                          torch_dtype=torch.float16, variant="fp16").to(
                            "cuda")
                    else:
                        model = StableDiffusionXLPipeline.from_pretrained(repo_id,
                                                                          torch_dtype=torch.float16, variant="fp16").to(
                            "cuda")

            elif model_type == "sdxl-turbo":
                if function_choice == "img2img":
                    model = AutoPipelineForImage2Image.from_pretrained(repo_id, torch_dtype=torch.float16,
                                                                      variant="fp16").to('cuda')
                else:
                    model = AutoPipelineForText2Image.from_pretrained(repo_id, torch_dtype=torch.float16,
                                                                      variant="fp16").to('cuda')
            elif model_type in normal_model_list:
                if model_type=="Ghibli-Diffusion":
                    model = StableDiffusionPipeline.from_pretrained(repo_id,
                                                              torch_dtype=torch.float16, variant="fp16").to("cuda")
                else:
                    model = DiffusionPipeline.from_pretrained(repo_id,
                                                                    torch_dtype=torch.float16, variant="fp16").to(
                        "cuda")

            else:
                raise "Unsupported model_path or repo_id"
        else: #using controlnet
            if control_model_type != "stable-diffusion-xl-1.0-inpainting-0.1":
                controlnet = ControlNetModel.from_pretrained(controlnet_repo_id, torch_dtype=torch.float16,
                                          variant="fp16").to("cuda")
            else:
                controlnet=""
            if model_type in xl_model_support:
                if unet_model in sdxl_lightning_list:
                    light_path = os.path.join(file_path, "models", "unet", unet_model)
                    ckpt = get_instance_path(light_path)
                    if unet_model.rsplit('.', 1)[-1] == "bin":
                        unet = UNet2DConditionModel.from_config(repo_id, subfolder="unet").to("cuda", torch.float16)
                        unet.load_state_dict(torch.load(ckpt))
                    else:
                        unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet").to("cuda", torch.float16)
                        unet.load_state_dict(load_file(ckpt, device="cuda"), strict=False, )
                    if control_model_type == "stable-diffusion-xl-1.0-inpainting-0.1":
                        model = AutoPipelineForInpainting.from_pretrained(repo_id, unet=unet,
                                                                          torch_dtype=torch.float16,
                                                                          variant="fp16").to("cuda")
                    elif function_choice == "img2img":
                        model = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(repo_id, unet=unet,
                                                                                           controlnet=controlnet,
                                                                                           torch_dtype=torch.float16,
                                                                                           variant="fp16").to("cuda")
                    else:
                        model = StableDiffusionXLControlNetPipeline.from_pretrained(repo_id, unet=unet,
                                                                                    controlnet=controlnet,
                                                                                    torch_dtype=torch.float16,
                                                                                    variant="fp16").to("cuda")
                else:
                    if control_model_type == "stable-diffusion-xl-1.0-inpainting-0.1":
                        model = AutoPipelineForInpainting.from_pretrained(repo_id,
                                                                          torch_dtype=torch.float16,
                                                                          variant="fp16").to("cuda")
                    elif function_choice == "img2img":
                        model = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(repo_id,
                                                                                           controlnet=controlnet,
                                                                                           torch_dtype=torch.float16,
                                                                                           variant="fp16").to("cuda")
                    else:
                        model = StableDiffusionXLControlNetPipeline.from_pretrained(repo_id,
                                                                                    controlnet=controlnet,
                                                                                    torch_dtype=torch.float16,
                                                                                    variant="fp16").to("cuda")

            elif model_type == "sdxl-turbo":
                model = StableDiffusionXLControlNetPipeline.from_pretrained(repo_id, controlnet=controlnet,
                                                                            torch_dtype=torch.float16,
                                                                          variant="fp16").to('cuda')
            else:
                raise "only support SDXL controlnet"
        model.scheduler = scheduler_used.from_config(model.scheduler.config, timestep_spacing="trailing")
        if lora!="none":
            lora_path = folder_paths.get_full_path("loras", lora)
            lora_path = get_instance_path(lora_path)
            # if lora in lightning_lora or lora in lightning_xl_lora:
            #     model.load_lora_weights(lora_path)
            #     model.fuse_lora()
            model.load_lora_weights(lora_path, adapter_name=trigger_words)
            model.fuse_lora(lora_scale=lora_scale, adapter_names=[trigger_words,])

        # Optional. enable_xformers_memory_efficient_attention can save memory usage and increase inference
        # speed. enable_model_cpu_offload and enable_vae_tiling can save memory usage.
        # Apply hidiffusion with a single line of code.
        apply_hidiffusion(model,apply_window_attn=apply_window_attn)
        model.enable_xformers_memory_efficient_attention()
        model.enable_model_cpu_offload()
        model.enable_vae_tiling()

        model_info = str(";".join([model_type, unet_model, control_model_type, function_choice,lora,trigger_words]))

        return (model, model_info,)


class Hi_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_info": ("STRING", {"forceInput": True}),
                "prompt": ("STRING", {"multiline": True,
                                      "default": "a girl,8k,smile,best quality"}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "blurry, ugly, duplicate, poorly drawn face, deformed, "
                                                          "mosaic, artifacts, bad limbs"}),

                "controlnet_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "clip_skip": ("INT", {"default": 1, "min": -5, "max": 100,"step": 1}),
                "pre_input": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "width": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number"}),
            },
            "optional": {"image": ("IMAGE",),
                "control_image": ("IMAGE",)}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "hi_sampler"
    CATEGORY = "Hidiffusion_Pro"


    def hi_sampler(self, model, model_info, prompt, negative_prompt,  controlnet_scale,clip_skip,pre_input,
                   seed,steps, cfg,  width,height,**kwargs):
        model_type, unet_model, control_net, function_choice, lora, trigger_words = model_info.split(";")
        # control_image = np.transpose(control_image, (
        #         #     0, 3, 1, 2))  # 将comfy输入的（batch_size,channels,rows,cols）改成条件需求的（3通道，16输出，3，1 ，pading=1）
        if lora!="none":
            prompt = prompt + " " + trigger_words
        #print(model_type, unet_model, control_net, function_choice)
        if control_net == "none":
            if function_choice == "img2img":
                image = kwargs["image"]
                image = input_size_adaptation_output(image, pre_input, width, height)
                image = \
                    model(prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=steps,
                          guidance_scale=cfg, clip_skip=clip_skip,
                          height=height, width=width, seed=seed, ).images[0]
            else:
                image = \
                    model(prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                          guidance_scale=cfg, clip_skip=clip_skip,
                          height=height, width=width, seed=seed, ).images[0]
        else:
            control_image = kwargs["control_image"]
            if control_net == "controlnet-tile-sdxl-1.0":
                control_image = input_size_adaptation_output(control_image, pre_input, width, height)
                controlnet_img = cv2.cvtColor(np.asarray(control_image), cv2.COLOR_RGB2BGR)
                new_height, new_width, _ = controlnet_img.shape
                ratio = np.sqrt(1024. * 1024. / (new_width * new_height))
                W, H = int(new_width * ratio), int(new_height * ratio)

                crop_w, crop_h = 0, 0
                controlnet_img = cv2.resize(controlnet_img, (W, H))

                blur_strength = random.sample([i / 10. for i in range(10, 201, 2)], k=1)[0]
                radius = random.sample([i for i in range(1, 40, 2)], k=1)[0]
                eps = random.sample([i / 1000. for i in range(1, 101, 2)], k=1)[0]
                scale_factor = random.sample([i / 10. for i in range(10, 181, 5)], k=1)[0]

                if random.random() > 0.5:
                    controlnet_img = apply_gaussian_blur(controlnet_img, ksize=int(blur_strength),
                                                         sigmaX=blur_strength / 2)

                if random.random() > 0.5:
                    # Apply Guided Filter
                    controlnet_img = apply_guided_filter(controlnet_img, radius, eps, scale_factor)

                # Resize image
                controlnet_img = cv2.resize(controlnet_img, (int(W / scale_factor), int(H / scale_factor)),
                                            interpolation=cv2.INTER_AREA)
                controlnet_img = cv2.resize(controlnet_img, (W, H), interpolation=cv2.INTER_CUBIC)

                controlnet_img = cv2.cvtColor(controlnet_img, cv2.COLOR_BGR2RGB)
                control_image = Image.fromarray(controlnet_img)
            else:
                control_image=input_size_adaptation_output(control_image, pre_input, width, height)

            if function_choice == "img2img":
                image = kwargs["image"]
                image=input_size_adaptation_output(image, pre_input, width, height)
                if control_net == "stable-diffusion-xl-1.0-inpainting-0.1":
                    image = \
                        model(prompt, negative_prompt=negative_prompt, image=image, mask_image=control_image,
                              num_inference_steps=steps, guidance_scale=cfg, height=height, clip_skip=clip_skip,
                              width=width, controlnet_conditioning_scale=controlnet_scale,
                              seed=seed, ).images[0]
                else:
                    image = model(prompt, negative_prompt=negative_prompt, image=image, control_image=control_image,
                              num_inference_steps=steps, guidance_scale=cfg, height=height, width=width,
                              clip_skip=clip_skip,
                              controlnet_conditioning_scale=controlnet_scale,
                              seed=seed, ).images[0]
            else:
                image = model(prompt, negative_prompt=negative_prompt, image=control_image,
                              num_inference_steps=steps, guidance_scale=cfg, height=height, width=width,
                              clip_skip=clip_skip,
                              controlnet_conditioning_scale=controlnet_scale,
                              seed=seed, ).images[0]

        output_image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
        model.unfuse_lora()
        del model
        return (output_image,)


NODE_CLASS_MAPPINGS = {
    "HI_Diffusers_Model_Loader": HI_Diffusers_Model_Loader,
    "Hi_Sampler": Hi_Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HI_Diffusers_Model_Loader": "HI_Diffusers_Model_Loader",
    "Hi_Hi_Sampler": "Hi_Sampler"
}
