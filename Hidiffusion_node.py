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
                       StableDiffusionXLImg2ImgPipeline, AutoPipelineForImage2Image,
                       AutoPipelineForText2Image, StableDiffusionXLControlNetImg2ImgPipeline, KDPM2DiscreteScheduler,
                       EulerAncestralDiscreteScheduler, UniPCMultistepScheduler, AutoencoderKL,
                       StableDiffusionXLControlNetPipeline, DDPMScheduler, TCDScheduler, LCMScheduler,
                       StableDiffusionPipeline, StableDiffusionControlNetPipeline, StableDiffusionXLInpaintPipeline)
from diffusers.models.unets import unet_2d_condition
from diffusers.loaders.single_file_utils import load_single_file_checkpoint,infer_diffusers_model_type
from huggingface_hub import hf_hub_download

from .hidiffusion.hidiffusion import apply_hidiffusion,remove_hidiffusion
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
from .ip_adapter import IPAdapterXL,IPAdapter

dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)


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
                "function_choice": (["txt2img", "img2img", ],),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_id": (["none"] + folder_paths.get_filename_list("vae"),),
                "unet_model": (["none"] + folder_paths.get_filename_list("unet"),),
                "controlnet_model": (["none"] + folder_paths.get_filename_list("controlnet"),),
                "lora": (["none"] + folder_paths.get_filename_list("loras"),),
                "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "trigger_words": ("STRING", {"default": "best quality"}),
                "scheduler": (scheduler_list,),
                "apply_window_attn":("BOOLEAN", {"default": False},),
                "ip_adapter": ("BOOLEAN", {"default": False},),
            }
        }
     
   
    RETURN_TYPES = ("HIDIF_MODEL", )
    RETURN_NAMES = ("pipe", )
    FUNCTION = "loader_models"
    CATEGORY = "Hidiffusion_Pro"

    def loader_models(self,function_choice, ckpt_name,vae_id,unet_model, controlnet_model,
                      lora,lora_scale,trigger_words,scheduler,apply_window_attn,ip_adapter):
        
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name) if ckpt_name!="none" else None
        sd_type=""
        if ckpt_path:
            sd = load_single_file_checkpoint(ckpt_path)
            sd_type = infer_diffusers_model_type(sd)
            del sd
            
        
        vae_id=vae_id if vae_id!="none" else None
        controlnet_path=folder_paths.get_full_path("controlnet", controlnet_model) if controlnet_model!="none" else None
        unet_ckpt = folder_paths.get_full_path("unet", unet_model)  if unet_model!="none" else None
        
        scheduler_used = get_sheduler(scheduler)
        
        ip_path = os.path.join(folder_paths.models_dir, "photomaker")
        sdxl_models_pre = os.path.join(ip_path, "sdxl_models", "image_encoder")
        if not os.path.exists(sdxl_models_pre):
            os.makedirs(sdxl_models_pre)
        sd_models_pre = os.path.join(ip_path, "models", "image_encoder")
        if not os.path.exists(sd_models_pre):
            os.makedirs(sd_models_pre)
            
        if sd_type == "v1" or sd_type == "v2":
            model_type="stable-diffusion-v1-5"
            model_config=os.path.join(dir_path,"sd15_config")
            original_config_file = os.path.join(folder_paths.models_dir, "configs", "v1-inference.yaml")
            if dif_version_int >= 28:
                model = StableDiffusionPipeline.from_single_file(
                    ckpt_path, config=model_config,original_config=original_config_file, torch_dtype=torch.float16).to("cuda")
            else:
                model = StableDiffusionPipeline.from_single_file(
                    ckpt_path,config=model_config, original_config_file=original_config_file, torch_dtype=torch.float16).to("cuda")
    
        elif sd_type =="playground-v2-5":
            model_type = "playground-v2-1024px-aesthetic"
            model_config ="playgroundai/playground-v2.5-1024px-aesthetic"
            model = StableDiffusionXLPipeline.from_single_file(ckpt_path, config=model_config,torch_dtype=torch.float16).to("cuda")

        elif sd_type == "xl_inpaint":
            model_type ="stable-diffusion-xl-1.0-inpainting-0.1"
            model_config = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
            original_config_file = os.path.join(dir_path, "weights", "sd_xl_base.yaml")
            if dif_version_int >= 28:
                model = StableDiffusionXLInpaintPipeline.from_single_file(ckpt_path,config=model_config,
                                                                          original_config=original_config_file,
                                                                          torch_dtype=torch.float16,
                                                                        )
            else:
                model = StableDiffusionXLInpaintPipeline.from_single_file(ckpt_path,config=model_config,
                                                                          original_config_file=original_config_file,
                                                                          torch_dtype=torch.float16,
                                                                         )
            if unet_model in sdxl_lightning_list:
                if unet_model.rsplit('.', 1)[-1] == "bin":
                    model.unet.load_state_dict(torch.load(unet_ckpt),strict=False,)
                else:
                    model.unet.load_state_dict(load_file(unet_ckpt), strict=False, )
        elif sd_type == "xl_base":
            model_type = "stable-diffusion-xl-base-1.0"
            model_config=os.path.join(dir_path,"sdxl_config")
            original_config_file = os.path.join(dir_path, "weights", "sd_xl_base.yaml")
            
            if dif_version_int >= 28:
                model = StableDiffusionXLPipeline.from_single_file(
                    pretrained_model_link_or_path=ckpt_path, config=model_config,original_config=original_config_file, torch_dtype=torch.float16)
            else:
                model = StableDiffusionXLPipeline.from_single_file(
                    ckpt_path,config=model_config, original_config_file=original_config_file, torch_dtype=torch.float16)
                
            if controlnet_path:
                controlnet = ControlNetModel.from_unet(model.unet)
                cn_state_dict = load_file(controlnet_path)
                controlnet.load_state_dict(cn_state_dict, strict=False)
                if function_choice == "img2img":
                    model = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(ckpt_path,config=model_config,original_config=original_config_file,controlnet=controlnet,
                                                                                       torch_dtype=torch.float16,
                                                                                      )
                else:
                    model = StableDiffusionXLControlNetPipeline.from_single_file(
                        ckpt_path, config=model_config,original_config=original_config_file, controlnet=controlnet,
                        torch_dtype=torch.float16)
                    
            if unet_model in sdxl_lightning_list:
                if unet_model.rsplit('.', 1)[-1] == "bin":
                    model.unet.load_state_dict(torch.load(unet_ckpt), strict=False,)
                else:
                    model.unet.load_state_dict(load_file(unet_ckpt), strict=False, )
        else:
            raise "unsupport model!!"
        if vae_id:
            vae_id = folder_paths.get_full_path("vae", vae_id)
            if sd_type == "xl_base" or sd_type == "xl_inpaint":
                vae_config=os.path.join(dir_path,"sdxl_config","vae")
            elif sd_type == "v1" or sd_type == "v2" :
                vae_config=os.path.join(dir_path, "sd15_config","vae")
            elif sd_type == "playground-v2-5" :
                vae_config=os.path.join(dir_path,"weights/playground")
            else:
                raise "vae not support"
            model.vae = AutoencoderKL.from_single_file(vae_id,config=vae_config, torch_dtype=torch.float16).to("cuda")
        if  sd_type == "xl_inpaint":
            model.scheduler =scheduler_used.from_pretrained(os.path.join(dir_path,"sdxl_config"), subfolder="scheduler")
        else:
            model.scheduler = scheduler_used.from_config(model.scheduler.config, timestep_spacing="trailing")
        
        if lora!="none":
            lora_path = folder_paths.get_full_path("loras", lora)
            lora_path = get_instance_path(lora_path)
            model.load_lora_weights(lora_path, adapter_name=trigger_words)
            model.fuse_lora(lora_scale=lora_scale, adapter_names=[trigger_words,])
       
        model.enable_xformers_memory_efficient_attention()
        model.enable_vae_tiling()
        apply_hidiffusion(model,apply_window_attn=apply_window_attn,model_type_str=model_type)
        model.enable_model_cpu_offload()  # need below apply_hidiffusion(model)
     
        if ip_adapter:
            model.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
            ip_model_dir_origin = get_instance_path(ip_path)
            device = "cuda"
            if model_type == "stable-diffusion-xl-base-1.0":  # #ip-adapter_sdxl.bin
                ip_model_path_xl = os.path.join(ip_path, "sdxl_models", "ip-adapter_sdxl.bin")
               
                if not os.path.exists(ip_model_path_xl):
                    adapter_path_xl = hf_hub_download(
                        repo_id="h94/IP-Adapter",
                        subfolder="sdxl_models",
                        filename="ip-adapter_sdxl.bin",
                        local_dir=ip_model_dir_origin,
                    )
                else:
                    adapter_path_xl = get_instance_path(ip_model_path_xl)
                image_encoder_path_xl = get_instance_path(
                    os.path.join(ip_path, "sdxl_models", "image_encoder", "model.safetensors"))
                if not os.path.exists(ip_model_path_xl):
                    os.makedirs(ip_model_path_xl)
                ip_model_path_dir = get_instance_path(os.path.join(ip_path, "sdxl_models", "image_encoder"))
                if not os.path.exists(image_encoder_path_xl):
                    adapter_encoder_xl = hf_hub_download(
                        repo_id="h94/IP-Adapter",
                        subfolder="sdxl_models/image_encoder",
                        local_dir=ip_model_dir_origin,
                    )
                else:
                    adapter_encoder_xl = get_instance_path(ip_model_path_dir)
                model = IPAdapterXL(model, adapter_encoder_xl, adapter_path_xl, device,
                                    target_blocks=["up_blocks.0.attentions.1"])

            elif model_type == "stable-diffusion-v1-5":  # SD1.5 ip-adapter_sd15.bin
                # sd1.5
                ip_model_path = os.path.join(ip_path, "models", "ip-adapter_sd15.bin")
                if not os.path.exists(ip_model_path):
                    adapter_path_sd = hf_hub_download(
                        repo_id="h94/IP-Adapter",
                        subfolder="models",
                        filename="ip-adapter_sd15.bin",
                        local_dir=ip_model_dir_origin,
                    )
                else:
                    adapter_path_sd = get_instance_path(ip_model_path)

                image_encoder_path = get_instance_path(
                    os.path.join(ip_path, "models", "image_encoder", "model.safetensors"))
                ip_model_encoder_dir = get_instance_path(os.path.join(ip_path, "models", "image_encoder"))
                if not os.path.exists(image_encoder_path):
                    adapter_encoder = hf_hub_download(
                        repo_id="h94/IP-Adapter",
                        subfolder="models/image_encoder",
                        filename="model.safetensors",
                        local_dir=ip_model_dir_origin,
                    )
                else:
                    adapter_encoder = get_instance_path(ip_model_encoder_dir)
                model = IPAdapter(model, adapter_encoder, adapter_path_sd, device, target_blocks=["block"])
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        pipe={"model":model,"controlnet_path":controlnet_path,"sd_type":sd_type,"lora":lora,"trigger_words":trigger_words,"ip_adapter":ip_adapter,"function_choice":function_choice}
        torch.cuda.empty_cache()
        return (pipe,)
    

class Hi_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("HIDIF_MODEL",),
                "prompt": ("STRING", {"multiline": True,
                                      "default": "a girl,8k,smile,best quality"}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "default": "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"}),
                "controlnet_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "clip_skip": ("INT", {"default": 1, "min": -5, "max": 100,"step": 1}),
                "pre_input": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "width": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64, "display": "number"}),
                "adapter_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1,}),
            },
            "optional": {"image": ("IMAGE",),
                "control_image": ("IMAGE",),
                "ip_image": ("IMAGE",)}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "hi_sampler"
    CATEGORY = "Hidiffusion_Pro"


    def hi_sampler(self, pipe, prompt, negative_prompt,controlnet_scale,clip_skip,pre_input,
                   seed,steps, cfg, width,height,adapter_scale,**kwargs):
        model=pipe.get("model",None)
        controlnet_path = pipe["controlnet_path"]
        sd_type = pipe["sd_type"]
        lora = pipe["lora"]
        trigger_words = pipe["trigger_words"]
        ip_adapter = pipe["ip_adapter"]
        function_choice =pipe["function_choice"]
        
        if ip_adapter:
            ip_image = kwargs.get("ip_image")
            ip_image = input_size_adaptation_output(ip_image, pre_input, width, height)
            if lora != "none":
                prompt = prompt + " " + trigger_words
            if  controlnet_path is None:
                if function_choice == "img2img":
                    image = kwargs.get("image")
                    image = input_size_adaptation_output(image, pre_input, width, height)
                    images = \
                        model.generate(prompt=prompt, negative_prompt=negative_prompt,pil_image=ip_image, image=image, scale=adapter_scale,num_inference_steps=steps,
                              guidance_scale=cfg, clip_skip=clip_skip,
                              height=height, width=width, seed=seed, )
                else:
                    images = \
                        model.generate(pil_image=ip_image,prompt=prompt, negative_prompt=negative_prompt,scale=adapter_scale, num_inference_steps=steps,
                              guidance_scale=cfg, clip_skip=clip_skip,
                              height=height, width=width, seed=seed, )
    
            else:
                control_image = kwargs.get("control_image")
                if "tile" in controlnet_path:
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
                    control_image = input_size_adaptation_output(control_image, pre_input, width, height)
                if function_choice == "img2img":
                    image = kwargs["image"]
                    image = input_size_adaptation_output(image, pre_input, width, height)
                    if sd_type == "xl_inpaint":
                        images = model.generate(prompt=prompt, negative_prompt=negative_prompt, image=image,pil_image=ip_image, scale=adapter_scale,mask_image=control_image,
                                  num_inference_steps=steps, guidance_scale=cfg, height=height, clip_skip=clip_skip,
                                  width=width, controlnet_conditioning_scale=controlnet_scale,
                                  seed=seed, )
                    else:
                        images =  model.generate(prompt=prompt, negative_prompt=negative_prompt, image=image, pil_image=ip_image,scale=adapter_scale,control_image=control_image,
                                      num_inference_steps=steps, guidance_scale=cfg, height=height, width=width,
                                      clip_skip=clip_skip,
                                      controlnet_conditioning_scale=controlnet_scale,
                                      seed=seed, )
                else:
                    images = model.generate(prompt=prompt, negative_prompt=negative_prompt, pil_image=ip_image,scale=adapter_scale,control_image=control_image,
                                  num_inference_steps=steps, guidance_scale=cfg, height=height, width=width,
                                  clip_skip=clip_skip,
                                  controlnet_conditioning_scale=controlnet_scale,
                                  seed=seed, )
            images = images[0]
        else:
            if lora != "none":
                prompt = prompt + " " + trigger_words
            # print(model_type, unet_model, control_net, function_choice)
            if controlnet_path is None:
                if function_choice == "img2img":
                    image = kwargs["image"]
                    image = input_size_adaptation_output(image, pre_input, width, height)
                    images = \
                        model(prompt, negative_prompt=negative_prompt, image=image, num_inference_steps=steps,
                              guidance_scale=cfg, clip_skip=clip_skip,
                              height=height, width=width, seed=seed, ).images[0]
                else:
                    images = \
                        model(prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                              guidance_scale=cfg, clip_skip=clip_skip,
                              height=height, width=width, seed=seed, ).images[0]
            else:
                control_image = kwargs["control_image"]
                if "tile" in controlnet_path:
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
                    control_image = input_size_adaptation_output(control_image, pre_input, width, height)
                
                if function_choice == "img2img":
                    image = kwargs["image"]
                    image = input_size_adaptation_output(image, pre_input, width, height)
                    if sd_type == "xl_inpaint":
                        print("controlnet inpainting")
                        images = \
                            model(prompt, negative_prompt=negative_prompt, image=image, mask_image=control_image,
                                  num_inference_steps=steps, guidance_scale=cfg, height=height, clip_skip=clip_skip,
                                  width=width, controlnet_conditioning_scale=controlnet_scale,
                                  seed=seed, ).images[0]
                    else:
                        print("controlnet img2img")
                        images = model(prompt, negative_prompt=negative_prompt, image=image, control_image=control_image,
                                      num_inference_steps=steps, guidance_scale=cfg, height=height, width=width,
                                      clip_skip=clip_skip,
                                      controlnet_conditioning_scale=controlnet_scale,
                                      seed=seed, ).images[0]
                else:
                    print("controlnet txt2img")
                    images = model(prompt,control_image=control_image, negative_prompt=negative_prompt,
                                  num_inference_steps=steps, guidance_scale=cfg, height=height, width=width,
                                  clip_skip=clip_skip,
                                  controlnet_conditioning_scale=controlnet_scale,
                                  seed=seed, ).images[0]
        

        output_image = torch.from_numpy(np.array(images).astype(np.float32) / 255.0).unsqueeze(0)
        if lora != "none":
            if ip_adapter is None:
                model.unfuse_lora()
        torch.cuda.empty_cache()
        return (output_image,)


NODE_CLASS_MAPPINGS = {
    "HI_Diffusers_Model_Loader": HI_Diffusers_Model_Loader,
    "Hi_Sampler": Hi_Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HI_Diffusers_Model_Loader": "HI_Diffusers_Model_Loader",
    "Hi_Hi_Sampler": "Hi_Sampler"
}
