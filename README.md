# ComfyUI_HiDiffusion_Pro
A  HiDiffusion node for ComfyUI

HiDiffusion  From: [link](https://github.com/megvii-research/HiDiffusion)  
----

Update 
----
07/27
修复模型加载过慢的问题

07/24
--新增社区单体模型支持，单体controlnet模型支持   
-- 通过菜单model_category来选择使用不同模型；   
--VAE可以填写你喜欢的vae地址 
--Added support for community monolithic models and monolithic ControlNet models     
--Select different models through the menu “model_category”;  
--VAE can fill in your favorite VAE address   

--
--增加adapter_style支持，SDXL需求的显存较大，虽然能跑CPU，但是不推荐，会爆显存，SD1.5测试没问题。  
--Adding adapter_style support, SDXL requires a large amount of graphics memory. Although it can run on CPU, it is not recommended as it may cause graphics memory "explosion". SD1.5 testing is not a problem.   

My ComfyUI node list：
-----

1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)      
13、ComfyUI_Streamv2v_Plus node:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   
15、ComfyUI_AnyDoor node: [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node: [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
17、ComfyUI_EchoMimic node:  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
18、ComfyUI_FollowYourEmoji node: [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   

Notice（节点的特殊功能说明 Special Function Description of Nodes）  
-----    
-- 增加 manne加速Lora  
-- 加入controlnet-tile-sdxl的支持，内置图片预处理，默认512尺寸，新增apply_window_attn 条件控制。  
--修复节点连接逻辑，现在文生图模式，无需接入image，无controlnet也无需接入control_image   
--支持SDXL-lighting\Hyper\LCM\DMD2\的加速Unet，建议适当提高步数；    
--基于官方的更新，加入lora支持，需要填关键词；    
--加入skip，去掉意义不大的其他参数；                       

-- add manne lighting lora  
--Added support for control net file sdxl, built-in image preprocessing, default size of 512, and added condition control for apply_window_attn.   
--Fix node connection logic, now in text-based graphics mode, there is no need to connect to image, no controllnet, and no need to connect to controll_image   
--Support acceleration Unet for SDXL lighting, Hyper, LCM, and DMD2. It is recommended to increase the number of steps appropriately;   
--Based on official updates, adding support for Lora requires filling in keywords;   
--Add skip and remove other parameters that are not significant;   


1.Installation
-----
  1.1 In the .\ComfyUI \ custom_node directory, run the following:   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_HiDiffusion_Pro.git   
  ```
  1.2 using it
  
2.requirements  
----
diffusers >=0.28.0   is best 
yaml

3 About models    
----
3.1 使用社区单体模型  Using the Community Monomer Model  

3.2 填写repo_id 和controlnet_repo_id     
    可以是stabilityai/stable-diffusion-xl-base-1.0   
    也可是“x:/xx/xx/stabilityai/stable-diffusion-xl-base-1.0”   
 
controlnet_repo_id 如果使用controlnet_repo_id，本地模型存放示例：
```
├── your path/   
|     ├──xinsir/controlnet-openpose-sdxl-1.0    
|     ├──xinsir/controlnet-scribble-sdxl-1.0   
|     ├──xinsir/controlnet-tile-sdxl-1.0  
|     ├──diffusers/controlnet-canny-sdxl-1.0   
|     ├──diffusers/controlnet-depth-sdxl-1.0   
|     ├──TheMistoAI/MistoLine 
```
ip_adapter model 模型存放示例： 

如果不存放对应，运行时会自动下载。  If the corresponding file is not stored, it will be automatically downloaded at runtime.  
```
├── ComfyUI/models/photomaker
|     ├──models/
|         ├── ip-adapter_sd15.bin
|         ├── image_encoder/
|             ├── config.json
|             ├── model.safetensors
|     ├──sdxl_models/
|         ├── ip-adapter_sdxl.bin
|         ├── image_encoder/
|             ├── config.json
|             ├── model.safetensors
```

4 other
----
部分支持的模型请查看model.yaml 文件    
For partially supported models, please refer to the model.yaml file   

5 example
-----
 sd1.5 using ip_adapter_style  使用ip_adapter_style  
 ![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/sd15ipstyle.png)

img2img use  lora     图生图和lora   
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/img2img_lora.png)


img2img + controlnet  图生图加controlnet      
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/controlnet_img2img.png)

img2img  use Hyper unet   图生图加加速unet   
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/lightingUnet.png)

6 Citation
------

``` python  
@article{zhang2023hidiffusion,
  title={HiDiffusion: Unlocking Higher-Resolution Creativity and Efficiency in Pretrained Diffusion Models},
  author={Zhang, Shen and Chen, Zhaowei and Zhao, Zhenyu and Chen, Yuhao and Tang, Yao and Liang, Jiajun},
  journal={arXiv preprint arXiv:2311.17528},
  year={2023}
}
