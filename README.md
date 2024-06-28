# ComfyUI_HiDiffusion_Pro
A  HiDiffusion node for ComfyUI

HiDiffusion  From: [link](https://github.com/megvii-research/HiDiffusion)  
----

Update 
----
-- 加入controlnet-tile-sdxl的支持，内置图片预处理，默认512尺寸，新增apply_window_attn 条件控制。  
--修复节点连接逻辑，现在文生图模式，无需接入image，无controlnet也无需接入control_image   
----Added support for control net file sdxl, built-in image preprocessing, default size of 512, and added condition control for apply_window_attn.   
--Fix node connection logic, now in text-based graphics mode, there is no need to connect to image, no controllnet, and no need to connect to controll_image   

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

Notice（节点的特殊功能说明 Special Function Description of Nodes）  
-----    

--支持SDXL-lighting\Hyper\LCM\DMD2\的加速Unet，建议适当提高步数；    
--基于官方的更新，加入lora支持，需要填关键词；    
--加入skip，去掉意义不大的其他参数；    
--支持所有的SDXL controlnet模型（仅支持配置config文件的diffuser加载模式）；         
--你可以修改model.yaml文件，添加其他的可能支持SDXL“扩散模型”或者“controlnet”或者SDXL“unet”模型；                 


--Support acceleration Unet for SDXL lighting, Hyper, LCM, and DMD2. It is recommended to increase the number of steps appropriately;   
--Based on official updates, adding support for Lora requires filling in keywords;   
--Add skip and remove other parameters that are not significant;   
--Supports all SDXL controllnet models (only supports diffuser loading mode for configuring config files);   
--You can modify the model.yaml file and add other models that may support SDXL "diffusion models" or "controllnet" or SDXL "unet" models;   


1.Installation
-----
  1.1 In the .\ComfyUI \ custom_node directory, run the following:   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_HiDiffusion_Pro.git   
  ```
  1.2 using it
  
2.requirements  
----
need diffusers >=0.27.0  
yaml

3 About models    
----
  3种调用模型节点的方法   
  一种是,在repo_id填写诸如"stabilityai/stable-diffusion-xl-base-1.0" 这样的标准抱脸的repo_id,系统会自己下载或者查找.cache路径，这个要开全局外网或者你开启了hf镜像  
  
  一种是，把你下载好的模型放在comfyUI的"models/diffusers"路径下，记得不要改最尾的模型路径名“比如stable-diffusion-xl-base-1.0”，就可以用菜单直接调用。（注意：要用菜单的方式，必须删掉repo_id默认的"stabilityai/stable-diffusion-xl-base-1.0"，让repo_id留空，controlnet_id一样，要调用菜单，必须留空controlnet_id） 
  
  一种是，你在repo_id或controlnet_id直接填写诸如“x:/xx/xx/stabilityai/stable-diffusion-xl-base-1.0” 这样的已经下载好的扩散模型的绝对路径也能用

controlnet模型存放示例：
```
├── ComfyUI/models/diffusers/   
|     ├──xinsir/controlnet-openpose-sdxl-1.0    
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors   
|     ├──xinsir/controlnet-scribble-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
|     ├──xinsir/controlnet-tile-sdxl-1.0  
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors  
|     ├──diffusers/controlnet-canny-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors   
|     ├──diffusers/controlnet-depth-sdxl-1.0   
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
|     ├──TheMistoAI/MistoLine 
|         ├── config.json   
|         ├── diffusion_pytorch_model.fp16.safetensors
```
  
  Three methods for calling model nodes   
  Filling in the corresponding repo_id will download the model   
  or   
  in the diffuse directory, store and use the downloaded model  （Notice:in this method,repo_id or controlnet_repo_id must be empty）  
  or  
  using repo_id,and Fill in the absolute path of your diffusers model,using"/"   

4 other
----
部分支持的模型请查看model.yaml 文件    
For partially supported models, please refer to the model.yaml file   

5 example
-----
img2img use controlnet and lora     图生图加controlnet和lora   
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/img2imgcontrolnetlora.png)

use inpainting XL model  内绘  
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/inpainting.png)

txt2img  use XL/sd1.5/XL turbo/background...    文生图   
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/txt2img.png)

txt2img + controlnet  文生图加controlnet  .     
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/txt2imgcontrolnet.png
 )

img2img  use Hyper unet   图生图加加速unet   
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/img2imgunet.png)

6 Citation
------

``` python  
@article{zhang2023hidiffusion,
  title={HiDiffusion: Unlocking Higher-Resolution Creativity and Efficiency in Pretrained Diffusion Models},
  author={Zhang, Shen and Chen, Zhaowei and Zhao, Zhenyu and Chen, Yuhao and Tang, Yao and Liang, Jiajun},
  journal={arXiv preprint arXiv:2311.17528},
  year={2023}
}
