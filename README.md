# ComfyUI_HiDiffusion_Pro
A  HiDiffusion node for ComfyUI

Update
-----

2025-05-31 update  
--增加了DMD2加速模型，但是还有bug暂时无法使用。
--修复controlnet图片的控制，减少一个图片输入几点
--增加了SDXL-scribble 模型的支持

Previous updates
--增加了model.yaml文件，你可以修改该文件，添加其他的可能支持“扩散模型”或者“controlnet”或者“unet”模型 ，所有目前支持的模型的repo_id都在该文件的注释里,除了openpose和inpainting   
--增加了SDXL flash 加速扩散模型的支持，也就是说controlnet节点可以的SDXL模型，还可以用SDXL flash 扩散模型  
--增加了对anyline（基于mistuline模型）和controlnet 线稿类的canny支持，增加了openposeXL模型的支持（须搭配controlnet openpose）   
--增加了模型输入节点，该节点的菜单用于已下载的本地模型，repo_id更方便，但需要网络支持
--所有节点都支持XL-lighting\Hyper\LCM Unet加速，但是低步数的效果一般

--Added model.yaml file, you can modify it and add other models that may support “diffusers model”, “controllnet”, or “unet” models
--Added support for the SDXL flash accelerated diffusion model, which means that the control net node can use the SDXL model, and the SDXL flash diffusion model can also be used  
--Added Canny support for Anyline (based on the pipeline model) and Controlnet draft classes, and added support for Openpose XL model (to be paired with Controlnet Openpose)   
--Added a model input node, whose menu is used for downloaded local models. Repo_id is more convenient, but requires network support   
--All nodes support XL lighting \ Hyper \ LCM Unet acceleration, but the effect of low steps is average   


NOTICE
----
注意，如果使用本地下载的模型，注意要配套config文件，以及有些模型是需要bin模型，或者模型名称要一样，注意看报错信息提示缺失什么模型。  使用openpose的大模型会预下载VAE模型，无梯子的注意这里会报错。  

HiDiffusion  From: [link](https://github.com/megvii-research/HiDiffusion)  
----

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


3 Download the model 
----
  3种调用模型节点的方法   
  一种是,在repo_id填写诸如"stabilityai/stable-diffusion-xl-base-1.0" 这样的标准抱脸的repo_id,系统会自己下载或者查找.cache路径，这个要开全局外网或者你开启了hf镜像  
  
  一种是，把你下载好的模型放在comfyUI的"models/diffusers"路径下，记得不要改最尾的模型路径名“比如stable-diffusion-xl-base-1.0”，就可以用菜单直接调用。（注意：要用菜单的方式，必须删掉repo_id默认的"stabilityai/stable-diffusion-xl-base-1.0"，让repo_id留空，controlnet_id一样，要调用菜单，必须留空controlnet_id） 
  
  一种是，你在repo_id或controlnet_id直接填写诸如“x:/xx/xx/stabilityai/stable-diffusion-xl-base-1.0” 这样的已经下载好的扩散模型的绝对路径也能用
  
  Three methods for calling model nodes   
  Filling in the corresponding repo_id will download the model   
  or   
  in the diffuse directory, store and use the downloaded model  （Notice:in this method,repo_id or controlnet_repo_id must be empty）  
  or  
  using repo_id,and Fill in the absolute path of your diffusers model,using"/"   

4 other
----
可用的模型可用加载默认的工作流文件，查看其注释  
Supported models can refer to the comments in the example JSON file  

5 example
-----
use openpose XL model    
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/example_pose.png)


use canny XL model or mistoline or scribble（anyline）   
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/mistoline.png)
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/canny.png)


use inpainting XL model  
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/inpainting.png)

txt2img  use XL/sd1.5/XL turbo/background...   
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/txt2img.png)
 

6 Citation
------

``` python  
@article{zhang2023hidiffusion,
  title={HiDiffusion: Unlocking Higher-Resolution Creativity and Efficiency in Pretrained Diffusion Models},
  author={Zhang, Shen and Chen, Zhaowei and Zhao, Zhenyu and Chen, Yuhao and Tang, Yao and Liang, Jiajun},
  journal={arXiv preprint arXiv:2311.17528},
  year={2023}
}
