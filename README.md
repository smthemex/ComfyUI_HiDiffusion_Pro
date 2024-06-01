# ComfyUI_HiDiffusion_Pro
A  HiDiffusion node for ComfyUI

2024-06-01 update 
-----
--修复使用加速模型时，controlnet无效的错误。   
--DMD2加速模型已可以正常使用   
--内置图片裁切功能   
--模型加载节点连接controlnet_repo节点时，可以为空或者其他不支持的模型，此时，text2img 和img2img生效，就是文生图和图生图，没有controlnet功能，也就是说文生图节点其实可以删掉了    
--菜单加入timestpes，对于LCM模型，可以试着改成例如390，128或者其他小于999的数字，或许效果会更佳。   

Previous updates       
--增加了SDXL-scribble 模型的支持。     
--增加了model.yaml文件，你可以修改该文件，添加其他的可能支持“扩散模型”或者“controlnet”或者“unet”模型 ，所有目前支持的模型的repo_id都在该文件的注释里,除了openpose和inpainting        
--增加了SDXL flash 加速扩散模型的支持，也就是说controlnet节点可以的SDXL模型，还可以用SDXL flash 扩散模型       
--增加了对anyline（基于mistuline模型）和controlnet 线稿类的canny支持，增加了openposeXL模型的支持（须搭配controlnet openpose）       
--所有节点都支持XL-lighting\Hyper\LCM\DMD2\ Unet加速，建议适当提高步数   

--Fix the error where controllnet is invalid when using accelerated models.   
--The DMD2 acceleration model is now functioning properly   
--Built in image cropping function   
--When the model loading node is connected to the controllet-repo node, it can be empty or other unsupported models. In this case, text2img and img2img take effect, which are the text generated graph and graph generated graph. There is no controllnet function, which means that the text generated graph node can actually be deleted   
--Add timestepes to the menu. For the LCM model, you can try changing it to 390 or other numbers smaller than 999 for better results.   
Previous updates        
--Added support for SDXL capable models.    
--Added model.yaml file, you can modify it and add other models that may support "diffusion model", "controllet", or "unet". The repo_id of all currently supported models is in the comments of this file, except for openpose and repainting    
--Added support for the SDXL flash accelerated diffusion model, which means that the control net node can use the SDXL model, and the SDXL flash diffusion model can also be used    
--Added Canny support for Anyline (based on the pipeline model) and Controlnet draft classes, and added support for Openpose XL model (to be paired with Controlnet Openpose)    
--All nodes support XL lighting, Hyper, LCM, DMD2, Unet acceleration, it is recommended to increase the number of steps appropriately    

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
部分支持的模型请查看model.yaml 文件    
For partially supported models, please refer to the model.yaml file   

5 example
-----
use openpose XL model    
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/openpose%20and%20hyper%204%20step.png)


use canny XL model or mistoline or scribble（anyline）   
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/mistoline.png)
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/canny%20and%20DMD2.png)


use inpainting XL model  
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/inpainting.png)

txt2img  use XL/sd1.5/XL turbo/background...   
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/txt2img.png)
 
img2img  use XL XL-flash or base XL
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/img2img.png)

6 Citation
------

``` python  
@article{zhang2023hidiffusion,
  title={HiDiffusion: Unlocking Higher-Resolution Creativity and Efficiency in Pretrained Diffusion Models},
  author={Zhang, Shen and Chen, Zhaowei and Zhao, Zhenyu and Chen, Yuhao and Tang, Yao and Liang, Jiajun},
  journal={arXiv preprint arXiv:2311.17528},
  year={2023}
}
