# ComfyUI_HiDiffusion_Pro
A  HiDiffusion node for ComfyUI

Update
-----
2025-05-15 update  
add XL-lighting\Hyper\LCM Unet model support, it's working,but get bad result. LCM is only valid in text2img node，Lightning and Hyper require selecting an XL large model to take effect   
增加了lighting、Hyper、LCM的Unet模型支持，但是效果真一般，就当玩玩吧。LCM只在文生图有效。 闪电和Hyper需要选XL大模型才生效   


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


3 Download the model 
----
  3种调用模型的方法   
  一种是,在repo_id填写诸如"stabilityai/stable-diffusion-xl-base-1.0" 这样的id,系统自己下载或者查找.cache路径，这个要开全局外网或者你开启了hf镜像  
  
  一种是，把你下载好的模型放在comfyUI的"models/diffusers"路径下，记得不要改最尾的模型路径名“比如stable-diffusion-xl-base-1.0”，就可以用菜单直接调用。（注意：要用菜单的方式，必须删掉repo_id默认的"stabilityai/stable-diffusion-xl-base-1.0"，让repo_id留空，controlnet_id一样，要调用菜单，必须留空controlnet_id） 
  
  一种是，你在repo_id或controlnet_id直接填写诸如“x:/xx/xx/stabilityai/stable-diffusion-xl-base-1.0” 这样的绝对路径也能用

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
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example%20text2img0.png)

![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example%20img2img.png) 

![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example%20text2img.png) 


6 Citation
------

``` python  
@article{zhang2023hidiffusion,
  title={HiDiffusion: Unlocking Higher-Resolution Creativity and Efficiency in Pretrained Diffusion Models},
  author={Zhang, Shen and Chen, Zhaowei and Zhao, Zhenyu and Chen, Yuhao and Tang, Yao and Liang, Jiajun},
  journal={arXiv preprint arXiv:2311.17528},
  year={2023}
}
