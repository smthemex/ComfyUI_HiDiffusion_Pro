# ComfyUI_HiDiffusion_Pro
A  HiDiffusion node for ComfyUI


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
  Filling in the corresponding repo_id will download the model   
  or   
  in the diffuse directory, store and use the downloaded model

4 example
-----
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example%20text2img0.png)

![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example%20img2img.png) 

![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example%20text2img.png) 


  Citation
------

``` python  
@article{zhang2023hidiffusion,
  title={HiDiffusion: Unlocking Higher-Resolution Creativity and Efficiency in Pretrained Diffusion Models},
  author={Zhang, Shen and Chen, Zhaowei and Zhao, Zhenyu and Chen, Yuhao and Tang, Yao and Liang, Jiajun},
  journal={arXiv preprint arXiv:2311.17528},
  year={2023}
}
