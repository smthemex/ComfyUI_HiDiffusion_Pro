# ComfyUI_HiDiffusion_Pro
A  HiDiffusion node for ComfyUI

HiDiffusion  From: [link](https://github.com/megvii-research/HiDiffusion)  
----

Update 
----

**09/08**   
* adapter style now using  single file.
* adapter style改成单体文件模式
 

**Previous updates**
*修复runwaybug / 去掉repo加载模型的方式  /自动选择模型的类别  
* 增加adapter_style支持，SDXL需求的显存较大，虽然能跑CPU，但是不推荐，会爆显存，SD1.5测试没问题。  
* 增加 manne加速Lora  
* 加入controlnet-tile-sdxl的支持，内置图片预处理，默认512尺寸，新增apply_window_attn 条件控制。  
* 修复节点连接逻辑，现在文生图模式，无需接入image，无controlnet也无需接入control_image   
* 支持SDXL-lighting\Hyper\LCM\DMD2\的加速Unet，建议适当提高步数；    
* 基于官方的更新，加入lora支持，需要填关键词；    
* 加入skip，去掉意义不大的其他参数；     
     
 *fix runway error/del repo /auto choice model type             
* Adding adapter_style support, SDXL requires a large amount of graphics memory. Although it can run on CPU, it is not recommended as it may cause graphics memory " explosion". SD1.5 testing is not a problem.   
* add manne lighting lora  
* Added support for control net file sdxl, built-in image preprocessing, default size of 512, and added condition control for apply_window_attn.   
* Fix node connection logic, now in text-based graphics mode, there is no need to connect to image, no controllnet, and no need to connect to controll_image   
* Support acceleration Unet for SDXL lighting, Hyper, LCM, and DMD2. It is recommended to increase the number of steps appropriately;   
* Based on official updates, adding support for Lora requires filling in keywords;   
* Add skip and remove other parameters that are not significant;   


1.Installation
-----
  1.1 In the .\ComfyUI \ custom_node directory, run the following:   
  
  ``` python 
  git clone https://github.com/smthemex/ComfyUI_HiDiffusion_Pro.git   
  ```
  1.2 using it
  
2.requirements  
----
diffusers >=0.28.0   #is best 
yaml

3 About models     
----
3.1 base ckpt  
```
├──comfyUI/models/checkpoints/   
|     ├──sd1.5 or sd2.1 or sdxl  or playground   
├──comfyUI/models/vae/   
|     ├──any vae fit ckpt 
```
3.2 if using SDXL controlnet     
```
├──comfyUI/models/controlnet/   
|     ├──any SDXL controlnet
```
3.3 if using lighting  Unet   
```
├──comfyUI/models/unet/   
|     ├──any SDXL lighting  Unet 
```
3.4 if using adapter style   

```
├── ComfyUI/models/photomaker
|         ├── ip-adapter_sd15.bin
|         ├── ip-adapter_sdxl.bin
├── ComfyUI/models/clip_vision
|             ├── sdxl_model.safetensors  # rename from   sd15/ncoder/model.safetensors
|             ├── sdm_model.safetensors  # rename from  sdxl/encoder/model.safetensors

```

4 example
-----

new workflow example   new
 ![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/new.png)

 sd1.5 using ip_adapter_style  使用ip_adapter_style  
 ![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/sd15ipstyle1.png)

img2img use  lora     图生图和lora   
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/img2img_lora1.png)


img2img + controlnet  图生图加controlnet      
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/controlnet_img2img1.png)

img2img  use Hyper unet   图生图加加速unet   
![](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro/blob/main/example/lightingUnet1.png)

6 Citation
------

``` python  
@article{zhang2023hidiffusion,
  title={HiDiffusion: Unlocking Higher-Resolution Creativity and Efficiency in Pretrained Diffusion Models},
  author={Zhang, Shen and Chen, Zhaowei and Zhao, Zhenyu and Chen, Yuhao and Tang, Yao and Liang, Jiajun},
  journal={arXiv preprint arXiv:2311.17528},
  year={2023}
}
