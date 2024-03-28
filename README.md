# SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free Anti-Aliasing
  
[[Paper](https://kevinsong729.github.io/project-pages/SA-GS/)] | [[Project Page](https://kevinsong729.github.io/project-pages/SA-GS/)] | [[3DGS Model](https://drive.google.com/drive/folders/10DC8iPt1RE5cp_b6b1naMoRlR2bsvlAa?usp=drive_link)]

This repository is an official implementation for:

**SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free Anti-Aliasing**

> Authors:  [_Xiaowei Song_*](https://kevinSONG729.github.io/), [_Jv Zheng_*](https://zsy1987.github.io/), _Shiran Yuan_, [_Huan-ang Gao_](https://c7w.tech/about/), _Jingwei Zhao_, _Xiang He_, _Weihao Gu_, [_Hao Zhao_](https://sites.google.com/view/fromandto)

<p align="center">
  <a href="">
    <img src="./img/bicycle_zoomoutin.gif" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
We introduce SA-GS, a training-free approach that can be directly applied to the inference process of any pretrained 3DGS model to resolve its visual artefacts at drastically changed rendering settings.
</p>
<br>

# Introduction
3DGS has gained attention in the industry due to its high-quality view rendering and fast speeds. However, view quality degradation can occur during rendering depending on settings such as resolution, distance, and focal length. Existing methods address this issue by adding regularity to Gaussian primitives in both 3D and 2D space during training. However, these methods overlook a significant drawback of 3DGS when used with different rendering settings: the scale ambiguity problem. This issue directly results in the inability of 3DGS to utilise conventional anti-aliasing techniques. We propose and analyse this problem for the first time and correct this shortcoming by using only 2D scale-adaptive filters. Based on this, we use conventional antialiasing methods such as integration and super-sampling to solve the aliasing effect caused by insufficient sampling frequency. It is worth noting that our method is the first Gaussian anti-aliasing technique that does not require training. Therefore, it can be directly integrated into existing 3DGS models to enhance their anti-aliasing capabilities. The method was validated in both bounded and unbounded scenarios, and the experimental results demonstrate that it achieves robust anti-aliasing performance enhancement in the most efficient way, surpassing or equaling the current optimal settings.

# Installation

```
cd SA-GS
conda create -y -n SA-GS python=3.8
conda activate SA-GS
pip install -r requirements.txt
pip install submodules/simple-knn/
pip install submodules/diff-gaussian-rasterization_new
```

# Dataset
## Blender Dataset
Please download and unzip nerf_synthetic.zip from the [NeRF's official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Then generate multi-scale blender dataset with
```
python convert_blender_data.py --blender_dir nerf_synthetic/ --out_dir multi-scale
```

## Mip-NeRF 360 Dataset
Please download the data from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) and request the authors for the treehill and flowers scenes.


## Model
Please download and unzip models.zip from the [Google Drive](https://drive.google.com/drive/folders/10DC8iPt1RE5cp_b6b1naMoRlR2bsvlAa?usp=drive_link).
Eventually, **model** folder should look like this:

```
<your/model/path>
|-- point_cloud
    |-- iteration_xxxx
        |-- point_cloud.ply
|-- cameras.json
|-- cfg_args
```

# Train(Vinilla 3D Gaussian Splatting)
```
# single-scale training on NeRF-Synthetic dataset
python train.py -s ./SA-GS/nerf_synthetic_multiscale/chair -m ./out_blender/chair/single_scale --save_iterations 30000 --mode source GS
# multi-scale training on NeRF-Synthetic dataset

# single-scale training on Mip-NeRF 360 dataset

```

# Test(Our SA-GS Rendering)
```
# single-scale training and single-scale testing on NeRF-synthetic dataset
python render_blender.py -s ./SA-GS/nerf_synthetic_multiscale/chair -m ./out_blender/chair/single_scale --save_name output --eval --load_allres --mode integration



# single-scale training and single-scale testing on the mip-nerf 360 dataset
python render_360.py -s ./SA-GS/360v2/bonsai -m ./out_360v2/bonsai/single_s4 --save_name outputs -r 8 --mode integration

# mode "only filter" ,"source GS", "integration", "super sampling"

```


# Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). [Mip-splatting](https://github.com/autonomousvision/SA-GS). Please follow the license of 3DGS. We thank all the authors for their great work and repos. 


