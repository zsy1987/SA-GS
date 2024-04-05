# SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free Anti-Aliasing
  
[[Paper](https://drive.google.com/file/d/1uVSdYOXreEuntpswW3HXV-TypKi-ZopQ/view?usp=drive_link)] | [[Project Page](https://kevinsong729.github.io/project-pages/SA-GS/)] | [[3DGS Model](https://drive.google.com/drive/folders/10DC8iPt1RE5cp_b6b1naMoRlR2bsvlAa?usp=drive_link)]

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

# Train
Our code integrates the training process of the vinilla 3DGS, which can be trained using the following code. Of course, you can also use a pre-trained 3DGS model, e.g. downloaded from [here](https://drive.google.com/drive/folders/10DC8iPt1RE5cp_b6b1naMoRlR2bsvlAa?usp=drive_link), or a model that you have trained separately (satisfying the model catalogue specification above).
```
# single-scale training on NeRF-Synthetic dataset
python train.py -s /your/dataset/scene/path -m /your/output/path --data_type blender --save_iterations 30000 --r 1

# multi-scale training on NeRF-Synthetic dataset
python train.py -s /your/dataset/scene/path -m /your/output/path --data_type blender --save_iterations 30000 --load_allres

# single-scale training on Mip-NeRF 360 dataset
python train.py -s /your/dataset/scene/path -m /your/output/path --data_type 360v2 --save_iterations 30000 --r 1
```

# Render
## Render on Training Dataset
Render using our method. There are four modes to choose from: source-GS, only-filter, integration and super-sampling:
```
# Multi-scale testing on NeRF-synthetic dataset
python render_blender.py -s /your/data/path -m /your/model/path --save_name OUTPUT --load_allres --mode integration

# Single-scale testing on NeRF-synthetic dataset
python render_blender.py -s /your/data/path -m /your/model/path --save_name OUTPUT --r 8 --focal_rate 0.5 --mode integration 

# Single-scale testing on Mip-NeRF 360 dataset
python render_360.py -s /your/data/path -m /your/model/path --save_name OUTPUT --r 8 --focal_rate 0.5 --mode integration
```
## Render with user-defined camera tracks(parameters)
We support user-defined camera tracks and camera parameters for scene rendering：
```
python render_custom.py -s /your/data/path -m /your/model/path --save_name OUTPUT --camera_trajectory /your/tracks/file/path.json --mode integration
```
We provide functions to generate camera track json files from the , which you can modify manually to generate the track effects you want (pose interpolation, wrap around, forward, backward, etc.):
```
python ./utils/generate_tracks.py 
```
# Citing
If you have used our work in your research, please consider citing our paper. This will be very helpful to us in conducting follow-up research and tracking the impact of this work.
```
@article{song2024sa,
  title={SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free Anti-Aliasing},
  author={Song, Xiaowei and Zheng, Jv and Yuan, Shiran and Gao, Huan-ang and Zhao, Jingwei and He, Xiang and Gu, Weihao and Zhao, Hao},
  journal={arXiv preprint arXiv:2403.19615},
  year={2024}
}
```

# Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [Mip-splatting](https://github.com/autonomousvision/mip-splatting). Please follow the license of 3DGS and Mip-splatting. We thank all the authors for their great work and repos. 
