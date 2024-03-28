<p align="center">

  <h1 align="center">SA-GS: Scale-Adaptive Gaussian Splatting for Training-Free Anti-Aliasing</h1>
  
</p>
<p align="center">

[Project Page](https://kevinsong729.github.io/project-pages/SA-GS/) / [3DGS Model](https://kevinsong729.github.io/project-pages/SA-GS/)

</p>
<p align="center">
  <a href="">
    <img src="./img/bicycle_zoomoutin.gif" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
We introduce a scale-adaptive 2D filter and intregation(super sampling) method for 3D Gaussian Splatting (3DGS), eliminating multiple artifacts and achieving alias-free renderings.  
</p>
<br>


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
Please download and unzip models.zip from the [Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).
Eventually, **model** folder should look like this:

```
<your/model/path>
|-- point_cloud
    |-- iteration_xxxx
        |-- point_cloud.ply
|-- cameras.json
|-- cfg_args
```

# Train 3D gaussian splatting
```
# single-scale training and single-scale testing on NeRF-synthetic dataset
CUDA_VISIBLE_DEVICES=0 python train.py -s ./SA-GS/nerf_synthetic_multiscale/chair -m ./out_blender/chair/single_scale --save_iterations 30000 --mode source GS

```

# Testing
```
# single-scale training and single-scale testing on NeRF-synthetic dataset
CUDA_VISIBLE_DEVICES=0  python render_blender.py -s /data15/DISCOVER_winter2024/zhengj2401/gaussian-splatting3/nerf_synthetic_multiscale/chair -m /data15/DISCOVER_winter2024/zhengj2401/gaussian-splatting3/out_blender/chair/single_scale_s1 --save_name output --eval --load_allres --mode integration



# single-scale training and single-scale testing on the mip-nerf 360 dataset
CUDA_VISIBLE_DEVICES=0 python render_360.py -s /data15/DISCOVER_winter2024/zhengj2401/360v2/bonsai -m /data15/DISCOVER_winter2024/zhengj2401/gaussian-splatting/out_360v2/bonsai/single_s4 --save_name outputs -r 8 --mode integration

# mode "only filter" ,"source GS", "integration", "super sampling"

```


# Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). [Mip-splatting](https://github.com/autonomousvision/SA-GS). Please follow the license of 3DGS. We thank all the authors for their great work and repos. 


