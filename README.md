<p align="center">

  <h1 align="center">SA-GS: Alias-free 3D Gaussian Splatting</h1>
  <h3 align="center"><a href="https://drive.google.com/file/d/1Q7KgGbynzcIEyFJV1I17HgrYz6xrOwRJ/view?usp=sharing">Paper</a> | <a href="https://arxiv.org/pdf/2311.16493.pdf">arXiv</a> | <a href="https://niujinshuchong.github.io/mip-splatting/">Project Page</a>  | <a href="https://niujinshuchong.github.io/mip-splatting-demo/">Online Viewer</a> </h3>
  <div align="center"></div>
</p>


<p align="center">
  <a href="">
    <img src="./media/bicycle_3dgs_vs_ours.gif" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
We introduce a 3D smoothing filter and a 2D Mip filter for 3D Gaussian Splatting (3DGS), eliminating multiple artifacts and achieving alias-free renderings.  
</p>
<br>


# Installation

```
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

# Training and Evaluation
```
# single-scale training and single-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_stmt.py 

# multi-scale training and multi-scale testing on NeRF-synthetic dataset
python scripts/run_nerf_synthetic_mtmt.py 

# single-scale training and single-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360.py 

# single-scale training and multi-scale testing on the mip-nerf 360 dataset
python scripts/run_mipnerf360_stmt.py 
```

# Online viewer
After training, you can fuse the 3D smoothing filter to the Gaussian parameters with
```
python create_fused_ply.py -m {model_dir}/{scene} --output_ply fused/{scene}_fused.ply"
```
Then use our [online viewer](https://niujinshuchong.github.io/mip-splatting-demo) to visualize the trained model.

# Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Please follow the license of 3DGS. We thank all the authors for their great work and repos. 


