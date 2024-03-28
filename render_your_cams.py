#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.graphics_utils import *
from utils.camera_utils import *
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import torch.nn.functional as F
import time
import json
method = "blendertest_cc"
model_path='/data15/DISCOVER_winter2024/zhengj2401/gaussian-splatting3/out_blender/chair/single_scale_s1'


def get_train_cams(model_path):
    jsonpath = os.path.join(model_path,'cameras.json')
    with open(jsonpath, 'r', encoding='utf-8') as file:
        data = json.load(file)
        train_meta_data=dict(
            train_width = data[0]['width'],
            train_height = data[0]['height'],
            train_fx = data[0]['fx'],
            train_fy = data[0]['fy']
        )
        
        train_position = [d['position'] for d in data]
        train_rotations = [d['rotation'] for d in data]

get_train_cams(model_path)




# def render_set(model_path, name, views, gaussians, pipeline, background, kernel_ratio):
#     render_path = os.path.join(model_path, name, method, "renders")
#     makedirs(render_path, exist_ok=True)

#     start_time = time.perf_counter()

#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
#         rendering = render(view, gaussians, pipeline, background,kernel_ratio=1/kernel_ratio[idx])["render"]
#         print(kernel_ratio[idx])
#         gt = view.original_image[0:3, :, :]
#         torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

#     end_time = time.perf_counter()

#     # 计算并打印执行时间
#     elapsed_time = end_time - start_time
#     print(f"代码执行时间: {elapsed_time} 秒")

# def rot_angle_z(angle):
#     return np.array([[np.cos(angle), np.sin(angle), 0],
#                    [-np.sin(angle), np.cos(angle), 0],
#                    [0, 0, 1]])
# resolution = 1
# def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
#     with torch.no_grad():
#         gaussians = GaussianModel(dataset.sh_degree)
#         scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,resolution_scales=[resolution])

#         cam_op_list = []
#         cam_list_video = []
#         for idx, cam in enumerate(scene.getTrainCameras(scale=1)):
#             cam_op_list.append(list(cam.T))
#             if(idx==0): first_cam = cam
        
#         cam_op_array = np.array(cam_op_list)
#         center = np.mean(cam_op_array, axis=0)
#         # center=center+np.array([-0.3,0,0])
#         # center = torch.mean(gaussians._xyz, dim=0).cpu().numpy()
#         # center = np.array([-0.37159,-1.076916,0.073908])
#         mean_d = np.mean(np.linalg.norm(cam_op_array - center, ord=2, axis=-1), axis=0)
#         mean_h = np.mean(cam_op_array[:,-1] - center[-1])
#         radius = np.sqrt(mean_d**2 - mean_h**2)
#         up = np.array([0.,0.,-1.])
        

#         aa1 = np.logspace(np.log10(first_cam.image_height/4), np.log10(first_cam.image_width), 100)
#         aa2 = np.logspace(np.log10(first_cam.image_height), np.log10(first_cam.image_width/4), 100)
#         kernel_ratio_list = [8 for _ in range(200)]+ list(np.logspace(np.log10(8), np.log10(1/2), 100)) +[1/2 for _ in range(200)]+ list(np.logspace(np.log10(1/2), np.log10(8), 100))+[8 for _ in range(200)]
#         Fovs = [first_cam.FoVx*2 for _ in range(200)] + list(np.logspace(np.log10(first_cam.FoVx*2),np.log10(first_cam.FoVx/2),100))+[first_cam.FoVx/2 for _ in range(200)]+list(np.logspace(np.log10(first_cam.FoVx/2),np.log10(first_cam.FoVx*2),100))+[first_cam.FoVx*2 for _ in range(200)] 
#         sizes = [F.interpolate(first_cam.original_image[None,...], size=(int(first_cam.image_height/4), int(first_cam.image_width/4))).squeeze(0) for _ in range(200)]+\
#                 [F.interpolate(first_cam.original_image[None,...], size=(int(aa1[i]), int(aa1[i]))).squeeze(0) for i in range(100)]+\
#                 [F.interpolate(first_cam.original_image[None,...], size=(int(first_cam.image_height), int(first_cam.image_width))).squeeze(0) for i in range(200)]+\
#                 [F.interpolate(first_cam.original_image[None,...], size=(int(aa2[i]), int(aa2[i]))).squeeze(0) for i in range(100)]+\
#                 [F.interpolate(first_cam.original_image[None,...], size=(int(first_cam.image_height/4), int(first_cam.image_width/4))).squeeze(0) for _ in range(200)]
        
#         for idx, theta in enumerate(np.linspace(0, 2*math.pi, 800)):
#             if idx>200 and idx<=300:
#                 center=center+np.array([0,0,0.018])
#             if idx>500 and idx<=600:
#                 center=center-np.array([0,0,0.018])
                

#             cam_pos = np.array([radius*np.cos(theta), radius*np.sin(theta),-0.0000001]) + center
#             l = (center - cam_pos) / np.linalg.norm(center - cam_pos, ord=2, axis=0)
#             s = np.cross(l, up) / np.linalg.norm(np.cross(l, up), ord=2, axis=0)
#             u = np.cross(s, l) / np.linalg.norm(np.cross(s, l), ord=2, axis=0)
#             R = np.stack((s,u,-l), axis=1)
#             # cam_list_video.append(Camera(first_cam.colmap_id, R , cam_pos, first_cam.FoVx, first_cam.FoVy, \
#             #     sizes[idx], None, first_cam.image_name, first_cam.uid))
#             cam_list_video.append(Camera(first_cam.colmap_id, R , cam_pos, Fovs[idx],  Fovs[idx], \
#                 sizes[idx], None, first_cam.image_name, first_cam.uid))
#         bg_color = [1,1,1] #if dataset.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#         render_set(dataset.model_path, "video", cam_list_video, gaussians, pipeline, background, kernel_ratio=kernel_ratio_list)
#     # Set up command line argument parser
        
# if __name__ == "__main__":        
#     parser = ArgumentParser(description="Testing script parameters")
#     model = ModelParams(parser, sentinel=True)
#     pipeline = PipelineParams(parser)
#     x=np.logspace(np.log10(4), np.log10(1), 100)
#     parser.add_argument("--iteration", default=-1, type=int)
#     parser.add_argument("--skip_train", action="store_true")
#     parser.add_argument("--skip_test", action="store_true")
#     parser.add_argument("--quiet", action="store_true")
#     args = get_combined_args(parser)
#     print("Rendering " + args.model_path)
#     safe_state(args.quiet)
#     render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)