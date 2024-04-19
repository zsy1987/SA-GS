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
        train_cam_center = np.mean(np.array(train_position),axis=0)
        diff = np.abs(np.array(train_position)-train_cam_center)
        train_distance = np.mean(np.sqrt(np.sum(diff**2,axis=1)))
    return train_meta_data, train_distance, train_rotations,train_cam_center,train_position

def get_render_cams(jsonpath):
    with open(jsonpath, 'r', encoding='utf-8') as file:
        views = json.load(file)
    return views



def render_set(save_name,model_path, name, gaussians, pipeline, background,resolution,mode):
    render_path = os.path.join(model_path, name, save_name, "renders")

    makedirs(render_path, exist_ok=True)


    train_meta_data, train_distance, train_rotations,train_cam_center,train_position = get_train_cams(model_path)

    fovx = 2 * np.arctan(train_meta_data['train_width']/train_meta_data['train_fx']/2)
    fovy = 2 * np.arctan(train_meta_data['train_height']/train_meta_data['train_fy']/2)


    if train_meta_data['train_width'] > 1600:
        global WARNED
        if not WARNED:
            print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                "If this is not desired, please explicitly specify '--resolution/-r' as 1")
            WARNED = True
        global_down = train_meta_data['train_width'] / 1600.0
    else:
        global_down = 1.0
             
    train_meta_data['train_width'] = int(train_meta_data['train_width'] / global_down)
    train_meta_data['train_height'] = int(train_meta_data['train_height'] / global_down)
    train_meta_data['train_fx'] = train_meta_data['train_fx'] / global_down
    train_meta_data['train_fy'] = train_meta_data['train_fy'] / global_down


    #------------------------define your cameras---------------------------
    # you can define your cameras here.
    # For example, We use 1 / 8 resolution train cameras. 
    res_down_rate = 8
    train_meta_data['train_fx'] /= res_down_rate
    train_meta_data['train_fy'] /= res_down_rate
    
    render_cameras=list()
    for R0,T0 in zip(train_rotations,train_position):
        RT= np.concatenate((np.array(R0),np.array(T0).reshape(3,1)),axis=1)
        extension_row = np.array([0,0,0,1]).reshape(1,4)
        c2w = np.vstack((RT, extension_row))
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  
        T = w2c[:3, 3]
     
        render_cameras.append(Camera(None, R, T, fovx, fovy, \
                torch.ones((3, int(train_meta_data['train_height'] / res_down_rate), int(train_meta_data['train_width'] / res_down_rate))), None, None, None))
    #----------------------------------------------------------------------


    for idx, view in enumerate(tqdm(render_cameras, desc="Rendering progress")):
        kernel_ratio = view.image_width / train_meta_data['train_width'] * \
                       train_distance / np.sqrt(np.sum((view.T-train_cam_center)**2)) * \
                       view.focal_x / train_meta_data['train_fx']
        rendering = render(view, gaussians, pipeline, background, kernel_ratio=kernel_ratio,mode=mode)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        resolution=dataset.resolution
        dataset.resolution = -1
        mode = dataset.mode
        assert mode in["only-filter" ,"source-GS","integration","super-sampling"]
        if mode == "only-filter": mode=3
        elif mode=="source-GS": mode=0
        elif mode=="integration": mode=1
        elif mode=="super-sampling": mode=2
        else: raise Exception("Not allowed this mode")
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,resolution_scales=[resolution])
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_set(dataset.save_name,dataset.model_path, "val", gaussians, pipeline, background,resolution,mode)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    # parser.add_argument("--camera_trajectory", default="n.json", type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
    