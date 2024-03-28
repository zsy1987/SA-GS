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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time
def render_set(save_name,model_path, name, iteration, views, gaussians, pipeline, background,resolution,mode):
    render_path = os.path.join(model_path, name, save_name, "renders")
    gts_path = os.path.join(model_path, name, save_name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    res_train = int(model_path[-1])

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :]
        kernel_ratio=res_train/resolution
        rendering = render(view, gaussians, pipeline, background, kernel_ratio=kernel_ratio,mode=mode)["render"]
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


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

        render_set(dataset.save_name,dataset.model_path, "val", scene.loaded_iter, scene.getTrainCameras(scale=resolution), gaussians, pipeline, background,resolution,mode)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    # args.model_path = '/mnt/share_disk/zhengjv/miner_outputs/0221_mip0.3'
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)