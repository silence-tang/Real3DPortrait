# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
import argparse
import torch
from tqdm import tqdm
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

parser.add_argument('-i', '--image_path', default='TestSamples/conan_obrien.mp4', type=str,
                    help='path to input image')
parser.add_argument('-e', '--exp_path', default='TestSamples/lecun_conan_obrien.mp4', type=str, 
                    help='path to expression')
parser.add_argument('-s', '--savefolder', default='TestSamples/animation_results', type=str,
                    help='path to the output directory, where results(obj, txt files) will be stored.')
parser.add_argument('--device', default='cuda', type=str,
                    help='set device, cpu for using cpu' )
# rendering option
parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                    help='rasterizer type: pytorch3d or standard' )
# process test images
parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to crop input image, set false only when the test image are well cropped' )
parser.add_argument('--detector', default='fan', type=str,
                    help='detector for cropping face, check detectos.py for details' )
# save
parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to use FLAME texture model to generate uv texture map, \
                        set it to True only if you downloaded texture model' )
parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save visualization of output' )
parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save 2D and 3D keypoints' )
parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save depth image' )
parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save outputs as .obj' )
parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save outputs as .mat' )
parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                    help='whether to save visualization output as seperate images' )


args=parser.parse_args(args=[])

savefolder = args.savefolder
device = args.device
os.makedirs(savefolder, exist_ok=True)

# load test images
# 对于视频数据, dataset类会将其抽帧, 然后返回的len为总帧数
# 操作时可以通过
# for i, sample in enumerate(src_frames, tgt_frames):
#     ...
srcdata = datasets.TestData(args.image_path, iscrop=args.iscrop, face_detector=args.detector)
tgtdata = datasets.TestData(args.exp_path, iscrop=args.iscrop, face_detector=args.detector)

# run DECA
deca_cfg.model.use_tex = args.useTex
deca_cfg.rasterizer_type = args.rasterizer_type
deca = DECA(config = deca_cfg, device=device)



l2_dist_sum = 0
cnt = 0

# 使用 tqdm 包装循环
with torch.no_grad():
    for i, sample in enumerate(tqdm(zip(srcdata, tgtdata), desc="Processing", unit="sample")):
        src_data = sample[0]
        tgt_data = sample[1]
        
        # 将图像数据移动到设备（如 GPU）
        src_img = src_data['image'].to(device)[None, ...]
        tgt_img = tgt_data['image'].to(device)[None, ...]
        
        # 编码获取表情代码
        src_exp_code = deca.encode(src_img)['exp']
        tgt_exp_code = deca.encode(tgt_img)['exp']
        
        # 计算 L2 距离
        l2_dist = torch.norm(src_exp_code - tgt_exp_code, p=2)
        l2_dist_sum += l2_dist
        cnt += 1

# 计算平均 L2 距离
if cnt > 0:
    avg_l2_dist = l2_dist_sum / cnt
    print(f"平均表情系数L2距离: {avg_l2_dist}")
else:
    print("没有样本可用")