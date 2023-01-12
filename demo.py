# sample
# 3d60: python test_disparity.py --dataset 3D60 --dataset_root ../../datasets/3D60/ --pair_3d60 lr --width 256 --height 512 --max_depth 20 --checkpoint_disp ./checkpoints/disp/ModeDisparity/3D60/ckpt_disp_ModeDisparity_3D60_65.tar --save_output_path ./outputs/testMode3D60_lr
import os

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import time
from datetime import datetime
import math
import random
import cv2
from PIL import Image

from tqdm import tqdm
import re
import prettytable as pt

from models import ModeDisparity
from utils import evaluation, geometry
from dataloader import get_transform_stage1

parser = argparse.ArgumentParser(description='MODE Disparity estimation testing')

parser.add_argument('--model_disp', default='ModeDisparity', help='select model')
parser.add_argument("--left_name", default="Deep360", type=str, help="dataset name")
parser.add_argument("--right_name", default="../../datasets/Deep360/", type=str, help="dataset root directory.")
parser.add_argument("--save_name", default="../../datasets/Deep360/", type=str, help="dataset root directory.")

parser.add_argument('--width', default=512, type=int, help="width of omnidirectional images in Cassini domain")
parser.add_argument('--height', default=1024, type=int, help="height of omnidirectional images in Cassini domain")
parser.add_argument('--baseline', default=0.9882608712548844, type=float, help="height of omnidirectional images in Cassini domain")
#0.39431598474785656
# stereo
parser.add_argument('--max_disp', type=int, default=192, help='maxium disparity')
parser.add_argument('--max_depth', default=1000, type=float, help="max valid depth")
# hyper parameters

parser.add_argument('--checkpoint_disp', default=None, help='load checkpoint of disparity estimation path')

parser.add_argument('--save_output_path', type=str, default=None, help='path to save output files. if set to None, will not save')

args = parser.parse_args()

# if args.dataset == '3D60':
#   os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # enable openexr

heightC, widthC = args.height, args.width  #Cassini shape
heightE, widthE = args.width, args.height  #ERP shape

save_out = args.save_output_path is not None
#args.cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if args.cuda else "cpu")


def saveOutputOriValue(pred, rootDir, name):
  saveimg = pred
  np.save(os.path.join(rootDir, name + '_pred.npy'), saveimg)


def saveOutput(saveimg, rootDir, name, log=True):
  if log:
    saveimg = np.log(saveimg + 1.0)
  saveimg = (saveimg - np.min(saveimg)) / (np.max(saveimg) - np.min(saveimg)) * 255
  saveimg = saveimg.astype(np.uint8)
  saveimg = cv2.applyColorMap(saveimg, cv2.COLORMAP_JET)
  cv2.imwrite(os.path.join(rootDir, name + '_pred.png'), saveimg)


def disp2depth(disp):

  output_h = disp.shape[0]
  output_w = disp.shape[1]

  phi_l_start = 0.5 * math.pi - (0.5 * math.pi / output_w)
  phi_l_end = -0.5 * math.pi
  phi_l_step = math.pi / output_w
  phi_l_range = np.arange(phi_l_start, phi_l_end, -phi_l_step)
  phi_l_map = np.array([phi_l_range for j in range(output_h)]).astype(np.float32)

  mask_disp_is_0 = disp == 0
  disp_not_0 = np.ma.array(disp, mask=mask_disp_is_0)

  phi_r_map = disp_not_0 * math.pi / output_w + phi_l_map

  # sin theory
  depth_l = args.baseline * np.sin(math.pi / 2 - phi_r_map) / np.sin(phi_r_map - phi_l_map)
  depth_l = depth_l.filled(args.max_depth)
  depth_l[depth_l > args.max_depth] = args.max_depth
  depth_l[depth_l < 0] = 0
  return depth_l


def demo(modelDisp, pre_p):
  os.makedirs(args.save_output_path, exist_ok=True)
  modelDisp.eval()
  with torch.no_grad():
    leftImg = Image.open(args.left_name).convert('RGB')
    leftImg_erp = Image.fromarray(geometry.cassini2Equirec(np.array(leftImg)))
    leftImg_erp.save(os.path.join(args.save_output_path, args.save_name + '_rgb_erp.png'))
    leftImg = pre_p(leftImg).unsqueeze(0).cuda()
    rightImg = Image.open(args.right_name).convert('RGB')
    rightImg = pre_p(rightImg).unsqueeze(0).cuda()
    b, c, h, w = leftImg.shape

    output = modelDisp(leftImg, rightImg)

    pred_disp = output.cpu().numpy().squeeze()
    pred_depth = disp2depth(pred_disp)
    pred_depth_erp = geometry.cassini2Equirec(pred_depth)

    saveOutputOriValue(pred_depth, args.save_output_path, args.save_name)  # save npy
    saveOutput(pred_depth, args.save_output_path, args.save_name)
    saveOutput(pred_depth_erp, args.save_output_path, args.save_name + '_erp')


# def main():
#   # model
#   model_disp = ModeDisparity(maxdisp=args.max_disp, conv='Sphere', in_height=args.height, in_width=args.width, sphereType='Cassini', out_conf=False)
#   if (args.parallel):
#     model_disp = nn.DataParallel(model_disp)
#   if args.cuda:
#     model_disp.cuda()
#   if (args.checkpoint_disp is not None):
#     state_dict = torch.load(args.checkpoint_disp)
#     model_disp.load_state_dict(state_dict['state_dict'])
#   else:
#     raise ValueError("disp model checkpoint is not defined")

#   # data
#   if args.dataset == 'Deep360':  # deep 360
#     test_left_img, test_right_img, test_left_disp = list_deep360_disparity_test(args.dataset_root, soiled=args.soiled)
#     testDispData = Deep360DatasetDisparity(leftImgs=test_left_img, rightImgs=test_right_img, disps=test_left_disp)
#   elif args.dataset == '3D60':
#     testDispData = Dataset3D60Disparity(filenamesFile='./dataloader/3d60_test.txt',
#                                         rootDir=args.dataset_root,
#                                         curStage='testing',
#                                         shape=(512,
#                                                256),
#                                         crop=False,
#                                         pair=args.pair_3d60,
#                                         flip=False,
#                                         maxDepth=20.0)
#   else:
#     raise NotImplementedError("dataset must be Deep360 or 3D60!")
#   testDispDataLoader = torch.utils.data.DataLoader(testDispData, batch_size=args.batch_size, num_workers=args.batch_size, pin_memory=False, shuffle=False)

#   # testing
#   testDisp(model_disp, testDispDataLoader, args.checkpoint_disp, len(testDispData))

if __name__ == '__main__':
  model_disp = ModeDisparity(maxdisp=args.max_disp, conv='Sphere', in_height=args.height, in_width=args.width, sphereType='Cassini', out_conf=False)
  model_disp = nn.DataParallel(model_disp)
  model_disp.cuda()
  if (args.checkpoint_disp is not None):
    state_dict = torch.load(args.checkpoint_disp)
    model_disp.load_state_dict(state_dict['state_dict'])
  else:
    raise ValueError("disp model checkpoint is not defined")
  preprocess = get_transform_stage1(augment=False)
  demo(model_disp, preprocess)
