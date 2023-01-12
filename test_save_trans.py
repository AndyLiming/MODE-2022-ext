from __future__ import print_function
import argparse
from ast import arg
import os
import os.path as osp
import re
import random
from utils.geometry import cassini2Equirec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
import cv2
from dataloader import list_file as lt
from dataloader import deep360_loader as DA
from dataloader import Dataset3D60Fusion_3view
from models import Baseline, ModeFusion
from utils import evaluation
import prettytable as pt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='MODE Trans testing')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='maximum depth in meters')
parser.add_argument('--model', default='ModeFusion', help='select model')
parser.add_argument('--dbname', default="Deep360", help='dataset name')
parser.add_argument('--soiled', action='store_true', default=False, help='test fusion network on soiled data (only for Deep360)')

parser.add_argument('--datapath-input', default='./outputs/Deep360PredDepth/', help='the path of the input of stage2, which is just the output of stage1')
parser.add_argument('--datapath-dataset', default='../../datasets/Deep360/', help='the path of the dataset')

parser.add_argument('--outpath', type=str, default='./outputs/MODE_trans_test_clear', help='the output path for fusion results')
parser.add_argument('--num_view', type=int, default=4, help='num of views in fusion')

args = parser.parse_args()

test_depthes, test_confs, test_rgbs, test_gt = lt.list_deep360_fusion_test(args.datapath_input, args.datapath_dataset, args.soiled)

test_depth_pair = test_depthes[2]

assert len(test_depth_pair) == len(test_gt)
print("test num: {}".format(len(test_gt)))

result_dir = osp.join(args.outpath, args.dbname)
depth_pred_path = os.path.join(result_dir, "depth_pred")
if not os.path.exists(depth_pred_path):
  os.makedirs(depth_pred_path)
gt_png_path = os.path.join(result_dir, "gt_png")
if not os.path.exists(gt_png_path):
  os.makedirs(gt_png_path)
#------------- TESTING -------------------------
total_eval_metrics = np.zeros(8)

tb = pt.PrettyTable()
tb.field_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
for i in tqdm(range(len(test_gt))):
  pred = np.load(test_depth_pair[i])['arr_0'].astype(np.float32)
  gt = np.load(test_gt[i])['arr_0'].astype(np.float32)
  pred = torch.from_numpy(pred).unsqueeze(0).cuda()
  gt = torch.from_numpy(gt).unsqueeze(0).cuda()

  pred = cassini2Equirec(pred.unsqueeze(1)).squeeze()
  gt = cassini2Equirec(gt.unsqueeze(1)).squeeze()

  #print(pred.min(), pred.max(), gt.min(), gt.max())

  #---------
  mask = gt <= args.maxdepth  # includes sky area, to exclude sky set mask=gt<args.maxdepth
  #mask = (gt <= args.maxdepth) & (gt > 0) & (~torch.isnan(gt)) & (~torch.isinf(gt))
  #----
  eval_metrics = []
  eval_metrics.append(evaluation.mae(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.rmse(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.absrel(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.sqrel(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.silog(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.delta_acc(1, pred[mask], gt[mask]))
  eval_metrics.append(evaluation.delta_acc(2, pred[mask], gt[mask]))
  eval_metrics.append(evaluation.delta_acc(3, pred[mask], gt[mask]))

  eval_metrics = np.array(eval_metrics)
  total_eval_metrics += eval_metrics
  depth_pred = pred.data.cpu().numpy()
  depth_gt = gt.data.cpu().numpy()
  name = osp.splitext(osp.basename(test_gt[i]))[0]
  ep_name = re.findall(r'ep[0-9]_', test_gt[i])[0]
  name = ep_name + name

  # save gt png
  depth_gt = np.log(depth_gt - np.min(depth_gt) + 1)
  depth_gt = 255 * depth_gt / np.max(depth_gt)
  depth_gt = np.clip(depth_gt, 0, 255)
  depth_gt = depth_gt.astype(np.uint8)
  depth_gt = cv2.applyColorMap(depth_gt, cv2.COLORMAP_JET)
  cv2.imwrite(gt_png_path + '/' + name + "_gt.png", depth_gt)

  # save depth pred
  np.save(depth_pred_path + '/' + name + "_pred.npy", depth_pred)
  depth_pred = np.log(depth_pred - np.min(depth_pred) + 1)
  depth_pred = 255 * depth_pred / np.max(depth_pred)
  depth_pred = np.clip(depth_pred, 0, 255)
  depth_pred = depth_pred.astype(np.uint8)
  depth_pred = cv2.applyColorMap(depth_pred, cv2.COLORMAP_JET)
  cv2.imwrite(depth_pred_path + '/' + name + "_pred.png", depth_pred)

eval_metrics = total_eval_metrics / len(test_gt)
tb.add_row(list(eval_metrics))
print('\nTest Results:\n')
print(tb)
