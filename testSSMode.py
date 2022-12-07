# sample script: python testSSMode.py --checkpoint ./checkpoints/ckpt_disp_SSMODE_sphere_Deep360_40_soiled.tar --save_output_path ./outputs/test_ssmode_sphere_soiled_40_b2_192 --soiled

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
from tqdm import tqdm
import prettytable as pt

from models import SphereSweepMODE

from utils import evaluation
from dataloader import list_deep360_ssmode_test, Deep360DatasetSsmode

from sphereSweepCassini import CassiniSweepViewTrans

from utils.geometry import cassini2Equirec

parser = argparse.ArgumentParser(description='Multi View Omnidirectional Depth Estimation')

parser.add_argument('--model', default='SSMODE', help='select model')
# data
parser.add_argument("--dataset", default="Deep360", type=str, help="dataset name")
parser.add_argument("--dataset_root", default="../../datasets/Deep360/", type=str, help="dataset root directory.")
parser.add_argument('--width', default=512, type=int, help="width of omnidirectional images in Cassini domain")
parser.add_argument('--height', default=1024, type=int, help="height of omnidirectional images in Cassini domain")
# stereo
parser.add_argument('--num_index', type=int, default=192, help='maxium disparity')
parser.add_argument('--max_depth', default=1000, type=float, help="max valid depth")
# hyper parameters

parser.add_argument('--batch_size', type=int, default=1, help='number of batch to train')

parser.add_argument('--checkpoint', default=None, help='path to load checkpoint of disparity estimation model')

parser.add_argument('--parallel', action='store_true', default=False, help='model parallel')

parser.add_argument('--num_cam', type=int, default=4, help='num of cameras')

parser.add_argument('--soiled', action='store_true', default=False, help='test soiled image')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--save_output_path', type=str, default=None, help='path to save output files. if set to None, will not save')
parser.add_argument('--save_ori', action='store_true', default=False, help='save original disparity or depth value map')

# saving

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heightC, widthC = args.height, args.width
heightE, widthE = args.width, args.height

save_out = args.save_output_path is not None
args.cuda = not args.no_cuda and torch.cuda.is_available()


def saveOutputOriValue(pred, gt, mask, rootDir, id, names=None, cons=True):
  b, h, w = pred.shape
  pred[~mask] = 0
  gt[~mask] = 0
  for i in range(b):
    predSave = pred[i, ::].cpu()
    gtSave = gt[i, ::].cpu()
    maskSave = mask[i, ::].cpu()
    saveimg = predSave.squeeze_(0).numpy()
    if names is None:
      prefix = "{:0>4}_test".format(id + i)
    else:
      oriName = names[i]
      if isinstance(oriName, list) or isinstance(oriName, tuple):
        oriName = oriName[0]
      oriName = oriName.replace(args.dataset_root, '')
      oriName = oriName.replace('../', '')
      oriName = oriName.replace('./', '')
      oriName = oriName.replace('/', '+')
      prefix = oriName.split('.')[0]
    # cv2.imwrite(os.path.join(rootDir, prefix + '.exr'), saveimg)
    np.save(os.path.join(rootDir, prefix + '.npy'), saveimg)


def saveOutput(pred, gt, mask, rootDir, id, names=None, log=True, cons=True, savewithGt=True):
  b, h, w = pred.shape
  div = torch.ones([h, 10])
  if log:
    div = torch.log10(div * 1000 + 1.0)
    pred[mask] = torch.log10(pred[mask] + 1.0)
    gt[mask] = torch.log10(gt[mask] + 1.0)
  pred[~mask] = 0
  gt[~mask] = 0
  for i in range(b):
    predSave = pred[i, ::].cpu()
    gtSave = gt[i, ::].cpu()
    maskSave = mask[i, ::].cpu()
    if savewithGt:
      saveimg = torch.cat([gtSave, div, predSave], dim=1).numpy()
    else:
      saveimg = predSave.numpy()
    saveimg = (saveimg - np.min(saveimg)) / (np.max(saveimg) - np.min(saveimg)) * 255

    saveimg = saveimg.astype(np.uint8)
    if names is None:
      prefix = "{:0>4}_test".format(id + i)
    else:
      oriName = names[i]
      if isinstance(oriName, list) or isinstance(oriName, tuple):
        oriName = oriName[0]
      oriName = oriName.replace(args.dataset_root, '')
      oriName = oriName.replace('../', '')
      oriName = oriName.replace('./', '')
      oriName = oriName.replace('/', '+')

      prefix = oriName.split('.')[0]
    saveimg = cv2.applyColorMap(saveimg, cv2.COLORMAP_JET)
    #print(os.path.join(rootDir, prefix + '.png'))
    cv2.imwrite(os.path.join(rootDir, prefix + '.png'), saveimg)


def test(model, testDispDataLoader, modelNameDisp, numTestData):
  sphereSweep = CassiniSweepViewTrans(maxDepth=args.max_depth, minDepth=0.5, numInvs=args.num_index, scaleDown=4, numInvDown=4)
  grids = sphereSweep.genCassiniSweepGrids()
  grids = grids[:args.num_cam]
  test_metrics = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
  total_eval_metrics = np.zeros(len(test_metrics))  # mae,rmse,px1,px3,px5,d1
  if save_out:
    os.makedirs(args.save_output_path, exist_ok=True)
  print("Testing of ss_mode. Model: {}".format(modelNameDisp))
  print("num of test files: {}".format(numTestData))
  counter = 0
  model.eval()
  with torch.no_grad():
    for batch_idx, batchData in enumerate(tqdm(testDispDataLoader, desc='Test iter')):
      rgbImgs = [x.cuda() for x in batchData['rgbImgs']]
      rgbImgs = rgbImgs[:args.num_cam]
      #print(torch.max(rgbImgs[0]), torch.min(rgbImgs[0]))
      depthGT = batchData['depthMap'].cuda()
      mask = (~torch.isnan(depthGT) & (depthGT > 0) & (depthGT <= args.max_depth))
      # for fish eye datasets
      b, c, h, w = rgbImgs[0].shape
      output_pred = model(rgbImgs, grids)
      _, output_pred, __ = sphereSweep.invIndex2depth(output_pred)  # to depth

      # to ERP
      output_pred = cassini2Equirec(output_pred)
      depthGT = cassini2Equirec(depthGT)
      depthGT[depthGT > args.max_depth] = args.max_depth
      mask = (~torch.isnan(depthGT) & (depthGT > 0) & (depthGT <= args.max_depth))
      # compute errors
      eval_metrics = []
      eval_metrics.append(evaluation.mae(output_pred[mask], depthGT[mask]))
      eval_metrics.append(evaluation.rmse(output_pred[mask], depthGT[mask]))
      eval_metrics.append(evaluation.absrel(output_pred[mask], depthGT[mask]))
      eval_metrics.append(evaluation.sqrel(output_pred[mask], depthGT[mask]))
      eval_metrics.append(evaluation.silog(output_pred[mask], depthGT[mask]))
      eval_metrics.append(evaluation.delta_acc(1, output_pred[mask], depthGT[mask]))
      eval_metrics.append(evaluation.delta_acc(2, output_pred[mask], depthGT[mask]))
      eval_metrics.append(evaluation.delta_acc(3, output_pred[mask], depthGT[mask]))
      if save_out:
        if args.save_ori: saveOutputOriValue(output_pred.clone(), depthGT.clone(), mask, args.save_output_path, counter, names=batchData['depthName'])  # save npz
        saveOutput(output_pred.clone(), depthGT.clone(), mask, args.save_output_path, counter, names=batchData['depthName'], log=True,savewithGt=False)
      total_eval_metrics += eval_metrics
    mean_errors = total_eval_metrics / len(testDispDataLoader)
    mean_errors = ['{:^.4f}'.format(x) for x in mean_errors]
  tb = pt.PrettyTable()
  tb.field_names = test_metrics
  tb.add_row(list(mean_errors))
  print('\nTest Results on SSMODE using model {}:\n'.format(args.checkpoint))
  print(tb)


def main():
  # model
  model = SphereSweepMODE(conv='Sphere', in_height=args.height, in_width=args.width, sphereType='Cassini', numCam=args.num_cam, numIndex=args.num_index)
  if (args.parallel):
    model = nn.DataParallel(model)
  if args.cuda:
    model.cuda()
  if (args.checkpoint is not None):
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict['state_dict'])
  else:
    raise ValueError("disp model checkpoint is not defined")

  # data
  if args.dataset == 'Deep360':  # deep 360
    test_rgbs, test_gt = list_deep360_ssmode_test(args.dataset_root, soiled=args.soiled)
    testDispData = Deep360DatasetSsmode(rgbs=test_rgbs, gt=test_gt, resize=False)
    testDispDataLoader = torch.utils.data.DataLoader(testDispData, batch_size=args.batch_size, num_workers=args.batch_size, pin_memory=False, shuffle=False)

  # testing
  test(model, testDispDataLoader, args.checkpoint, len(testDispData))


if __name__ == '__main__':
  main()