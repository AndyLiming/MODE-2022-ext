from __future__ import print_function
import argparse
from ast import arg
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import list_file as lt
from dataloader import deep360_loader as DA
from dataloader import Dataset3D60Fusion_3view  # 3D60 fusion 3 views
from models import Baseline, ModeFusion
from utils import evaluation
import prettytable as pt

from utils.geometry import cassini2Equirec  # val erp projection

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='MODE Fusion training')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='maximum depth in meters')
parser.add_argument('--model', default='ModeFusion', help='select model')
parser.add_argument('--dbname', default="Deep360", help='dataset name')
parser.add_argument('--soiled', action='store_true', default=False, help='train fusion network from soiled data (only for Deep360)')
parser.add_argument('--resize', action='store_true', default=False, help='resize the input by downsampling to 1/2 of its original size')
parser.add_argument('--datapath-input', default='./outputs/Deep360PredDepth/', help='the path of the input of stage2, which is just the output of stage1')
parser.add_argument('--datapath-dataset', default='../../datasets/Deep360/', help='the path of the dataset')
parser.add_argument('--epochs', type=int, default=150, help='the number of epochs for training')
parser.add_argument('--epoch-start', type=int, default=0, help='change this if the training was broken and you want to continue from the breakpoint')
parser.add_argument('--batch-size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--num_view', type=int, default=4, help='num of views in fusion')
parser.add_argument('--loadmodel', default=None, help='load model path')
parser.add_argument('--savemodel', default='./checkpoints/fusion/', help='save model path')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def select_deep360_views(depthes, confs, rgbs, num_view):
  if num_view == 4:  # all views
    return depthes, confs, rgbs
  elif num_view == 3:  # 1,2,3
    depthes = [depthes[0], depthes[1], depthes[3]]  #12,13,23
    confs = [confs[0], confs[1], confs[3]]  #12,13,23
    rgbs = [rgbs[0], rgbs[1], rgbs[2]]  #1,2,3
    return depthes, confs, rgbs
  elif num_view == 2:  # 1,2
    depthes = [depthes[0]]  #12
    confs = [confs[0]]  #12
    rgbs = [rgbs[0], rgbs[1]]  #1,2
    return depthes, confs, rgbs
  else:
    raise NotImplementedError("num of views must in [2,4] !")


torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)

if args.dbname == 'Deep360':
  train_depthes, train_confs, train_rgbs, train_gt, val_depthes, val_confs, val_rgbs, val_gt = lt.list_deep360_fusion_train(args.datapath_input, args.datapath_dataset, args.soiled)
  # select inputs based on num of views
  train_depthes, train_confs, train_rgbs = select_deep360_views(train_depthes, train_confs, train_rgbs, args.num_view)
  val_depthes, val_confs, val_rgbs = select_deep360_views(val_depthes, val_confs, val_rgbs, args.num_view)

  d_channel = args.num_view * (args.num_view - 1)  # depth channel = 2 * C(n,2)=2*n*(n-1)/2=n*(n-1)
  c_channel = 3 * args.num_view  # color channel = 3 * n
  print("num of views: {}. Depth channel: {}. Color channel: {}".format(args.num_view, d_channel, c_channel))
  print("train. len depth: {}, len confs: {}, len colors: {}".format(len(train_depthes), len(train_confs), len(train_rgbs)))
  print("val. len depth: {}, len confs: {}, len colors: {}".format(len(val_depthes), len(val_confs), len(val_rgbs)))

  TrainImgLoader = torch.utils.data.DataLoader(DA.Deep360DatasetFusion(train_depthes,
                                                                       train_confs,
                                                                       train_rgbs,
                                                                       train_gt,
                                                                       resize=args.resize,
                                                                       training=True),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.batch_size,
                                               drop_last=False)

  ValImgLoader = torch.utils.data.DataLoader(DA.Deep360DatasetFusion(val_depthes,
                                                                     val_confs,
                                                                     val_rgbs,
                                                                     val_gt,
                                                                     resize=False,
                                                                     training=False),
                                             batch_size=8,
                                             shuffle=False,
                                             num_workers=8,
                                             drop_last=False)
elif args.dbname == '3D60':
  os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # enable openexr
  train_data = Dataset3D60Fusion_3view(filenamesFile='./dataloader/3d60_train.txt', rootDir=args.datapath_dataset, inputDir=args.datapath_input, curStage='training')
  val_data = Dataset3D60Fusion_3view(filenamesFile='./dataloader/3d60_val.txt', rootDir=args.datapath_dataset, inputDir=args.datapath_input, curStage='validation')
  TrainImgLoader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_size, drop_last=False)
  ValImgLoader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, num_workers=8, drop_last=False)

  d_channel = 3 * 2 * 2
  c_channel = 3 * 3
  print("Depth channel: {}. Color channel: {}".format(d_channel, c_channel))

if args.model == 'Baseline':
  model = Baseline(args.maxdepth)
elif args.model == 'ModeFusion':
  if args.dbname == 'Deep360':
    model = ModeFusion(args.maxdepth, [32, 64, 128, 256], {'depth': d_channel, 'rgb': c_channel})
  elif args.dbname == '3D60':
    model = ModeFusion(args.maxdepth, [32, 64, 128, 256], {'depth': d_channel, 'rgb': c_channel})
else:
  print('no model')

if args.cuda:
  model = nn.DataParallel(model)
  model.cuda()

if args.loadmodel is not None:
  print('Load pretrained model')
  pretrain_dict = torch.load(args.loadmodel)
  model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def silog_loss(lamda, pred, gt):
  mask1 = gt > 0
  mask2 = pred > 0
  mask = mask1 * mask2
  d = torch.log(pred[mask]) - torch.log(gt[mask])
  return torch.mean(torch.square(d)) - lamda * torch.square(torch.mean(d))


def train(depthes, confs, rgbs, gt):
  model.train()

  if args.cuda:
    depthes = [depth.cuda() for depth in depthes]
    confs = [conf.cuda() for conf in confs]
    rgbs = [rgb.cuda() for rgb in rgbs]
    gt = gt.cuda()

  #---------
  mask = gt <= args.maxdepth  # includes sky area, to exclude sky set mask=gt<args.maxdepth
  mask.detach_()
  #----

  optimizer.zero_grad()

  if args.model == 'Baseline':
    output = model(depthes)
  elif args.model == 'ModeFusion':
    output = model(depthes, confs, rgbs)
  output = torch.squeeze(output, 1)

  loss = silog_loss(0.5, output[mask], gt[mask])
  loss.backward()
  optimizer.step()

  return loss.data


def val(depthes, confs, rgbs, gt):
  model.eval()

  if args.cuda:
    depthes = [depth.cuda() for depth in depthes]
    confs = [conf.cuda() for conf in confs]
    rgbs = [rgb.cuda() for rgb in rgbs]
    gt = gt.cuda()

  #---------
  mask = gt <= args.maxdepth
  #----

  with torch.no_grad():
    if args.model == 'Baseline':
      output = model(depthes)
    elif args.model == 'ModeFusion':
      output = model(depthes, confs, rgbs)
    pred = torch.squeeze(output, 1)

  eval_metrics = []
  eval_metrics.append(evaluation.mae(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.rmse(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.absrel(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.sqrel(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.silog(pred[mask], gt[mask]))
  eval_metrics.append(evaluation.delta_acc(1, pred[mask], gt[mask]))
  eval_metrics.append(evaluation.delta_acc(2, pred[mask], gt[mask]))
  eval_metrics.append(evaluation.delta_acc(3, pred[mask], gt[mask]))
  return np.array(eval_metrics)


def adjust_learning_rate(optimizer, epoch):
  lr = args.lr
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def main():
  log_path = os.path.join(args.savemodel, args.model, args.dbname, 'log')
  if not os.path.exists(log_path):
    os.makedirs(log_path)
  writer = SummaryWriter(log_path, purge_step=args.epoch_start)

  start_full_time = time.time()
  min_mae, ep_id = 1e9 + 7, -1
  for epoch in range(0, args.epochs):
    print('This is %d-th epoch' % (epoch + args.epoch_start))
    total_train_loss = 0
    adjust_learning_rate(optimizer, epoch + args.epoch_start)

    #--- TRAINING ---#
    for batch_idx, (_, depthes, confs, rgbs, gt) in enumerate(TrainImgLoader):
      loss = train(depthes, confs, rgbs, gt)
      print("\rFusion Stage Epoch" + str(epoch + args.epoch_start) + ": {:.2f}%".format(100 * (batch_idx + 1) / len(TrainImgLoader)), end='')
      total_train_loss += loss
    writer.add_scalar('Training Loss', total_train_loss / len(TrainImgLoader), epoch + args.epoch_start)

    #--- SAVING ---#
    savefilename = os.path.join(args.savemodel, args.model, args.dbname, 'ckpt_fusion_%dviews_epoch%d.tar' % (args.num_view, epoch + args.epoch_start))
    torch.save({'state_dict': model.state_dict()}, savefilename)

    #--- VALIDATION ---#
    total_eval_metrics = np.zeros(8)
    for batch_idx, (_, depthes, confs, rgbs, gt) in enumerate(ValImgLoader):
      print("\rStage2 Epoch" + str(epoch + args.epoch_start) + ": {:.2f}%".format(100 * (batch_idx + 1) / len(ValImgLoader)), end='')
      eval_metrics = val(depthes, confs, rgbs, gt)
      total_eval_metrics += eval_metrics

    eval_metrics = total_eval_metrics / len(ValImgLoader)
    eval_metrics = np.around(eval_metrics, decimals=6)
    tb = pt.PrettyTable()
    tb.field_names = ["MAE", "RMSE", "AbsRel", "SqRel", "SILog", "δ1 (%)", "δ2 (%)", "δ3 (%)"]
    tb.add_row(list(eval_metrics))
    print('\n')
    print(tb)
    writer.add_scalar('MAE', eval_metrics[0], epoch + args.epoch_start)
    writer.add_scalar('RMSE', eval_metrics[1], epoch + args.epoch_start)
    writer.add_scalar('AbsRel', eval_metrics[2], epoch + args.epoch_start)
    writer.add_scalar('SqRel', eval_metrics[3], epoch + args.epoch_start)
    writer.add_scalar('SILog', eval_metrics[4], epoch + args.epoch_start)
    writer.add_scalar('δ1', eval_metrics[5], epoch + args.epoch_start)
    if eval_metrics[0] < min_mae:
      min_mae = eval_metrics[0]
      ep_id = epoch
    print("min MAE :{}, at epoch {}".format(min_mae, ep_id))

  print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

  writer.close()


if __name__ == '__main__':
  main()
