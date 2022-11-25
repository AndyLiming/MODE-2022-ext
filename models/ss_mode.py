# sphere sweeping MODE network with one cost volume built with cassini spherical sweep
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .submodule import feature_extraction, sphere_feature_extraction  # spherical feature extraction
from .submodule import convbn, convbn_3d, disparityregression
from .mode_disparity import hourglass


# 1. feature extraction from 4 views in Cassini domain
# 2. grid sample to generate caoncat cost volume
# 3. cost volume squeeze dimension
# 4. regression
class SphereSweepMODE(nn.Module):
  def __init__(self, conv='Sphere', in_height=1024, in_width=512, sphereType='Cassini', numCam=4, numIndex=192):
    super(SphereSweepMODE, self).__init__()
    self.numCam = numCam
    self.numIndex = numIndex
    if conv == 'Regular':
      self.feature_extraction = feature_extraction()
    elif conv == 'Sphere':
      self.feature_extraction = sphere_feature_extraction(in_height, in_width, sphereType)
    else:
      raise NotImplementedError("Convolution Type must be Regular or Sphere!")

    self.multi_view_channel_conbine = nn.Sequential(nn.Conv3d(in_channels=32 * self.numCam,
                                                              out_channels=64,
                                                              kernel_size=(1,
                                                                           1,
                                                                           1),
                                                              stride=(1,
                                                                      1,
                                                                      1),
                                                              padding=(0,
                                                                       0,
                                                                       0)),
                                                    nn.ReLU(inplace=True),
                                                    nn.Conv3d(in_channels=64,
                                                              out_channels=64,
                                                              kernel_size=(1,
                                                                           3,
                                                                           3),
                                                              stride=(1,
                                                                      1,
                                                                      1),
                                                              padding=(0,
                                                                       1,
                                                                       1)),
                                                    nn.ReLU(inplace=True))
    self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))

    self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1))

    self.dres2 = hourglass(32)

    self.dres3 = hourglass(32)

    self.dres4 = hourglass(32)

    self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.Conv3d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()

  def forward(self, imgs, grids):
    assert (len(imgs) == self.numCam and len(grids) == self.numCam), "num of Cams:{},imgs:{},grids:{}".format(self.numCam, len(imgs), len(grids))
    assert (grids[0].shape[0] == self.numIndex // 4), "num of index:{},grids:{}".format(self.numIndex, grids[0].shape[0])
    b, c, h, w = imgs[0].shape
    cost = []
    for i in range(self.numCam):
      feature_i = self.feature_extraction(imgs[i])  #b,c,h/4,w/4
      features = []
      for j in range(self.numIndex // 4):
        grid = (grids[i][j:j + 1, ...]).repeat(b, 1, 1, 1)
        features.append((F.grid_sample(feature_i, grid, align_corners=True)).unsqueeze(2))  #b,c,1,h/2,w/2
      features = torch.cat(features, dim=2)
      cost.append(features)
    cost = torch.cat(cost, dim=1)  #b,c=128,d=ind/2,h/2,w/2
    cost = cost.contiguous()
    cost = self.multi_view_channel_conbine(cost)  #b,c=64,d=ind/2,h/2,w/2

    # stack hourglass used in
    cost0 = self.dres0(cost)
    cost0 = self.dres1(cost0) + cost0

    out1, pre1, post1 = self.dres2(cost0, None, None)
    out1 = out1 + cost0

    out2, pre2, post2 = self.dres3(out1, pre1, post1)
    out2 = out2 + cost0

    out3, pre3, post3 = self.dres4(out2, pre1, post2)
    out3 = out3 + cost0

    cost1 = self.classif1(out1)
    cost2 = self.classif2(out2) + cost1
    cost3 = self.classif3(out3) + cost2

    if self.training:
      cost1 = F.upsample(cost1, [self.numIndex, h, w], mode='trilinear', align_corners=True)
      cost2 = F.upsample(cost2, [self.numIndex, h, w], mode='trilinear', align_corners=True)

      cost1 = torch.squeeze(cost1, 1)
      pred1 = F.softmax(cost1, dim=1)
      pred1 = disparityregression(self.numIndex)(pred1)

      cost2 = torch.squeeze(cost2, 1)
      pred2 = F.softmax(cost2, dim=1)
      pred2 = disparityregression(self.numIndex)(pred2)

    cost3 = F.upsample(cost3, [self.numIndex, h, w], mode='trilinear', align_corners=True)
    cost3 = torch.squeeze(cost3, 1)
    pred3 = F.softmax(cost3, dim=1)

    #For your information: This formulation 'softmax(c)' learned "similarity"
    #while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
    #However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
    pred3 = disparityregression(self.numIndex)(pred3)

    if self.training:
      return pred1, pred2, pred3
    else:
      return pred3
