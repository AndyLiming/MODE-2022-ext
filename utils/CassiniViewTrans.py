import os
import cv2
import numpy as np
import math
import torch
from numba import jit


class CassiniDepthDispTransformer:
  def __init__(self, height, width, maxDisp, maxDepth, baseline, device):
    self.height = height
    self.width = width
    self.maxDisp = maxDisp
    self.maxDepth = maxDepth
    self.baseline = baseline
    self.device = device
    phi_1_start = 0.5 * math.pi - (0.5 * math.pi / width)
    phi_1_end = -0.5 * math.pi
    phi_1_step = math.pi / width
    phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
    self.phi_l_map = np.array([phi_1_range for j in range(height)]).astype(np.float32)
    self.phi_l_map_torch = torch.from_numpy(self.phi_l_map).unsqueeze_(0).unsqueeze_(0).to(device)

  def depth2disp(self, depthMap):
    if (type(depthMap) == torch.Tensor):
      return self.__depth2dispTorch(depthMap)
    else:
      return self.__depth2dispNumpy(depthMap)

  def disp2depth(self, dispMap):
    if (type(dispMap) == torch.Tensor):
      return self.__disp2depthTorch(dispMap)
    else:
      return self.__disp2depthNumpy(dispMap)

  def __depth2dispNumpy(self, depthMap):
    # input: depth map,1-channel numpy array, shape(h,w)
    # output: disp map,1-channel numpy array, shape(h,w)
    h, w = depthMap.shape
    assert h == self.height and w == self.width, "request shape is ({},{}), while input shape is ({},{})".format(self.height, self.width, h, w)
    invMask = (depthMap <= 0) | (depthMap > self.maxDepth)
    depth_not_0 = np.ma.array(depthMap, mask=invMask)
    phi_l_map = self.phi_l_map
    disp = self.width * (np.arcsin(
        np.clip(
            (depth_not_0 * np.sin(phi_l_map) + self.baseline) / np.sqrt(depth_not_0 * depth_not_0 + self.baseline * self.baseline - 2 * depth_not_0 * self.baseline * np.cos(phi_l_map + np.pi / 2)),
            -1,
            1)) - phi_l_map) / np.pi
    disp = disp.filled(np.nan)
    disp[depthMap >= self.maxDepth] = 0
    disp[disp < 0] = 0
    return disp

  def __depth2dispNumpy2(self, depthMap):
    # input: depth map,1-channel numpy array, shape(h,w)
    # output: disp map,1-channel numpy array, shape(h,w)
    h, w = depthMap.shape
    assert h == self.height and w == self.width, "request shape is ({},{}), while input shape is ({},{})".format(self.height, self.width, h, w)
    invMask = (depthMap <= 0) | (depthMap > self.maxDepth)
    depth_not_0 = np.ma.array(depthMap, mask=invMask)
    phi_l_map = self.phi_l_map
    disp = self.width * (np.arcsin(
        np.clip(
            (depth_not_0 * np.sin(phi_l_map) + self.baseline) / np.sqrt(depth_not_0 * depth_not_0 + self.baseline * self.baseline - 2 * depth_not_0 * self.baseline * np.cos(phi_l_map + np.pi / 2)),
            -1,
            1)) - phi_l_map) / np.pi
    disp = disp.filled(np.nan)
    disp[depthMap >= self.maxDepth] = 0
    disp[disp < 0] = 0
    return disp

  def __depth2dispTorch(self, depthMap):
    b, c, h, w = depthMap.shape
    assert h == self.height and w == self.width and c == 1, "request shape is (1,{},{}), while input shape is ({},{},{})".format(self.height, self.width, c, h, w)
    invMask = (depthMap <= 0) | (depthMap > self.maxDepth) | (torch.isnan(depthMap)) | (torch.isinf(depthMap))
    phi_l_map = self.phi_l_map_torch.repeat(b, 1, 1, 1)
    disp = torch.zeros_like(depthMap)
    disp[~invMask] = self.width * (torch.asin(
        torch.clamp((depthMap[~invMask] * torch.sin(phi_l_map[~invMask]) + self.baseline) /
                    torch.sqrt(depthMap[~invMask] * depthMap[~invMask] + self.baseline * self.baseline - 2 * depthMap[~invMask] * self.baseline * torch.cos(phi_l_map[~invMask] + np.pi / 2)),
                    -1,
                    1)) - phi_l_map[~invMask]) / np.pi
    disp[invMask] = np.nan
    disp[depthMap >= self.maxDepth] = 0
    disp[disp < 0] = 0
    return disp

  def __disp2depthNumpy(self, dispMap):
    mask_disp_is_0 = dispMap == 0
    disp_not_0 = np.ma.array(dispMap, mask=mask_disp_is_0)
    phi_l_map = self.phi_l_map
    phi_r_map = disp_not_0 * math.pi / self.width + phi_l_map
    # sin theory
    depth_l = self.baseline * np.sin(math.pi / 2 - phi_r_map) / np.sin(phi_r_map - phi_l_map)
    depth_l = depth_l.filled(self.maxDepth)
    depth_l[depth_l > self.maxDepth] = self.maxDepth
    depth_l[depth_l < 0] = 0
    return depth_l

  def __disp2depthTorch(self, dispMap):
    b, c, h, w = dispMap.shape
    assert h == self.height and w == self.width and c == 1, "request shape is (1,{},{}), while input shape is ({},{},{})".format(self.height, self.width, c, h, w)
    depth = torch.ones_like(dispMap) * self.maxDepth
    phi_l_map = self.phi_l_map_torch.repeat(b, 1, 1, 1)
    dispPhi = dispMap * np.pi / self.width
    phi_r_map = phi_l_map + dispPhi
    invMask = (dispMap <= 0) | (torch.isnan(dispMap)) | (torch.isinf(dispMap))
    depth[~invMask] = self.baseline * torch.sin(torch.ones_like(phi_r_map[~invMask]) * np.pi / 2 - phi_r_map[~invMask]) / torch.sin(phi_r_map[~invMask] - phi_l_map[~invMask])
    depth[depth < 0] = 0
    depth[invMask] = self.maxDepth
    depth[depth > self.maxDepth] = self.maxDepth
    return depth


def dispBasedLeft2RightTorch(left, disp, device):
  b, c, h, w = left.shape
  v_range = (
      torch.arange(0,
                   h)  # [0 - h]
      .view(1,
            h,
            1)  # [1, [0 - h], 1]
      .expand(1,
              h,
              w)  # [1, [0 - h], W]
      .type(torch.float32)  # [1, H, W]
  ).to(device)
  u_range = (
      torch.arange(0,
                   w)  # [0 - w]
      .view(1,
            1,
            w)  # [1, 1, [0 - w]]
      .expand(1,
              h,
              w)  # [1, H, [0 - w]]
      .type(torch.float32)  # [1, H, W]
  ).to(device)
  v_range = v_range.repeat(b, 1, 1, 1)  # [B,1,H,W]
  u_range = u_range.repeat(b, 1, 1, 1)  # [B,1,H,W]
  u_range -= disp
  # u_range[u_range < 0] = 0
  # u_range[u_range >= w] = w - 1
  # u_range = u_range / (w - 1) * 2 - 1
  # v_range = v_range / (h - 1) * 2 - 1
  uvgrid = torch.cat([u_range, v_range], dim=1).to(device)
  #uvgrid = uvgrid.permute((0, 2, 3, 1))
  #right = torch.nn.functional.grid_sample(left, uvgrid, align_corners=True)
  right = torch.zeros_like(left)
  right = __splat__(left, uvgrid, right)
  return right


def __splat__(values, coords, splatted):
  b, c, h, w = splatted.size()
  uvs = coords
  u = uvs[:, 0, :, :].unsqueeze(1)
  v = uvs[:, 1, :, :].unsqueeze(1)

  u0 = torch.floor(u)
  u1 = u0 + 1
  v0 = torch.floor(v)
  v1 = v0 + 1

  u0_safe = torch.clamp(u0, 0.0, w - 1)
  v0_safe = torch.clamp(v0, 0.0, h - 1)
  u1_safe = torch.clamp(u1, 0.0, w - 1)
  v1_safe = torch.clamp(v1, 0.0, h - 1)

  u0_w = (u1 - u) * (u0 == u0_safe).detach().type(values.dtype)
  u1_w = (u - u0) * (u1 == u1_safe).detach().type(values.dtype)
  v0_w = (v1 - v) * (v0 == v0_safe).detach().type(values.dtype)
  v1_w = (v - v0) * (v1 == v1_safe).detach().type(values.dtype)

  top_left_w = u0_w * v0_w
  top_right_w = u1_w * v0_w
  bottom_left_w = u0_w * v1_w
  bottom_right_w = u1_w * v1_w

  weight_threshold = 1e-3
  top_left_w *= (top_left_w >= weight_threshold).detach().type(values.dtype)
  top_right_w *= (top_right_w >= weight_threshold).detach().type(values.dtype)
  bottom_left_w *= (bottom_left_w >= weight_threshold).detach().type(values.dtype)
  bottom_right_w *= (bottom_right_w >= weight_threshold).detach().type(values.dtype)

  splatted_weights = torch.zeros_like(splatted)

  for channel in range(c):
    top_left_values = values[:, channel, :, :].unsqueeze(1) * top_left_w
    top_right_values = values[:, channel, :, :].unsqueeze(1) * top_right_w
    bottom_left_values = values[:, channel, :, :].unsqueeze(1) * bottom_left_w
    bottom_right_values = values[:, channel, :, :].unsqueeze(1) * bottom_right_w

    top_left_values = top_left_values.reshape(b, -1)
    top_right_values = top_right_values.reshape(b, -1)
    bottom_left_values = bottom_left_values.reshape(b, -1)
    bottom_right_values = bottom_right_values.reshape(b, -1)

    top_left_indices = (u0_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
    top_right_indices = (u1_safe + v0_safe * w).reshape(b, -1).type(torch.int64)
    bottom_left_indices = (u0_safe + v1_safe * w).reshape(b, -1).type(torch.int64)
    bottom_right_indices = (u1_safe + v1_safe * w).reshape(b, -1).type(torch.int64)

    splatted_channel = splatted[:, channel, :, :].unsqueeze(1)
    splatted_channel = splatted_channel.reshape(b, -1)
    splatted_channel.scatter_add_(1, top_left_indices, top_left_values)
    splatted_channel.scatter_add_(1, top_right_indices, top_right_values)
    splatted_channel.scatter_add_(1, bottom_left_indices, bottom_left_values)
    splatted_channel.scatter_add_(1, bottom_right_indices, bottom_right_values)

    splatted_weights_channel = splatted_weights[:, channel, :, :].unsqueeze(1)
    splatted_weights_channel = splatted_weights_channel.reshape(b, -1)
    splatted_weights_channel.scatter_add_(1, top_left_indices, top_left_w.reshape(b, -1))
    splatted_weights_channel.scatter_add_(1, top_right_indices, top_right_w.reshape(b, -1))
    splatted_weights_channel.scatter_add_(1, bottom_left_indices, bottom_left_w.reshape(b, -1))
    splatted_weights_channel.scatter_add_(1, bottom_right_indices, bottom_right_w.reshape(b, -1))
  splatted = splatted.reshape(b, c, h, w)
  splatted_weights = splatted_weights.reshape(b, c, h, w)
  #print(torch.max(splatted_weights), torch.min(splatted_weights))
  mask = (splatted_weights < 1e-3)
  #print(torch.max(splatted))
  splatted = splatted / splatted_weights
  splatted[mask] = 0
  #print(torch.max(splatted))
  return splatted


def depthBasedLeft2RightTorch(left, depth, baseline=1.0, maxDepth=10000):
  b, c, h, w = left.shape
  t = np.array([[-baseline], [0], [0]])
  theta_1_start = math.pi - (math.pi / h)
  theta_1_end = -math.pi
  theta_1_step = 2 * math.pi / h
  theta_1_range = np.arange(theta_1_start, theta_1_end, -theta_1_step)
  theta_1_map = np.array([theta_1_range for i in range(w)]).astype(np.float32).T

  phi_1_start = 0.5 * math.pi - (0.5 * math.pi / w)
  phi_1_end = -0.5 * math.pi
  phi_1_step = math.pi / w
  phi_1_range = np.arange(phi_1_start, phi_1_end, -phi_1_step)
  phi_1_map = np.array([phi_1_range for j in range(h)]).astype(np.float32)

  right_list = []
  for i in range(b):
    leftImg = left[i, ::].cpu().numpy()  # numpy [C x H x W]
    r_1 = depth[i, 0, ::].cpu().numpy()  # numpy [H x W]
    x_1 = r_1 * np.sin(phi_1_map)
    y_1 = r_1 * np.cos(phi_1_map) * np.sin(theta_1_map)
    z_1 = r_1 * np.cos(phi_1_map) * np.cos(theta_1_map)
    X_2 = np.expand_dims(np.dstack((x_1, y_1, z_1)), axis=-1) - t
    r_2 = np.sqrt(np.square(X_2[:, :, 0, 0]) + np.square(X_2[:, :, 1, 0]) + np.square(X_2[:, :, 2, 0]))
    theta_2_map = np.arctan2(X_2[:, :, 1, 0], X_2[:, :, 2, 0])
    phi_2_map = np.arcsin(np.clip(X_2[:, :, 0, 0] / r_2, -1, 1))

    view_2 = np.ones((h, w)).astype(np.float32) * maxDepth
    rightImg = np.zeros((c, h, w)).astype(np.float32)

    I_2 = np.clip(np.rint(h / 2 - h * theta_2_map / (2 * math.pi)), 0, h - 1).astype(np.int16)
    J_2 = np.clip(np.rint(w / 2 - w * phi_2_map / math.pi), 0, w - 1).astype(np.int16)

    rightImg = iter_pixels(h, w, r_2, view_2, leftImg, rightImg, I_2, J_2)

    invMask = (view_2 == maxDepth)
    view_2[invMask] = 0
    view_2 = view_2.astype(np.float32)
    invMask = np.repeat(np.expand_dims(invMask, 0), c, 0)
    rightImg[invMask] = 0
    right_list.append(torch.from_numpy(np.expand_dims(rightImg, 0)))

  return torch.cat(right_list, dim=0)  # tensor [B x C x H x W]


@jit(nopython=True)
def iter_pixels(output_h, output_w, r_2, view_2, leftImg, rightImg, I_2, J_2):
  for i in range(output_h):
    for j in range(output_w):
      flag = r_2[i, j] < view_2[I_2[i, j], J_2[i, j]]
      view_2[I_2[i, j], J_2[i, j]] = flag * r_2[i, j] + (1 - flag) * view_2[I_2[i, j], J_2[i, j]]
      rightImg[:, I_2[i, j], J_2[i, j]] = flag * leftImg[:, i, j] + (1 - flag) * rightImg[:, I_2[i, j], J_2[i, j]]
  return rightImg


if __name__ == '__main__':
  """
  # NOTE: depth->disp->depth testing
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  CDDT = CassiniDepthDispTransformer(height=1024, width=512, maxDisp=192, maxDepth=1000, baseline=1.0, device=device)
  ra = range(4801, 4802)
  num = len(ra)
  total_mean = 0
  total_mean_tensor = 0
  for r in ra:
    depthName = '../../datasets/CarlaFisheye/down/testing/depth_cassini/0' + str(r) + '.npy'
    depthGT = np.load(depthName)
    dispGT = CDDT.depth2disp(depthGT)
    depthback = CDDT.disp2depth(dispGT)
    total_mean += np.mean(np.abs(depthGT - depthback) / depthGT)
    np.save('depthback.npy', depthback.astype(np.float32))
    # print(np.mean(np.abs(depthGT - depthback) / depthGT))

    depthGT_tensor = torch.from_numpy(depthGT).unsqueeze_(0).unsqueeze_(0).to(device)
    dispGT_tensor = CDDT.depth2disp(depthGT_tensor)
    depthback_tensor = CDDT.disp2depth(dispGT_tensor)

    total_mean_tensor += torch.mean(torch.abs(depthback_tensor - depthGT_tensor) / depthGT_tensor)

    # print(torch.mean(torch.abs(depthback_tensor - depthGT_tensor) / depthGT_tensor))
  print(total_mean / num, total_mean_tensor / num)
  """
  # NOTE: left->right testing
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  leftImgName = "../CarlaFisheyeCassini/down/testing/cam1/04801.png"
  left = cv2.imread(leftImgName).transpose((2, 0, 1)).astype(np.float32)
  left = torch.from_numpy(left).unsqueeze_(0).to(device)
  dispName = "../CarlaFisheyeCassini/down/testing/disp_gt/04801.npy"
  disp = np.load(dispName).astype(np.float32)
  print(disp.max())
  disp = torch.from_numpy(disp).unsqueeze_(0).unsqueeze_(0)
  right = dispBasedLeft2RightTorch(left, disp, device).to(device)
  right = right.squeeze(0).cpu().numpy().transpose((1, 2, 0))
  print(right.max())
  cv2.imwrite('trans_right.png', right.astype(np.uint8))

  depthName = "../../datasets/CarlaFisheye/down/testing/depth_cassini/04801.npy"
  depth = np.load(depthName).astype(np.float32)
  depth = torch.from_numpy(depth).unsqueeze_(0).unsqueeze_(0)
  right = depthBasedLeft2RightTorch(left, depth)
  right = right.squeeze(0).cpu().numpy().transpose((1, 2, 0))
  cv2.imwrite('trans_right2.png', right.astype(np.uint8))
