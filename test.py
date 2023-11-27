import torch
import numpy as np
#frustum
ogfH, ogfW = 128,352
# 16 下采样
downsample = 16
  # 下采样16倍后图像大小  fH: 8  fW: 22
fH, fW = ogfH // downsample, ogfW //downsample
dbound=[4.0, 45.0, 1.0]
# self.grid_conf['dbound'] = [4, 45, 1]
# 在深度方向上划分网格 ds: DxfHxfW (41x8x22)
ds = torch.arange(*dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
# D: 41 表示深度方向上网格的数量
D, _, _ = ds.shape 
"""
1. torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
tensor([0.0000, 16.7143, 33.4286, 50.1429, 66.8571, 83.5714, 100.2857,
        117.0000, 133.7143, 150.4286, 167.1429, 183.8571, 200.5714, 217.2857,
        234.0000, 250.7143, 267.4286, 284.1429, 300.8571, 317.5714, 334.2857,
        351.0000])
        
2. torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
tensor([0.0000, 18.1429, 36.2857, 54.4286, 72.5714, 90.7143, 108.8571,
        127.0000])
"""

# 在0到351上划分22个格子 xs: DxfHxfW(41x8x22)
xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

# D x H x W x 3
# 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
# xs ys ds 代表的是 像素camera坐标系上xs ys ds 的坐标对应vcs下的坐标(栅格坐标)
frustum = torch.stack((xs, ys, ds), -1)

# get_geometry
# trans =  np.ones((1,1,9))
# post_trans = trans.copy()
# B, N, _ = trans.shape
# points = frustum - post_trans.view(B, N, 1, 1, 1, 3)
# post_rots = np.ones((1,1,9))
# points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
# points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
#                         points[:, :, :, :, :, 2:3]
#                         ), 5)
# rots = np.ones((3,3))
# intrins = rots.copy()
# combine = rots.matmul(torch.inverse(intrins))
# points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
# points += trans.view(B, N, 1, 1, 1, 3)
# def get_geometry(rots, trans, intrins, post_rots, post_trans,frustum):
#         """Determine the (x,y,z) locations (in the ego frame)
#         of the points in the point cloud.
#         Returns B x N x D x H/downsample x W/downsample x 3
#         """
#         B, N, _ = trans.shape

#         # undo post-transformation
#         # B x N x D x H x W x 3
#         # 抵消数据增强及预处理对像素的变化
#         points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
#         # 从r矩阵求逆获得
#         points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
#         # cam_to_ego
#         #
#         points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
#                             points[:, :, :, :, :, 2:3]
#                             ), 5)
        
#         combine = rots.matmul(torch.inverse(intrins))
#         points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
#         points += trans.view(B, N, 1, 1, 1, 3)

#         return points