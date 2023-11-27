"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # 获得输出 下采样
        x = self.get_eff_depth(x)
        # Depth
        # B*N 512 fH fW
        x = self.depthnet(x)
        # B*N,D+C，fH,fW
        # depthnet 
        # 拆分 D C
        # 经过softmax
        depth = self.get_depth_dist(x[:, :self.D])
        # new_x 生成是生成per pixel outer product
        # B x 1 x D x fH x fW
        # B x C x 1 x fH x fW
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        # new_x 会生成 B x C x D x fH x fW 
        # depth (BxN) x D x fH x fw
        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        # 获得截获输出点的 output tensor 如下所示
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        
        #dx xbound=[-50.0, 50.0, 0.5],
        #bx ybound=[-50.0, 50.0, 0.5],
        #nx zbound=[-10.0, 10.0, 20.0],
        self.downsample = 16
        self.camC = 64
        # 视锥的创建 
        self.frustum = self.create_frustum()
        # D 41
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        # 16 下采样
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        # 如果什么都没增强这边可以忽略不计，是rt矩阵变换,但是要需要注意增加一维度
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        # 从r矩阵求逆获得
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # 使用了转换矩阵以后 会拓展一维
        # B x N x D x H x W x 3 x 1 - (1,1,41,8,22,3,1)
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        # cat 拼接第五维度 (1,1,41,8,22,3,1)
        # cam-> vcs(ego)
        combine = rots.matmul(torch.inverse(intrins))
        # R 矩阵 配合 K 内参矩阵
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # 去除最后一维度 B x N x D x H x W x 3
        #   (1,1,41,8,22,3)
        points += trans.view(B, N, 1, 1, 1, 3)
        # 配合使用T矩阵回到
        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        # B: batch size N：环视相机个数
        B, N, C, imH, imW = x.shape
        # BN拼接，多视角特征强行拼接，图像还是C
        x = x.view(B*N, C, imH, imW)
        # 直接进入编码
        x = self.camencode(x)
        # output B x D x C x fH x fw
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        # B x N x C x D x fH x fW
        x = x.permute(0, 1, 3, 4, 5, 2)
        # B x N x D x fH x fW x C
        return x

    def voxel_pooling(self, geom_feats, x):
        # geom_feats：(B x N x D x H x W x 3)：在ego坐标系下的坐标点；
        # x：(B x N x D x fH x fW x C)：图像点云特征
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        # ego下的空间坐标转换到体素坐标（计算栅格坐标并取整）
        # bx tensor([-49.7500, -49.7500,   0.0000])
        # dx tensor([ 0.5000,  0.5000, 20.0000])
        # 平移到中间把输出的X Y Z 需要拍平 全部>0
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        # 将体素坐标同样展平，geom_feats: (B*N*D*H*W, 3)
        geom_feats = geom_feats.view(Nprime, 3)
        
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        # geom_feats shape Nprime(B x N x D x H x W) x 4 最后一维度带 batch_ix 信息
        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~199  y: 0~199  z: 0
        # 对 x 模型输出的feat 进行过滤
        # 对 geom_feats 外参转换回来ego坐标进行过滤
        # kept 获得bool值
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        # 不同相机点投射到同一个点
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        # 根据每个batch_idx索引偏移
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        # 删除 Z 轴
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)
        # B x N x D x fH x fW x C
        # 关键的体素池化 借鉴Lidar的思维
        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
# if __name__ == "__main__":
    
    # make grid in image plane
   