
import sys
sys.path.append('/home/mxh24/codes/PPF_Tracker_release/models')
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sprin import GlobalInfoProp, SparseSO3Conv
import numpy as np
import copy
from models.layers import EquivariantLayer, JointBranch2
import models.gcn3d_hs as gcn3d_hs
import torch.optim.lr_scheduler as lr_sched

import os
os.chdir(sys.path[0])
from Pointnet2_PyTorch_master.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule

from IPython import embed
from utils.easy_utils import sample_points, generate_rd_transform, easy_trans
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

def print_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        # print('Learning rate: ', param_group['lr'])
        return param_group['lr']
    
class PoseEstimator(torch.nn.Module):
    def __init__(self, num_parts, init_r, init_t, device, joint_axes):
        super(PoseEstimator, self).__init__()
        self.num_parts = num_parts
        self.device = device
        self.joint_axes = joint_axes
        
        self.rot_quat_s = []
        self.tra_s = []
        for idx in range(self.num_parts):
            x, y, z, w = R.from_matrix(init_r[idx].cpu().numpy()).as_quat()
            rot_quat = torch.nn.Parameter(torch.tensor(
                [w, x, y, z], device=device, dtype=torch.float), requires_grad=True)  # q=a+bi+ci+di
            tra = torch.nn.Parameter(init_t[idx].reshape(3,1).clone().detach().to(device), requires_grad=True)
            self.rot_quat_s.append(rot_quat)
            self.tra_s.append(tra)

        self.rot_quat_s = nn.ParameterList([torch.nn.Parameter(torch.tensor( \
            [w, x, y, z], device=device, dtype=torch.float), requires_grad=True) for idx in range(self.num_parts)])
        self.tra_s = nn.ParameterList([torch.nn.Parameter(init_t[idx].reshape(3,1).clone().detach().to(device), requires_grad=True) for idx in range(self.num_parts)])
       
        
   
    def E_geo(self, x, y):
        try:
            x = x.to(torch.float64)
            y = y.to(torch.float64)
            dist_matrix = torch.cdist(x, y)  # 计算x和y的二范数

            min_dist_x_to_y, _ = torch.min(dist_matrix, dim=1)
            Dxy = torch.mean(min_dist_x_to_y, dim=0)

            min_dist_y_to_x, _ = torch.min(dist_matrix, dim=0)
            Dyy = torch.mean(min_dist_y_to_x, dim=0)

            e_geo = torch.mean(Dxy + Dyy)

            return e_geo

        except:
            print(x.shape, y.shape)
            tensor = torch.tensor(1.0, dtype=torch.float64, device='cuda:0', requires_grad=True)
            return tensor
   
    def E_kin(self, transforms, joint_axes):   
        qj_homo = torch.from_numpy(joint_axes).float().to(self.device) 
        # qj_homo = torch.cat([qj, torch.ones(1, device=self.device)])   
        Tj_q = transforms[0] @ qj_homo[0].T                          
        Tj1_q = transforms[1] @ qj_homo[1].T                         
        diff = Tj_q - Tj1_q
        # e_kin = torch.norm(diff,p=2)                      
        norm = torch.norm(diff, p=2)
        e_kin = torch.sigmoid(-5 * norm)
        return e_kin

    def forward(self, camera_pts, cad_pts, part_weight):
        all_objective = 0.
        e_geo = 0.
        transforms = []
        scad_pts = [_ for _ in range(self.num_parts)]
        scamera_pts = [_ for _ in range(self.num_parts)]
        for idx in range(self.num_parts):
            c_mask = sample_points(cad_pts[idx], 3, True)
            scad_pts[idx] = cad_pts[idx][c_mask]
            scamera_pts[idx] = camera_pts[idx][c_mask]
            # scad_pts[idx] = cad_pts[idx]
            # scamera_pts[idx] = camera_pts[idx]


        scad_pts = [torch.cat([pts.to(self.device), torch.ones(pts.shape[0], 1, device=self.device)], dim=-1) for pts in scad_pts]
        scamera_pts = [torch.cat([pts.to(self.device), torch.ones(pts.shape[0], 1, device=self.device)], dim=-1) for pts in scamera_pts]
        for idx in range(self.num_parts):

            base_r_quat = self.rot_quat_s[idx] / torch.norm(self.rot_quat_s[idx])
            a, b, c, d = base_r_quat[0], base_r_quat[1], base_r_quat[2], base_r_quat[3]  # q=a+bi+ci+di
            base_rot_matrix = torch.stack([1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                                        2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b,
                                        2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d,
                                        1 - 2 * b * b - 2 * c * c]).reshape(3, 3)
            base_transform = torch.cat([torch.cat([base_rot_matrix, self.tra_s[idx]], dim=1),
                                        torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0)
            transforms.append(base_transform)
            cad_base = base_transform.matmul(scad_pts[idx].T).T
            camera_base = scamera_pts[idx]
            # base_objective = self.E_geo(base_transform, scamera_pts[idx], scad_pts[idx])
            base_objective = self.E_geo(cad_base, camera_base)
            # if not choosen_opt[idx]:
            #     base_objective = 0.

            e_geo += part_weight[idx] * base_objective
        e_kin = self.E_kin(transforms, self.joint_axes)
        all_objective = e_geo + part_weight[-1] * e_kin
        transforms = torch.stack(transforms, axis=0)
        return all_objective, e_kin, transforms.detach()

class Pose_Ts_part_size(nn.Module):
    def __init__(self):
        super(Pose_Ts_part_size, self).__init__()
        self.f = 1289
        #self.k = FLAGS.Ts_c
        self.k = 3     # num_parts*3

        #self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)
        #self.conv1 = torch.nn.Conv1d(1283, 1024, 1)
        num_points = 1024*3
        self.conv1 = torch.nn.Conv1d(1795, num_points, 1)
        self.conv2 = torch.nn.Conv1d(num_points, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(num_points)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.conv3(x)) # outshape 1,256,1
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        #xt = x[:, 0:3]
        xs = x[:, 0:]
        return  xs


class PointSeg(nn.Module):
    def __init__(self, in_channel=3,num_parts=2,use_background=False):
        super(PointSeg, self).__init__()
        use_xyz = True if in_channel == 3 else False
        self.num_parts = num_parts
        self.num_classes = self.num_parts + 1 if use_background else self.num_parts
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[in_channel - 3, 32, 32, 64],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=use_xyz,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + in_channel - 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        
        self.seg_layer = nn.Conv1d(128, self.num_parts, kernel_size=1, padding=0)


    def forward(self, xyz, features):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        # xyz, features = self._break_up_pc(pointcloud)
        # embed()
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
        dense_pts_feat = self.fc_layer(l_features[0])
        # dense_cls_score = self.cls_2(self.cls_1(dense_pts_feat)).transpose(1, 2).contiguous()  # (bs, n, K)
        pred_seg_per_point = self.seg_layer(dense_pts_feat).transpose(1,2)

        # Process the predicted things
        pred_seg_per_point = F.softmax(pred_seg_per_point, dim=2)
        return pred_seg_per_point


class ResLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bn=False) -> None:
        super().__init__()
        assert(bn is False)
        self.fc1 = torch.nn.Linear(dim_in, dim_out)
        if bn:
            self.bn1 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn1 = lambda x: x
        self.fc2 = torch.nn.Linear(dim_out, dim_out)
        if bn:
            self.bn2 = torch.nn.BatchNorm1d(dim_out)
        else:
            self.bn2 = lambda x: x
        if dim_in != dim_out:
            self.fc0 = torch.nn.Linear(dim_in, dim_out)
        else:
            self.fc0 = None
    
    def forward(self, x):
        x_res = x if self.fc0 is None else self.fc0(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x + x_res
    
class PointEncoder(nn.Module):
    def __init__(self, k, spfcs, out_dim1, num_layers=2, num_nbr_feats=2) -> None:
        super().__init__()
        self.k = k
        self.spconvs = nn.ModuleList()
        self.spconvs.append(SparseSO3Conv(32, num_nbr_feats, out_dim1, *spfcs))
        self.aggrs = nn.ModuleList()
        self.aggrs.append(GlobalInfoProp(out_dim1, out_dim1 // 4))
        for _ in range(num_layers - 1):
            self.spconvs.append(SparseSO3Conv(32, out_dim1 + out_dim1 // 4, out_dim1, *spfcs))
            self.aggrs.append(GlobalInfoProp(out_dim1, out_dim1 // 4))
      

    def forward(self, pc, pc_normal, dist):
        nbrs_idx = torch.topk(dist, self.k, largest=False, sorted=False)[1]  #[..., N, K]
        pc_nbrs = torch.gather(pc.unsqueeze(-3).expand(*pc.shape[:-1], *pc.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc.shape[-1]))  #[..., N, K, 3]
        pc_nbrs_centered = pc_nbrs - pc.unsqueeze(-2)  #[..., N, K, 3]
        pc_nbrs_norm = torch.norm(pc_nbrs_centered, dim=-1, keepdim=True)
        
        pc_normal_nbrs = torch.gather(pc_normal.unsqueeze(-3).expand(*pc_normal.shape[:-1], *pc_normal.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc_normal.shape[-1]))  #[..., N, K, 3]
        pc_normal_cos = torch.sum(pc_normal_nbrs * pc_normal.unsqueeze(-2), -1, keepdim=True)
        
        feat = self.aggrs[0](self.spconvs[0](pc_nbrs, torch.cat([pc_nbrs_norm, pc_normal_cos], -1), pc))
        for i in range(len(self.spconvs) - 1):
            spconv = self.spconvs[i + 1]
            aggr = self.aggrs[i + 1]
            feat_nbrs = torch.gather(feat.unsqueeze(-3).expand(*feat.shape[:-1], *feat.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, feat.shape[-1]))
            feat = aggr(spconv(pc_nbrs, feat_nbrs, pc))  # [1, X, 40]
        
        return feat 
    
    def forward_nbrs(self, pc, pc_normal, nbrs_idx):
        pc_nbrs = torch.gather(pc.unsqueeze(-3).expand(*pc.shape[:-1], *pc.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc.shape[-1]))  #[..., N, K, 3]
        pc_nbrs_centered = pc_nbrs - pc.unsqueeze(-2)  #[..., N, K, 3]
        pc_nbrs_norm = torch.norm(pc_nbrs_centered, dim=-1, keepdim=True)
        
        pc_normal_nbrs = torch.gather(pc_normal.unsqueeze(-3).expand(*pc_normal.shape[:-1], *pc_normal.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc_normal.shape[-1]))  #[..., N, K, 3]
        pc_normal_cos = torch.sum(pc_normal_nbrs * pc_normal.unsqueeze(-2), -1, keepdim=True)
        
        feat = self.aggrs[0](self.spconvs[0](pc_nbrs, torch.cat([pc_nbrs_norm, pc_normal_cos], -1), pc))
        for i in range(len(self.spconvs) - 1):
            spconv = self.spconvs[i + 1]
            aggr = self.aggrs[i + 1]
            feat_nbrs = torch.gather(feat.unsqueeze(-3).expand(*feat.shape[:-1], *feat.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, feat.shape[-1]))
            feat = aggr(spconv(pc_nbrs, feat_nbrs, pc))
        return feat


class PPFEncoder(nn.Module):
    def __init__(self, ppffcs, out_dim2, k, spfcs, out_dim1, num_layers=2, num_nbr_feats=2, weighted=False) -> None:
        super().__init__()
        self.k = k
        self.spconvs = nn.ModuleList()
        self.spconvs.append(SparseSO3Conv(32, num_nbr_feats, out_dim1, *spfcs))
        self.aggrs = nn.ModuleList()
        self.aggrs.append(GlobalInfoProp(out_dim1, out_dim1 // 4))
        for _ in range(num_layers - 1):
            self.spconvs.append(SparseSO3Conv(32, out_dim1 + out_dim1 // 4, out_dim1, *spfcs))
            self.aggrs.append(GlobalInfoProp(out_dim1, out_dim1 // 4))
       
        ##########################################################################
        self.weighted = weighted
        self.res_layers = nn.ModuleList()
        for i in range(len(ppffcs) - 1):
            dim_in, dim_out = ppffcs[i], ppffcs[i + 1]
            self.res_layers.append(ResLayer(dim_in, dim_out, bn=False))
        self.final = nn.Linear(ppffcs[-1], out_dim2)

        #############################################################################
        self.neighbor_num = 20
        self.support_num = 7
        self.conv_0 = gcn3d_hs.HSlayer_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = gcn3d_hs.HS_layer(128, 128, support_num=self.support_num)
        self.pool_1 = gcn3d_hs.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = gcn3d_hs.HS_layer(128, 256, support_num=self.support_num)
        self.conv_3 = gcn3d_hs.HS_layer(256, 256, support_num=self.support_num)
        self.pool_2 = gcn3d_hs.Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = gcn3d_hs.HS_layer(256, 512, support_num=self.support_num)
        self.conv_5 = gcn3d_hs.HS_layer(512, 512, support_num=self.support_num)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.get_size = Pose_Ts_part_size()

    def forward(self, pc, pc_normal, dist=None, idxs=None, vertices=None):
        nbrs_idx = torch.topk(dist, self.k, largest=False, sorted=False)[1]  #[..., N, K]
        pc_nbrs = torch.gather(pc.unsqueeze(-3).expand(*pc.shape[:-1], *pc.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc.shape[-1]))  #[..., N, K, 3]
        pc_nbrs_centered = pc_nbrs - pc.unsqueeze(-2)  #[..., N, K, 3]
        pc_nbrs_norm = torch.norm(pc_nbrs_centered, dim=-1, keepdim=True)
        
        pc_normal_nbrs = torch.gather(pc_normal.unsqueeze(-3).expand(*pc_normal.shape[:-1], *pc_normal.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc_normal.shape[-1]))  #[..., N, K, 3]
        pc_normal_cos = torch.sum(pc_normal_nbrs * pc_normal.unsqueeze(-2), -1, keepdim=True)
        
        feat = self.aggrs[0](self.spconvs[0](pc_nbrs, torch.cat([pc_nbrs_norm, pc_normal_cos], -1), pc)) 
        for i in range(len(self.spconvs) - 1):
            spconv = self.spconvs[i + 1]
            aggr = self.aggrs[i + 1]
            feat_nbrs = torch.gather(feat.unsqueeze(-3).expand(*feat.shape[:-1], *feat.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, feat.shape[-1]))
            feat = aggr(spconv(pc_nbrs, feat_nbrs, pc))  # [1, X, 40]
        ###################################################################################
        
        if idxs is not None:
            final_output = self.forward_with_idx(pc[0], pc_normal[0], feat[0], idxs)[None]
        else:
            xx = pc.unsqueeze(-2) - pc.unsqueeze(-3)
            xx_normed = xx / (dist[..., None] + 1e-7)

            # used to weighted PPF
            if self.weighted:
                lambda_ = 0.5
                cos_theta = torch.sum(
                    pc_normal[..., :, None, :] * pc_normal[..., None, :, :], dim=-1
                ).clamp(-1.0, 1.0)
                weight = 1.0 - lambda_ * torch.abs(cos_theta)

            outputs = []
            for idx in torch.chunk(torch.arange(pc.shape[1]), 5):
                feat_chunk = feat[..., idx, :]
                target_shape = [*feat_chunk.shape[:-2], feat_chunk.shape[-2], feat.shape[-2], feat_chunk.shape[-1]]  # B x NC x N x F
                xx_normed_chunk = xx_normed[..., idx, :, :]
                ppf = torch.cat([
                    torch.sum(pc_normal[..., idx, :].unsqueeze(-2) * xx_normed_chunk, -1, keepdim=True), 
                    torch.sum(pc_normal.unsqueeze(-3) * xx_normed_chunk, -1, keepdim=True), 
                    torch.sum(pc_normal[..., idx, :].unsqueeze(-2) * pc_normal.unsqueeze(-3), -1, keepdim=True), 
                    dist[..., idx, :, None],
                ], -1)
                # ppf.zero_()
                final_feat = torch.cat([feat_chunk[..., None, :].expand(*target_shape), feat[..., None, :, :].expand(*target_shape), ppf], -1)
            
                # 应用加权
                if self.weighted:
                    w = weight[..., idx, :].unsqueeze(-1)
                    final_feat = final_feat * w


                output = final_feat
                for res_layer in self.res_layers:
                    output = res_layer(output)
                outputs.append(output)
            
            output = torch.cat(outputs, dim=-3)
            final_output = self.final(output)

        if vertices is not None:
            # 推理阶段主动输入逆变换后的点云
            Pred_s = self.estimate_size(vertices)
        else:
            # 训练阶段，直接用标准空间点云就行  pc.shape   torch.Size([1, 3072, 3])
            Pred_s = self.estimate_size(pc)    # [1, 3]

        return final_output, Pred_s

    def forward_with_idx(self, pc, pc_normal, feat, idxs):
        a_idxs = idxs[:, 0]
        b_idxs = idxs[:, 1]
        xy = pc[a_idxs] - pc[b_idxs]
        xy_norm = torch.norm(xy, dim=-1)
        xy_normed = xy / (xy_norm[..., None] + 1e-7)
        pnormal_cos = pc_normal[a_idxs] * pc_normal[b_idxs]
        ppf = torch.cat([
            torch.sum(pc_normal[a_idxs] * xy_normed, -1, keepdim=True),
            torch.sum(pc_normal[b_idxs] * xy_normed, -1, keepdim=True),
            torch.sum(pnormal_cos, -1, keepdim=True),
            xy_norm[..., None],
        ], -1)
        # ppf.zero_()
        
        final_feat = torch.cat([feat[a_idxs], feat[b_idxs], ppf], -1)
        
        output = final_feat
        for res_layer in self.res_layers:
            output = res_layer(output)
        return self.final(output)
    
    def point_forward_nbrs(self, pc, pc_normal, nbrs_idx):
        pc_nbrs = torch.gather(pc.unsqueeze(-3).expand(*pc.shape[:-1], *pc.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc.shape[-1]))  #[..., N, K, 3]
        pc_nbrs_centered = pc_nbrs - pc.unsqueeze(-2)  #[..., N, K, 3]
        pc_nbrs_norm = torch.norm(pc_nbrs_centered, dim=-1, keepdim=True)
        
        pc_normal_nbrs = torch.gather(pc_normal.unsqueeze(-3).expand(*pc_normal.shape[:-1], *pc_normal.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, pc_normal.shape[-1]))  #[..., N, K, 3]
        pc_normal_cos = torch.sum(pc_normal_nbrs * pc_normal.unsqueeze(-2), -1, keepdim=True)
        
        feat = self.aggrs[0](self.spconvs[0](pc_nbrs, torch.cat([pc_nbrs_norm, pc_normal_cos], -1), pc))
        for i in range(len(self.spconvs) - 1):
            spconv = self.spconvs[i + 1]
            aggr = self.aggrs[i + 1]
            feat_nbrs = torch.gather(feat.unsqueeze(-3).expand(*feat.shape[:-1], *feat.shape[-2:]), -2, nbrs_idx[..., None].expand(*nbrs_idx.shape, feat.shape[-1]))
            feat = aggr(spconv(pc_nbrs, feat_nbrs, pc))
        return feat

    def estimate_size(self, vertices):
        # 这个地方就不该传标准空间的点云，就算要用也应该使用优化之后的变换矩阵逆回去
        if type(vertices) is np.ndarray:            
            # 转化成torch格式，并且增加一个batch维度
            vertices = torch.from_numpy(vertices).to('cuda').unsqueeze(0)

        vertices -= vertices.mean(dim=1, keepdim=True)
        # 对点云进行一个扰动增强鲁棒性
        _, rd_R, rd_t = generate_rd_transform([-5,5],[-0.02,0.02])
        rd_R, rd_t = torch.from_numpy(rd_R).to('cuda').float(), torch.from_numpy(rd_t).to('cuda').float()
        # 对vertices进行rd_RT变换
        vertices = torch.matmul(vertices, rd_R.T) + rd_t  # [1, 3072, 3]
        
  
        fm_0 = F.relu(self.conv_0(vertices, self.neighbor_num), inplace=True)#bs,n,128
        fm_1 = F.relu(self.bn1(self.conv_1(vertices, fm_0, self.neighbor_num).transpose(1, 2)).transpose(1, 2), inplace=True)#bs,n,128
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        fm_2 = F.relu(self.bn2(self.conv_2(v_pool_1, fm_pool_1, 
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1, 2), inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(v_pool_1, fm_2, 
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1, 2), inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        fm_4 = self.conv_4(v_pool_2, fm_pool_2, min(self.neighbor_num, v_pool_2.shape[1] // 8))#
        fm_5 = F.relu(self.bn4(self.conv_5(v_pool_2, fm_4, 
                                       min(self.neighbor_num, v_pool_2.shape[1] // 8)).transpose(1, 2)).transpose(1, 2), inplace=True)

        nearest_pool_1 = gcn3d_hs.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d_hs.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d_hs.indexing_neighbor_new(fm_2, nearest_pool_1).squeeze(2)#bs,n,256
        fm_3 = gcn3d_hs.indexing_neighbor_new(fm_3, nearest_pool_1).squeeze(2)#bs,n,256
        fm_4 = gcn3d_hs.indexing_neighbor_new(fm_4, nearest_pool_2).squeeze(2)#bs,n,512
        fm_5 = gcn3d_hs.indexing_neighbor_new(fm_5, nearest_pool_2).squeeze(2)#bs,n,512
        
        f_s = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4,fm_5], dim=2)#bs,n,1280 1792
  
        ###size estimation###
        feat_for_size = torch.cat([f_s, vertices], dim=2)  #bs 1024 1795
        Pred_s = self.get_size(feat_for_size.permute(0, 2, 1))#bs 3*num_parts
        return Pred_s
    
    @staticmethod
    def optimization(estimator, camera_pts, cad_pts, part_weight, k=False):
        """
        使用观测点云和预测点云计算chamfer_distance, 将变换矩阵参数化, 进行优化更新
        pose_estimator = pose_optimizer.PoseEstimator(num_parts=n_parts, init_r=init_base_r, init_t=init_base_t,
                                                device=device, joint_type='revolute')

        loss, optim_transforms = pose_optimizer.optimize_pose(pose_estimator, xyz_camera, cad, part_weight)
        
        """
        estimator.rot_quat_s.requires_grad_(True)
        estimator.tra_s.requires_grad_(True)
        # params_to_optimize = []
        # for i, c in enumerate(choosen_opt):
        #     estimator.rot_quat_s[i].requires_grad_(c)
        #     estimator.tra_s[i].requires_grad_(c)
        #     if c:
        #         params_to_optimize.append(estimator.rot_quat_s[i])
        #         params_to_optimize.append(estimator.tra_s[i])

        lr = 0.1
        if k:
            MAX_EPOCH = 100
            et_lr = 0.05
        else:
            MAX_EPOCH = 30 # 100
            et_lr = 0.01
        optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=et_lr)
        best_loss = float('inf')
        current_lr = lr
        transforms = None
        with tqdm(range(MAX_EPOCH), desc='optim') as tbar:
            for iter in tbar:
                loss, e, transform = estimator(camera_pts, cad_pts, part_weight)
                tbar.set_postfix(loss=loss.item(),e=e.item(),lr=current_lr, x=iter)
                tbar.update()  # 默认参数n=1，每update一次，进度+n
                
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    transforms = transform
                if best_loss <= 0.3:
                    break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # change lr
                current_lr = print_learning_rate(optimizer)
                scheduler.step()
    
        return loss, transforms
