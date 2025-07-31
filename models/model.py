
import sys

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
    
class FeatNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=40, input_points=2048, output_points=1024):
        super(FeatNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels)
        )
        self.output_points = output_points

    def forward(self, x):
        # x: [B, 2048, 2]
        _, N, _ = x.shape
        x = self.mlp(x)  # [B, 2048, 40]
        
        if N > self.output_points:
            idx = torch.randperm(N)[:self.output_points]
            x = x[:, idx, :]  # [B, 1024, 40]
        else:
            raise ValueError("Input point number must be greater than output.")

        return x
    
class IOUHead(nn.Module):
    def __init__(self, n_parts, npoints):
        super(IOUHead, self).__init__()
        self.n_parts = n_parts
        self.npoints = npoints
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 3, kernel_size=1)  # 输出维度是3，对应每个part的xyz
        )

    def forward(self, dense_pts_feat):
        """
        参数:
            dense_pts_feat: [B, 128, n*1024]
        返回:
            [B, 3*n]
        """
        B, C, N = dense_pts_feat.shape
        assert C == 128, "输入通道应为128"
        # assert N % 1024 == 0, "输入点数应为n*1024"
        n = N // self.npoints

        feat = self.decoder(dense_pts_feat)  # [B, 3, n*1024]
        feat = feat.view(B, 3, n, 1024)
        feat = feat.mean(dim=-1)  # 对每个part内部的1024个点平均 => [B, 3, n]
        out = feat.view(B, 3 * n)  # 展平成 [B, 3*n]
        return out
    
class PointSeg(nn.Module):
    def __init__(self, in_channel=3,npts=1024,num_parts=2,use_background=False):
        super(PointSeg, self).__init__()
        use_xyz = True if in_channel == 3 else False
        self.num_parts = num_parts
        self.num_classes = self.num_parts + 1 if use_background else self.num_parts
        self.featnet = FeatNet(input_points=int(npts*num_parts),output_points=npts)
       
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
        self.size_layer = IOUHead(n_parts=num_parts,npoints=npts)
        self.seg_layer = nn.Conv1d(128, self.num_parts, kernel_size=1, padding=0)


    def forward(self, xyz, features=None):
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
        dense_pts_feat = self.fc_layer(l_features[0])  # [1,128,parts*1024]
        # 3D IOU
        
        Pred_s = self.size_layer(dense_pts_feat)

        # segmentation
        pred_seg_feat = self.seg_layer(dense_pts_feat).transpose(1,2)  # [1,parts*1024,2]
        cls_feat = self.featnet(pred_seg_feat)
        pred_seg_per_point = F.softmax(pred_seg_feat, dim=2)
        return pred_seg_per_point, cls_feat, Pred_s


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
    
class PPFDecoder(nn.Module):
    def __init__(self, ppffcs, out_dim2, k, spfcs, out_dim1, num_layers=2, num_nbr_feats=2, weighted=False, npts=1024, nparts=2) -> None:
        super().__init__()
        self.k = k
        self.classifier = PointSeg(npts=npts,num_parts=nparts)
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


    def forward(self, pc, pc_normal, dist=None, idxs=None, vertices=None, aux=False):
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
            feat = aggr(spconv(pc_nbrs, feat_nbrs, pc))  # [1, 1024, 40]
        ###################################################################################
        if aux:
            logits, cls_feat, Pred_s = self.classifier(vertices)  # [1, parts*1024, 2]  [1,1024,40]
            feat = feat + cls_feat
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
            
               
                if self.weighted:
                    w = weight[..., idx, :].unsqueeze(-1)
                    final_feat = final_feat * w


                output = final_feat
                for res_layer in self.res_layers:
                    output = res_layer(output)
                outputs.append(output)
            
            output = torch.cat(outputs, dim=-3)
            final_output = self.final(output)
        if aux:
            return [final_output, logits, Pred_s]
        return [final_output, Pred_s]

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

    @staticmethod
    def optimization(estimator, camera_pts, cad_pts, part_weight, lim):
        # print(cad_pts.requires_grad) # False
        # print(cad_pts.grad_fn)       # None
        cad_pts = cad_pts.clone()
        camera_pts = camera_pts.clone()
        estimator.rot_quat_s.requires_grad_(True)
        estimator.tra_s.requires_grad_(True)

        lr = 5e-2  # (3e-2 dishwasher)  
        MAX_EPOCH = 100   # 5e-3
        et_lr = 1e-2
       
        optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=et_lr)
        best_loss = float('inf')
        best_dif = float('inf')
        current_lr = lr
        transforms = None
        # cad_pts
        
        for iter in range(MAX_EPOCH):
            loss, es, transform = estimator(camera_pts, cad_pts, part_weight)
            if loss > lim[0]:
                loss = loss * 0.5
               
            dif = max(es[:-1]).item() - min(es[:-1]).item()
            if loss < lim[0] and dif <lim[1]:
                transforms = transform
                break

            if loss < best_loss and dif < best_dif:
                best_loss = loss
                best_dif = dif 
                transforms = transform
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    
        return loss, transforms
    
class PoseEstimator(torch.nn.Module):
    def __init__(self, num_parts, init_r, init_t, device, joint_axes, rt_k):
        super(PoseEstimator, self).__init__()
        self.num_parts = num_parts
        self.device = device
        self.joint_axes = joint_axes
        if isinstance(rt_k,list):
            self.rt_k = torch.from_numpy(np.array(rt_k)).float().to(device)
        else:
            self.rt_k = rt_k
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
   
    def E_kin(self, transforms, joint_axes,x, y): 
        distances = []
        for i in range(self.num_parts):
            distances.append(torch.norm(transforms[i].matmul(x[i].T).T - y[i], dim=-1).mean())
       
        distances = sum(distances) / self.num_parts
        
        qj_homo = torch.from_numpy(joint_axes).float().to(self.device) # [parts-1, 2, 4]
        transforms = torch.stack(transforms,axis=0)   # [parts, 4, 4]
        diff = []

        for i in range(self.num_parts-1):
            Tj_q = self.rt_k[0] @ transforms[0] @ qj_homo[0][0].T                          
            Tj1_q = self.rt_k[i+1] @ transforms[i+1] @ qj_homo[i][1].T  
            diff.append((Tj_q - Tj1_q)[:3])
                                     
        norm = (torch.norm(torch.stack(diff), p=2)) / self.num_parts
        e_kin = torch.log(0.1*norm + 1) + distances 
        return e_kin
    
    def forward(self, camera_pts, cad_pts, part_weight):
        all_objective = 0.
        transforms = []
        
        scad_pts = cad_pts.clone()
        scamera_pts = camera_pts.clone()

        scad_pts = [torch.cat([pts.to(self.device), torch.ones(pts.shape[0], 1, device=self.device)], dim=-1) for pts in scad_pts]
        scamera_pts = [torch.cat([pts.to(self.device), torch.ones(pts.shape[0], 1, device=self.device)], dim=-1) for pts in scamera_pts]
        eners = []
        errors = []
        for idx in range(self.num_parts):

            base_r_quat = self.rot_quat_s[idx] / torch.norm(self.rot_quat_s[idx])
            a, b, c, d = base_r_quat[0], base_r_quat[1], base_r_quat[2], base_r_quat[3]  # q=a+bi+ci+di
            base_rot_matrix = torch.stack([1 - 2 * c * c - 2 * d * d, 2 * b * c - 2 * a * d, 2 * a * c + 2 * b * d,
                                        2 * b * c + 2 * a * d, 1 - 2 * b * b - 2 * d * d, 2 * c * d - 2 * a * b,
                                        2 * b * d - 2 * a * c, 2 * a * b + 2 * c * d,
                                        1 - 2 * b * b - 2 * c * c]).reshape(3, 3)
            base_transform = torch.cat([torch.cat([base_rot_matrix, self.tra_s[idx]], dim=1),
                                        torch.tensor([[0., 0., 0., 1.]], device=self.device)], dim=0).float()
            transforms.append(base_transform)
            cad_base = base_transform.matmul(scad_pts[idx].T).T
            camera_base = scamera_pts[idx]
           
            base_objective = self.E_geo(cad_base, camera_base)
            eners.append(5.* base_objective)
           
       
        eners_tensor = torch.tensor(eners, device=self.device)
        min_val, max_val = torch.min(eners_tensor),torch.max(eners_tensor)
        eners_tensor = (eners_tensor-min_val)/(max_val-min_val+1e-8)  # Normalize to [0, 1]
        eners_softmax = F.softmax(eners_tensor, dim=0)
        for idx in range(self.num_parts): 
            errors.append(part_weight[idx] * eners_softmax[idx] * eners[idx])
            all_objective += eners_softmax[idx] * eners[idx]
        e_kin = max(errors) * self.E_kin(transforms, self.joint_axes,scad_pts, scamera_pts)
        errors.append(e_kin)
        all_objective += e_kin
        transforms = torch.stack(transforms, axis=0)
        return all_objective, errors, transforms.detach(),
