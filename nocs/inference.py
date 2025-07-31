import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import glob
import open3d as o3d
from models.model import PPFDecoder
import time
import pickle
import argparse
from torchvision.models import segmentation
from utils.util import backproject, fibonacci_sphere, convert_layers, estimate_normals
from utils.dataset_inference import PPF_infer_Dataset
from utils.dataset_train import PM_Dataset
import cv2
import numpy as np
import omegaconf
import MinkowskiEngine as ME
import torch
from tqdm import tqdm
import cupy as cp
from models.voting import rot_voting_kernel, backvote_kernel, ppf_kernel
from tqdm import tqdm
from utils.easy_utils import vis_cloud,compose_rt,save_pts,rot_diff_degree,tr_diff

def tracking_func(input_pc, ppf_decoder,p_args,n_pairs=10000):
    cfg             = p_args['cfg'] 
    cp_device       = p_args['cp_device'] 
    num_rots        = p_args['num_rots']  
    n_threads       = p_args['n_threads'] 
    sphere_pts      = p_args['sphere_pts']
    adaptive_voting = p_args['adaptive_voting']
    angle_tol       = p_args['angle_tol']
    cls_pts         = p_args['cls_pts'].cuda()
    bcelogits = torch.nn.BCEWithLogitsLoss()  
    RTs = np.eye(4, dtype=np.float32)
    scales = np.ones((3,), dtype=np.float32)
    # high_res_indices = ME.utils.sparse_quantize(np.ascontiguousarray(input_pc), return_index=True, quantization_size=cfg.res)[1]
    # input_pc = input_pc[high_res_indices].astype(np.float32)
    # # assert len(input_pc)>512
    pc_normal = estimate_normals(input_pc, cfg.knn).astype(np.float32)
    pc_normals = torch.from_numpy(pc_normal[None]).cuda()
    pcs = torch.from_numpy(input_pc[None]).cuda()

    point_idxs = np.random.randint(0, input_pc.shape[0], (n_pairs, 2))
        
    with torch.no_grad():
        dist = torch.cdist(pcs, pcs)
        # sprin_feat = point_encoder(pcs, pc_normals, dist, cano_p)
        preds_out = ppf_decoder(pcs, pc_normals, dist, idxs=point_idxs, vertices=cls_pts, aux=cfg.aux_cls)
        scales = preds_out[-1]
        preds = preds_out[0]
        preds_tr = preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins)
   
    preds_tr = torch.softmax(preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins), -1)
    preds_tr = torch.cat([torch.multinomial(preds_tr[:, 0], 1), torch.multinomial(preds_tr[:, 1], 1)], -1).float()[None]
    preds_tr[0, :, 0] = preds_tr[0, :, 0] / (cfg.tr_num_bins - 1) * 2 * cfg.vote_range[0] - cfg.vote_range[0]
    preds_tr[0, :, 1] = preds_tr[0, :, 1] / (cfg.tr_num_bins - 1) * cfg.vote_range[1]
        
    # vote for center
    with cp.cuda.Device(cp_device):
        block_size = (input_pc.shape[0] ** 2 + 512 - 1) // 512

        corners = np.stack([np.min(input_pc, 0), np.max(input_pc, 0)])
        grid_res = ((corners[1] - corners[0]) / cfg.res).astype(np.int32) + 1  
        grid_res = np.clip(grid_res, a_min= 0, a_max=50)
        grid_obj = cp.asarray(np.zeros(grid_res, dtype=np.float32))
        ppf_kernel(
            (block_size, 1, 1),
            (512, 1, 1),
            (
                cp.asarray(input_pc).astype(cp.float32), cp.asarray(preds_tr[0].cpu().numpy()).astype(cp.float32), cp.asarray(np.ones((input_pc.shape[0],))).astype(cp.float32),
                cp.asarray(point_idxs).astype(cp.int32), grid_obj, cp.asarray(corners[0]), cp.float32(cfg.res), 
                point_idxs.shape[0], num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2], True if adaptive_voting else False
            )
        )
        
        grid_obj = grid_obj.get()
        cand = np.array(np.unravel_index([np.argmax(grid_obj, axis=None)], grid_obj.shape)).T[::-1]
        cand_world = corners[0] + cand * cfg.res
        
    T_est = cand_world[-1]

    corners = np.stack([np.min(input_pc, 0), np.max(input_pc, 0)])
    RTs[:3, -1] = T_est
    
    # back vote filtering
    block_size = (point_idxs.shape[0] + n_threads - 1) // n_threads

    pred_center = T_est
    with cp.cuda.Device(cp_device):
        output_ocs = cp.zeros((point_idxs.shape[0], 3), cp.float32)
        backvote_kernel(
            (block_size, 1, 1),
            (n_threads, 1, 1),
            (
                cp.asarray(input_pc), cp.asarray(preds_tr[0].cpu().numpy()), output_ocs, cp.asarray(point_idxs).astype(cp.int32), cp.asarray(corners[0]), cp.float32(cfg.res), 
                point_idxs.shape[0], num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2], cp.asarray(pred_center).astype(cp.float32), cp.float32(3 * cfg.res)
            )
        )
    oc = output_ocs.get()
    mask = np.any(oc != 0, -1)
    point_idxs = point_idxs[mask]  
        
    with torch.cuda.device(0):
        with torch.no_grad():
            # sprin_feat = point_encoder.forward_nbrs(input_pc[None], pc_normal[None], torch.from_numpy(knn_idxs).cuda()[None])[0]
            preds_out = ppf_decoder(pcs, pc_normals, dist, idxs=point_idxs, vertices=cls_pts, aux=cfg.aux_cls)
            preds = preds_out[0]
            preds_tr = preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins)
            preds_up = preds[..., 2 * cfg.tr_num_bins:2 * cfg.tr_num_bins + cfg.rot_num_bins]
            preds_right = preds[..., 2 * cfg.tr_num_bins + cfg.rot_num_bins:2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins]
            preds_up_aux = preds[..., -5]
            preds_right_aux = preds[..., -4]
            preds_scale = preds[..., -3:]
            
            preds_tr = torch.softmax(preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins), -1)
            preds_tr = torch.cat([torch.multinomial(preds_tr[:, 0], 1), torch.multinomial(preds_tr[:, 1], 1)], -1).float()[None]
            preds_tr[0, :, 0] = preds_tr[0, :, 0] / (cfg.tr_num_bins - 1) * 2 * cfg.vote_range[0] - cfg.vote_range[0]
            preds_tr[0, :, 1] = preds_tr[0, :, 1] / (cfg.tr_num_bins - 1) * cfg.vote_range[1]
            
            preds_up = torch.softmax(preds_up[0], -1)
            preds_up = torch.multinomial(preds_up, 1).float()[None]
            preds_up[0] = preds_up[0] / (cfg.rot_num_bins - 1) * np.pi
            
            preds_right = torch.softmax(preds_right[0], -1)
            preds_right = torch.multinomial(preds_right, 1).float()[None]
            preds_right[0] = preds_right[0] / (cfg.rot_num_bins - 1) * np.pi

    final_directions = []
    for j, (direction, aux) in enumerate(zip([preds_up, preds_right], [preds_up_aux, preds_right_aux])):
        if j == 1 and not cfg.regress_right:
            continue
            
        # vote for orientation
        with cp.cuda.Device(cp_device):
            candidates = cp.zeros((point_idxs.shape[0], num_rots, 3), cp.float32)

            block_size = (point_idxs.shape[0] + 512 - 1) // 512
            rot_voting_kernel(
                (block_size, 1, 1),
                (512, 1, 1),
                (
                    cp.asarray(input_pc), cp.asarray(preds_tr[0].cpu().numpy()), cp.asarray(direction[0].cpu().numpy()), candidates, cp.asarray(point_idxs).astype(cp.int32), cp.asarray(corners[0]).astype(cp.float32), cp.float32(cfg.res), 
                    point_idxs.shape[0], num_rots, grid_obj.shape[0], grid_obj.shape[1], grid_obj.shape[2]
                )
            )
            sph_cp = torch.tensor(sphere_pts.T, dtype=torch.float32).cuda()
            start = np.arange(0, point_idxs.shape[0] * num_rots, num_rots)
            np.random.shuffle(start)
            sub_sample_idx = (start[:10000, None] + np.arange(num_rots)[None]).reshape(-1)
            candidates = torch.as_tensor(candidates, device='cuda').reshape(-1, 3)
            candidates = candidates[torch.LongTensor(sub_sample_idx).cuda()]
            cos = candidates.mm(sph_cp)
            counts = torch.sum(cos > np.cos(angle_tol / 180 * np.pi), 0).cpu().numpy()
        best_dir = np.array(sphere_pts[np.argmax(counts)])
        
        # filter up
        ab = input_pc[point_idxs[:, 0]] - input_pc[point_idxs[:, 1]]
        distsq = np.sum(ab ** 2, -1)
        ab_normed = ab / (np.sqrt(distsq) + 1e-7)[..., None]
        
        pairwise_normals = pc_normal[point_idxs[:, 0]]
        pairwise_normals[np.sum(pairwise_normals * ab_normed, -1) < 0] *= -1
            
        with torch.no_grad():
            target = torch.from_numpy((np.sum(pairwise_normals * best_dir, -1) > 0).astype(np.float32)).cuda()
            up_loss = bcelogits(aux[0], target).item()
            down_loss = bcelogits(aux[0], 1. - target).item()
            
        if down_loss < up_loss:
            final_dir = -best_dir
        else:
            final_dir = best_dir
        final_directions.append(final_dir)
        
    up = final_directions[0]
    if up[1] < 0:
        up = -up
        
    if cfg.regress_forward:
        forward = final_directions[1]
        if forward[2] < 0:
            forward = -forward
        forward -= np.dot(up, forward) * up
        forward /= (np.linalg.norm(forward) + 1e-9)
    else:
        forward = np.array([-up[1], up[0], 0.])
        forward /= (np.linalg.norm(forward) + 1e-9)
        
    if np.linalg.norm(forward) < 1e-7: # right is zero
        # input('right real?')
        forward = np.random.randn(3)
        forward -= forward.dot(up) * up
        forward /= np.linalg.norm(forward)

    R_est = np.stack([np.cross(up, forward), up, forward], -1)
       
    # pred_scale = np.exp(preds_scale[0].mean(0).cpu().numpy()) # * scale_mean * 2   
    # # scale_norm = np.linalg.norm(pred_scale)
    scale_norm = 1.
    # # assert scale_norm > 0
        
    RTs[:3, :3] = R_est * scale_norm
    
    # scales = pred_scale / scale_norm
    
    return RTs, scales

class compute_3d_iou_size(torch.nn.Module):

    def __init__(self):
        super(compute_3d_iou_size, self).__init__()   
    def forward(self,scales_1,scales_2,num_part):
        iou_size = [0. for _ in range(num_part)]

        scales_1_new = scales_1.reshape(-1,3)
        scales_2_new = scales_2.reshape(-1,3)
        for idx in range(num_part):
            iou = compute_iou(scales_1_new[idx],scales_2_new[idx])
            iou_size[idx] = iou
        iou_size = np.array(iou_size)

        return iou_size  

def compute_iou(scales_1,scales_2):
    noc_cube_1 = get_3d_bbox(scales_1, 0)
    noc_cube_2 = get_3d_bbox(scales_2, 0)
    bbox_1_max = np.amax(noc_cube_1, axis=0)
    bbox_1_min = np.amin(noc_cube_1, axis=0)
    bbox_2_max = np.amax(noc_cube_2, axis=0)
    bbox_2_min = np.amin(noc_cube_2, axis=0)
    overlap_min = np.maximum(bbox_1_min, bbox_2_min)
    overlap_max = np.minimum(bbox_1_max, bbox_2_max)
    if np.amin(overlap_max - overlap_min) < 0:
        intersections = 0
    else:
        intersections = np.prod(overlap_max - overlap_min)
    union = np.prod(bbox_1_max - bbox_1_min) + \
        np.prod(bbox_2_max - bbox_2_min) - intersections
    overlaps = intersections / union
    return overlaps

def get_3d_bbox(scale, shift=0):
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                            [scale / 2, +scale / 2, -scale / 2],
                            [-scale / 2, +scale / 2, scale / 2],
                            [-scale / 2, +scale / 2, -scale / 2],
                            [+scale / 2, -scale / 2, scale / 2],
                            [+scale / 2, -scale / 2, -scale / 2],
                            [-scale / 2, -scale / 2, scale / 2],
                            [-scale / 2, -scale / 2, -scale / 2]]) + shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d   
