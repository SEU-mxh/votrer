import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append('/home/mxh24/codes/PPF_Tracker_release')
import glob
import open3d as o3d
from models.model import PPFEncoder, PointEncoder, PointSeg
import time
import pickle
import argparse
from torchvision.models import segmentation
from utils.util import backproject, fibonacci_sphere, convert_layers, estimate_normals
from utils.dataset_inference import PPF_infer_Dataset
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seg_dir', default='/home/mxh24/codes/MM_vote/voting_ppf/data/nocs_seg', help='Segmentation PKL files for NOCS')
    # parser.add_argument('--nocs_dir', default='/home/mxh24/codes/MM_vote/voting_ppf/data/nocs', help='NOCS real test image path')
    # parser.add_argument('--out_dir', default='/home/mxh24/codes/MM_vote/voting_ppf/data/nocs_prediction', help='Output directory for predictions')
    parser.add_argument('--cp_device', type=int, default=0, help='GPU device number for custom voting algorithms')
    parser.add_argument('--ckpt_path', default='/home/mxh24/codes/PPF_Tracker_release/checkpoint2/dishwasher/seg', help='Model checkpoint path')
    parser.add_argument('--angle_prec', type=float, default=1.5, help='Angle precision in orientation voting')
    parser.add_argument('--num_rots', type=int, default=72, help='Number of candidate center votes generated for a given point pair')
    parser.add_argument('--n_threads', type=int, default=512, help='Number of cupy threads')
    parser.add_argument('--bbox_mask', action='store_true', help='Whether to use bbox mask instead of instance segmentations')
    parser.add_argument('--adaptive_voting', action='store_false', help='Whether to use adaptive center voting')
    args = parser.parse_args()

    cp_device = args.cp_device


    path = os.path.join(args.ckpt_path)
    nepoch = 'best'
    cfg = omegaconf.OmegaConf.load(f'{path}/.hydra/config.yaml')
    cls_name = cfg.category
    point_seg = PointSeg().cuda().eval()
    
    
    point_seg.load_state_dict(torch.load(f'{path}/point_seg_epoch{nepoch}.pth'))
    
    dataset = PPF_infer_Dataset(cfg=cfg,
                                mode='val',
                                data_root='/home/mxh24/codes/dataset1',
                                num_pts=cfg.num_points,
                                num_cates=5,
                                num_parts=2,
                                device='cpu',
                                )
   
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i, data in enumerate(tqdm(data_loader)):
        # pc, test_RT =\
        #     data['pc'],data['RT']
        # pc, test_RT = pc[0].numpy(), test_RT[0].numpy()
        pc= data['camera_pts'][0].numpy()
               
        # high_res_indices = ME.utils.sparse_quantize(np.ascontiguousarray(pc), return_index=True, quantization_size=cfg.res)[1]
        # pc = pc[high_res_indices].astype(np.float32)
        # # if len(pc)<512:
        # #     continue
        # # pc_normal = estimate_normals(pc, cfg.knn).astype(np.float32)

        pcs = torch.from_numpy(pc[None]).cuda()
        # pc_normals = torch.from_numpy(pc_normal[None]).cuda()
        # point_idxs = np.random.randint(0, pc.shape[0], (100000, 2))
        if i % 10 == 0:
            with torch.no_grad():
                dense_part_cls_score = point_seg(pcs, None).contiguous()
                segmentation_result = torch.argmax(dense_part_cls_score, dim=-1)
                print(dense_part_cls_score.shape,segmentation_result.shape)
                # 选择一个特定的类别索引，例如类别 1
                selected_class_index = 1
                segmentation_result = segmentation_result.cpu().numpy()
                # 提取属于该类别的点云
                selected_points_mask = segmentation_result[0] == selected_class_index
                selected_points = pcs.squeeze()[selected_points_mask]
                selected_points = selected_points.cpu().numpy()
       
                # np.savetxt('/home/mxh24/codes/PPF_Tracker_release/pc_out.txt',selected_points)

                base_points_mask = segmentation_result[0] == 0
                base_points = pcs.squeeze()[base_points_mask]
                base_points = base_points.cpu().numpy()
                # np.savetxt('/home/mxh24/codes/PPF_Tracker_release/pc_out1.txt',base_points)
                vis_cloud(cloud_canonical=base_points, cloud_camera=selected_points)
                print()
