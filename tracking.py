import os

import sys

sys.path.append('/home/xxx/codes/PPF_Tracker_release')

import math
import time
import open3d as o3d
import torch
import numpy as np
import copy
import os.path as osp
import argparse
from utils.iou import *
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

import omegaconf
from utils.dataset_inference import PPF_infer_Dataset
from utils.easy_utils import *
from utils.util import backproject, fibonacci_sphere, convert_layers, estimate_normals
from utils.refrect import *
from models.model import PPFDecoder, PointSeg, PoseEstimator
from models.voting import rot_voting_kernel, backvote_kernel, ppf_kernel
from nocs.inference import tracking_func, compute_3d_iou_size



CLASSES = ['laptop', 'eyeglasses', 'dishwasher', 'drawer', 'scissors']



def cal_theta(a,b):
    # 计算点积
    dot_product = np.dot(a, b)
    # 计算模长
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_a * norm_b)
    # 计算夹角（以弧度为单位）
    theta = np.arccos(cos_theta)
    # 将夹角转换为度（如果需要）
    theta_degrees = np.degrees(theta)  
    return theta_degrees    
            

class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        # use center as initial point
        init_point = (np.max(pts, axis=0, keepdims=True) + np.min(pts, axis=0, keepdims=True)) / 2
        distances = self.calc_distances(init_point, pts)
        for i in range(0, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts

def calErr(pred_r, pred_t, gt_r, gt_t, sort_part):
    num_parts = gt_r.shape[0]
    r_errs = []
    t_errs = []
    for i in range(num_parts):
        
        r_err = rot_diff_degree(gt_r[i], pred_r[i])
        if r_err > 90:
            r_err = 180-r_err
        if r_err > 45:
            r_err = 90 - r_err
        t_err = np.linalg.norm(gt_t[i] - pred_t[i])   # 平方距离（L2范数）
        r_errs.append(r_err)
        t_errs.append(t_err)
    return r_errs, t_errs


def get_cloudscenter(obj_pts):
    if obj_pts.shape[0] == 2:
        obj_pts = np.concatenate(obj_pts, axis=0) 
    
    xmin, xmax = np.min(obj_pts[:, 0]), np.max(obj_pts[:, 0])
    ymin, ymax = np.min(obj_pts[:, 1]), np.max(obj_pts[:, 1])
    zmin, zmax = np.min(obj_pts[:, 2]), np.max(obj_pts[:, 2])
    center = np.array([(xmin + xmax)/2., (ymin + ymax)/2., (zmin + zmax)/2.])
    
    return center



#####################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--cp_device', type=int, default=0, help='GPU device number for custom voting algorithms')
    parser.add_argument('--ckpt_path', default='/home/xxx/codes/PPF_Tracker_release/checkpoint/scissors', help='Model checkpoint path')
    parser.add_argument('--angle_prec', type=float, default=1.5, help='Angle precision in orientation voting')
    parser.add_argument('--num_rots', type=int, default=72, help='Number of candidate center votes generated for a given point pair')
    parser.add_argument('--n_threads', type=int, default=512, help='Number of cupy threads')
    parser.add_argument('--optim', default=True,action='store_true', help='Whether to use bbox mask instead of instance segmentations')
    parser.add_argument('--bbox_mask', action='store_true', help='Whether to use bbox mask instead of instance segmentations')
    parser.add_argument('--adaptive_voting',default=True, action='store_true', help='Whether to use adaptive center voting')
    
    
    
    args = parser.parse_args()
    
    cp_device = args.cp_device
    device = torch.device("cuda")
    
    nepoch = 'best'
    cfg = omegaconf.OmegaConf.load(f'{args.ckpt_path}/.hydra/config.yaml')
    cls_name = cfg.category
    num_parts = cfg.num_parts
    angle_tol = args.angle_prec
    num_samples = int(4 * np.pi / (angle_tol / 180 * np.pi))
    sphere_pts = np.array(fibonacci_sphere(num_samples))

    num_rots = args.num_rots
    n_threads = args.n_threads
    bcelogits = torch.nn.BCEWithLogitsLoss()
    compute_iou_3d = compute_3d_iou_size()
    point_segs = []
    point_encoders = []
    ppf_decoders = []
        
    for i in range(num_parts):
        path = os.path.join(args.ckpt_path, str(i))
        ppf_decoder = PPFDecoder(ppffcs=[84, 32, 32, 16], out_dim2=2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins + 2 + 3, k=cfg.knn, spfcs=[32, 64, 32, 32], num_layers=1, out_dim1=32).cuda().eval()
        ppf_decoder.load_state_dict(torch.load(f'{path}/ppf_decoder_epoch{nepoch}.pth'))
        ppf_decoders.append(ppf_decoder)
    
    para_ppf = {}
    para_ppf['cfg'] = cfg  
    para_ppf['cp_device'] = cp_device
    para_ppf['num_rots'] = num_rots
    para_ppf['n_threads'] = n_threads
    para_ppf['adaptive_voting'] = args.adaptive_voting
    para_ppf['sphere_pts'] = sphere_pts
    para_ppf['angle_tol'] = angle_tol
    dataset = PPF_infer_Dataset(cfg=cfg,
                        mode='val',
                        data_root='/home/xxx/codes/dataset1',
                        num_pts=cfg.num_points,
                        num_cates=5,
                        num_parts=num_parts,
                        device='cpu',
                        )
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=True)

    k_fix = []
    test_count = 0

    sample_id = 0
    farthest_sampler = FarthestSampler()

    part_weight = cfg.part_weighted
    # ini_base_r_error_all = 0.
    # ini_sort_child_r_error_all = 0.
    # ini_base_t_error_all = 0.
    # ini_sort_child_t_error_all = 0.
    ini_r_error_lst = [_ for _ in range(num_parts)]
    ini_t_error_lst = [_ for _ in range(num_parts)]
    ini_r_error_all = [_ for _ in range(num_parts)]
    ini_t_error_all = [_ for _ in range(num_parts)]
    # optim_base_r_error_all = 0.
    # optim_sort_child_r_error_all = 0.
    # optim_base_t_error_all = 0.
    # optim_sort_child_t_error_all = 0.
    optim_r_error_lst = [_ for _ in range(num_parts)]
    optim_t_error_lst = [_ for _ in range(num_parts)]
    optim_r_error_all = [_ for _ in range(num_parts)]
    optim_t_error_all = [_ for _ in range(num_parts)]

    iou3d_all = [0. for _ in range(num_parts)]
    rt_key = [_ for _ in range(num_parts)]
    r_key = [_ for _ in range(num_parts)]
    t_key = [_ for _ in range(num_parts)]
    key_j_state_gt = torch.tensor(0.)
    turns = 0
    video_num = 0
    key_dis = 0
    gt_rt_list = []
    xi = [_ for _ in range(num_parts)]

    time1 = time.time()
    for i, data in enumerate(test_dataloader):  # batch_size=1， 每一个data仅仅包含一条数据，因此关键帧可以通过i来计算
        turns += 1
        # cloud 相机空间下
        clouds ,gt_part_r, gt_part_t, gt_part_rt = \
            data['clouds'],data['part_target_r'],data['part_target_t'],data['part_target_rt']
        homo_q = data['homo_q'][0].numpy()  # [numparts-1, 2, 4]  2表示NPCS导致的分裂
        color_path = data['color_path'][0]
        sort_part = data['sort_part'][0]
        canonical_pts = data['clouds_part_cano'][0].numpy()
        gt_size = data['fsnet_scale'].numpy()
        mean_shape_part = data['mean_shape_part'].numpy()
       

        para_ppf['cls_pts'] = data['cano_cloud']  # [1, 2048, 3] tensor

        clouds = clouds[0].numpy()
        assert len(canonical_pts) == len(clouds)
       
      
        camera_pts = copy.deepcopy(clouds)  # [2, 4]
        
       
        ori_gt_part_r = copy.deepcopy(gt_part_r[0].numpy())  # [2,3,3]
        ori_gt_part_t = copy.deepcopy(gt_part_t[0].numpy().reshape(num_parts,3)) # reshape将[2,3,1]转换为[2,3]
        ori_gt_part_rt = [compose_rt(ori_gt_part_r[p_idx], ori_gt_part_t[p_idx]) for p_idx in range(num_parts)]

      
        # angles = np.array([np.random.uniform(-5., 5.),
        #                 np.random.uniform(-5., 5.),
        #                 np.random.uniform(-5., 5.)])
        # base_fix_t = np.array([np.random.uniform(-0.1, 0.1),
        #                             np.random.uniform(-0.1, 0.1),
        #                             np.random.uniform(-0.1, 0.1)])
        angles = np.array([0,0,0])
        base_fix_t = np.array([0,0,0])
        base_fix_r = Rotation.from_euler('yxz', angles, degrees=True).as_matrix() #
        base_fix_rt = compose_rt(base_fix_r, base_fix_t)

        flag = 0
        inner_index = (i+1) % 15
        
        print(f'i: {i}')
        # print(f'inner_index: {inner_index}')
        pred_rt = None
        if inner_index == 0 or i==0:  
            k_fix = base_fix_rt
            flag = 1
            pre_errdis = np.inf
            update_num=0
            r_key = copy.deepcopy(gt_part_r[0].numpy())  # [2,3,3]  绝对量
            t_key = copy.deepcopy(gt_part_t[0].numpy())  # [2,1,3]
            abs_pose = copy.deepcopy(gt_part_rt[0].numpy())  # [2,4,4]          

            for part_idx in range(num_parts):
                rt_key[part_idx] = compose_rt(r_key[part_idx], t_key[part_idx])
                rt_key[part_idx] = rt_key[part_idx] @ k_fix
                xi[part_idx] = se3_log(rt_key[part_idx])  # 李代数向量

        key_inner_id = ((inner_index - 1) // 5) * 5 if flag == 0 else 0  
        key_dis = inner_index - key_inner_id
        # print('key_dis: ',key_dis)
        
        
        gt_part_r = gt_part_r[0].numpy()
        gt_part_t = gt_part_t[0].numpy()
        gt_part_rt = [_ for _ in range(num_parts)]

        
        for part_idx in range(num_parts):
            gt_part_rt[part_idx] = compose_rt(gt_part_r[part_idx], gt_part_t[part_idx])
            gt_part_rt[part_idx] = np.linalg.inv(rt_key[part_idx]) @ gt_part_rt[part_idx]  
            
            # gt_R = gt_part_r[part_idx]
            # gt_t = gt_part_t[part_idx]
            # key_R = rt_key[part_idx][:3,:3]
            # key_t = rt_key[part_idx][:3,3]
            # gt_RT = np.eye(4)
            # gt_RT[:3,:3] = np.linalg.inv(key_R) @ gt_R
            # gt_RT[:3,3] = gt_t - key_t
            # gt_part_rt[part_idx] = gt_RT

            gt_part_rt[part_idx] = base_fix_rt @ gt_part_rt[part_idx]  # 
            gt_part_r[part_idx] = gt_part_rt[part_idx][:3, :3]  #
            gt_part_t[part_idx] = gt_part_rt[part_idx][:3, 3]


            cloud_pcd = o3d.geometry.PointCloud()
            cloud_pcd.points = o3d.utility.Vector3dVector(clouds[part_idx])
            cloud_pcd.transform(np.linalg.inv(rt_key[part_idx]))  
            cloud_pcd.transform(base_fix_rt) # 噪声量
            clouds[part_idx] = np.asarray(cloud_pcd.points)   
          


        pred_rt_list = [_ for _ in range(num_parts)]
        optim_pred_rt_list = [_ for _ in range(num_parts)]
        pred_s_list = [0. for _ in range(num_parts)]
        d = np.inf
        d_turn = 0
        cloud = np.concatenate(clouds,axis=0)
      
        for idx in range(num_parts):
            pred_rt_list[idx], pred_s_list[idx] = tracking_func(clouds[idx], ppf_decoders[idx],para_ppf,10000)
        
        
        Pred_s = torch.mean(torch.stack(pred_s_list),axis=0).cpu().numpy()
        gt_size = gt_size + mean_shape_part # (1, 6)
        Pred_s =  Pred_s + mean_shape_part
        iou_3d = compute_iou_3d(gt_size, Pred_s, num_parts)
        for i in range(num_parts):
            iou3d_all[i] += iou_3d[i]
            
        update_k = False
        # part_weight = [4, 4, 2]  # dishwasher
        init_base_rt = np.array(optimize_rotation(pred_rt_list))  
        init_base_rt = torch.from_numpy(init_base_rt)
        init_base_r = init_base_rt[...,:3,:3]
        init_base_t = init_base_rt[...,:3,3]
        n_parts = len(init_base_r)
        pose_estimator = PoseEstimator(num_parts=n_parts, init_r=init_base_r, init_t=init_base_t,
                                            device=device, joint_axes=homo_q, rt_k=rt_key)
        
        xyz_camera = torch.from_numpy(copy.deepcopy(clouds)).cuda()
        cad = torch.from_numpy(copy.deepcopy(canonical_pts)).cuda()   # NPCS
    
        
        loss, optim_transforms = PPFDecoder.optimization(pose_estimator, xyz_camera, cad, part_weight, cfg.lim)
        
        optim_transforms = optim_transforms.cpu().numpy()
       
        optim_pred_r_list = [optim_transforms[i][:3,:3] for i in range(num_parts)]  
        optim_pred_t_list = [optim_transforms[i][:3,3] for i in range(num_parts)]  
        optim_pred_rt_list = optim_transforms         
        optim_errs = calErr(optim_pred_r_list, optim_pred_t_list, gt_part_r, gt_part_t, sort_part)   
            
        for i in range(num_parts):
            # Rotation ERR and Translation ERR
            r_ERR = optim_errs[0][i]
            t_ERR = optim_errs[1][i]
            optim_r_error_lst[i] = r_ERR
            optim_t_error_lst[i] = t_ERR

            optim_r_error_all[i] += r_ERR  
            optim_t_error_all[i] += t_ERR  
            # print(f'optim part{i}.  r_err: {r_ERR}, t_err: {t_ERR}')
        

        errdis = energy_function(camera_pts, easy_trans(canonical_pts,np.array(operation_at(rt_key,pred_rt_list,seperate=False))))
 
        update_num += 1
        if errdis < 10:
            update_k = True    

        if update_k or update_num==3:  
            k_fix = base_fix_rt
            # print("Updating KeyFrame............................")
            if not update_k: 
                print('absolute pose predicting...')      
                key_pred = [_ for _ in range(num_parts)]
                kRT = [_ for _ in range(num_parts)]
                for i in range(num_parts):
                    kRT[i] = rt_key[i] @ optim_pred_rt_list[i]
                kRT = np.array(ori_gt_part_rt)  # kRT
                init_base_r = torch.from_numpy(kRT[:,:3,:3])   
                init_base_t = torch.from_numpy(kRT[:,:3,3])   
                    
                n_parts = len(init_base_r)
                pose_estimator = PoseEstimator(num_parts=n_parts, init_r=init_base_r, init_t=init_base_t,
                                                device=device, joint_axes=homo_q, rt_k=torch.eye(4, device=device).repeat(n_parts, 1, 1))
                
                xyz_camera = torch.from_numpy(copy.deepcopy(camera_pts)).cuda()
                cad = torch.from_numpy(copy.deepcopy(canonical_pts)).cuda()   # canonical_pts
            
                ener, key_pred = PPFDecoder.optimization(pose_estimator, xyz_camera, cad, part_weight, cfg.lim)
                
                key_pred = key_pred.cpu().numpy()
                
                for idx in range(num_parts): 
                    rt_key[idx] = key_pred[idx]
                    r_key[idx] = rt_key[idx][:3, :3]
                    t_key[idx] = rt_key[idx][:3, 3]  
                    xi[idx] = se3_log(rt_key[idx]) 
            else:     
                pred_rt1 = optim_pred_rt_list if args.optim else pred_rt_list                 
                for part_idx in range(num_parts):      
                    xi_pred = se3_log(pred_rt1[part_idx]) 
                    xi[part_idx] += xi_pred
                    rt_key[part_idx] = se3_exp(xi[part_idx])
                    r_key[part_idx] = rt_key[part_idx][:3, :3]
                    t_key[part_idx] = rt_key[part_idx][:3, 3]       
            update_num = 0
            update_k = False

        
    print(f'turn:{turns}')
    for i in range(num_parts):
        print(f'optim error mean: part{i}.  r_err: {optim_r_error_all[i]/turns}, t_err: {optim_t_error_all[i]/turns}, 3D IOU: {100*iou3d_all[i]/turns}%')
        print('------------------------------------------------------------------------------------------')
    

    time2 = time.time()
    print((time2-time1)/turns)

