import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["CXX"] = "/usr/bin/g++-9"
os.environ["CC"] = "/usr/bin/gcc-9"
import sys

sys.path.append('/home/mxh24/codes/PPF_Tracker_release')

import math
import time
import open3d as o3d
import torch
import numpy as np
import copy
import os.path as osp
import argparse
import mmcv
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
import tqdm
import omegaconf
from utils.dataset_inference import PPF_infer_Dataset
from utils.easy_utils import vis_cloud,easy_inv_trans,easy_trans,save_pts,compose_rt,rot_diff_degree,\
    tr_diff,RotateAnyAxis_np,update_kframe,operation_at,energy_function,generate_rd_transform
from utils.util import backproject, fibonacci_sphere, convert_layers, estimate_normals
from models.model import PPFEncoder, PointSeg, PoseEstimator
from models.voting import rot_voting_kernel, backvote_kernel, ppf_kernel
from nocs.inference import tracking_func, compute_3d_iou_size
# from utils import complex_utils
import sympy as sp

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
        if i in [0, sort_part]:
            r_err = rot_diff_degree(gt_r[i], pred_r[i])
            if r_err > 90:
                r_err = 180-r_err
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


def se3_log(T):
   
    R = T[:3, :3]
    t = T[:3, 3]
    theta = np.arccos((np.trace(R) - 1) / 2)
    if theta < 1e-6:
        omega = np.zeros(3)
        V_inv = np.eye(3)
    else:
        lnR = (theta / (2 * np.sin(theta))) * (R - R.T)
        omega = np.array([lnR[2,1], lnR[0,2], lnR[1,0]])
        A = np.sin(theta)/theta
        B = (1-np.cos(theta))/(theta**2)
        V_inv = np.eye(3) - 0.5*lnR + (1/(theta**2))*(1-A/(2*B)) * (lnR @ lnR)
    v = V_inv @ t
    return np.concatenate([omega, v])

def se3_exp(xi):
    
    omega = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(omega)
    if theta < 1e-6:
        R = np.eye(3)
        V = np.eye(3)
    else:
        wx, wy, wz = omega
        wx_mat = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
        R = sp.Matrix(np.eye(3)) + sp.sin(theta)/theta * sp.Matrix(wx_mat) + (1-sp.cos(theta))/(theta**2) * sp.Matrix(wx_mat) @ sp.Matrix(wx_mat)
        R = np.array(R).astype(np.float64)
        V = np.eye(3) + (1-np.cos(theta))/(theta**2)*wx_mat + (theta-np.sin(theta))/(theta**3)*wx_mat@wx_mat
    t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
#####################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument('--cp_device', type=int, default=0, help='GPU device number for custom voting algorithms')
    parser.add_argument('--ckpt_path', default='/home/mxh24/codes/PPF_Tracker_release/checkpoint5_hebing/dishwasher', help='Model checkpoint path')
    parser.add_argument('--num_parts', type=int, default=2, help='Angle precision in orientation voting')
    parser.add_argument('--angle_prec', type=float, default=1.5, help='Angle precision in orientation voting')
    parser.add_argument('--num_rots', type=int, default=72, help='Number of candidate center votes generated for a given point pair')
    parser.add_argument('--n_threads', type=int, default=512, help='Number of cupy threads')
    parser.add_argument('--optim', default=True,action='store_true', help='Whether to use bbox mask instead of instance segmentations')
    parser.add_argument('--bbox_mask', action='store_true', help='Whether to use bbox mask instead of instance segmentations')
    parser.add_argument('--adaptive_voting',default=True, action='store_true', help='Whether to use adaptive center voting')
    
    
    args = parser.parse_args()
    
    cp_device = args.cp_device
    device = torch.device("cuda")
    num_parts = args.num_parts
    
    nepoch = 'best'
    # cfg文件不影响 里面就centric不一样，在推理的时候用不上
    cfg = omegaconf.OmegaConf.load(f'{args.ckpt_path}/base/.hydra/config.yaml')
    cls_name = cfg.category
    angle_tol = args.angle_prec
    num_samples = int(4 * np.pi / (angle_tol / 180 * np.pi))
    sphere_pts = np.array(fibonacci_sphere(num_samples))

    num_rots = args.num_rots
    n_threads = args.n_threads
    bcelogits = torch.nn.BCEWithLogitsLoss()
    compute_iou_3d = compute_3d_iou_size()
    point_segs = []
    point_encoders = []
    ppf_encoders = []
    assert num_parts == 2
    sel_part = ['base','child'] # 这个地方需要设计一下当part不等于2的情况
    for each_part in sel_part:
        path = os.path.join(args.ckpt_path, each_part)
        point_seg = PointSeg().cuda().eval()
        # point_encoder = PointEncoder(k=cfg.knn, spfcs=[32, 64, 32, 32], num_layers=1, out_dim=32).cuda().eval()
        ppf_encoder = PPFEncoder(ppffcs=[84, 32, 32, 16], out_dim2=2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins + 2 + 3, k=cfg.knn, spfcs=[32, 64, 32, 32], num_layers=1, out_dim1=32).cuda().eval()
        # point_seg.load_state_dict(torch.load(f'{path}/point_seg_epoch{nepoch}.pth'))
        # point_encoder.load_state_dict(torch.load(f'{path}/point_encoder_epoch{nepoch}.pth'))
        ppf_encoder.load_state_dict(torch.load(f'{path}/ppf_encoder_epoch{nepoch}.pth'))
        point_segs.append(point_seg)
        # point_encoders.append(point_encoder)
        ppf_encoders.append(ppf_encoder)
    
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
                        data_root='/home/mxh24/codes/dataset1',
                        num_pts=cfg.num_points,
                        num_cates=5,
                        num_parts=args.num_parts,
                        device='cpu',
                        )
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=True)

    k_fix = []
    test_count = 0

    sample_id = 0
    farthest_sampler = FarthestSampler()


    ini_base_r_error_all = 0.
    ini_sort_child_r_error_all = 0.
    ini_base_t_error_all = 0.
    ini_sort_child_t_error_all = 0.
    ini_j_state_error_all = 0.


    optim_base_r_error_all = 0.
    optim_sort_child_r_error_all = 0.
    optim_base_t_error_all = 0.
    optim_sort_child_t_error_all = 0.
    optim_j_state_error_all = 0.


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
        gt_joint_state, homo_q = \
            data['joint_state'], data['homo_q']
        color_path = data['color_path'][0]
        sort_part = data['sort_part'][0]
        canonical_pts = data['clouds_part_cano'][0].numpy()
        # gt_size = data['fsnet_scale'].numpy()
        # mean_shape_part = data['mean_shape_part'].numpy()
        gt_size = data['targets_scale'].numpy()  # [1,6]

        clouds = clouds[0].numpy()
        assert len(canonical_pts) == len(clouds)
        # rd_tran = generate_rd_transform([-5,5],[-0.1,0.1],just_zero=True)
        # eng1 = energy_function(easy_trans(clouds,np.array([rd_tran,rd_tran])),clouds)
        # print(eng1)
        # vis_cloud(cloud_canonical=easy_trans(clouds,np.array([rd_tran,rd_tran])),cloud_trans=clouds)
      
        camera_pts = copy.deepcopy(clouds)  # [2, 4]
        homo_q = torch.squeeze(homo_q).numpy()
        # ori_gt_joint_loc = gt_norm_joint_loc[0][0].numpy()  # [3,]
        # ori_gt_joint_axis = gt_norm_joint_axis[0][0].numpy()  # [3,]
        ori_gt_state = copy.deepcopy(gt_joint_state[0][1].numpy()) # gt_joint_state.shape is [1,2] 第二个值存的才是夹角 

        ori_gt_part_r = copy.deepcopy(gt_part_r[0].numpy())  # [2,3,3]
        ori_gt_part_t = copy.deepcopy(gt_part_t[0].numpy().reshape(2,3)) # reshape将[2,3,1]转换为[2,3]
        ori_gt_part_rt = [compose_rt(ori_gt_part_r[p_idx], ori_gt_part_t[p_idx]) for p_idx in range(num_parts)]

        # canonical_pts = []
        # canonical_pts.append((np.linalg.inv(ori_gt_part_r[0]) @ (camera_pts[0] - ori_gt_part_t[0]).T).T)
        # canonical_pts.append((np.linalg.inv(ori_gt_part_r[1]) @ (camera_pts[1] - ori_gt_part_t[1]).T).T)
        # canonical_pts = np.array(canonical_pts)
        # 这个随机噪声的影响不大，不会造成part之间不满足运动学约束的情况
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
        inner_index = i % 29
        
        print(f'i: {i}')
        print(f'inner_index: {inner_index}')
        pred_rt = None
        if inner_index == 0:  # 起始帧的情况下，关键帧的变换矩阵直接读取
            k_fix = base_fix_rt
            flag = 1
            pre_errdis = np.inf
            update_num=0
            r_key = copy.deepcopy(gt_part_r[0].numpy())  # [2,3,3]  绝对量
            t_key = copy.deepcopy(gt_part_t[0].numpy())  # [2,1,3]
            # print('gt_joint_state', gt_joint_state)
            key_j_state_gt = ori_gt_state   # 取绝对量
            key_j_state = ori_gt_state    # key_j_state_gt 是GT，key_j_state是带有累计误差的

            for part_idx in range(num_parts):
                rt_key[part_idx] = compose_rt(r_key[part_idx], t_key[part_idx])
                rt_key[part_idx] = rt_key[part_idx] @ k_fix
                xi[part_idx] = se3_log(rt_key[part_idx])  # 李代数向量

        key_inner_id = ((inner_index - 1) // 5) * 5 if flag == 0 else 0  # 初始帧的肯定为0
        key_dis = inner_index - key_inner_id
        # print('key_dis: ',key_dis)
        
        
        gt_part_r = gt_part_r[0].numpy()
        gt_part_t = gt_part_t[0].numpy()
        gt_part_rt = [_ for _ in range(num_parts)]

        
        for part_idx in range(num_parts):
            gt_part_rt[part_idx] = compose_rt(gt_part_r[part_idx], gt_part_t[part_idx])
          
            gt_part_rt[part_idx] = np.linalg.inv(rt_key[part_idx]) @ gt_part_rt[part_idx]  # 这里的gt是用于变换到伪标准空间的
            # gt_part_rt[part_idx][:3,:3] = np.linalg.inv(rt_key[part_idx][:3,:3]) @ gt_part_rt[part_idx][:3,:3]
            # gt_part_rt[part_idx][:3,3] = gt_part_rt[part_idx][:3,3] - rt_key[part_idx][:3,3]
            # gt_part_rt[part_idx] = gt_part_rt[part_idx] @ np.linalg.inv(rt_key[part_idx])  # T2 @ T1^-1 这样取delta才能满足： ΔT @ key_cloud = current_cloud （相机空间下）

            gt_part_rt[part_idx] = base_fix_rt @ gt_part_rt[part_idx]  # base_fix_rt是随机生成的噪声矩阵
            gt_part_r[part_idx] = gt_part_rt[part_idx][:3, :3]  # 从这里开始 gt 变成了delta量
            gt_part_t[part_idx] = gt_part_rt[part_idx][:3, 3]

            if part_idx == sort_part:  # 如果是childpart，则跟新角度差，变成delta量
                gt_joint_state[0][part_idx] = copy.deepcopy(gt_joint_state[0][part_idx] - key_j_state_gt)
            else:
                gt_joint_state[0][part_idx] = torch.tensor(0.)  # 对于有多个part的物体，只计算了一个child_part
            
            cloud_pcd = o3d.geometry.PointCloud()
            cloud_pcd.points = o3d.utility.Vector3dVector(clouds[part_idx])
            cloud_pcd.transform(np.linalg.inv(rt_key[part_idx]))   # clouds是当前帧相机空间下的点云，cloud是伪标准空间下的点云
            cloud_pcd.transform(base_fix_rt) # 噪声量
            clouds[part_idx] = np.asarray(cloud_pcd.points)   # clouds被覆盖，变成了伪标准空间下的点云
            if part_idx not in [0, sort_part]:
                rt_key[part_idx] = rt_key[0]


        # vis_cloud(cloud_canonical=clouds)
        ##########################  PPF 获得预测值  ###################################
       
        pred_rt_list = [_ for _ in range(num_parts)]
        optim_pred_rt_list = [_ for _ in range(num_parts)]
        pred_s = [0. for _ in range(num_parts)]
        d = np.inf
        d_turn = 0
        cloud = np.concatenate(clouds,axis=0)
        # for idx in range(num_parts):
        #     pred_rt_list[idx] = tracking_func(clouds[idx], point_segs[idx], ppf_encoders[idx], para_ppf, 10000)
            
        
        while d > 4:
            for idx in range(num_parts):
                pred_rt_list[idx] = tracking_func(clouds[idx], point_segs[idx], ppf_encoders[idx],para_ppf,10000)
            d = energy_function(easy_trans(canonical_pts,pred_rt_list),clouds)
            d_turn += 1
            if d_turn>3:
                break
        
        ###################################################################

        pred_r_list = np.array([pred_rt_list[0][:3,:3], pred_rt_list[1][:3,:3]])
        pred_t_list = np.array([pred_rt_list[0][:3,3],  pred_rt_list[1][:3,3]])      
        part_weight = [4, 4, 2]
        
                        
        init_base_r = torch.from_numpy(pred_r_list)
        init_base_t = torch.from_numpy(pred_t_list)
        n_parts = len(pred_r_list)
        pose_estimator = PoseEstimator(num_parts=n_parts, init_r=init_base_r, init_t=init_base_t,
                                            device=device, joint_axes=homo_q)
        
        xyz_camera = torch.from_numpy(copy.deepcopy(clouds)).cuda()
        cad = torch.from_numpy(copy.deepcopy(canonical_pts)).cuda()   # canonical_pts
    
        
        loss, optim_transforms = PPFEncoder.optimization(pose_estimator, xyz_camera, cad, part_weight)
        # print("loss:{}".format(loss.item()))
        optim_transforms = optim_transforms.cpu().numpy()
        optim_pred_r_list = [optim_transforms[0][:3,:3], optim_transforms[1][:3,:3]]
        optim_pred_t_list = [optim_transforms[0][:3,3], optim_transforms[1][:3,3]]
        optim_pred_rt_list = optim_transforms         
        optim_errs = calErr(optim_pred_r_list, optim_pred_t_list, gt_part_r, gt_part_t, sort_part)   # gt_part_t [2,1,3]  gt_part_r [2,3,3]
            
        optim_base_r_err = optim_errs[0][0]
        optim_child_r_err = optim_errs[0][1]
        optim_base_t_err = optim_errs[1][0]   
        optim_child_t_err = optim_errs[1][1]

                
        optim_base_r_error_all += optim_base_r_err
        optim_sort_child_r_error_all += optim_child_r_err
        optim_base_t_error_all += optim_base_t_err
        optim_sort_child_t_error_all += optim_child_t_err
        # 感觉的err计算有问题，可视化出来发现optim之后的没什么误差
        print(f'base r_err: {optim_base_r_err}')
        print(f'child r_err: {optim_child_r_err}')
        print(f'base t_err: {optim_base_t_err}')
        print(f'child t_err: {optim_child_t_err}')
        
        print()


        errdis = energy_function(camera_pts, easy_trans(canonical_pts,np.array(operation_at(rt_key,pred_rt_list,seperate=False))))
        update_k = False
        update_num += 1
        if errdis < 3:
            update_k = True    

        if update_k or update_num==6:  
            # print("更新关键帧, update_num=",update_num)          
            # pred_rt1 = pred_rt_list
            k_fix = base_fix_rt
            
            if update_num == 6:
                ener = np.inf
                best_ener = np.inf
                key_temp = [_ for _ in range(num_parts)]
                key_pred = [_ for _ in range(num_parts)]
                ener_num = 0   
                print("Updating KeyFrame............................")
                while ener > 1. and ener_num < 3:            
                    for idx in range(num_parts):
                        key_temp[idx] = tracking_func(camera_pts[idx], point_segs[idx], ppf_encoders[idx], para_ppf, 10000)
                    init_base_r = torch.from_numpy(np.array([key_temp[0][:3,:3], key_temp[1][:3,:3]]))
                    init_base_t = torch.from_numpy(np.array([key_temp[0][:3,3], key_temp[1][:3,3]]))
                    
                    n_parts = len(init_base_r)
                    pose_estimator = PoseEstimator(num_parts=n_parts, init_r=init_base_r, init_t=init_base_t,
                                                    device=device, joint_axes=homo_q)
                    
                    xyz_camera = torch.from_numpy(copy.deepcopy(camera_pts)).cuda()
                    cad = torch.from_numpy(copy.deepcopy(canonical_pts)).cuda()   # canonical_pts
                
                    ener, key_pred_0 = PPFEncoder.optimization(pose_estimator, xyz_camera, cad, part_weight, True)
                    if ener < best_ener:
                        best_ener = ener
                        key_pred = key_pred_0
                    ener_num += 1

                key_pred = key_pred.cpu().numpy()
                for idx in range(num_parts):
                    rt_key[idx] = key_pred[idx]
                    r_key[idx] = rt_key[idx][:3, :3]
                    t_key[idx] = rt_key[idx][:3, 3]  
                    xi[idx] = se3_log(rt_key[idx]) 
            else:     
                pred_rt1 = optim_pred_rt_list if args.optim else pred_rt_list                 
                for part_idx in range(num_parts):      
                    # rt_key[part_idx] = rt_key[part_idx] @ pred_rt1[part_idx]
                    # # rt_key[part_idx] = ori_gt_part_rt[part_idx]
                    # r_key[part_idx] = rt_key[part_idx][:3, :3]
                    # t_key[part_idx] = rt_key[part_idx][:3, 3] 
                    
                    # 1.李代数转化
                    xi_pred = se3_log(pred_rt1[part_idx]) 
                    # 2.李代数相加
                    xi[part_idx] += xi_pred
                    # 3.指数映射回SE3
                    rt_key[part_idx] = se3_exp(xi[part_idx])
                    r_key[part_idx] = rt_key[part_idx][:3, :3]
                    t_key[part_idx] = rt_key[part_idx][:3, 3]  
            # rt_key[part_idx] = ori_gt_part_rt[part_idx]
            # r_key[part_idx] = rt_key[part_idx][:3, :3]
            # t_key[part_idx] = rt_key[part_idx][:3, 3]         
            update_num=0

        if turns == 50:          
            print(f'turn:{turns}')
      
            print(f"optim base r error mean:{optim_base_r_error_all/turns}")
            print(f"optim child r error mean:{optim_sort_child_r_error_all/turns}")
            print(f"optim base t error mean:{optim_base_t_error_all/turns}")
            print(f"optim child t error mean:{optim_sort_child_t_error_all/turns}")
            print()
            time2 = time.time()
            print((time2-time1)/turns)
            exit(1)

