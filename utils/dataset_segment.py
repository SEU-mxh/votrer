import sys
sys.path.append('/home/xxx/codes/PPF_Tracker_release')
import math
from matplotlib import cm
import torch.utils.data as data
import os
import random

import os.path as osp
import torch
import numpy as np
import open3d as o3d
import copy
import xml.etree.ElementTree as ET
import mmcv
import cv2
from scipy.spatial.transform import Rotation
import pycocotools.mask as maskUtils

from scipy.spatial.transform import Rotation
import hydra
import MinkowskiEngine as ME
from utils.easy_utils import compose_rt,vis_cloud,save_pts,easy_inv_trans,easy_trans,sample_points,o3d_trans
from sklearn.neighbors import NearestNeighbors

def augment_rgb(rgb, bg_color):
    fg_mask = np.any(rgb != bg_color, -1)
    rgb[fg_mask] *= (1 + 0.4 * np.random.random(3) - 0.2) # brightness change for each channel
    rgb[fg_mask] += np.expand_dims((0.05 * np.random.random(rgb[fg_mask].shape[:-1]) - 0.025), -1) # jittering on each pixel
    rgb[fg_mask] = np.clip(rgb[fg_mask], 0, 1)
    return rgb

class SegAuxDataset(torch.utils.data.Dataset):
    CLASSES = ('background', 'laptop', 'eyeglasses', 'dishwasher', 'drawer', 'scissors')
    TEST_URDF_IDs = (10040, 10885, 11242, 11030, 11156) + (101300, 101859, 102586, 102590, 102596, 102612) + \
                    (11700, 12530, 12559, 12580) + (46123, 45841, 46440) + \
                    (10449, 10502, 10569, 10907)
    def __init__(self, cfg, mode, data_root,
                 num_pts, num_cates, num_parts,
                 device='cuda:0', data_tag='train',
                 use_background=False,debug=False):
        assert mode in ('train', 'val')
        self.data_root = data_root  # '/home/xxx/codes/a_datas/dataset1'
        self.mode = mode
        self.num_pts = num_pts  # self.num_points
        self.num_cates = num_cates
        self.num_parts = num_parts
        self.device = device
        self.data_tag = data_tag
        self.use_background=use_background
        cate_id = self.CLASSES.index(cfg.category)
        self.cate_id = cate_id   

        os.environ.update(
            OMP_NUM_THREADS = '1',
            OPENBLAS_NUM_THREADS = '1',
            NUMEXPR_NUM_THREADS = '1',
            MKL_NUM_THREADS = '1',
            PYOPENGL_PLATFORM = 'osmesa',
            PYOPENGL_FULL_LOGGING = '0'
        )
        self.cfg = cfg
               
        valid_flag_path = osp.join(self.data_root, self.CLASSES[cate_id], 'train.txt'
            if mode == 'train' else 'test.txt') # '/home/xxx/codes/a_datas/ArtImage/laptop/train.txt'
        self.annotation_valid_flags = dict()
        with open(valid_flag_path, 'r') as f:
            self.annotation_valid_flags[self.cate_id] = f.readlines()
        for idx in range(len(self.annotation_valid_flags[self.cate_id])):
            self.annotation_valid_flags[self.cate_id][idx] = self.annotation_valid_flags[self.cate_id][idx].split('\n')[0]

        self.obj_list = {}
        self.obj_name_list = {}

        intrinsics_path = osp.join(self.data_root, 'camera_intrinsic.json')
        self.camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsics_path)
        self.data_dir = osp.join(self.data_root, self.CLASSES[cate_id], self.data_tag)
        self.obj_annotation_list = []
        self.obj_urdf_id_list = []
        self.num_samples = 0
        # print(sorted(os.listdir(self.annotation_dir)))
        # pause = input()
        bad_data = ['00013','00026','00293']
        for video in sorted(os.listdir(self.data_dir)): # ArtImage/laptop/train
            # if video in bad_data:
            #     continue
            video_path = osp.join(self.data_dir, video) # '/home/xxx/codes/a_datas/ArtImage/laptop/train/00000'
            for file in sorted(os.listdir(osp.join(video_path, 'annotations'))):
                if '.json' in file and file in self.annotation_valid_flags[self.cate_id]:
                # if '.json' in file:
                    # print(file)
                    # pause = input()
                    anno_path = osp.join(video_path, 'annotations', file) # 具体的 00001.json
                    annotation = mmcv.load(osp.join(anno_path))
                    annotation['annotation_path'] = anno_path
                    # 相对路径转换为绝对路径
                    annotation['mask_path'] = osp.join(video_path,
                                                    annotation['depth_path'].replace('depth', 'category_mask'))
                    annotation['depth_path'] = osp.join(video_path,
                                                        annotation['depth_path'])
                    annotation['color_path'] = osp.join(video_path,
                                                        annotation['color_path'])
                    instances = annotation['instances']
                    assert len(instances) == 1, 'Only support one instance per image'
                    instance = instances[0]
                    urdf_id = instance['urdf_id']
                    if (self.mode == 'train' and urdf_id not in self.TEST_URDF_IDs) \
                            or (self.mode == 'val' and urdf_id in self.TEST_URDF_IDs):
                        self.obj_annotation_list.append(annotation)
                        self.obj_urdf_id_list.append(urdf_id)
                        self.num_samples += 1
        print('Finish loading {} annotations!'.format(self.num_samples))

        self.raw_part_obj_pts_dict = dict()
        self.raw_part_obj_pts_dict[self.cate_id] = dict()
        self.urdf_ids_dict = dict()
        self.urdf_ids_dict[self.cate_id] = []
        self.urdf_dir = osp.join(self.data_root, self.CLASSES[cate_id], 'urdf')
        self.rest_state_json = mmcv.load(osp.join(self.urdf_dir, 'rest_state.json'))
        self.urdf_rest_transformation_dict = dict()
        self.urdf_rest_transformation_dict[self.cate_id] = dict()
        self.raw_urdf_joint_loc_dict = dict()
        self.raw_urdf_joint_loc_dict[self.cate_id] = dict()
        self.bbox_dict = dict()
        self.bbox_dict[self.cate_id] = dict()
        self.raw_urdf_joint_axis_dict = dict()
        self.raw_urdf_joint_axis_dict[self.cate_id] = dict()
        self.all_norm_obj_joint_loc_dict = dict()
        self.all_norm_obj_joint_loc_dict[self.cate_id] = dict()  # (joint anchor - center) -> normalized joint anchor
        self.all_norm_obj_joint_axis_dict = dict()
        self.all_norm_obj_joint_axis_dict[self.cate_id] = dict()  # numpy array, the same as raw joint axis
        self.all_obj_raw_scale = dict()
        self.all_obj_raw_scale[self.cate_id] = dict()  # raw mesh scale(rest state)
        self.all_obj_raw_center = dict()  # raw mesh center(rest state)
        self.all_obj_raw_center[self.cate_id] = dict()  # raw mesh center(rest state)
        self.norm_part_obj_corners = dict()
        self.norm_part_obj_corners[self.cate_id] = dict()  # raw mesh corners(rest state), part-level
        self.all_raw_obj_pts_dict = dict()  # raw complete obj pts(rest state)
        self.all_raw_obj_pts_dict[self.cate_id] = dict()
        self.norm_part_obj_pts_dict = dict()  # zero centered complete obj pts(rest state)
        self.norm_part_obj_pts_dict[self.cate_id] = dict()
        for dir in sorted(os.listdir(self.urdf_dir)):
            if osp.isdir(osp.join(self.urdf_dir, dir)):
                urdf_id = int(dir)
                if (self.mode == 'train' and urdf_id not in self.TEST_URDF_IDs) \
                        or (self.mode == 'val' and urdf_id in self.TEST_URDF_IDs):
                    self.raw_part_obj_pts_dict[self.cate_id][urdf_id] = [None for _ in range(self.num_parts)]
                    if urdf_id not in self.urdf_ids_dict[self.cate_id]:
                        self.urdf_ids_dict[self.cate_id].append(urdf_id)
                    new_urdf_file = osp.join(self.urdf_dir, dir, 'mobility_for_unity_align.urdf')
                    # more flexible
                    self.bbox_dict[self.cate_id][urdf_id] = mmcv.load(osp.join(self.urdf_dir, dir, 'bounding_box.json'))

                    compute_relative = True if cate_id == 5 else False  # only applied for scissors
                    self.urdf_rest_transformation_dict[self.cate_id][urdf_id], \
                        self.raw_urdf_joint_loc_dict[self.cate_id][urdf_id], \
                        self.raw_urdf_joint_axis_dict[self.cate_id][urdf_id] = \
                        self.parse_joint_info(urdf_id, new_urdf_file, self.rest_state_json, compute_relative)
                    for file in sorted(os.listdir(osp.join(self.urdf_dir, dir, 'part_point_sample'))):
                        assert '.xyz' in file
                        if '.meta' in file:
                            continue
                        part_idx = int(file.split('.xyz')[0])
                        self.raw_part_obj_pts_dict[self.cate_id][urdf_id][part_idx] = np.asarray(o3d.io.read_point_cloud(
                            osp.join(self.urdf_dir, dir, 'part_point_sample', file), print_progress=False, format='xyz').points)
                        num_pts = self.raw_part_obj_pts_dict[self.cate_id][urdf_id][part_idx].shape[0]
                        # handle tree structure with depth > 2
                        if part_idx in self.urdf_rest_transformation_dict[self.cate_id][urdf_id]:
                            homo_obj_pts = np.concatenate([self.raw_part_obj_pts_dict[self.cate_id][urdf_id][part_idx], np.ones((num_pts, 1))], axis=1)
                            new_homo_obj_pts = (self.urdf_rest_transformation_dict[self.cate_id][urdf_id][part_idx] @ homo_obj_pts.T).T
                            self.raw_part_obj_pts_dict[self.cate_id][urdf_id][part_idx] = new_homo_obj_pts[:, :3]

                    self.all_raw_obj_pts_dict[self.cate_id][urdf_id] = np.concatenate(
                        self.raw_part_obj_pts_dict[self.cate_id][urdf_id], axis=0)
                    # print(list(self.raw_part_obj_pts_dict[self.cate_id].keys()))
                    # pause = input()

                    center, scale, _ = self.get_norm_factor(self.all_raw_obj_pts_dict[self.cate_id][urdf_id])
                    self.norm_part_obj_pts_dict[self.cate_id][urdf_id] = [
                        (self.raw_part_obj_pts_dict[self.cate_id][urdf_id][part_idx] - center[np.newaxis, :])
                        for part_idx in range(self.num_parts)]
                    self.norm_part_obj_corners[self.cate_id][urdf_id] = [None for _ in range(self.num_parts)]
                    for part_idx in range(self.num_parts):
                        _, _, self.norm_part_obj_corners[self.cate_id][urdf_id][part_idx] = \
                            self.get_norm_factor(self.norm_part_obj_pts_dict[self.cate_id][urdf_id][part_idx])
                    self.norm_part_obj_corners[self.cate_id][urdf_id] = np.stack(
                        self.norm_part_obj_corners[self.cate_id][urdf_id], axis=0)

                    self.all_obj_raw_center[self.cate_id][urdf_id] = center
                    self.all_obj_raw_scale[self.cate_id][urdf_id] = scale

                    self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id] = []
                    self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id] = []
                    for part_idx in range(self.num_parts):
                        if part_idx in self.raw_urdf_joint_loc_dict[self.cate_id][urdf_id]:
                            # 注意 joint_loc 减了一个 center 进行了中心化
                            self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id].append(
                                (self.raw_urdf_joint_loc_dict[self.cate_id][urdf_id][part_idx] - center))
                            self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id].append(
                                self.raw_urdf_joint_axis_dict[self.cate_id][urdf_id][part_idx]
                            )
                    self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id] = np.stack(
                        self.all_norm_obj_joint_loc_dict[self.cate_id][urdf_id], axis=0)
                    self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id] = np.stack(
                        self.all_norm_obj_joint_axis_dict[self.cate_id][urdf_id], axis=0)

        self.num_objs = len(self.raw_part_obj_pts_dict[self.cate_id])
        self.samples_per_obj = self.num_samples // self.num_objs
        print('Finish loading {} objects!'.format(self.num_objs))

        self.cam_cx, self.cam_cy = self.camera_intrinsic.get_principal_point()
        self.cam_fx, self.cam_fy = self.camera_intrinsic.get_focal_length()
        self.width = self.camera_intrinsic.width
        self.height = self.camera_intrinsic.height

        self.xmap = np.array([[j for _ in range(self.width)] for j in range(self.height)])
        self.ymap = np.array([[i for i in range(self.width)] for _ in range(self.height)])



    @staticmethod
    def load_depth(depth_path):
        depth = cv2.imread(depth_path, -1)

        if len(depth.shape) == 3:
            depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2])
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'

        return depth16

    def parse_joint_info(self, urdf_id, urdf_file, rest_state_json, compute_relative=False):
        # support kinematic tree depth > 2
        tree = ET.parse(urdf_file)
        root_urdf = tree.getroot()
        rest_transformation_dict = dict()
        joint_loc_dict = dict()
        joint_axis_dict = dict()
        for i, joint in enumerate(root_urdf.iter('joint')): # root_urdf.iter('joint')创建一个迭代器，遍历xml树的所有joint节点
            if joint.attrib['type'] == 'fixed' or joint.attrib['type'] == '0':
                continue
            child_name = joint.attrib['name'].split('_')[-1]
            for origin in joint.iter('origin'):
                x, y, z = [float(x) for x in origin.attrib['xyz'].split()][::-1]
                a, b, c = y, x, z  # 这里是原始的轴点位置，但数据集加载最后返回时进行了去中心化
                joint_loc_dict[int(child_name)] = np.array([a, b, c])
            for axis in joint.iter('axis'):
                r, p, y = [float(x) for x in axis.attrib['xyz'].split()][::-1]
                axis = np.array([p, r, y])
                axis /= np.linalg.norm(axis)
                u, v, w = axis
                joint_axis_dict[int(child_name)] = np.array([u, v, w])
            if joint.attrib['type'] == 'prismatic':
                delta_state = rest_state_json[str(urdf_id)][child_name]['state']
                delta_transform = np.concatenate([np.concatenate([np.eye(3), np.array([[u*delta_state, v*delta_state, w*delta_state]]).T],
                                                 axis=1), np.array([[0., 0., 0., 1.]])], axis=0)
            elif joint.attrib['type'] == 'revolute':
                if str(urdf_id) in rest_state_json:
                    delta_state = -rest_state_json[str(urdf_id)][child_name]['state'] / 180 * np.pi
                else:
                    delta_state = 0.
                cos = np.cos(delta_state)
                sin = np.sin(delta_state)

                delta_transform = np.concatenate(
                    [np.stack([u * u + (v * v + w * w) * cos, u * v * (1 - cos) - w * sin, u * w * (1 - cos) + v * sin,
                                  (a * (v * v + w * w) - u * (b * v + c * w)) * (1 - cos) + (b * w - c * v) * sin,
                                  u * v * (1 - cos) + w * sin, v * v + (u * u + w * w) * cos, v * w * (1 - cos) - u * sin,
                                  (b * (u * u + w * w) - v * (a * u + c * w)) * (1 - cos) + (c * u - a * w) * sin,
                                  u * w * (1 - cos) - v * sin, v * w * (1 - cos) + u * sin, w * w + (u * u + v * v) * cos,
                                  (c * (u * u + v * v) - w * (a * u + b * v)) * (1 - cos) + (a * v - b * u) * sin]).reshape(
                        3, 4),
                     np.array([[0., 0., 0., 1.]])], axis=0)
            rest_transformation_dict[int(child_name)] = delta_transform
        if not compute_relative:  # 出了剪刀  都是False
            return rest_transformation_dict, joint_loc_dict, joint_axis_dict
        else:
            # support structure with more than 1 depth
            urdf_dir = os.path.dirname(urdf_file)
            urdf_ins_old = self.get_urdf_mobility(urdf_dir, filename='mobility_for_unity.urdf')
            urdf_ins_new = self.get_urdf_mobility(urdf_dir, filename='mobility_for_unity_align.urdf')

            joint_old_rpy_base = [-urdf_ins_old['joint']['rpy'][0][0], urdf_ins_old['joint']['rpy'][0][2],
                                  -urdf_ins_old['joint']['rpy'][0][1]]
            joint_old_xyz_base = [-urdf_ins_old['joint']['xyz'][0][0], urdf_ins_old['joint']['xyz'][0][2],
                                  -urdf_ins_old['joint']['xyz'][0][1]]
            joint_new_rpy_base = [-urdf_ins_new['joint']['rpy'][0][0], urdf_ins_new['joint']['rpy'][0][2],
                                  -urdf_ins_new['joint']['rpy'][0][1]]
            joint_new_xyz_base = [-urdf_ins_new['joint']['xyz'][0][0], urdf_ins_new['joint']['xyz'][0][2],
                                  -urdf_ins_new['joint']['xyz'][0][1]]

            joint_rpy_relative = np.array(joint_new_rpy_base) - np.array(joint_old_rpy_base)
            joint_xyz_relative = np.array(joint_new_xyz_base) - np.array(joint_old_xyz_base)

            transformation_base_relative = self.compose_rt(
                Rotation.from_euler('ZXY', joint_rpy_relative.tolist()).as_matrix(), joint_xyz_relative)

            for child_name in rest_transformation_dict.keys():
                rest_transformation_dict[child_name] = transformation_base_relative @ rest_transformation_dict[
                    child_name]
            rest_transformation_dict[0] = transformation_base_relative

            for child_name in joint_loc_dict:
                # support kinematic tree depth > 2
                homo_joint_loc = np.concatenate([joint_loc_dict[child_name], np.ones(1)], axis=-1)
                joint_loc_dict[child_name] = (rest_transformation_dict[0] @ homo_joint_loc.T).T[:3]
                joint_axis_dict[child_name] = (rest_transformation_dict[0][:3, :3] @ joint_axis_dict[child_name].T).T

            return rest_transformation_dict, joint_loc_dict, joint_axis_dict

    @staticmethod
    def get_urdf_mobility(dir, filename='mobility_for_unity_align.urdf'):
        urdf_ins = {}
        tree_urdf = ET.parse(os.path.join(dir, filename))
        num_real_links = len(tree_urdf.findall('link'))
        root_urdf = tree_urdf.getroot()

        rpy_xyz = {}
        list_type = [None] * (num_real_links - 1)
        list_parent = [None] * (num_real_links - 1)
        list_child = [None] * (num_real_links - 1)
        list_xyz = [None] * (num_real_links - 1)
        list_rpy = [None] * (num_real_links - 1)
        list_axis = [None] * (num_real_links - 1)
        list_limit = [[0, 0]] * (num_real_links - 1)
        # here we still have to read the URDF file
        for joint in root_urdf.iter('joint'):
            joint_index = int(joint.attrib['name'].split('_')[1])
            list_type[joint_index] = joint.attrib['type']

            for parent in joint.iter('parent'):
                link_name = parent.attrib['link']
                if link_name == 'base':
                    link_index = 0
                else:
                    # link_index = int(link_name.split('_')[1]) + 1
                    link_index = int(link_name) + 1
                list_parent[joint_index] = link_index
            for child in joint.iter('child'):
                link_name = child.attrib['link']
                if link_name == 'base':
                    link_index = 0
                else:
                    # link_index = int(link_name.split('_')[1]) + 1
                    link_index = int(link_name) + 1
                list_child[joint_index] = link_index
            for origin in joint.iter('origin'):
                if 'xyz' in origin.attrib:
                    list_xyz[joint_index] = [float(x) for x in origin.attrib['xyz'].split()]
                else:
                    list_xyz[joint_index] = [0, 0, 0]
                if 'rpy' in origin.attrib:
                    list_rpy[joint_index] = [float(x) for x in origin.attrib['rpy'].split()]
                else:
                    list_rpy[joint_index] = [0, 0, 0]
            for axis in joint.iter('axis'):  # we must have
                list_axis[joint_index] = [float(x) for x in axis.attrib['xyz'].split()]
            for limit in joint.iter('limit'):
                list_limit[joint_index] = [float(limit.attrib['lower']), float(limit.attrib['upper'])]

        rpy_xyz['type'] = list_type
        rpy_xyz['parent'] = list_parent
        rpy_xyz['child'] = list_child
        rpy_xyz['xyz'] = list_xyz
        rpy_xyz['rpy'] = list_rpy
        rpy_xyz['axis'] = list_axis
        rpy_xyz['limit'] = list_limit

        urdf_ins['joint'] = rpy_xyz
        urdf_ins['num_links'] = num_real_links

        return urdf_ins
    @staticmethod
    def get_norm_factor(obj_pts):
        xmin, xmax = np.min(obj_pts[:, 0]), np.max(obj_pts[:, 0])
        ymin, ymax = np.min(obj_pts[:, 1]), np.max(obj_pts[:, 1])
        zmin, zmax = np.min(obj_pts[:, 2]), np.max(obj_pts[:, 2])

        x_scale = xmax - xmin
        y_scale = ymax - ymin
        z_scale = zmax - zmin

        center = np.array([(xmin + xmax)/2., (ymin + ymax)/2., (zmin + zmax)/2.])
        scale = np.array([x_scale, y_scale, z_scale])
        corners = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        return center, scale, corners
    @staticmethod
    def filter_points(point_cloud):
        # 假设 point_cloud 是你的点云数据，形状为 [n_points, n_dimensions]

        # 创建最近邻搜索对象
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(point_cloud)  # 设置 K 值为 10

        # 计算每个点的 K 个最近邻点的距离
        distances, indices = nbrs.kneighbors(point_cloud)

        # 取每个点到第 K 个最近邻点的距离（排除自身）
        knn_distances = distances[:, -1]

        # 计算统计量（均值和标准差）
        mean_knn_dist = np.mean(knn_distances)
        std_knn_dist = np.std(knn_distances)

        # 设置阈值，例如均值加 2 倍标准差
        threshold = mean_knn_dist + 2 * std_knn_dist

        # 筛选离群点的索引
        outlier_indices = np.where(knn_distances > threshold)[0]

        # 去除离群点
        filtered_point_cloud = np.delete(point_cloud, outlier_indices, axis=0)
        return filtered_point_cloud
    
    def __len__(self):
        return self.num_samples
    
    def backproject(self, depth, mask):
        sensor_height = self.resolution_y / self.resolution_x * self.camera_sensor_width
        u, v = np.meshgrid(np.arange(self.resolution_x), np.arange(self.resolution_y))
        
        u = u[mask]
        v = v[mask]
        
        x = (0.5 - u / self.resolution_x) * self.camera_sensor_width / self.camera_lens
        y = (0.5 - v / self.resolution_y) * sensor_height / self.camera_lens
        z = np.ones_like(x)
        
        norm = np.linalg.norm(np.stack([x, y, z], -1), axis=-1)
        
        u = (0.5 - x * self.camera_lens / self.camera_sensor_width) * self.resolution_x
        v = (0.5 - y * self.camera_lens / sensor_height) * self.resolution_y
        
        intrinsics_inv = np.linalg.inv(self.intrinsics)
        
        grid = np.stack([u, v], -1).T

        # shape: height * width
        # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
        # mesh_grid = np.reshape(mesh_grid, [2, -1])
        length = grid.shape[1]
        ones = np.ones([1, length])
        uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

        xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
        xyz = np.transpose(xyz) #[num_pixel, 3]

        z = depth[mask] / norm

        # print(np.amax(z), np.amin(z))
        pts = xyz * z[:, np.newaxis]/xyz[:, -1:]
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]

        return pts
    def get_frame(self, choose_frame_annotation):
        link_category_to_idx_map = [_ for _ in range(self.num_parts)]
        for idx in range(self.num_parts):
            link_category_id = choose_frame_annotation['instances'][0]['links'][idx]['link_category_id']
            link_category_to_idx_map[link_category_id] = idx

        # which part to sort
        try:  
            sort_part = choose_frame_annotation['instances'][0]['links'][choose_frame_annotation['sort_part']]['link_category_id']
        except:
            sort_part = 1

        raw_transform_matrix = [np.array(choose_frame_annotation['instances'][0]['links']
                                         [link_category_to_idx_map[i]]['transformation'])
                                         for i in range(self.num_parts)]
            

        rest_transform_matrix = [np.diag([1., 1., 1., 1.]) for _ in range(self.num_parts)]
        joint_state = [0. for _ in range(self.num_parts)]
        
        choosen_urdf_id = choose_frame_annotation['instances'][0]['urdf_id']
        # more flexible
        joint_type = 'prismatic' if self.cate_id == 4 else 'revolute'
        
        for link_idx, transform in self.urdf_rest_transformation_dict[self.cate_id][choosen_urdf_id].items():
            rest_transform_matrix[link_idx] = transform
            if 'state' in choose_frame_annotation['instances'][0]['links'][link_category_to_idx_map[link_idx]]:
                joint_state[link_idx] = \
                choose_frame_annotation['instances'][0]['links'][link_category_to_idx_map[link_idx]]['state']
        joint_state = np.array(joint_state)        
        rest_transform_matrix = np.array(rest_transform_matrix)
        

        all_center = copy.deepcopy(self.all_obj_raw_center[self.cate_id][choosen_urdf_id][np.newaxis, :])
        # new transform matrix for zero-centerd pts in rest state
        part_transform_matrix = [_ for _ in range(self.num_parts)]
        
        # X' = R_rest @ X_raw - Center(C)   转换到canonical空间，这段代码不要动，为什么这样处理和数据集本身的生成有关
        for part_idx in range(self.num_parts):
            # R' = R @ (R_rest)^-1   用于转换到canonical空间的旋转矩阵
            transform_matrix = raw_transform_matrix[part_idx] @ np.linalg.inv(rest_transform_matrix[part_idx])
            # transform_matrix = raw_transform_matrix[part_idx]
            # T' = T + R' @ C
            transform_matrix[:3, -1] = transform_matrix[:3, -1] + (transform_matrix[:3, :3] @ all_center.T)[:, 0]  
            part_transform_matrix[part_idx] = transform_matrix   # 这个是修正之后的变换矩阵

        
        part_target_r = copy.deepcopy(np.stack([part_transform_matrix[i][:3, :3] for i in range(self.num_parts)], axis=0))  # 修正后的变换矩阵，绝对量
        part_target_t = copy.deepcopy(np.stack([part_transform_matrix[i][:3, 3] for i in range(self.num_parts)], axis=0)[:, np.newaxis, :])

        part_target_rt = []
        for idx in range(self.num_parts):
            part_target_rt.append(compose_rt(part_target_r[idx], part_target_t[idx]))
        part_target_rt = np.array(part_target_rt)

        rgb = cv2.imread(choose_frame_annotation['color_path'])[:, :, ::-1]
        depth = np.array(self.load_depth(choose_frame_annotation['depth_path'])) / 1000.0   # [640,640]
        rgb[depth == 0] = np.full((3,), 255)
        rgb = (rgb/255.).astype(np.float32)
        label = np.full((rgb.shape[0], rgb.shape[1]), -100, np.int64)
        # 这个mask标注的是整个物体的，而不是part级别的
        part_mask = mmcv.imread(choose_frame_annotation['mask_path'])[:, :, 0]  # [640,640]  最后一维取[0]，在这相当于降维了

        if self.use_background:
            x1, y1, x2, y2 = choose_frame_annotation['instances'][0]['bbox']
            depth = depth[y1:y2, x1:x2]
            part_mask = part_mask[y1:y2, x1:x2]

        img_height, img_width = choose_frame_annotation['height'], choose_frame_annotation['width']  

        # multi_part
        cam_scale = 1.0
        clouds = [_ for _ in range(self.num_parts)]
    
        for i in range(self.num_parts):
            clouds[i] = np.zeros([self.num_pts, 3])
           
            # part_seg标注的mask是part级别的
            part_seg = choose_frame_annotation['instances'][0]['links'][link_category_to_idx_map[i]]['segmentation']

            if len(part_seg) == 0:
                continue

            rle = maskUtils.frPyObjects(part_seg, img_height, img_width)

            # maskUtils.decode(rle) 是 [img_height, img_width ,1] 的掩码矩阵
            # np.sum( , axis=2) 相当于消除了最后一个维度 变成了 [img_height, img_width]  clip限制了最大值为1
            part_label = np.sum(maskUtils.decode(rle), axis=2).clip(max=1).astype(np.uint8)
            part_depth = depth * part_label  # 取出这个part的深度图（恢复这个part的三维点云）
            choose0 = (part_depth != 0.0).nonzero()
            label[choose0] = i

            choose = (part_depth.flatten() != 0.0).nonzero()[0] # nonzero()用于提取出矩阵非零元素的行列，取索引[0]代表行
           
            if self.use_background:
                xmap_masked = self.xmap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = self.ymap[y1:y2, x1:x2].flatten()[choose][:, np.newaxis].astype(np.float32)
                
            else: # flatten()函数用于展平， newaxis将维度从[1024,] -> [1024,1]
                xmap_masked = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
               
            # xmap_masked 和 ymap_masked 是点云的像素坐标，depth_masked是深度值
            depth_masked = part_depth.flatten()[choose][:, np.newaxis].astype(np.float32)
            pt2 = depth_masked / cam_scale  # 点的顺序取决于pt2
            pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
            pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy

            clouds[i] = np.concatenate((pt0, pt1, pt2), axis=1)  # 相机空间 
            clouds[i] = self.filter_points(clouds[i])  # 去掉离得很远的噪声点
            clouds[i] = sample_points(clouds[i],self.num_pts)


            # part_mask = part_mask[y1:y2, x1:x2] 
        cloud = np.concatenate([clouds[i] for i in range(len(clouds))], axis=0)  # (num_pts, 3)    
        return rgb, label, cloud
        
    
            
    def __getitem__(self, index):    
        # torch.cuda.empty_cache()
        choose_urdf_id = self.obj_urdf_id_list[index]
        # norm_joint_loc = copy.deepcopy(self.all_norm_obj_joint_loc_dict[self.cate_id][choose_urdf_id])
        # norm_joint_axis = copy.deepcopy(self.all_norm_obj_joint_axis_dict[self.cate_id][choose_urdf_id])
        choose_frame_annotation = self.obj_annotation_list[index]
        color_path = choose_frame_annotation['color_path']
        depth_path = choose_frame_annotation['depth_path']
        rgb, label, pc = self.get_frame(choose_frame_annotation)
        return rgb, label,pc, color_path, depth_path
        
@hydra.main(config_path='/home/xxx/codes/PPF_Tracker_release/config', config_name='config')
def main(cfg):
    dataset = SegAuxDataset(cfg=cfg,
                        mode='val',
                        data_root='/home/xxx/codes/dataset1',
                        num_pts=224*224,
                        num_cates=5,
                        num_parts=2,
                        device='cpu',
                        debug = True
                        )
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=True)

    i = 0
    from tqdm import tqdm

    for i, data in enumerate(tqdm(data_loader)):
        d = data

if __name__ == '__main__':
    main()