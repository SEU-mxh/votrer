import os
import os.path as osp
import numpy as np

data_root = '/home/xxx/codes/dataset1'
category = 'dishwasher'
urdf_path = osp.join(data_root,category,'urdf')
if category == 'laptop':
    name = 'part_point_sample_rest'
else:
    name = 'part_point_sample'
urdf_ids = os.listdir(urdf_path)
# print(urdf_ids)
urdf_xyzs = {}
urdf_centers = {}
for urdf_id in urdf_ids:
    if urdf_id.endswith('.json') or urdf_id.endswith('.meta'):
        continue
    rest_dir = osp.join(urdf_path,urdf_id,name) 
    rest_xyz = os.listdir(rest_dir)
    xyzs = []
    centers = []
    for xyz in rest_xyz:
        if not xyz.endswith('.xyz'):
            continue
        xyz_path = osp.join(rest_dir, xyz)
        pc = np.loadtxt(xyz_path)
        xyzs.append(pc)
        centers.append((np.min(pc,axis=0)+np.max(pc,axis=0))/2)
    urdf_xyzs[urdf_id] = np.concatenate(xyzs,axis=0)
    urdf_centers[urdf_id] = centers

print(urdf_centers)
print()