import open3d as o3d
import numpy as np

from scipy.spatial.distance import directed_hausdorff
import os.path as osp
import copy
from scipy.spatial import distance
from scipy.spatial.transform import Rotation


def update_kframe(frame1, frame2): 
    assert frame1.shape[0] == frame2.shape[0]
    frame1 = np.concatenate(frame1, axis=0)
    frame2 = np.concatenate(frame2, axis=0)
    return directed_hausdorff(frame1, frame2)[0] / frame1.shape[0]
def estimate_normals(pc, knn=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    # 较小的 knn 值对噪声更敏感，但能更好地捕捉局部细节；较大的 knn 值对噪声更鲁棒，但可能会丢失局部细节。
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.array(pcd.normals)
def energy_function(point_cloud1:np.ndarray,point_cloud2:np.ndarray):
    if point_cloud1.ndim == 3:
        point_cloud1 = np.concatenate(point_cloud1, axis=0)
        point_cloud2 = np.concatenate(point_cloud2, axis=0)
    assert point_cloud1.shape[0] == point_cloud2.shape[0]
    # c_mask = sample_points(point_cloud1,64,True)
    # point_cloud1 = point_cloud1[c_mask]
    # point_cloud2 = point_cloud2[c_mask]
    distances = np.array([distance.euclidean(p1, p2) for p1, p2 in zip(point_cloud1, point_cloud2)]) 
    normals1 = estimate_normals(point_cloud1)
    normals2 = estimate_normals(point_cloud2)
    dot_products = np.sum(normals1 * normals2, axis=1)
    norms1 = np.linalg.norm(normals1, axis=1)
    norms2 = np.linalg.norm(normals2, axis=1)
    cos_theta = dot_products / (norms1 * norms2)
    theta = np.arccos(cos_theta)
    mean_theta = np.mean(theta)
    theta_rad = np.deg2rad(mean_theta)
    
    # 10°，0.1m的误差下，距离能量和角度能量分别为1，0.008
    # print('旋转能量:{} ; 平移能量:{}'.format(theta_rad, np.sum(distances) / point_cloud1.shape[0]))
    return 300 * theta_rad + np.sum(distances) / point_cloud1.shape[0]

def generate_rd_transform(ang_range:list,t_range:list,just_zero=False):
    """
    Rotation.from_euler()函数
    利用欧拉角生成旋转矩阵
    如果 angles 是 [30, 45, 60]，那么这段代码将生成一个旋转矩阵，
    该矩阵表示首先绕 y 轴旋转 30 度，然后绕 x 轴旋转 45 度，最后绕 z 轴旋转 60 度的组合旋转。
    """
    a_low = ang_range[0]
    a_high = ang_range[1]
    t_low = t_range[0]
    t_high = t_range[1]

    if just_zero:
        angles = np.array([0,0,0])
        base_fix_t = np.array([0.,0.,0.])
    else:
        angles = np.array([np.random.uniform(a_low, a_high),
                           np.random.uniform(a_low, a_high),
                           np.random.uniform(a_low, a_high)])
        base_fix_t = np.array([np.random.uniform(t_low, t_high),
                               np.random.uniform(t_low, t_high),
                               np.random.uniform(t_low, t_high)])
    base_fix_r = Rotation.from_euler('yxz', angles, degrees=True).as_matrix() # as_matrix用于将旋转对象转换为旋转矩阵
    base_fix_rt = compose_rt(base_fix_r, base_fix_t)
    return base_fix_rt, base_fix_r, base_fix_t

def vis_cloud(describe:str=None, key_dis=None, p1=None,p2=None, ax=None, cloud_trans=None, cloud_canonical=None, cloud_camera=None, RT=None):
    """
    标准空间  红色；
    相机空间  蓝色；
    变换空间  紫色；
    """
    if describe:
        print(describe)

    vis_lst = []
    # 创建一个球体来表示点，并设置其位置和颜色
    if p1 is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(p1)
        sphere.paint_uniform_color([0, 0, 0])
        vis_lst.append(sphere)

    # 可视化轴
    if p1 is not None and (ax is not None or p2 is not None):
        line_pcd = o3d.geometry.LineSet()
        if p2 is None:
            p2 = p1 + ax
        p1 = np.array(p1).reshape(3)
        p2 = np.array(p2).reshape(3)
        lines = np.array([[0, 1]])  # 线段的索引
        points = np.vstack([p1, p2])  # 线段的端点
        colors = [[0, 0, 1] for i in range(len(lines))]
        line_pcd.points = o3d.utility.Vector3dVector(points)
        line_pcd.lines = o3d.utility.Vector2iVector(lines)
        line_pcd.colors = o3d.utility.Vector3dVector(colors)
        vis_lst.append(line_pcd)

    if cloud_canonical is not None:
        # 可视化点云
        if cloud_canonical.shape[0] == 2:
            cloud_canonical = np.concatenate(cloud_canonical,axis=0)
        point_pcd = o3d.geometry.PointCloud()
        point_pcd.points = o3d.utility.Vector3dVector(cloud_canonical)
        point_pcd.paint_uniform_color([1, 0, 0])   # 标准空间 红色
        # point_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        vis_lst.append(point_pcd)
    if cloud_camera is not None:
        # 可视化点云
        if cloud_camera.shape[0] == 2:
            cloud_camera = np.concatenate(cloud_camera,axis=0)
        point_pcd = o3d.geometry.PointCloud()
        point_pcd.points = o3d.utility.Vector3dVector(cloud_camera)
        point_pcd.paint_uniform_color([0, 0, 1])   # 相机空间 蓝色
        # point_pcd.paint_uniform_color([180/255.0, 211/255.0, 231/255.0]) 
        # point_pcd.paint_uniform_color([0.5, 0.5, 0.5])  
        vis_lst.append(point_pcd)
    if cloud_trans is not None:
        # 可视化点云
        if cloud_trans.shape[0] == 2:
            cloud_trans = np.concatenate(cloud_trans,axis=0)
        point_pcd = o3d.geometry.PointCloud()
        point_pcd.points = o3d.utility.Vector3dVector(cloud_trans)
        point_pcd.paint_uniform_color([1, 0, 1])   # 紫色
        # point_pcd.paint_uniform_color([206/255.0, 228/255.0, 181/255.0])   # 紫色
        # point_pcd.paint_uniform_color([0.5, 0.5, 0.5])  
        vis_lst.append(point_pcd)

    # 可视化坐标轴
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3,origin=[0.0, 0.0, 0.0])
    vis_lst.append(axis)
    # if RT is not None:
    #     for i in range(len(RT)):
    #         axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    #         axis.transform(RT[i])
    #         vis_lst.append(axis)
    

    # o3d.visualization.draw_geometries(vis_lst)
    # 创建一个 Visualizer 对象
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D', width=800, height=600)  # 设置窗口名称和大小

    # 添加点云到可视化器
    for v in vis_lst:
        vis.add_geometry(v)
    # 运行可视化器
    vis.run()
    # 关闭可视化器
    vis.destroy_window()
    if key_dis:
        print("key_dis={}  当前帧点云可视化结束！！！".format(key_dis))

def o3d_inv_trans(o3d_cloud, transform):
    cloud_pcd = o3d.geometry.PointCloud()  
    cloud_pcd.points = o3d.utility.Vector3dVector(o3d_cloud)
    cloud_pcd.transform(np.linalg.inv(transform))  
    o3d_cloud = np.asarray(cloud_pcd.points) 
    return o3d_cloud

def o3d_trans(o3d_cloud, transform):
    cloud_pcd = o3d.geometry.PointCloud()  
    cloud_pcd.points = o3d.utility.Vector3dVector(o3d_cloud)
    cloud_pcd.transform(transform)
    o3d_cloud = np.asarray(cloud_pcd.points) 
    return o3d_cloud

def easy_inv_trans(pts:np.ndarray, transform=None, rot=None, tran=None):
    
    assert pts.shape[-1] == 3
    if pts.ndim == 2:
        if transform is not None:
            assert isinstance(transform, np.ndarray)
            rot = transform[:3,:3]
            tran = transform[:3,3]
        return (np.linalg.inv(rot) @ (pts - tran.reshape(1,3)).T).T
    elif pts.ndim == 3:
        if transform is not None:
            if not isinstance(transform, np.ndarray):
                transform = np.array(transform)
            rot = transform[:,:3,:3]
            tran = transform[:,:3,3]
        trans_pts = []
        for i in range(pts.shape[0]):
            trans_pts.append((np.linalg.inv(rot[i]) @ (pts[i] - tran[i].reshape(1,3)).T).T)
        trans_pts = np.array(trans_pts)
        return trans_pts
    else:
        raise ValueError("pts shape error!!!")
    
def easy_trans(pts:np.ndarray, transform=None, rot=None, tran=None):  # 输入要求都是np.ndarray
    assert pts.shape[-1] == 3
    if pts.ndim == 2:
        if transform is not None:
            assert isinstance(transform, np.ndarray)
            rot = transform[:3,:3]
            tran = transform[:3,3]
        return (rot @ pts.T).T + tran.reshape(3)
    elif pts.ndim == 3:
        if transform is not None:
            if not isinstance(transform, np.ndarray):
                transform = np.array(transform)
            rot = transform[:,:3,:3]
            tran = transform[:,:3,3]
        trans_pts = []
        for i in range(pts.shape[0]):
            trans_pts.append((rot[i] @ pts[i].T).T + tran[i].reshape(3))
        trans_pts = np.array(trans_pts)
        return trans_pts
    else:
        raise ValueError("pts shape error!!!")
    
def operation_at(matrix1,matrix2,seperate=False):
    """
    matrix1 * matrix2
    """
    m = []
    if isinstance(matrix1, list) and isinstance(matrix2, list):
        for i in range(len(matrix1)):
            if seperate:
                mr = matrix1[i][:3,:3] @ matrix2[i][:3,:3]
                mt = matrix1[i][:3,3] + matrix2[i][:3,3] 
                m.append(compose_rt(mr,mt))
            else:
                m.append(matrix1[i] @ matrix2[i])
        return m
    
    if isinstance(matrix1, np.ndarray) and isinstance(matrix2, np.ndarray):
        for i in range(matrix1.shape[0]):
            if seperate:
                mr = matrix1[i][:3,:3] @ matrix2[i][:3,:3]
                mt = matrix1[i][:3,3] + matrix2[i][:3,3] 
                m.append(compose_rt(mr,mt))
            else:
                m.append(matrix1[i] @ matrix2[i])
        return np.array(m)
    
def RotateAnyAxis_np(v1, v2, step):   
    # 实际上只需要axis（旋转轴）和step（旋转角度）就可以计算出旋转矩阵，这里轴点的位置是将旋转的原点平移到了v1
    axis = v2 - v1  # v1是旋转轴的起点，v2是旋转轴的终点
    axis = axis / np.linalg.norm(axis)

    a, b, c = v1[0], v1[1], v1[2]    
    u, v, w = axis[0], axis[1], axis[2]  # 轴长度单位化为1

    cos = np.cos(-step)  # step是旋转的角度Θ
    sin = np.sin(-step)

    # 返回的是4*4的矩阵，包含了平移项，确保了旋转发生在v1和v2定义的轴上，而不是绕原点的旋转
    rot = np.concatenate([np.stack([u*u+(v*v+w*w)*cos, u*v*(1-cos)-w*sin, u*w*(1-cos)+v*sin,
                                                   (a*(v*v+w*w)-u*(b*v+c*w))*(1-cos)+(b*w-c*v)*sin,
                                                   u*v*(1-cos)+w*sin, v*v+(u*u+w*w)*cos, v*w*(1-cos)-u*sin,
                                                   (b*(u*u+w*w)-v*(a*u+c*w))*(1-cos)+(c*u-a*w)*sin,
                                                   u*w*(1-cos)-v*sin, v*w*(1-cos)+u*sin, w*w+(u*u+v*v)*cos,
                                                   (c*(u*u+v*v)-w*(a*u+b*v))*(1-cos)+(a*v-b*u)*sin]).reshape(3, 4),
                                                   np.array([[0., 0., 0., 1.]])], axis=0)

    return rot

def save_pts(pc0:np.ndarray, path=None):
    """
    用这个函数进行可视化，可以打出512个点，不容易卡住
    """
    pc = copy.deepcopy(pc0)
    if pc.ndim > 2:
        pc = np.concatenate(pc,axis=0)
    num_point = 512
    c_mask = np.random.choice(np.arange(len(pc)), num_point, replace=True)
    pc = pc[c_mask]
    if path == None:
        np.savetxt('/home/xxx/codes/PPF_Tracker_release/pc_out.txt',pc)
    else:
        np.savetxt(osp.join(path,'pc_out.txt'),pc)
        
def compose_rt(rotation, translation):
    aligned_RT = np.zeros((4, 4), dtype=np.float32)
    aligned_RT[:3, :3] = rotation[:3, :3]
    aligned_RT[:3, 3] = translation
    aligned_RT[3, 3] = 1
    return aligned_RT
def rot_diff_rad(rot1, rot2):
    if np.abs((np.trace(np.matmul(rot1, rot2.T)) - 1) / 2) > 1.:
        print('Something wrong in rotation error!')
        return 0.
    return np.arccos((np.trace(np.matmul(rot1, rot2.T)) - 1) / 2) % (2*np.pi)
def rot_diff_degree(rot1, rot2):
    return rot_diff_rad(rot1, rot2) / np.pi * 180
def tr_diff(tr1, tr2):
    return np.linalg.norm(tr1 - tr2)

def sample_points(points,num_pts=1024,get_mask=False):
    
    c_len = len(points)
    if c_len < num_pts:
        raise ValueError(f"Not enough points to sample. Available: {c_len}, Required: {num_pts}")
   
    c_mask = np.random.choice(np.arange(c_len), num_pts, replace=False)
    if get_mask:
        return c_mask

    return points[c_mask]
