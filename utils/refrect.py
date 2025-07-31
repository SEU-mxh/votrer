import open3d as o3d
import torch.utils.data
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from easy_utils import sample_points
import sympy as sp
from util import rotation_matrix_log,rotation_matrix_exp  
def optimize_rotation(RT):
    if isinstance(RT, list):
        RT = np.array(RT)
    RT = torch.from_numpy(RT)
    single = False
    if RT.ndim == 2:
        RT = RT.unsqueeze(0)
        single = True

    R = RT[:, :3, :3]
    T = RT[:, :3, 3]

    logR = rotation_matrix_log(R)     # [n, 3, 3]
    R_new = rotation_matrix_exp(logR)  # [n, 3, 3]

    RT_new = torch.eye(4, device=RT.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
    RT_new[:, :3, :3] = R_new
    RT_new[:, :3, 3] = T

    return RT_new.squeeze(0) if single else RT_new

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