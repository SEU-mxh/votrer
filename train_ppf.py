import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["CXX"] = "/usr/bin/g++-9"
os.environ["CC"] = "/usr/bin/gcc-9"
import sys

from glob import glob

import hydra
import torch
from models.model import PPFEncoder, PointSeg

from utils.dataset_train import PM_Dataset,get_PPF_GT

import numpy as np
import logging
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from utils.util import AverageMeter, typename2shapenetid
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter 

def estimate_normals(pc, knn):
    pc = pc.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    return np.array(pcd.normals,dtype=np.float32)
#   unset LD_LIBRARY_PATH
@hydra.main(config_path='./config', config_name='config')
def main(cfg):
    logger = logging.getLogger(__name__)
           
    # ds = PPF_dataset(cfg)
    ds = PM_Dataset(cfg=cfg,
                        mode='train',
                        data_root='/home/mxh24/codes/dataset1',
                        num_pts=cfg.num_points,
                        num_cates=5,
                        num_parts=2,
                        device='cpu',
                        use_p1_aug=False,
                        rescaled=False,
                       )
    
    df = torch.utils.data.DataLoader(ds, pin_memory=True, batch_size=cfg.batch_size, shuffle=True, num_workers=8)
    assert cfg.batch_size == 1
    point_seg = PointSeg().cuda()
  
    ppf_encoder = PPFEncoder(ppffcs=[84, 32, 32, 16], out_dim2=2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins + 2 + 3,k=cfg.knn, spfcs=[32, 64, 32, 32], num_layers=1, out_dim1=32).cuda()
    
    opt = optim.Adam([*ppf_encoder.parameters()], lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epoch, eta_min=1e-3)
    writer = SummaryWriter(log_dir=os.path.join(cfg.root_dir,cfg.ckpdir,cfg.category,cfg.centric)) 
    total_iter = 0
    kldiv = nn.KLDivLoss(reduction='batchmean')
    bcelogits = nn.BCEWithLogitsLoss()
    get_size_loss = nn.L1Loss()
    # crologits = nn.CrossEntropyLoss()
    logger.info('Train')
    best_loss = np.inf
    for epoch in range(cfg.max_epoch):
        n = 0
        writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch)

        loss_meter = AverageMeter()
        loss_tr_meter = AverageMeter()
        loss_up_meter = AverageMeter()
        loss_up_aux_meter = AverageMeter()
        loss_right_meter = AverageMeter()
        loss_right_aux_meter = AverageMeter()
        loss_scale_meter = AverageMeter()
        loss_seg_meter = AverageMeter()
        point_seg.train()
        # point_encoder.train()
        ppf_encoder.train()
        with tqdm(df) as t:
            for d_dict in t:
                pcs             = d_dict['pc'].cuda()
                # camera_pts      = d_dict['camera_pts'].cuda()
                pc_normals      = d_dict['normals'].cuda() 
                targets_tr      = d_dict['targets_tr'].cuda()
                targets_rot     = d_dict['targets_rot'].cuda()
                targets_rot_aux = d_dict['targets_rot_aux'].cuda()
                targets_scale   = d_dict['targets_scale'].cuda()
                point_idxs      = d_dict['point_idxs'].cuda()
                gt_labels       = d_dict['label'].cuda()

                opt.zero_grad()

               
                # dense_part_cls_score = point_seg(camera_pts, None).contiguous()


                with torch.no_grad():
                    dist = torch.cdist(pcs, pcs)
                
                # sprin_feat = point_encoder(pcs, pc_normals, dist)
                preds = ppf_encoder(pcs, pc_normals, dist, idxs=point_idxs[0])
                
                preds_tr = preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins)
                preds_up = preds[..., 2 * cfg.tr_num_bins:2 * cfg.tr_num_bins + cfg.rot_num_bins]
                preds_right = preds[..., 2 * cfg.tr_num_bins + cfg.rot_num_bins:2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins]
                
                preds_up_aux = preds[..., -5]
                preds_right_aux = preds[..., -4]
                
                preds_scale = preds[..., -3:]

                loss_tr = kldiv(F.log_softmax(preds_tr[:, 0], dim=-1), targets_tr[0, :, 0]) + kldiv(F.log_softmax(preds_tr[:, 1], dim=-1), targets_tr[0, :, 1])
                loss_up = kldiv(F.log_softmax(preds_up[0], dim=-1), targets_rot[0, :, 0])
                loss_up_aux = bcelogits(preds_up_aux[0], targets_rot_aux[0, :, 0])
                loss_scale = F.mse_loss(preds_scale, targets_scale[:, None])
                # loss_scale = get_size_loss(Pred_s, gt_size)
                """Classification(segmentation) Loss"""
                # epsilon = 1e-5
                # torch.clamp(dense_part_cls_score, epsilon)
                # torch.clamp(gt_labels, epsilon)
                # num_parts = 2
                # loss_classify = torch.mean(F.cross_entropy(dense_part_cls_score.view(-1, num_parts), gt_labels.view(-1)))
                loss_classify = torch.tensor(0.)
                loss = loss_up + loss_tr + loss_up_aux + loss_scale #+ loss_classify 
                # loss = loss_scale
                if cfg.regress_right:
                    loss_right = kldiv(F.log_softmax(preds_right[0], dim=-1), targets_rot[0, :, 1])
                    loss_right_aux = bcelogits(preds_right_aux[0], targets_rot_aux[0, :, 1])
                    
                    loss += loss_right + loss_right_aux
                    loss_right_meter.update(loss_right.item())
                    loss_right_aux_meter.update(loss_right_aux.item())
                    
                loss.backward(retain_graph=False)
                
                # torch.nn.utils.clip_grad_norm_([*point_encoder.parameters(), *ppf_encoder.parameters()], 1.)
                opt.step()
                
                loss_meter.update(loss.item())
                loss_tr_meter.update(loss_tr.item())
                loss_up_meter.update(loss_up.item())
                loss_up_aux_meter.update(loss_up_aux.item())
                loss_scale_meter.update(loss_scale.item())
                loss_seg_meter.update(loss_classify.item())    
                n += 1
                total_iter += 1
                if n % 10 == 0:
                    writer.add_scalar('loss', loss_meter.avg, total_iter)
                    writer.add_scalar('loss_tr', loss_tr_meter.avg, total_iter)
                    writer.add_scalar('loss_up', loss_up_meter.avg, total_iter)
                    writer.add_scalar('loss_right', loss_right_meter.avg, total_iter)
                    writer.add_scalar('loss_up_aux', loss_up_aux_meter.avg, total_iter)
                    writer.add_scalar('loss_right_aux', loss_right_aux_meter.avg, total_iter)
                    writer.add_scalar('loss_scale', loss_scale_meter.avg, total_iter)
                    writer.add_scalar('loss_seg', loss_seg_meter.avg, total_iter)
                if cfg.regress_right:
                    t.set_postfix(epoch=epoch,loss=loss_meter.avg, loss_tr=loss_tr_meter.avg, 
                            loss_up=loss_up_meter.avg, loss_right=loss_right_meter.avg,
                            loss_up_aux=loss_up_aux_meter.avg, loss_right_aux=loss_right_aux_meter.avg,
                            loss_scale=loss_scale_meter.avg,loss_seg=loss_seg_meter.avg)
                else:
                    t.set_postfix(epoch=epoch,loss=loss_meter.avg, loss_tr=loss_tr_meter.avg, 
                            loss_up=loss_up_meter.avg,
                            loss_up_aux=loss_up_aux_meter.avg,
                            loss_scale=loss_scale_meter.avg)
        if epoch % 3 == 0:
            # torch.save(point_seg.state_dict(), f'point_seg_epoch{epoch}_{loss_meter.avg}.pth')
            # torch.save(point_encoder.state_dict(), f'point_encoder_epoch{epoch}_{loss_meter.avg}.pth')
            torch.save(ppf_encoder.state_dict(), f'ppf_encoder_epoch{epoch}_{loss_meter.avg}.pth')
            
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg 
            # torch.save(point_seg.state_dict(), f'point_seg_epochbest.pth')
            # torch.save(point_encoder.state_dict(), f'point_encoder_epochbest.pth')
            torch.save(ppf_encoder.state_dict(), f'ppf_encoder_epochbest.pth')
        logger.info('loss: {:.4f}, loss_tr: {:.4f}, loss_up: {:.4f}, loss_right: {:.4f}, loss_up_aux: {:.4f}, loss_right_aux: {:.4f}, loss_scale: {:.4f}'
                    .format(loss_meter.avg, loss_tr_meter.avg, loss_up_meter.avg, loss_right_meter.avg, loss_up_aux_meter.avg, loss_right_aux_meter.avg, loss_scale_meter.avg))
        scheduler.step(epoch)
    writer.close()
if __name__ == '__main__':
    main()