import os

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from concurrent.futures import ThreadPoolExecutor
import logging
from models.model import PPFDecoder
from utils.dataset_train import PM_Dataset
# from utils.dataset_inference import PPF_infer_Dataset
from utils.util import AverageMeter
import hydra
import time


@hydra.main(config_path='./config', config_name='config')
def main(cfg):
    logger = logging.getLogger(__name__)
    assert cfg.batch_size == 1, "每个 part 的数据是独立的，batch_size 必须为 1"
    assert torch.cuda.device_count() >= cfg.num_parts, f"至少需要 {cfg.num_parts} 块 GPU"

    # ---------------- Dataset ----------------
    dataset = PM_Dataset(cfg=cfg,
                         mode='train',
                         data_root='/home/xxx/codes/dataset1',
                         num_pts=cfg.num_points,
                         num_cates=5,
                         num_parts=cfg.num_parts,
                         device='cpu',
                         use_p1_aug=False,
                         rescaled=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        pin_memory=True,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=16
    )
  
    # ---------------- Per-GPU state ----------------
    models, opts, schedulers, writers = [], [], [], []
    best_losses = [np.inf] * cfg.num_parts
    total_iters = [0] * cfg.num_parts
    streams = [torch.cuda.Stream(device=torch.device(f'cuda:{j}')) for j in range(cfg.num_parts)]
    kldiv = nn.KLDivLoss(reduction='batchmean')
    bcelogits = nn.BCEWithLogitsLoss()
    get_size_loss = nn.L1Loss()
    for j in range(cfg.num_parts):
        device = torch.device(f'cuda:{j}')
        model = PPFDecoder(ppffcs=[84, 32, 32, 16],
                           out_dim2=2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins + 2 + 3,
                           k=cfg.knn,
                           spfcs=[32, 64, 32, 32],
                           num_layers=1,
                           out_dim1=32).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epoch, eta_min=1e-5)
        writer = SummaryWriter(log_dir=os.path.join(cfg.root_dir, cfg.ckpdir, cfg.category, f'{j}'))

        models.append(model)
        opts.append(opt)
        schedulers.append(scheduler)
        writers.append(writer)

    # ---------------- Per-GPU train function ----------------
    def train_step_on_gpu(j, data, model, opt, writer, stream):
        os.makedirs(f'{j}', exist_ok=True)   
        device = torch.device(f'cuda:{j}')
        with torch.cuda.stream(stream):
            pcs             = data[ 0].to(device, non_blocking=True)
            pc_normals      = data[ 1].to(device, non_blocking=True)
            targets_tr      = data[ 2].to(device, non_blocking=True)
            targets_rot     = data[ 3].to(device, non_blocking=True)
            targets_rot_aux = data[ 4].to(device, non_blocking=True)
            targets_scale   = data[ 5].to(device, non_blocking=True)
            point_idxs      = data[ 6].to(device, non_blocking=True)
            cls_pts         = data[ 7].to(device, non_blocking=True)
            gt_labels       = data[ 8].to(device, non_blocking=True)
            gt_size         = data[ 9].to(device, non_blocking=True)
            mean_shape_part = data[10].to(device, non_blocking=True)

            with torch.no_grad():
                dist = torch.cdist(pcs, pcs)

            preds_out = model(pcs, pc_normals, dist, idxs=point_idxs[0],vertices=cls_pts, aux=cfg.aux_cls)
            preds = preds_out[0]
            preds_tr = preds[..., :2 * cfg.tr_num_bins].reshape(-1, 2, cfg.tr_num_bins)
            preds_up = preds[..., 2 * cfg.tr_num_bins:2 * cfg.tr_num_bins + cfg.rot_num_bins]
            preds_right = preds[..., 2 * cfg.tr_num_bins + cfg.rot_num_bins:2 * cfg.tr_num_bins + 2 * cfg.rot_num_bins]
            preds_up_aux = preds[..., -5]
            preds_right_aux = preds[..., -4]
            # preds_scale = preds[..., -3:]

            loss_tr = kldiv(F.log_softmax(preds_tr[:, 0], dim=-1), targets_tr[0, :, 0]) + \
                      kldiv(F.log_softmax(preds_tr[:, 1], dim=-1), targets_tr[0, :, 1])
            loss_up = kldiv(F.log_softmax(preds_up[0], dim=-1), targets_rot[0, :, 0])
            loss_up_aux = bcelogits(preds_up_aux[0], targets_rot_aux[0, :, 0])
            # loss_scale = F.mse_loss(preds_scale.mean(1), targets_scale)
            Pred_size = preds_out[-1]
            loss_scale = get_size_loss(Pred_size, gt_size)
            if cfg.aux_cls:
                cls = preds_out[1]
                loss_classify = F.cross_entropy(cls.view(-1,2), gt_labels.view(-1))
            else:
                loss_classify = torch.tensor(0.)
            loss = loss_tr + loss_up + loss_up_aux + loss_scale + loss_classify

            if cfg.regress_right:
                loss_right = kldiv(F.log_softmax(preds_right[0], dim=-1), targets_rot[0, :, 1])
                loss_right_aux = bcelogits(preds_right_aux[0], targets_rot_aux[0, :, 1])
                loss += loss_right + loss_right_aux
            else:
                loss_right = torch.tensor(0.).to(device)
                loss_right_aux = torch.tensor(0.).to(device)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_iters[j] += 1
            if total_iters[j] % 100 == 0:
                writers[j].add_scalar('lr', opt.param_groups[0]['lr'], total_iters[j])
                writers[j].add_scalar('loss', loss.item(), total_iters[j])
                writers[j].add_scalar('loss_tr', loss_tr.item(), total_iters[j])
                writers[j].add_scalar('loss_up', loss_up.item(), total_iters[j])
                writers[j].add_scalar('loss_up_aux', loss_up_aux.item(), total_iters[j])
                writers[j].add_scalar('loss_scale', loss_scale.item(), total_iters[j])
                if cfg.regress_right:
                    writers[j].add_scalar('loss_right', loss_right.item(), total_iters[j])
                    writers[j].add_scalar('loss_right_aux', loss_right_aux.item(), total_iters[j])
                writers[j].add_scalar('loss_seg', loss_classify.item(), total_iters[j])
                
            if total_iters[j] % 1000 == 0:               
                torch.save(model.state_dict(), f'{j}/epoch{epoch}_iter{int(total_iters[j]/1000)}_{loss.item()}.pth')
                if loss.item() < best_losses[j]:
                    best_losses[j] = loss.item()
                    torch.save(model.state_dict(), f'{j}/ppf_decoder_epochbest.pth')
           
          

    # ---------------- Training Loop ----------------
    start_time = time.time()
    logger.info(f"Start training on {cfg.num_parts} GPUs...")
    for epoch in range(cfg.max_epoch):
        for model in models:
            model.train()

        for data_list in tqdm(dataloader, desc=f"Epoch {epoch:02d}"):
            with ThreadPoolExecutor(max_workers=cfg.num_parts) as executor:
                futures = []
                for j in range(cfg.num_parts):
                    futures.append(executor.submit(
                        train_step_on_gpu,
                        j,
                        data_list[j],
                        models[j],
                        opts[j],
                        writers[j],
                        streams[j]
                    ))
                for f in futures:
                    f.result()
        logger.info(f"[Epoch {epoch:02d}] " + " | ".join([
            f"GPU{j}: BestLoss={best_losses[j]:.4f}" for j in range(cfg.num_parts)
        ]))

        for j in range(cfg.num_parts):
            torch.cuda.synchronize(device=f'cuda:{j}')
            schedulers[j].step()

        

    for writer in writers:
        writer.close()

    finish_time = time.time()
    # 计算总时长（秒）
    total_seconds = finish_time - start_time

    # 将总时长转换为时分秒
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 格式化输出
    logger.info(f"Training completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.")

if __name__ == '__main__':
    main()