import argparse
import time
import os
import random
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import cv2
import numpy as np
from tensorboardX import SummaryWriter

from dataset.youtube_scribble import YtbScribble_MO_Train
from dataset.davis_scribble import DAVISScribble_MO_Train
from dataset.seq_sampler import SeqSampler
from model import build_model
import configs.default as default_config
from utils import ScribbleLoss, SmoothnessLoss
from utils.train_utils import reduce_mean, update_stats, print_stats, write_tb


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
parser.add_argument('--config',default='demo',type=str,help='config file for training')
parser.add_argument('-p','--print-freq',default=50,type=int,metavar='N',help='print frequency (default: 10)')
parser.add_argument('--resume',default=None, help='resume training from a pretrained model')


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic=True
    cudnn.benchmark=False


def adjust_learning_rate(init_lr, iteration, total_iteration, power = 0.9):
    lr = init_lr * pow((1 - 1.0 * iteration / total_iteration), power)
    return lr


def main():
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()
    args.save = './checkpoints/pretrain_weights_{}'.format(args.config)

    cfg = default_config.cfg
    default_config.update_config_from_file('./experiments/{:s}.yaml'.format(args.config))
    print(cfg)
    
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    args.nprocs = torch.cuda.device_count()

    main_worker(args.local_rank, args.nprocs, args, cfg)


def main_worker(local_rank, nprocs, args, cfg, base_seed=42):
    dist.init_process_group(backend='nccl')
    init_seeds(base_seed + local_rank)
    '''creat model'''
    model = build_model(cfg)
    if cfg.RESUME is not None:
        checkpoint = torch.load(cfg.RESUME, map_location='cpu')
        print("load from {}".format(cfg.RESUME))
        model.load_state_dict(checkpoint['state_dict'])

    start_epoch = 0
    print("start epoch: {}".format(start_epoch))

    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)  # TODO: 会报错，暂时设置find_unused_parameters=True

    '''loss, optimizer and scheduler'''
    pce_weight = torch.ones(12)
    pce_weight[0] = 0 # 0 for unlabeld region
    pce_weight[-1] = cfg.TRAIN.LOSS.MASK.BG_WEIGHT
    mask_loss = nn.CrossEntropyLoss(weight=pce_weight, reduction='mean').cuda(local_rank)
    smooth_loss = SmoothnessLoss(with_scribble=cfg.TRAIN.LOSS.SMOOTH.WITH_SCRIBBLE,
                                 alpha=cfg.TRAIN.LOSS.SMOOTH.ALPHA)
    consistency_loss = nn.MSELoss(reduction='mean').cuda(local_rank)
    scribble_loss = ScribbleLoss(use_focal_loss=cfg.TRAIN.LOSS.SCRIBBLE_LOSS.USE_FOCAL_LOSS,
                                       alpha=cfg.TRAIN.LOSS.SCRIBBLE_LOSS.ALPHA)

    criterions = [mask_loss, smooth_loss, consistency_loss, scribble_loss]
    # optimizer and scheduler
    param_dicts = [{"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]}, 
                   {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,}]

    if cfg.TRAIN.OPTIMIZER == 'AdamW':
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(param_dicts, lr = cfg.TRAIN.LR, eps=1e-8, betas=[0.9,0.999])
    else:
        raise ValueError("Unsupported Optimizer")

    '''data loading'''
    ytb_dataset = YtbScribble_MO_Train(root=cfg.DATASETS.YTB.ROOT,
                                    max_object_num=cfg.DATASETS.MAX_OBJECT_NUM,
                                    clip_length=cfg.DATASETS.YTB.CLIP_LENGTH,
                                    crop_size=cfg.DATASETS.CROP_SIZE,
                                    random_flip=cfg.DATASETS.YTB.RANDOM_FLIP,
                                    minimum_area=cfg.DATASETS.MINIMUM_AREA,
                                    init_skip=cfg.TRAIN.INIT_SKIP,
                                    bidirection=cfg.DATASETS.BIDIRECTION,
                                    dilate_kernel_size=cfg.DATASETS.DILATE_KERNEL_SIZE)

    davis_dataset = DAVISScribble_MO_Train(root=cfg.DATASETS.DAVIS.PATH,
                                    max_object_num=cfg.DATASETS.MAX_OBJECT_NUM,
                                    clip_length=cfg.DATASETS.DAVIS.CLIP_LENGTH,
                                    crop_size=cfg.DATASETS.CROP_SIZE,
                                    random_flip=cfg.DATASETS.DAVIS.RANDOM_FLIP,
                                    minimum_area=cfg.DATASETS.MINIMUM_AREA,
                                    init_skip=cfg.TRAIN.INIT_SKIP,
                                    bidirection=cfg.DATASETS.BIDIRECTION,
                                    dilate_kernel_size=cfg.DATASETS.DILATE_KERNEL_SIZE)
                                    
    train_dataset = SeqSampler(datasets=[ytb_dataset, davis_dataset], 
                               p_datasets=cfg.DATASETS.SAMPLE_PROBABILITY, 
                               samples_per_epoch=cfg.TRAIN.SAMPLES_PER_EPOCH,
                               init_skip=cfg.TRAIN.INIT_SKIP,
                               temporal_flip=cfg.DATASETS.VIDEO_TEMPORAL_FLIP)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                               num_workers=8, pin_memory=True, sampler=train_sampler, drop_last=True)

    '''tensorboard'''
    if args.local_rank == 0:
        writer = SummaryWriter('tensorboard/' + args.config)
    else:
        writer=None
    
    '''training'''
    for epoch in range(start_epoch, cfg.TRAIN.EPOCH):
        train_sampler.set_epoch(epoch)

        train(train_loader, model, criterions, optimizer, epoch, local_rank, args, cfg, writer)

        if args.local_rank == 0:
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': model.module.state_dict(),
            }, args)


def train(train_loader, model, criterions, optimizer, epoch, local_rank, args, cfg, writer):
    train_stats = OrderedDict()
    mask_loss_func, smooth_loss_func, consistency_loss_func, scribble_loss_func = criterions
    model.train()

    start_time = time.time()
    all_frames = 0
    for i, (Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R, num_objects, info) in enumerate(train_loader):
        prev_time = time.time()
        all_frames += Fs.shape[0]

        iter_number = epoch*(len(train_loader)) + i
        # update lr
        if (iter_number+1) % cfg.TRAIN.LR_DECREASE_STEP == 0:
            lr = adjust_learning_rate(cfg.TRAIN.LR, iter_number+1, cfg.TRAIN.EPOCH*len(train_loader))
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg.TRAIN.BACKBONE_MULTIPLIER
            if local_rank == 0:
                writer.add_scalars('LR', {'LR': lr}, iter_number+1)
                print('********** drop lr to {}. **********'.format(lr))
        # change skip
        if (iter_number+1) % cfg.TRAIN.SKIP_INCREASE_STEP == 0:
            curr_skip = train_loader.dataset.skip
            if curr_skip < cfg.TRAIN.MAX_SKIP:
                train_loader.dataset.change_skip(curr_skip+1)
                print('********** change skip gap to {}. **********'.format(curr_skip+1))

        
        Fs = Fs.cuda(local_rank, non_blocking=True) # [B, clip_length, 3, H, W]
        Gs = Gs.cuda(local_rank, non_blocking=True) # [B, clip_length, H, W]
        Init_Scr = Init_Scr.cuda(local_rank, non_blocking=True) # [B, clip_length, K, H, W]
        Ms = Ms.cuda(local_rank, non_blocking=True) # [B, clip_length, 12, H, W], self.K=12
        Ds_Ms = Ds_Ms.cuda(local_rank, non_blocking=True) # [B, clip_length, K, H//4, W//4], self.K=12
        Ignore_R = Ignore_R.cuda(local_rank, non_blocking=True) # [B, clip_length, K, H//4, W//4], self.K=12
        num_objects = num_objects.cuda(local_rank, non_blocking=True)   # [B]
        no = num_objects.max()
        B, _, K, im_h, im_w = Ms.shape
        ds_h, ds_w = Ds_Ms.shape[-2:]
        seq_name = info['name']
        num_frames = info['num_frames'][0].item()
        
        # forward prediction
        multi_pred_prob_list, multi_pred_logit_list, binary_pred_prob_list, binary_edge_logit_list, valid, \
            multi_scribble_prob_list, binary_scribble_logit_list = model(images=Fs, init_scribble=Init_Scr[:,0,...], num_objects=num_objects)
        pred_h, pred_w = multi_pred_prob_list[0][0].shape[-2:]
        
        # backward prediction
        Fs_inverse = Fs.flip(dims=(1,))
        Gs_inverse = Gs.flip(dims=(1,))
        Ms_inverse = Ms.flip(dims=(1,))
        with torch.no_grad():
            multi_pred_prob_list_inverse, multi_pred_logit_list_inverse, binary_pred_prob_list_inverse, binary_edge_logit_list_inverse, _, _, _ = \
                model(images=Fs_inverse, init_scribble=Init_Scr[:,-1,...], num_objects=num_objects)
        
        multi_pred_prob_list_backward = multi_pred_prob_list_inverse[::-1]
        multi_pred_logit_list_backward = multi_pred_logit_list_inverse[::-1]
        
        status = {}
        all_us_multi_logits = [] # list of [batch*clip_length, self.K, H, W]
        mask_label = Ms.reshape(-1, K, im_h, im_w).argmax(dim=1).long() # [B*clip_length, im_h, im_w]
        pce_losses = torch.zeros([]).to(Ms.device)
        smooth_losses = torch.zeros([]).to(Ms.device)
        for layer, multi_pred_logits in enumerate(zip(*multi_pred_logit_list)):
            #[batch, clip_length, self.K, H, W] --> #[batch*clip_length, self.K, H, W]
            multi_logits = torch.stack(multi_pred_logits, dim=1).reshape(-1, K, pred_h, pred_w)
            if cfg.MODEL.DECODER.UPSAMPLE_LOGITS:
                us_multi_logits = multi_logits
            else:
                us_multi_logits = F.interpolate(multi_logits, size=(im_h, im_w), mode='bilinear', align_corners=True)
            all_us_multi_logits.append(us_multi_logits)
            pce_loss = mask_loss_func(us_multi_logits.permute(0,2,3,1).reshape(-1, K), mask_label.reshape(-1))
            pce_losses += pce_loss
            status["pce_loss_layer{:d}".format(layer)] = pce_loss.item()
            
            if cfg.TRAIN.LOSS.SMOOTH.WEIGHT != 0:
                smooth_loss = smooth_loss_func(torch.softmax(us_multi_logits, dim=1), Gs.reshape(-1, im_h, im_w), mask_label)
                smooth_losses += smooth_loss
                status["smooth_loss_layer{:d}".format(layer)] = smooth_loss.item()

        if cfg.TRAIN.LOSS.CONSISTENCY.WEIGHT != 0:
            multi_probs_list = [torch.stack(multi_pred_probs, dim=1).reshape(-1, K, pred_h, pred_w) for multi_pred_probs in zip(*multi_pred_prob_list)]
            multi_probs_list_bw = [torch.stack(multi_pred_probs_bw, dim=1).reshape(-1, K, pred_h, pred_w) for multi_pred_probs_bw in zip(*multi_pred_prob_list_backward)]
            consistency_losses = torch.zeros([]).to(Ms.device)
            for layer in range(len(multi_probs_list)):
                consistency_loss = consistency_loss_func(multi_probs_list[layer], multi_probs_list_bw[layer])
                consistency_losses += consistency_loss
                status["consistency_loss_layer{:d}".format(layer)] = consistency_loss.item()

        if cfg.TRAIN.LOSS.SCRIBBLE_LOSS.WEIGHT != 0:
            # (B, no) --> (clip_length, B, no, 1, 1)
            valid_weights = valid.unsqueeze(0).repeat(num_frames, 1, 1).unsqueeze(-1).unsqueeze(-1).cuda(local_rank, non_blocking=True)
            # (B, clip_length, K, H//4, W//4) --> (clip_length, B, no, H//4, W//4)
            
            valid_regions = ((1 - Ignore_R).permute(1, 0, 2, 3, 4)[:, :, 1:no+1, :, :] * valid_weights).reshape(-1)
            scribble_labels = Ds_Ms.permute(1, 0, 2, 3, 4)[:, :, 1:no+1, :, :].reshape(-1)
            pred_scribble_logits = torch.stack(binary_scribble_logit_list, dim=0).reshape(-1)
            scribble_loss = scribble_loss_func(pred_scribble_logits, scribble_labels, valid_regions)
            status['scribble_loss'] = scribble_loss.item()

        total_losses = cfg.TRAIN.LOSS.MASK.WEIGHT * pce_losses
        if cfg.TRAIN.LOSS.SMOOTH.WEIGHT != 0:
            total_losses += cfg.TRAIN.LOSS.SMOOTH.WEIGHT * smooth_losses
        if cfg.TRAIN.LOSS.CONSISTENCY.WEIGHT != 0:
            total_losses += cfg.TRAIN.LOSS.CONSISTENCY.WEIGHT * consistency_losses
        if cfg.TRAIN.LOSS.SCRIBBLE_LOSS.WEIGHT != 0:
            total_losses += cfg.TRAIN.LOSS.SCRIBBLE_LOSS.WEIGHT * scribble_loss
        status['total_loss'] = total_losses.item()

        torch.distributed.barrier()

        train_stats = update_stats(old_stats=train_stats, new_stats=status, batch_size=B)
        
        optimizer.zero_grad()
        total_losses.backward()
        optimizer.step()

        # write tensorboard
        if i % args.print_freq == 0:
            if args.local_rank == 0:
                print_stats(epoch, i, train_loader, [start_time, prev_time], B, all_frames, train_stats, './logs/train_{:s}_log.txt'.format(args.config))
                train_stats = write_tb(writer, train_stats, iter_number)

            # visualize
            if args.local_rank == 0:
                layer_nums = len(all_us_multi_logits)
                all_us_multi_logits = torch.stack(all_us_multi_logits, dim=0) # [layer, B*L, K, H, W]
                mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,1,3,1,1).cuda(local_rank)
                std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,1,3,1,1).cuda(local_rank)
                colors = np.array([[0, 0, 255],[0, 255, 0],[0, 255, 255],[255, 0, 0],[255, 0, 255]])

                unnormed_Fs = Fs * std + mean   # [B, L, 3, H, W]
                np_images = (unnormed_Fs.permute(0,1,3,4,2).cpu().numpy()*255.).astype(np.uint8)   # [B, L, H, W, 3]
                np_labels = np.argmax(Ms.cpu().numpy(), axis=2).astype(np.uint8)  # [B, L, H, W]
                
                # input    # [B, H, W, 3]
                init_image, init_label = np_images[:, 0].copy(), np_labels[:, 0].copy()
                for o in range(cfg.DATASETS.MAX_OBJECT_NUM):
                    init_image_ = init_image * 0.4 + np.ones(init_image.shape) * 0.6 * colors[o]
                    init_image[init_label == o+1] = init_image_[init_label == o+1]
                init_image[init_label == 11] = 255
                init_image = init_image.reshape(-1, *init_image.shape[2:])  # [B*H, W, 3]
                for idx, name in enumerate(seq_name):
                    cv2.putText(init_image, name, (0, idx*im_h+30), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,255), 1)
                # pred
                masks = torch.argmax(F.softmax(all_us_multi_logits, dim=2), dim=2).cpu().numpy().reshape(layer_nums, *np_images.shape[:-1]) # [layer, B, L, H, W]
                for layer in range(layer_nums):
                    images = np_images.copy()  # [B, L, H, W, 3]
                    for o in range(cfg.DATASETS.MAX_OBJECT_NUM):
                        images_ = images * 0.4 + np.ones(images.shape) * 0.6 * colors[o]
                        images[masks[layer] == o+1] = images_[masks[layer] == o+1]
                    images = images.transpose([0, 2, 1, 3, 4])  # [B, H, L, W, 3]
                    images = images.reshape(images.shape[0]* images.shape[1], images.shape[2]*images.shape[3], -1)  # [B*H, L*W, 3]
                    vis = np.concatenate([init_image, images], axis=1)   # [B*H, (L+1)*W, 3]
                    writer.add_image('layer_{}'.format(layer+1), vis, iter_number, dataformats='HWC')

                scribble_images = np_images.copy()  #[B, L, H, W, 3]
                multi_scribble_probs = torch.stack(multi_scribble_prob_list, dim=1).reshape(-1, K, ds_h, ds_w)#(B*L, K, H, W)
                us_multi_scribble_probs = F.interpolate(multi_scribble_probs, size=(im_h, im_w), mode='bilinear', align_corners=True)
                scribbles = torch.argmax(us_multi_scribble_probs, dim=1).cpu().numpy().reshape(*np_images.shape[:-1])# [B, L, H, W]
                for o in range(cfg.DATASETS.MAX_OBJECT_NUM):
                    scribble_images_ = scribble_images * 0.4 + + np.ones(scribble_images.shape) * 0.6 * colors[o]
                    scribble_images[scribbles == o+1] = scribble_images_[scribbles == o+1]
                scribble_images = scribble_images.transpose([0, 2, 1, 3, 4])# [B, H, L, W, 3]
                scribble_images = scribble_images.reshape(scribble_images.shape[0]* scribble_images.shape[1], scribble_images.shape[2]*scribble_images.shape[3], -1)  # [B*H, L*W, 3]
                scribble_vis = np.concatenate([init_image, scribble_images], axis=1)   # [B*H, (L+1)*W, 3]
                writer.add_image('scribble', scribble_vis, iter_number, dataformats='HWC')

def save_checkpoint(state, args):
    save_dir = args.save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, 'pretrain_checkpoint_{}.pth.tar'.format(state['epoch']))
    torch.save(state, filename)

if __name__ =='__main__':
    main()
