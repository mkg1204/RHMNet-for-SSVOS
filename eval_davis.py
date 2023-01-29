import torch
import cv2
import numpy as np
import time
import tqdm
import os
import argparse
import sys

from dataset.davis_eval import DAVIS_MO_Test
from model import build_model, build_tracker
import configs.default as default_config

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from evaldavis2017.davis2017.davis import DAVIS
from evaldavis2017.davis2017.metrics import db_eval_boundary, db_eval_iou
from evaldavis2017.davis2017 import utils
from evaldavis2017.davis2017.results import Results
import skimage.morphology as sm
from scipy.ndimage.morphology import binary_erosion, binary_dilation

colors = np.array([[255, 0, 0],[0, 255, 0],[0, 0, 255],[255, 255, 0],[255, 0, 255]])

def denormalize(images):
    """
    images (Tensor) - (B, C, H, W)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)
    return images * std + mean

def Run_video(dataset, video, num_frames, num_objects, tracker):
    save_dir = os.path.join(args.save_root_path, video)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    F, M, num_objects = dataset.load_single_image(video, 0)

    # save init scribble
    '''
    init_image = (denormalize(F).cpu().permute(0,2,3,1).numpy() * 255.0).astype(np.uint8).squeeze(0)
    init_scribbles = M.squeeze().detach().cpu().numpy()
    for o in range(num_objects):
        init_image[init_scribbles[o+1] == 1] = colors[o]
    cv2.imwrite(os.path.join(save_dir, "init_scribble.jpg"), np.flip(init_image, axis=-1))
    '''
    tracker.initialize(F.cuda(), M.cuda(), num_objects)
    pred_prob, pred_scribble_prob, scores = tracker.track(F.cuda(), num_objects, frame_num=0)
    pred = np.zeros((num_frames, M.shape[-2], M.shape[-1]))
    scribble = np.zeros((num_frames, M.shape[-2], M.shape[-1]))
    all_Ms = []
    fpss = []
    for t in range(1, num_frames):
        start = time.time()
        F_, M_, num_objects = dataset.load_single_image(video, t)
        # the predicted mask
        pred_prob, pred_scribble_prob, scores = tracker.track(F_.cuda(), num_objects, frame_num=t)
        all_Ms.append(M_.cpu().numpy())
        pred[t] = torch.argmax(pred_prob[0], dim=0).cpu().numpy().astype(np.uint8)
        fpss.append(1/(time.time()-start))
        # the predicted scribble
        pred_scribble = torch.zeros_like(pred_scribble_prob[0])
        for i in range(1, 1+num_objects[0]):
            pred_scribble[i] = (pred_scribble_prob[0, i] >= 0.5).to(torch.int) * (pred_scribble_prob[0, i] >= 0.5).to(torch.int)
        pred_scribble[-1] = 1 - pred_scribble.sum(dim=0)
        for i in range(12):
            pred_scribble[i] = pred_scribble[i] * i
        scribble[t] = torch.argmax(pred_scribble, dim=0).cpu().numpy().astype(np.uint8)
    '''  
        vis_image = (denormalize(F_).cpu().permute(0,2,3,1).numpy() * 255.0).astype(np.uint8).squeeze(0) # w, h, 3
        vis_mask = pred[t]  # w, h
        for o in range(num_objects):
            binary_mask = vis_mask == o+1

            # _palette = Image.open('/datasets/DAVIS/00000.png').getpalette()
            # save = binary_mask.astype(np.uint8)
            # save[save == 1] = 2
            # save = Image.fromarray(save).convert('P')
            # save.putpalette(_palette)
            # save.save(os.path.join(save_dir, "{:0>5d}_mask.png".format(t)))
            
            vis_image_ = vis_image * 0.4 + np.ones(vis_image.shape) * 0.6 * colors[o]
            vis_image[binary_mask] = vis_image_[binary_mask]
            contours = binary_dilation(binary_mask, sm.square(5)) ^ binary_mask
            vis_image[contours] = 255
        cv2.imwrite(os.path.join(save_dir, "{:0>5d}.jpg".format(t)), np.flip(vis_image, axis=-1))

        scribble_vis_image = (denormalize(F_).cpu().permute(0,2,3,1).numpy() * 255.0).astype(np.uint8).squeeze(0) # w, h, 3
        vis_scribble = scribble[t]
        for o in range(num_objects):

            # _palette = Image.open('/datasets/DAVIS/00000.png').getpalette()
            # save = (vis_scribble == o+1).astype(np.uint8)
            # save[save == 1] = 2
            # save = Image.fromarray(save).convert('P')
            # save.putpalette(_palette)
            # save.save(os.path.join(save_dir, "{:0>5d}_scribble.png".format(t)))
            

            scribble_vis_image_ = scribble_vis_image * 0.4 + np.ones(scribble_vis_image.shape) * 0.6 * colors[o]
            scribble_vis_image[vis_scribble == o+1] = scribble_vis_image_[vis_scribble == o+1]
        cv2.imwrite(os.path.join(save_dir, "scribble_{:0>5d}.jpg".format(t)), np.flip(scribble_vis_image, axis=-1))
    '''
    Ms = np.stack(all_Ms, axis=2)
    return pred, Ms

def evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
    if all_res_masks.shape[0] > all_gt_masks.shape[0]:
        sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
        sys.exit()
    elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
        zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
        all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
    j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
    for ii in range(all_gt_masks.shape[0]):
        if 'J' in metric:
            j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        if 'F' in metric:
            f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
    return j_metrics_res, f_metrics_res

def evaluate(model, Testloader, metric, video_name=None):

    # Containers
    metrics_res = {}
    if 'J' in metric:
        metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
    if 'F' in metric:
        metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

    for V in tqdm.tqdm(Testloader):
        num_objects, info = V
        seq_name = info['name']
        num_frames = info['num_frames']
        if video_name is not None:
            if seq_name == video_name:
                pred, Ms = Run_video(Testloader, seq_name, num_frames, num_objects, model)
                return None        
            else:
                continue
        pred, Ms = Run_video(Testloader, seq_name, num_frames, num_objects, model)
        # all_res_masks = Es[0].cpu().numpy()[1:1+num_objects]
        all_res_masks = np.zeros((num_objects, pred.shape[0], pred.shape[1], pred.shape[2]))
        for i in range(1,num_objects+1):
            all_res_masks[i-1,:,:,:] = (pred == i).astype(np.uint8)
        all_res_masks = all_res_masks[:, 1:-1, :, :]
        all_gt_masks = Ms[0][1:1+num_objects]
        all_gt_masks = all_gt_masks[:, :-1, :, :]
        j_metrics_res, f_metrics_res = evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                metrics_res['J']["M"].append(JM)
                metrics_res['J']["R"].append(JR)
                metrics_res['J']["D"].append(JD)
            if 'F' in metric:
                [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                metrics_res['F']["M"].append(FM)
                metrics_res['F']["R"].append(FR)
                metrics_res['F']["D"].append(FD)

    J, F = metrics_res['J'], metrics_res['F']
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    return g_res, g_measures
	    
def test(Testloaders, metric, epoch, args, cfg):
    g_res_list, g_measures_list = [], []
    for scribble_group_id, Testloader in enumerate(Testloaders):
        args.save_root_path = os.path.join(args.save_path, '{:s}_{:d}_davis_g{}'.format(args.config, epoch, scribble_group_id))
        checkpoint_path = './checkpoints/pretrain_weights_{:s}/pretrain_checkpoint_{:d}.pth.tar'.format(args.config, epoch)
        model = build_model(cfg)
        if torch.cuda.is_available():
            model.cuda()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        tracker = build_tracker(cfg, model)
        g_res, g_measures = evaluate(tracker, Testloader, metric, args.video_name)
        g_res_list.append(g_res)
        g_measures_list.append(g_measures)
    return g_res_list, g_measures_list

if __name__ == "__main__":
    torch.set_grad_enabled(False) # Volatile
    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("--gpu", type=str, help="0; 0,1; 0,3; etc", required=True)
        parser.add_argument("--set", type=str, help="set", required=True)
        parser.add_argument("--year", type=int, help="year", required=True)
        parser.add_argument("--dataset", type=str, help="dataset_name",default='DAVIS')
        parser.add_argument("--config", type=str, default='default', help='config file name')
        parser.add_argument("--ckpt_num", type=int, help="index to epoch",default=0)
        parser.add_argument("--start", type=int, help="index to epoch",default=0)
        parser.add_argument("--end", type=int, help="index to epoch",default=0)
        parser.add_argument("--video_name", type=str, default=None)
        parser.add_argument("--save_path", type=str, default='test_vis')
        parser.add_argument("--with_bg", action='store_true')
        parser.add_argument("--mask_init", action='store_true')
        return parser.parse_args()

    args = get_arguments()
    cfg = default_config.cfg
    default_config.update_config_from_file('./experiments/{:s}.yaml'.format(args.config))
    print(cfg)
    GPU = args.gpu
    YEAR = args.year
    SET = args.set

    print(cfg.MODEL.TYPE, ': Testing on {:s}{:d}'.format(args.dataset, YEAR))

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())
    
    if args.dataset == "DAVIS":
        Testloaders = [DAVIS_MO_Test(cfg.DATASETS.DAVIS.PATH, resolution='480p', 
                                   imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16),
                                   scribble_path="valid_scribbles{}".format(scribble_group_id),
                                   mask_init=args.mask_init,with_bg=args.with_bg) for scribble_group_id in range(5)]
    else:
        raise Exception('Undefined dataset!')

    metric = ['J','F']
    if args.ckpt_num > 0:
        g_res_list, g_measures_list = test(Testloaders, metric, args.ckpt_num, args, cfg)
        for scribble_group_id in range(len(g_res_list)):
            g_res = g_res_list[scribble_group_id]
            print('epoch{:d}_group{:d}:'.format(args.ckpt_num, scribble_group_id), 'J&F-Mean: {:.3f}, J-Mean: {:.3f}, J-Recall: {:.3f}, J-Decay: {:.3f}, \
                            F-Mean: {:.3f}, F-Recall: {:.3f}, F-Decay: {:.3f}'.format(*g_res))
    else:
        all_results = {}
        for ep in range(args.start, args.end+1):
            try:
                g_res_list, g_measures_list = test(Testloaders, metric, ep, args, cfg)
                all_results[ep] = g_res_list
            except FileNotFoundError:
                print("Epoch {:d} does not exist".format(ep))
        for ep in all_results.keys():
            g_res_list = all_results[ep]
            for scribble_group_id in range(len(g_res_list)):
                g_res = g_res_list[scribble_group_id]
                print('epoch{:d}_group{:d}:'.format(args.ckpt_num, scribble_group_id), 'J&F-Mean: {:.3f}, J-Mean: {:.3f}, J-Recall: {:.3f}, J-Decay: {:.3f}, \
                                F-Mean: {:.3f}, F-Recall: {:.3f}, F-Decay: {:.3f}'.format(*g_res))
    