import os
import torch
import cv2 as cv
from PIL import Image
import numpy as np
import tqdm
import argparse
import warnings

from utils.helpers import pad_divide_by
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from dataset.youtube_eval import Ytb_MO_Test
from model import build_model, build_tracker
import configs.default as default_config


colors = np.array([[0, 0, 255],[0, 255, 0],[0, 255, 255],[255, 0, 0],[255, 0, 255],[255, 255, 0]])
_palette = Image.open('/datasets/Youtubu-VOS/valid/Annotations/0062f687f1/00000.png').getpalette()

def denormalize(images):
    """
    images (Tensor) - (B, C, H, W)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1)
    return images * std + mean


def Run_video(video, tracker):
    video_name = video.seq_name
    vis_dir = os.path.join(args.save_root_path, 'vis', video_name)
    results_dir = os.path.join(args.save_root_path, 'results', video_name)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    num_objects = torch.LongTensor([max(video.obj_indices[-1])])

    for f_idx, (Fs, Ms, info) in enumerate(video):
        # put new objects into the memory if Ms is not None
        if f_idx == 0:
            tracker.initialize(Fs.cuda(), Ms.cuda(), num_objects.cuda())
            cur_object_idx = video.obj_indices[f_idx]
            valid = torch.zeros(1, num_objects.item()).cuda()
            for obj_idx in cur_object_idx:
                valid[:, obj_idx-1] = 1
            tracker.valid = valid
        if f_idx != 0 and Ms is not None:
            pre_object_idx = video.obj_indices[f_idx-1]
            cur_object_idx = video.obj_indices[f_idx]
            new_object_idx = []
            for obj_idx in cur_object_idx:
                if obj_idx not in pre_object_idx:
                    new_object_idx.append(obj_idx)
            for obj_idx in new_object_idx:
                tracker.valid[:, obj_idx-1] = 1
            # run init
            F_, M_ = Fs.cuda().clone(), Ms.cuda().clone()
            (image, scribble), pad = pad_divide_by([F_, M_], tracker.net.stride, (tracker.im_h, tracker.im_w))
            valid = tracker.net.compute_valid(num_objects, num_objects.max(), 1, device=image.device)
            middle_feats = tracker.net.backbone(image)
            feat = tracker.net.bottleneck(middle_feats[-1])
            learned_pos = tracker.net.position_embedding(feat)
            new_memories = tracker.net.transformer.init_memory(feat, scribble, learned_pos, tracker.net.hr_embed_obj.weight,
                                                       tracker.net.hr_embed_bg.weight, tracker.net.stride, num_objects, 
                                                       tracker.net.use_similarity, tracker.net.local_similarity, feat)
            for layer_idx, new_memory in enumerate(new_memories):
                for obj_idx in new_object_idx:
                    tracker.memory['scribble'][layer_idx][:, obj_idx-1, :] = new_memory[:, obj_idx-1, :]
        pred_prob, pred_scribble_prob, scores = tracker.track(Fs.cuda(), num_objects.cuda(), frame_num=f_idx)
        pred = torch.argmax(pred_prob[0], dim=0).cpu().numpy().astype(np.uint8)

        pred_scribble = torch.zeros_like(pred_scribble_prob[0])
        for o in range(1, 1+num_objects.item()):
            pred_scribble[o] = (pred_scribble_prob[0, o] >= 0.5).to(torch.int) * (pred_scribble_prob[0, o] >= 0.5).to(torch.int)
        pred_scribble[-1] = 1 - pred_scribble.sum(dim=0)
        for o in range(12):
            pred_scribble[o] = pred_scribble[o] * o
        scribble = torch.argmax(pred_scribble, dim=0).cpu().numpy().astype(np.uint8)
        # vis and save results
        vis_image = (denormalize(Fs).cpu().permute(0,2,3,1).numpy()*255.).astype(np.uint8).squeeze(0)
        scr_image = vis_image.copy()
        result = np.zeros(vis_image.shape[:-1]).astype(np.uint8)
        for o in range(num_objects):
            vis_image_ = vis_image*0.4+np.ones(vis_image.shape)*0.6*colors[o]
            scr_image_ = scr_image*0.4+np.ones(scr_image.shape)*0.6*colors[o]
            vis_image[pred == o+1] = vis_image_[pred == o+1]
            scr_image[scribble == o+1] = scr_image_[scribble == o+1]
            result[pred == o+1] = o+1
        vis_image = cv.resize(vis_image, (info['original_width'], info['original_height']), interpolation=cv.INTER_LINEAR)
        scr_image = cv.resize(scr_image, (info['original_width'], info['original_height']), interpolation=cv.INTER_LINEAR)
        cv.imwrite(os.path.join(vis_dir, info['frame_name']), np.flip(vis_image, axis=-1))
        cv.imwrite(os.path.join(vis_dir, 'scribble_'+info['frame_name']), np.flip(scr_image, axis=-1))
        if info['save_result']:
            result = Image.fromarray(result).convert('P')
            result = result.resize((info['original_width'], info['original_height']), 0)
            result.putpalette(_palette)
            result.save(os.path.join(results_dir, info['frame_name'].replace('jpg', 'png')))


def test(TestLoader, epoch, args, cfg):
    args.save_root_path = os.path.join(args.save_path, '{:s}_{:d}_{}'.format(args.config, epoch, args.size))
    checkpoint_path = './checkpoints/pretrain_weights_{:s}/pretrain_checkpoint_{:d}.pth.tar'.format(args.config, epoch)
    print("checkpoint path: {}".format(checkpoint_path))
    print("Save results to: {}".format(args.save_root_path))
    model = build_model(cfg)
    if torch.cuda.is_available():
        model.cuda()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    tracker = build_tracker(cfg, model)
    for video in tqdm.tqdm(TestLoader):
        video_name = video.seq_name
        if args.video_name is not None:
            if video_name == args.video_name:
                Run_video(video, tracker)
                break
            else:
                continue
        Run_video(video, tracker)

if __name__ == "__main__":
    torch.set_grad_enabled(False) # Volatile
    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("--gpu", type=str, help="0; 0,1; 0,3; etc", required=True)
        parser.add_argument("--set", type=str, help="set", required=True)
        parser.add_argument("--dataset", type=str, help="dataset_name",default='Youtube-VOS')
        parser.add_argument("--config", type=str, default='default', help='config file name')
        parser.add_argument("--ckpt_num", type=int, help="index to epoch",default=0)
        parser.add_argument("--start", type=int, help="index to epoch",default=0)
        parser.add_argument("--end", type=int, help="index to epoch",default=0)
        parser.add_argument("--video_name", type=str, default=None)
        parser.add_argument("--save_path", type=str, default='test_vis')
        parser.add_argument("--size", type=int, default=None)
        return parser.parse_args()
    
    args = get_arguments()
    cfg = default_config.cfg
    default_config.update_config_from_file('./experiments/{:s}.yaml'.format(args.config))
    print(cfg)
    GPU = args.gpu
    SET = args.set
    print(cfg.MODEL.TYPE, ': Testing on {:s}'.format(args.dataset))

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    if torch.cuda.is_available():
        print('using Cuda devices, num:', torch.cuda.device_count())

    TestLoader = Ytb_MO_Test(root=cfg.DATASETS.YTB.ROOT, imset='val', size=args.size)

    if args.ckpt_num > 0:
        test(TestLoader, args.ckpt_num, args, cfg)
    else:
        for ep in range(args.start, args.end+1):
            try:
                test(TestLoader, ep, args, cfg)
            except FileNotFoundError:
                print("Epoch {:d} does not exist".format(ep))