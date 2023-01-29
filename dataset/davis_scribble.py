import os, cv2
import numpy as np
from PIL import Image

import torch
from torch.utils import data
import random
import glob
import cv2 as cv
from .aug import aug_heavy

class DAVISScribble_MO_Train(data.Dataset):
    def __init__(self, root, imset='2017/train.txt', resolution='480p', max_object_num=3, clip_length=3, crop_size=(384,384), 
                 random_flip=False, minimum_area=100, additional_bg=True, init_skip=0, bidirection=False, dilate_kernel_size=19, ds_stride=4):
        self.root = root
        self.max_object_num = max_object_num
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.minimum_area = minimum_area
        self.mask_dir = os.path.join(root, 'train_scribbles')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.num_frames = {}
        self.videos = []
        self.img_files = {}
        self.anno_files = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                tmp_imgs = glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))
                tmp_annos = glob.glob(os.path.join(self.mask_dir, _video, '*.bmp'))
                if len(tmp_imgs) != len(tmp_annos):
                    continue
                self.videos.append(_video)
                tmp_imgs.sort()
                tmp_annos.sort()
                self.img_files[_video] = tmp_imgs
                self.anno_files[_video] = tmp_annos
                self.num_frames[_video] = len(tmp_imgs)
        print('DAVIS Seq number:', len(self.videos))
        
        self.K = 12
        self.skip = init_skip
        self.additional_bg = additional_bg
        self.bidirection = bidirection
        self.crop_height, self.crop_width = crop_size
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                             (dilate_kernel_size, dilate_kernel_size))
        self.ds_stride = ds_stride

        self.aug = aug_heavy(crop_size=self.crop_size, minimum_area=self.minimum_area, in_clip_flip=self.random_flip, 
                             k=self.K, clip_length=self.clip_length, bidirection=self.bidirection)

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.videos)

    def change_skip(self,f):
        self.skip = f

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def mask_process(self, masks, f, num_object, ob_list, init_scrs):
        n = num_object
        mask = masks[f]
        init_scr = init_scrs[f]
        mask_ = np.zeros(mask.shape).astype(np.uint8)
        init_scr_ = np.zeros(init_scr.shape).astype(np.uint8)
        if f == 0:
            sample_range = self.K-1
            for i in range(1, sample_range):
                valid = np.sum(init_scrs[0] == i) > self.minimum_area and np.sum(init_scrs[-1] == i) > self.minimum_area if self.bidirection \
                    else np.sum(init_scrs[0] == i) > self.minimum_area
                if valid:
                    n += 1
                    ob_list.append(i)
            if n > self.max_object_num:
                n = self.max_object_num
                ob_list = random.sample(ob_list, n)

        for i, l in enumerate(ob_list):
            mask_[mask == l] = i+1
            init_scr_[init_scr == l] = i+1
        if self.additional_bg:
            other_mask = np.zeros_like(mask).astype(np.bool)
            for i in range(1, self.K-1):
                other_mask = other_mask + (mask == i) if i not in ob_list else other_mask
            mask_[other_mask] = self.K-1
        mask_[mask == self.K-1] = self.K-1
        init_scr_[init_scr == self.K-1] = self.K-1
        return mask_, n, ob_list, init_scr_

    def sample_frames(self, num_frames):
        frame_list = []
        for i in range(self.clip_length):
            post_frame_num = self.clip_length-i-1
            if i == 0: 
                frame_list.append(random.sample(range(0, num_frames-post_frame_num), 1)[0])
            else:
                frame_list.append(random.sample(range(frame_list[-1]+1, min(num_frames-post_frame_num, frame_list[-1]+2+self.skip)), 1)[0])
        return frame_list


    def get_sample(self, index):
        video = self.videos[index]
        image_files = self.img_files[video]
        anno_files = self.anno_files[video]
        info = {}
        info['name'] = video
        info['num_frames'] = self.clip_length

        N_frames = np.empty((self.clip_length,)+(self.crop_size[0],self.crop_size[1],)+(3,), dtype=np.float32)
        N_masks = np.empty((self.clip_length,)+(self.crop_size[0],self.crop_size[1],), dtype=np.uint8)
        N_init_scrs = np.empty((self.clip_length,)+(self.crop_size[0],self.crop_size[1],), dtype=np.uint8)
        valid_sample = False
        sample_time = 0
        while not valid_sample:
            if sample_time > 10:
                return None, None, None, None, None, None, 0, None
            frame_list = self.sample_frames(self.num_frames[video])
            num_object = 0
            ob_list = []
            frames_, masks_ = [], []
            for f_idx in range(self.clip_length):
                image_file = image_files[frame_list[f_idx]]
                tmp_frame = np.array(Image.open(image_file).convert('RGB'))
                try:
                    anno_file = anno_files[frame_list[f_idx]]
                    tmp_mask = np.array(Image.open(anno_file))
                    no = tmp_mask.max()
                    tmp_mask[tmp_mask == no] = self.K-1
                except:
                    tmp_mask = 255
                h, w = tmp_mask.shape
                if h < w:
                    tmp_frame = cv.resize(tmp_frame, (int(w/h*480), 480), interpolation=cv.INTER_LINEAR)
                    tmp_mask = Image.fromarray(tmp_mask).resize((int(w/h*480), 480), resample=Image.NEAREST)  
                else:
                    tmp_frame = cv.resize(tmp_frame, (480, int(h/w*480)), interpolation=cv.INTER_LINEAR)
                    tmp_mask = Image.fromarray(tmp_mask).resize((480, int(h/w*480)), resample=Image.NEAREST) 
                frames_.append(tmp_frame)
                masks_.append(np.array(tmp_mask))
            
            frames_, init_scrs_, masks_ = self.aug(frames_, masks_)
            if frames_ != None and init_scrs_ != None and masks_ != None:
                valid_sample = True
            sample_time += 1

        for f_idx in range(self.clip_length):
            masks_[f_idx], num_object, ob_list, init_scrs_[f_idx] = self.mask_process(masks_, f_idx, num_object, ob_list, init_scrs_)
            N_frames[f_idx], N_masks[f_idx], N_init_scrs[f_idx] = frames_[f_idx], masks_[f_idx], init_scrs_[f_idx]

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (0, 3, 1, 2)).copy()).float()
        Gs = torch.sum(Fs * torch.tensor([0.299, 0.587, 0.114])[None, :, None, None], dim=1)
        Fs = (Fs - self.mean.reshape(1, 3, 1, 1)) / self.std.reshape(1, 3, 1, 1)
        Init_Scr = torch.from_numpy(np.transpose(self.All_to_onehot(N_init_scrs).copy(), (1,0,2,3)).copy()).float()
        masks = self.All_to_onehot(N_masks)
        Ms = torch.from_numpy(np.transpose(masks.copy(), (1, 0, 2, 3)).copy()).float()
        
        ignore_regions = np.zeros((self.K, self.clip_length, self.crop_height, self.crop_width), dtype=np.uint8)
        for f in range(self.clip_length):
            for o in range(num_object):
                ignore_regions[o+1, f] = ((cv2.dilate(masks[o+1, f], self.dilate_kernel) + masks[o+1, f]) > 0).astype(np.uint8) - masks[o+1, f]
        ds_ignore_regions = self.downsample_label(ignore_regions)
        ds_masks = self.downsample_label(masks)
        Ds_Ms = torch.from_numpy(ds_masks.copy()).float().permute(1, 0, 2, 3)
        Ignore_R = torch.from_numpy(ds_ignore_regions.copy()).float().permute(1, 0, 2, 3)
        assert(torch.all(Ignore_R >=0) and torch.all(Ignore_R <= 1)), 'Expecting the value is either 0 or 1.'
        return Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R, num_object, info

    def downsample_label(self, labels):
        """label.shape: [K, clip_lengh, H, W]"""
        H, W = labels.shape[-2:]
        h, w = H//self.ds_stride, W//self.ds_stride
        ds_labels = np.stack([cv2.resize(l, (h, w), interpolation=cv2.INTER_AREA).transpose(2,0,1) \
                              for l in labels.transpose(1, 2, 3, 0)], axis=1)
        return ds_labels

    def __getitem__(self, index):
        valid_sample = False
        while not valid_sample:
            Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R, num_object, info = self.get_sample(index)
            if num_object > 0:
                valid_sample = True
            else:
                index = random.randint(0, len(self.videos)-1)
        return Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R, num_object, info