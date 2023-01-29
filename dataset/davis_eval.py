import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
from torch.utils import data

import glob

class DAVIS_MO_Test(data.Dataset):
    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False, scribble_path='default', mask_init=False, with_bg=True):
        self.root = root
        self.scribble_path = scribble_path
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.K = 12
        self.single_object = single_object

        self.with_bg = with_bg
        self.mask_init = mask_init
        print('*****Test {} background scribble.*****'.format('with' if self.with_bg else 'without'))

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.videos)


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

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)
        num_objects = torch.LongTensor([int(self.num_objects[video])])
        return num_objects, info

    def load_single_image(self,video, f):
        N_frames = np.empty((1,)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+self.shape[video], dtype=np.uint8)

        img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))

        N_frames[0] = np.array(Image.open(img_file).convert('RGB'))/255.
        try:
            if f == 0 and self.mask_init==False:
                mask_file = os.path.join(self.root, self.scribble_path, '{}.bmp'.format(video))
                N_masks[0] = np.array(Image.open(mask_file), dtype=np.uint8)
                no = N_masks[0].max() - 1
                if self.with_bg:
                    N_masks[0][N_masks[0] == no+1] = self.K-1
                else:
                    N_masks[0][N_masks[0] == no+1] = 0
            else:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))  
                N_masks[0] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        except Exception as e:
            print('Exception:', e)
            N_masks[0] = 255
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (0, 3, 1, 2)).copy()).float()
        Fs = (Fs - self.mean.reshape(1,3,1,1)) / self.std.reshape(1,3,1,1)
        if self.single_object:
            bg_masks = N_masks == (self.K-1)
            N_masks[N_masks != 1] = 0
            N_masks[bg_masks] = (self.K - 1)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float().permute(1,0,2,3)
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float().permute(1,0,2,3)
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects