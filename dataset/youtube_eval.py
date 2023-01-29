import os
import json
import cv2 as cv
import numpy as np
from PIL import Image
import torch
from torch.utils import data


class VOSTest(data.Dataset):
    def __init__(self, image_root, anno_root, seq_name, images, annos, need_save_results=None, resize_size=None):
        self.image_root = image_root
        self.anno_root = anno_root
        self.seq_name = seq_name
        self.images = images
        self.need_save_results = need_save_results if need_save_results is not None else [True]*len(self.images)
        self.annos = annos
        self.resize_size = resize_size
        self.num_frames = len(self.images)
        self.obj_nums = []
        self.obj_indices = []

        curr_objs = []
        for image_name in self.images:
            current_anno_name = image_name.split('.')[0]+'.bmp'
            if current_anno_name in self.annos:
                current_anno = np.array(Image.open(os.path.join(self.anno_root, self.seq_name, current_anno_name)), dtype=np.uint8)
                current_obj = sorted(list(np.unique(current_anno)))[1:-1]
                for obj_idx in current_obj:
                    if obj_idx not in curr_objs:
                        curr_objs.append(obj_idx)
            self.obj_nums.append(len(curr_objs))
            self.obj_indices.append(sorted(curr_objs.copy()))
        self.K = 12
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.images)
    
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

    def __getitem__(self, idx):
        img_name = self.images[idx]
        current_img = np.array(Image.open(os.path.join(self.image_root, self.seq_name, img_name)).convert('RGB'))
        h, w, _ = current_img.shape
        if self.resize_size is None:
            new_h, new_w = h, w
        else:
            if w < self.resize_size:
                new_h, new_w = h, w
            else:
                new_h, new_w = int(self.resize_size * h / w), self.resize_size
        current_img = cv.resize(current_img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        current_img = current_img.astype(np.float32)/255.
        Fs = torch.from_numpy(np.transpose(current_img[np.newaxis, ...].copy(), (0,3,1,2)).copy()).float()
        Fs = (Fs - self.mean.reshape(1,3,1,1)) / self.std.reshape(1,3,1,1)
        current_anno_name = img_name.split('.')[0]+'.bmp'
        obj_num = self.obj_nums[idx]
        obj_idx = self.obj_indices[idx]
        all_obj_num = max(self.obj_indices[-1])
        frame_info = {'frame_name':img_name, 'obj_num':obj_num, 'obj_idx':obj_idx, 
                      'original_height':h, 'original_width':w, 'save_result':self.need_save_results[idx]}

        if current_anno_name in self.annos:
            current_anno = np.array(Image.open(os.path.join(self.anno_root, self.seq_name, current_anno_name)))
            current_anno = cv.resize(current_anno, (new_w, new_h), interpolation=cv.INTER_NEAREST)
            current_anno = current_anno.astype(np.uint8)
            current_anno[current_anno == all_obj_num+1] = 0 # 不使用背景涂鸦
            # current_anno[current_anno == all_obj_num+1] = self.K-1
            Ms = torch.from_numpy(self.All_to_onehot(current_anno[np.newaxis, ...]).copy()).float().permute(1,0,2,3)
        else:
            Ms = None
        return Fs, Ms, frame_info

class Ytb_MO_Test(data.Dataset):
    def __init__(self, root, imset, size=None):
        if imset=='val':
            self.img_dir = os.path.join(root, 'valid', 'JPEGImages')
            self.anno_dir = os.path.join(root, 'valid_scribbles')
            meta_path = os.path.join(root, 'valid', 'meta.json')
        elif imset=='test':
            self.img_dir = os.path.join(root, 'test', 'JPEGImages')
            self.anno_dir = os.path.join(root, 'test_init_scr')
            meta_path = os.path.join(root, 'test', 'meta.json')
        else:
            raise Exception("imset should be either 'val' or 'test'.")
        with open(meta_path) as f:
            self.meta_data = json.load(f)['videos']
        self.seq_list = list(self.meta_data.keys())
        self.resize_size = size
        print('test size:', self.resize_size)

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, idx):
        seq_name = self.seq_list[idx]
        seq_info = self.meta_data[seq_name]['objects']
        obj_names = list(seq_info.keys())
        images = []
        annos = []
        for obj_n in obj_names:
            images += map(lambda x: x+'.jpg', list(seq_info[obj_n]['frames']))
            annos.append(seq_info[obj_n]['frames'][0]+'.bmp')
        images = sorted(list(set(images)))
        annos = sorted(list(set(annos)))
        
        seq_dataset = VOSTest(self.img_dir, self.anno_dir, seq_name, images, annos, resize_size=self.resize_size)
        
        return seq_dataset