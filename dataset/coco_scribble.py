import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils import data
from PIL import Image


class CocoSribble_MO_Train(data.Dataset):
    def __init__(self, img_dir, anno_dir, scribble_json_path,
                 clip_length=3, random_flip=True, transform_mode='perspective', affine_range=[8, 10], 
                 minimum_area=100, mem_bg=True, additional_bg=True, max_object_num=5, crop_size=[384, 384],
                 dilate_kernel_size=19, ds_stride=4):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.K = 12                 # max 10 objects + background + unlabeled region
        self.max_object_num = max_object_num
        self.clip_length = clip_length
        with open(scribble_json_path) as f:
            data = json.load(f)
        self.image_list = data['images']
        self.image_num = len(self.image_list)

        self.random_flip = random_flip          
        self.transform_mode = transform_mode    
        self.affine_range = affine_range        # [1st frame, 2nd frame, 3rd frame, ...]
        self.minimum_area = minimum_area        
        self.mem_bg = mem_bg                    
        self.additional_bg = additional_bg      # use unused foreground scribbles as background scribbles to cal loss

        self.crop_height, self.crop_width = crop_size
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                             (dilate_kernel_size, dilate_kernel_size))
        self.ds_stride = ds_stride

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406])
        self.std = torch.FloatTensor([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_list)
            
    def random_offset(self, h, w, zero_initial=False, affine_range=8):
        if zero_initial == False:
            interval_left, interval_right = -w, w
            interval_bottom, interval_top = -h, h
        else:
            interval_left, interval_right = 0, w
            interval_bottom, interval_top = 0, h
        tx1 = random.randint(interval_left // affine_range, interval_right // affine_range)
        ty1 = random.randint(interval_bottom // affine_range, interval_top // affine_range)
        tx2 = random.randint(interval_left // affine_range, interval_right // affine_range)
        ty2 = random.randint(interval_bottom // affine_range, interval_top // affine_range)
        tx3 = random.randint(interval_left // affine_range, interval_right // affine_range)
        ty3 = random.randint(interval_bottom // affine_range, interval_top // affine_range)
        tx4 = random.randint(interval_left // affine_range, interval_right // affine_range)
        ty4 = random.randint(interval_bottom // affine_range, interval_top // affine_range)
        return np.float32([[tx1, ty1], [tx2, ty2], [tx3, ty3], [tx4, ty4]])
            
    def Augmentation(self, image, label):
        """label (np.array) - [im_h, im_w]"""
        # Scaling
        h, w = label.shape
        if w<h:
            factor = 480/w
            image = cv2.resize(image, (480, int(factor*h)), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (480, int(factor*h)), interpolation=cv2.INTER_NEAREST)
        else:
            factor = 480/h
            image = cv2.resize(image, (int(factor*w), 480), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (int(factor*w), 480), interpolation=cv2.INTER_NEAREST)
        
        # Random flipping
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        
        h, w = label.shape

        # Affine or perspective transformation
        dst_points = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]])
        
        image_clip = [image]
        label_clip = [label]
        for i in range(self.clip_length-1):
            augmented_image = image.copy()
            augmented_label = label.copy()
            if i == 0:
                offset = self.random_offset(h, w, zero_initial=False, affine_range=self.affine_range[i])
                src_points = dst_points + offset
                last_offset = offset
            else:
                offset = self.random_offset(h, w, zero_initial=True, affine_range=self.affine_range[i])
                accumulate_offset = last_offset + offset * last_offset / (np.abs(last_offset) + 1e-8)
                src_points = dst_points + accumulate_offset
                last_offset = accumulate_offset

            if self.transform_mode == 'perspective':
                H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 10)
                augmented_image = cv2.warpPerspective(augmented_image, H, (w, h), flags=cv2.INTER_LINEAR)
                augmented_label = cv2.warpPerspective(augmented_label, H, (w, h), flags=cv2.INTER_NEAREST)
            elif self.transform_mode == 'affine':
                H, _ = cv2.estimateAffine2D(src_points, dst_points)
                augmented_image = cv2.warpAffine(augmented_image, H, (w, h), flags=cv2.INTER_LINEAR)
                augmented_label = cv2.warpAffine(augmented_label, H, (w, h), flags=cv2.INTER_NEAREST)
            image_clip.append(augmented_image)
            label_clip.append(augmented_label)
                 

        scr_loc = (np.stack(label_clip, axis=0).sum(axis=0) > 0).astype(np.uint8)

        box = cv2.boundingRect(scr_loc)
        x_min, x_max, y_min, y_max = box[0], box[0]+box[2], box[1], box[1]+box[3]

        valid_crop = False
        crop_time = 0
        while valid_crop == False:
            if crop_time > 20:
                return None, None, None, None
            if x_max - x_min > self.crop_width:
                start_w = random.randint(x_min, x_max-self.crop_width)
            elif x_max - x_min == self.crop_width:
                start_w = x_min
            else:
                start_w = random.randint(max(0, x_max-self.crop_width), min(x_min, w-self.crop_width))
            if y_max - y_min > self.crop_height:
                start_h = random.randint(y_min, y_max-self.crop_height)
            elif y_max - y_min == self.crop_height:
                start_h = y_min
            else:
                start_h = random.randint(max(0, y_max-self.crop_height), min(y_min, h-self.crop_height))
            end_h = start_h + self.crop_height
            end_w = start_w + self.crop_width
            
            valid_augmentation = [(label[start_h:end_h, start_w:end_w] == (self.K-1)).sum() > self.minimum_area and \
                                  ((label[start_h:end_h, start_w:end_w] > 0).sum() - (label[start_h:end_h, start_w:end_w] == (self.K-1)).sum()) > self.minimum_area
                                    for label in label_clip]
            if np.float32(valid_augmentation).sum() == self.clip_length:
                valid_crop = True
            crop_time += 1

        image_clip = [image[start_h:end_h, start_w:end_w] for image in image_clip]
        label_clip = [label[start_h:end_h, start_w:end_w] for label in label_clip]
        init_scr = label_clip[0]
        calculate_smooth = True

        image_clip = [image / 255. for image in image_clip]

        if not self.mem_bg:
            init_scr[init_scr == self.K-1] = 0

        if self.random_flip:
            for i in range(1, self.clip_length):    
                if random.random() < 0.5:
                    image_clip[i] = np.fliplr(image_clip[i]).copy()
                    label_clip[i] = np.fliplr(label_clip[i]).copy()

        return image_clip, init_scr, label_clip, calculate_smooth

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M

    def All_to_onehot(self, masks):
        """masks (np.array) - (clip_length, img_h, img_w)"""
        # masks.shape = [clip_length, H, W]
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:, n] = self.To_onehot(masks[n])
        return Ms
    
    def mask_process(self, mask, f, num_object, ob_list, init_scr=None):
        n = num_object
        mask_ = np.zeros(mask.shape).astype(np.uint8)
        if init_scr is not None:
            init_scr_ = np.zeros(init_scr.shape).astype(np.uint8)
        
            sample_range = self.K-1
            for i in range(1, sample_range):
                if np.sum(init_scr == i) > self.minimum_area:
                    n += 1
                    ob_list.append(i)
            if n > self.max_object_num:
                n = self.max_object_num
                ob_list = random.sample(ob_list, n)

        for i, l in enumerate(ob_list):
            mask_[mask == l] = i+1
            if init_scr is not None:
                init_scr_[init_scr == l] = i+1

        if self.additional_bg:
            other_mask = np.zeros_like(mask).astype(np.bool)
            for i in range(1, self.K-1):
                other_mask = other_mask + (mask == i) if i not in ob_list else other_mask
            mask_[other_mask] = self.K-1
        mask_[mask == self.K-1] = self.K-1
        
        if init_scr is not None:
            init_scr_[init_scr == self.K-1] = self.K-1
            return mask_, n, ob_list, init_scr_
        return mask_, n, ob_list


    def get_sample(self, index):
        images = self.image_list[index]
        image_name = images['file_name']
        anno_name = images['anno_name']
        n_objects = images['num_objects']

        img = Image.open(os.path.join(self.img_dir, image_name))
        frame = np.array(img.convert('RGB'))
        anno = np.array(Image.open(os.path.join(self.anno_dir, anno_name)))
        bg_anno = (anno == n_objects)
        anno[bg_anno] = 0
        anno[anno >= (self.K-1)] = 0
        anno[bg_anno] = self.K-1

        info = {}
        info['name'] = image_name
        info['num_frames'] = self.clip_length

        N_frames = np.empty((self.clip_length, self.crop_height, self.crop_width, 3), dtype=np.float32)
        N_masks = np.empty((self.clip_length, self.crop_height, self.crop_width), dtype=np.uint8)
        frames_, init_scr, masks_, cal_smooth = self.Augmentation(frame, anno)

        info['cal_smooth'] = cal_smooth
        if frames_ is None and masks_ is None:
            return None, None, None, None, None, None, 0, info
        
        num_object = 0
        ob_list = []
        for f in range(self.clip_length):
            if f == 0:
                tmp_mask, num_object, ob_list, init_scr = self.mask_process(masks_[f], f, num_object, ob_list, init_scr)
            else:
                tmp_mask, num_object, ob_list = self.mask_process(masks_[f], f, num_object, ob_list)
            N_frames[f] = frames_[f]# (clip_length, img_h, img_w, 3)
            N_masks[f] = tmp_mask# (clip_length, img_h, img_w)

        # N_frames: [clip_length, H, W, 3] -- > [clip_length, 3, H, W]
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (0, 3, 1, 2)).copy()).float()
        Gs = torch.sum(Fs * torch.tensor([0.299, 0.587, 0.114])[None, :, None, None], dim=1)
        # normalization
        Fs = (Fs - self.mean.reshape(1, 3, 1, 1)) / self.std.reshape(1, 3, 1, 1)
        
        masks = self.All_to_onehot(N_masks) # (K, clip_length, h, w)
        init_scr = self.To_onehot(init_scr) # (K, h, w)

        ignore_regions = np.zeros((self.K, self.clip_length, self.crop_height, self.crop_width), dtype=np.uint8)
        for f in range(self.clip_length):
            for o in range(num_object):
                ignore_regions[o+1, f] = ((cv2.dilate(masks[o+1, f], self.dilate_kernel) + masks[o+1, f]) > 0).astype(np.uint8) - masks[o+1, f]
        ds_ignore_regions = self.downsample_label(ignore_regions)
        ds_masks = self.downsample_label(masks)

        Ds_Ms = torch.from_numpy(ds_masks.copy()).float().permute(1, 0, 2, 3)
        Ignore_R = torch.from_numpy(ds_ignore_regions.copy()).float().permute(1, 0, 2, 3)
        assert(torch.all(Ignore_R >=0) and torch.all(Ignore_R <= 1)), 'Expecting the value is either 0 or 1.'
        Ms = torch.from_numpy(masks.copy()).float().permute(1, 0, 2, 3)# (self.K, clip_length, H, W) --> (clip_length, self.K, H, W)
        Init_Scr = torch.from_numpy(init_scr.copy()).float()

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
            try:
                Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R, num_object, info = self.get_sample(index)
            except Exception as e:
                print(e)
                num_object = 0

            if num_object > 0:
                valid_sample = True
            else:
                index = random.randint(0, self.image_num-1)

        return Fs, Gs, Init_Scr, Ms, Ds_Ms, Ignore_R, num_object, info