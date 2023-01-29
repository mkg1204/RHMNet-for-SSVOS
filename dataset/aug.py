import numpy as np
import imgaug.augmenters as iaa
import random
import cv2
from PIL import Image


class Flip(object):
    def __init__(self,rate):
        self.rate = rate
    def __call__(self, images, labels, in_clip=False):
        if not in_clip:
            if random.random() < self.rate:
                for i in range(len(images)):
                    # Random flipping
                    images[i] = np.fliplr(images[i]).copy()  # HWC
                    labels[i] = np.fliplr(labels[i]).copy()  # HW
        else:
            for i in range(len(images)):
                if random.random() < self.rate:
                    # Random flipping
                    images[i] = np.fliplr(images[i]).copy()  # HWC
                    labels[i] = np.fliplr(labels[i]).copy()  # HW
        return images, labels


class RandomSizedCrop(object):
    def __init__(self,scale,crop_size, minimum_area, K=12):
        self.scale = scale
        self.crop_size = crop_size
        self.minimum_area = minimum_area
        self.K = K
    def __call__(self,images,labels):
        scale_factor = random.uniform(self.scale[0],self.scale[1])
        for i in range(len(images)):
            h, w = labels[i].shape
            h, w = (max(self.crop_size[0],int(h * scale_factor)), max(self.crop_size[1],int(w * scale_factor)))
            images[i] = (cv2.resize(images[i], (w, h), interpolation=cv2.INTER_LINEAR))
            labels[i] = Image.fromarray(labels[i]).resize((w, h), resample=Image.NEAREST)
            labels[i] = np.asarray(labels[i], dtype=np.int8)

        ob_loc = ((labels[0] + labels[1] + labels[2]) > 0).astype(np.uint8)
        box = cv2.boundingRect(ob_loc)

        x_min = box[0]
        x_max = box[0] + box[2]
        y_min = box[1]
        y_max = box[1] + box[3]

        images_croped, labels_croped = [None]*len(images), [None]*len(images)
        valid_crop = False
        crop_time = 0
        while not valid_crop:
            if crop_time > 20:
                return None, None
            if x_max - x_min > self.crop_size[1]:
                start_w = random.randint(x_min,x_max - self.crop_size[1])
            elif x_max - x_min == self.crop_size[1]:
                start_w = x_min
            else:
                start_w = random.randint(max(0,x_max-self.crop_size[1]), min(x_min,w - self.crop_size[1]))
            if y_max - y_min > self.crop_size[0]:
                start_h = random.randint(y_min,y_max - self.crop_size[0])
            elif y_max - y_min == self.crop_size[0]:
                start_h = y_min
            else:
                start_h = random.randint(max(0,y_max-self.crop_size[0]), min(y_min,h - self.crop_size[0]))
            end_h = start_h + self.crop_size[0]
            end_w = start_w + self.crop_size[1]

            valid_frames = 0
            for i in range(len(images)):
                start_h = random.randint(start_h-20,start_h+20)
                start_h = max(0,start_h)
                start_h = min(h - self.crop_size[0],start_h)
                start_w = random.randint(start_w-20,start_w+20)
                start_w = max(0,start_w)
                start_w = min(w - self.crop_size[1],start_w)
                end_h = start_h + self.crop_size[0]
                end_w = start_w + self.crop_size[1]

                valid_augmentation = (labels[i][start_h:end_h, start_w:end_w] == (self.K-1)).sum() > self.minimum_area and \
                                    ((labels[i][start_h:end_h, start_w:end_w] > 0).sum() - (labels[i][start_h:end_h, start_w:end_w] == (self.K-1)).sum()) > self.minimum_area
                if not valid_augmentation:
                    break
                valid_frames += 1
                images_croped[i] = images[i][start_h:end_h, start_w:end_w]
                labels_croped[i] = labels[i][start_h:end_h, start_w:end_w]
            if valid_frames == len(images):
                valid_crop = True
            crop_time += 1
        return images_croped,labels_croped


class aug_heavy(object):
    def __init__(self, crop_size, minimum_area=100, in_clip_flip=False, k=12, clip_length=3, bidirection=False):
        self.affinity = iaa.Sequential([
            iaa.Sometimes(
                0.5,
                iaa.Affine(rotate=(-30, 30))
            ),
            iaa.Sometimes(
                0.5,
                iaa.Affine(shear=(-15, 15))
            ),
            iaa.Sometimes(
                0.5,
                iaa.Affine(translate_px={"x": (-15, 15), "y": (-15, 15)})
            ),
            iaa.Sometimes(
                0.5,
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
            ),   
            ], random_order=True)
        self.crop_height, self.crop_width = crop_size
        self.crop = RandomSizedCrop([0.80,1.1],crop_size, minimum_area)
        self.flip = Flip(0.5)
        self.in_clip_flip=in_clip_flip
        self.K = k
        self.clip_length = clip_length
        self.minimum_area = minimum_area
        self.bidirection = bidirection

    def __call__(self,images, labels):
        images,labels = self.flip(images,labels)
        for i in range(len(images)):
            images[i],labels[i] = self.affinity(image = images[i],segmentation_maps = labels[i][np.newaxis,:,:,np.newaxis])
            labels[i] = labels[i][0,:,:,0]
        images,labels = self.crop(images,labels)
        if images is None and labels is None:
            return None, None, None
        init_scrs = [label.copy() for label in labels]
        images = [image/255. for image in images]
        if self.in_clip_flip and images != None and labels != None: #TODO: init_scrs
            images, labels = self.flip(images, labels, in_clip=True)
        return images, init_scrs, labels