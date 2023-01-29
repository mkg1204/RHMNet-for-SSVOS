from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import cv2 as cv
import os


class ScribblePooling(nn.Module):
    def __init__(self, max_len):
        super(ScribblePooling, self).__init__()
        self.max_len = max_len
    
    def forward(self, feats, label, num_objects):
        '''
        param:
            feats: [B, C, h, w], feature of images from backbone
            label: [B, K, h, w], downsampled labels of these images, have same size with feats
            num_objects: [B], number of objects in each image.
        return:
            obj, bg, other  are all dict with keys: src, mask
            obj:
                src: [B*no, self.max_len, C], scribble feature for each object
                mask: [B*no, self.max_len], 1 for meaningless position
            bg:
                src: [B, self.max_len, C], scribble feature of background for each image 
                mask: [B, self.max_len], 1 for meaningless position
            obj:
                src: [B*no, self.max_len*no, C], scribble feature of other object and background for each object
                mask: [B*no, self.max_len*no], 1 for meaningless position
        '''
        assert feats.shape[-1] == label.shape[-1] and feats.shape[-2] == label.shape[-2]

        no = num_objects.max()
        B, C, H, W = feats.shape

        # 特征初始为零向量，初始mask为1
        src_obj = torch.zeros([B, no, self.max_len, C])  # 每个目标的特征
        mask_obj = torch.ones([B, no, self.max_len])
        src_bg = torch.zeros([B, self.max_len, C])  # 每张图片背景的特征
        mask_bg = torch.ones([B, self.max_len])
        src_other = torch.zeros([B, no, self.max_len*no, C]) # 相对于每个目标，其他目标和背景的特征
        mask_other = torch.ones([B, no, self.max_len*no])

        for b in range(B):
            # 提取每个目标的特征
            for o in range(1, no+1):
                if o > num_objects[b]:  # 对无意义的目标不做处理
                    continue
                else:
                    extracted_feat = feats[b, :, label[b, o] == 1].permute(1,0)  # [n, C]
                    extracted_len = extracted_feat.shape[0]
                    if extracted_len > self.max_len:    # 如果特征数超过max_len，从中随机抽取max_len个
                        index = torch.LongTensor(random.sample(range(extracted_len), self.max_len))
                        extracted_feat = torch.index_select(extracted_feat, dim=0, index=index)
                    src_obj[b, o-1, :min(extracted_len, self.max_len)] = extracted_feat
                    mask_obj[b, o-1, :min(extracted_len, self.max_len)] = 0
            # 提取每张图片背景涂鸦的特征
            extracted_feat = feats[b, :, label[b, -1] == 1].permute(1,0)
            extracted_len = extracted_feat.shape[0]
            if extracted_len > self.max_len:
                index = torch.LongTensor(random.sample(range(extracted_len), self.max_len))
                extracted_feat = torch.index_select(extracted_feat, dim=0, index=index)
            src_bg[b, :min(extracted_len, self.max_len)] = extracted_feat
            mask_bg[b, :min(extracted_len, self.max_len)] = 0

        # 对每个目标，将其他目标的涂鸦和背景涂鸦放在一起
        for b in range(B):
            for o in range(no):
                other_feat = torch.cat([src_obj[b, 0:o], src_obj[b, o+1:], src_bg[b:b+1]], dim=0).reshape(-1, C)
                other_mask = torch.cat([mask_obj[b, 0:o], mask_obj[b, o+1:], mask_bg[b:b+1]], dim=0).reshape(-1)
                src_other[b, o] = other_feat
                mask_other[b, o] = other_mask

        obj = {'src': src_obj.reshape(B*no, self.max_len, C), 'mask': mask_obj.reshape(B*no, self.max_len)}
        bg = {'src': src_bg, 'mask': mask_bg}
        other = {'src': src_other.reshape(B*no, self.max_len*no, C), 'mask': mask_other.reshape(B*no, self.max_len*no)}

        return obj, bg, other

class SoftLabelAttach(nn.Module):
    def __init__(self, stride, use_bg_ptt=True, threshold_mean_feat = 0.211, threshold_soft_label = 0.15):
        super(SoftLabelAttach, self).__init__()
        self.stride = stride
        self.use_bg_ptt = use_bg_ptt
        self.threshold_mean_feat = threshold_mean_feat
        self.threshold_soft_label = threshold_soft_label
        #self.ii=0

    def forward(self, feats_to_attach, soft_labels, embed_obj, embed_bg, stride,
                num_objects, use_similarity=False, local_similarity=False, feats_for_similarity=None):
        """
        args:
            feats_to_attach (Tensor) - [B, C, h, w] or [B*no, C, h, w], feature to put into memory
            feats_for_similarity (Tensor) - [B, C, h, w] or [B*no, C, h, w], feature of images from backbone, 不能带梯度
            soft_labels (Tensor) - [B, K, h, w], downsampled labels of these images, have same size with feats
            embed_obj (Tensor) - [1, C]
            embed_bg (Tensor) - [1, C]
            num_objects: [B], number of objects in each image.
        returns:
            attached_feats (Tensor) - [B*no, C, h, w]
        """
        # 第一帧有stride, B, 后续帧, 也有stride, B * no
        no = num_objects.max()
        B, C, H, W = feats_to_attach.shape
        B_, K_, H_, W_ = soft_labels.shape
        #scribbles = soft_labels# 仅用于可视化
        if not (H == H_ and W == W_):
            assert H * stride == H_ and W * stride == W_
        if use_similarity:# two_stage_prediction 分支之前都是只有初始化才用similariy计算soft_label
            B__, _, H__, W__ = feats_for_similarity.shape
            assert (B_ == B__ and H__ * stride == H_ and W__ * stride == W_)
            if local_similarity:
                soft_labels = self.local_similarity_downsample_label(soft_labels, feats_for_similarity, stride, no)
                # print('local')
            else:
                soft_labels = self.similarity_downsample_label(soft_labels, feats_for_similarity, stride, no)
                # print('global')
            #self.visualize(images, soft_labels, scribbles, no)# 仅用于可视化
        else:
            soft_labels = self.soft_downsample_label(soft_labels, stride)
        if B == B_:# 进行attach之前需要先复制特征
            feats_to_attach = feats_to_attach.unsqueeze(1).repeat(1, no, 1, 1, 1).reshape(-1, C, H, W)
        else:
            assert B == B_ * no
        
        obj_soft_labels = soft_labels[:, 1:1+no, :, :]
        extended_obj_embed = (obj_soft_labels[:, :, None, :, :] * embed_obj[:, None, :, None, None]).reshape(B_*no, C, H, W)
        
        if embed_bg is not None:
            bg_soft_labels = soft_labels[:, -1:, :, :]
            allbg_soft_labels = torch.stack([torch.cat([obj_soft_labels[:, torch.arange(0, no)!=i, :, :],
                                       bg_soft_labels], dim=1).sum(dim=1) for i in range(0, no)], dim=1)
            extended_bg_embed = (allbg_soft_labels[:, :, None, :, :] * embed_bg[:, None, :, None, None]).reshape(B_*no, C, H, W)
        if self.use_bg_ptt and embed_bg is not None:
            return feats_to_attach + extended_obj_embed + extended_bg_embed
        else:
            return feats_to_attach + extended_obj_embed
    
    def soft_downsample_label(self, labels, stride):
        '''
        args:
            labels (Tensor) - [B, K, im_h, im_w] labels need to downsample
            stride (int) dowmsample stride
        return:
            soft_downsampled_label (Tensor) - [B, K, im_h/stride, im_w/stride]
        '''
        B, K, H, W = labels.shape
        assert H % stride == 0 and W % stride == 0
        h, w = H//stride, W//stride
        soft_labels = labels.reshape(B, K, h, stride, w, stride)  # [B, K, H, W] ===> [B, K, h, stride, w, stride]
        soft_labels = soft_labels.permute(0,1,2,4,3,5).reshape(B, K, h, w, -1)    # ===> [B, K, h, w, stride, stride] ===> [B, K, h, w, stride*stride]
        soft_labels = soft_labels.sum(dim=-1) / (stride ** 2) # ===> [B, K, h, w]
        return soft_labels
    
    def similarity_downsample_label(self, labels, feats, stride, no):
        '''
        args:
            labels (Tensor) - [B, K, im_h, im_w] labels need to downsample
            feats (Tensor) - [B, C, h, w] feats from backbone layer3
        '''
        B, _, h, w = feats.shape
        _, K, _, _ = labels.shape
        soft_labels = self.soft_downsample_label(labels, stride)    # (B, K, h, w)
        similarity_labels = torch.zeros_like(soft_labels).to(soft_labels.device)
        for o in range(1, no+2):
            if o == no+1:
                o = K-1 # 处理背景
            soft_labels_o = soft_labels[:, o: o+1]    # (B, 1, h, w)
            soft_labels_o_ =  soft_labels_o.clone()# 每个图像块内的涂鸦比率
            soft_labels_o_binary = (soft_labels_o > 0).to(torch.float) # 用于筛选属于同一涂鸦的点计算相似度
            soft_labels_o_[soft_labels_o_ < self.threshold_mean_feat] = 0    # 用于卡0.211的阈值计算平均特征
            for b in range(B):  # 如果所有点都被阈值卡掉，保留值最大的点, 对于padding的物体, tmp的所有值都等于tmp.max()
                if soft_labels_o_[b].sum() == 0:
                    tmp = soft_labels_o[b].clone()
                    tmp[tmp < tmp.max()] = 0
                    soft_labels_o_[b] = tmp
            with torch.no_grad():
                mean_feats = (feats * soft_labels_o_).flatten(2).sum(dim=-1) / (soft_labels_o_.flatten(2).sum(dim=-1) + 1e-8)# (B, C) 涂鸦比率较高的点参与平均特征计算
                similarity = F.cosine_similarity(mean_feats.unsqueeze(-1), (feats * soft_labels_o_binary).flatten(2)).reshape(B, h, w)# (B, h, w) 有涂鸦的点都计算参与相似度计算
            #print('similarity', similarity.shape, 'soft_labels', soft_labels.shape)
            #print((soft_labels[:,o:o+1].squeeze() < self.threshold_soft_label).shape)
            similarity[soft_labels[:, o] < self.threshold_soft_label] = 0    # (B, h, w) 过滤掉涂鸦比率太低的点
            similarity_labels[:,o] = similarity
        return similarity_labels    # (B, K, h, w)

    def local_similarity_downsample_label(self, labels, feats, stride, no):
        B, _, h, w = feats.shape
        _, K, _, _ = labels.shape
        soft_labels = self.soft_downsample_label(labels, stride)    # (B, K, h, w)
        similarity_labels = torch.zeros_like(soft_labels).to(soft_labels.device)
        for o in range(1, no+2):
            if o == no+1:
                o = K-1 # 处理背景
            soft_labels_o = soft_labels[:, o: o+1]    # (B, 1, h, w)
            soft_labels_o_binary = (soft_labels_o > self.threshold_soft_label).to(torch.float)
            for b in range(B):  # 如果所有点都被阈值卡掉，保留值最大的点, 对于padding的物体, tmp的所有值都等于tmp.max()
                if soft_labels_o_binary[b].sum() == 0:
                    tmp = soft_labels_o[b].clone()
                    tmp[tmp < tmp.max()] = 0.
                    tmp[tmp != 0] = 1.
                    soft_labels_o_binary[b] = tmp
            with torch.no_grad():
                feats_o = feats * soft_labels_o_binary # (B, C, h, w)
                unfold_feats_o = self.unfold_wo_center(feats_o, radius=5, dilation=1) # (B, C, 24, h, w)
                similarity = F.cosine_similarity(feats_o.unsqueeze(2), unfold_feats_o)# (B, 24, h, w) 过滤后的点计算相似度
            similarity = similarity.max(dim=1)[0]  # (B, h, w)
            similarity_labels[:, o] = similarity
        return similarity_labels    # (B, K, h, w)

    def unfold_wo_center(self, x, radius, dilation):
        """radius实际上是kernel size"""
        B, C, H, W = x.shape
        padding = (radius + (dilation -1) * (radius - 1)) // 2  # 计算边缘处所需padding的大小

        unfolded_x = F.unfold(x, kernel_size=radius, padding=padding, dilation=dilation)    # unfold    [B, C*R, H*W], R为region内包含像素的个数
        unfolded_x = unfolded_x.reshape(B, C, -1, H, W) # [B, C*R, H*W] ===> [B, C, R, H, W]

        size = radius ** 2
        unfolded_x = torch.cat((unfolded_x[:, :, :size//2], unfolded_x[:, :, size//2+1:]), dim=2) # remove center pixels    [B, C, R-1, H, W]

        return unfolded_x

    def visualize(self, images, labels, scribbles, no):
        colors = np.array([[0, 0, 255],[0, 255, 0],[0, 255, 255],[255, 0, 0],[255, 0, 255],[255, 255, 0]])
        mean = torch.FloatTensor([0.485, 0.456, 0.406]).reshape(3,1,1)
        std = torch.FloatTensor([0.229, 0.224, 0.225]).reshape(3,1,1)
        image = images[0,0,...].cpu()
        h, w = images.shape[-2:]
        unnormed_image = image * std + mean   # [B, L, 3, H, W]
        np_images = (unnormed_image.permute(1,2,0).cpu().numpy()*255.).astype(np.uint8)
        similarities = labels.cpu().numpy()
        for i in range(1, no+1):
            similarity = similarities[0, i, ...]
            similarity = similarity[:,:,np.newaxis,np.newaxis].repeat(16,axis=-1).repeat(16,axis=-2).transpose(0,2,1,3).reshape(h,w)[:,:,np.newaxis].repeat(3,axis=-1)
            scribble = scribbles.cpu().numpy()[0, i, ...][:,:,np.newaxis].repeat(3,axis=-1)
            
            img_similarity = np_images.copy()
            
            img_similarity = img_similarity * (1-similarity) + colors[i] * similarity
            img_scribble = np_images.copy()
            img_scribble = img_scribble * (1-scribble) + colors[i] * scribble

            img = np.concatenate([img_scribble, img_similarity], axis=1)

            cv.imwrite(os.path.join('./test_vis/validate', '{:d}test.jpg'.format(self.ii)), img)
            self.ii += 1


def build_scribble_pooling(cfg):
    return ScribblePooling(max_len=cfg.MODEL.MAX_CAPACITY)

def build_softlabel_attaching(cfg):
    return SoftLabelAttach(stride=cfg.MODEL.TOTAL_STRIDE,
                           use_bg_ptt=cfg.MODEL.BG_PROTOTYPE,
                           threshold_mean_feat = cfg.MODEL.ATTACH.THRESHOLD_MEAN_FEAT,
                           threshold_soft_label = cfg.MODEL.ATTACH.THRESHOLD_SOFT_LABEL)

if __name__ == '__main__':
    M = SoftLabelAttach(16)
    feats = torch.randn((5,1024,24,24))
    labels = torch.randint(0,1,(5,12,384,384))
    no = 5
    similarity_labels = M.similarity_downsample_label(labels, feats, 16, 5)
    print(similarity_labels.shape)
    # M = ScribblePooling(100)
    # feats = torch.randn((5,256,24,24))
    # label = torch.randint(0, 2, (5,12,24,24))
    # num_objects = torch.tensor([2,2,1,3,5])
    # obj, bg, other = M(feats, label, num_objects)
    # print('obj_src:', obj['src'].shape)
    # print('obj_mask:', obj['mask'].shape)
    # print('bg_src:', bg['src'].shape)
    # print('bg_mask:', bg['mask'].shape)
    # print('other_src:', other['src'].shape)
    # print('other_mask:', other['mask'].shape)