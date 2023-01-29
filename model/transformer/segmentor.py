from logging import raiseExceptions
from numpy import short
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import ToCuda


class ConvN(nn.Module):
    def __init__(self, indim, outdim, kernel_size, norm='gn', gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(indim, outdim, kernel_size, padding=kernel_size//2)
        if norm == 'gn':
            self.norm = nn.GroupNorm(gn_groups, outdim)
        elif norm == 'bn':
            self.norm = nn.BatchNorm2d(outdim)
        else:
            print('Use bn or gn in decoder.')
            raise
    
    def forward(self, x):
        return self.norm(self.conv(x))


class Segmentor(nn.Module):
    def __init__(self, in_dim,  # 从Transformer输入特征的维度
                       out_dim, # 输出mask的维度
                       hidden_dim=256,
                       shortcut_dims=[256, 512, 1024],   # Backbone特征的维度
                       align_corners=True,
                       upsample_logits=True,
                       scale_factor = 4,
                       norm='gn',
                       pred_edge=False):  # bn or gn
        super().__init__()
        self.K = 12
        self.upsample_logits = upsample_logits
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.pred_edge = pred_edge

        self.conv_in = ConvN(in_dim, hidden_dim, 1, norm=norm)
        self.conv_16x = ConvN(hidden_dim, hidden_dim, 3, norm=norm)
        self.conv_8x = ConvN(hidden_dim, hidden_dim//2, 3, norm=norm)
        self.conv_4x = ConvN(hidden_dim//2, hidden_dim//2, 3, norm=norm)

        self.adapter_16x = nn.Conv2d(shortcut_dims[-1], hidden_dim, 1)
        self.adapter_8x = nn.Conv2d(shortcut_dims[-2], hidden_dim, 1)
        self.adapter_4x = nn.Conv2d(shortcut_dims[-3], hidden_dim//2, 1)

        self.conv_out = nn.Conv2d(hidden_dim//2, out_dim, 1)
        if self.pred_edge:
            self.edge_predictor = nn.Conv2d(hidden_dim//2, 1, 1)# 目前是按照二分类来计算
        self._init_weight()

    def forward(self, input, shortcuts, valid=None, multi_object_fusion=True):
        '''
        args:
            input: [B*no, C, h, w], input feature from transformer layers
            shortcuts: list of features, from backbone layer1, layer2, layer3
        return:
            [B*no, out_dim, H, W], output mask of each object
        '''
        x = F.relu_(self.conv_in(input))
        x = F.relu_(self.conv_16x(self.adapter_16x(shortcuts[-1]) + x)) # 1/16 hidden_dim

        x = F.interpolate(x, size=shortcuts[-2].size()[-2:], mode='bilinear', align_corners=self.align_corners)
        x = F.relu_(self.conv_8x(self.adapter_8x(shortcuts[-2]) + x))   # 1/8 hidden_dim/2

        x = F.interpolate(x, size=shortcuts[-3].size()[-2:], mode='bilinear', align_corners=self.align_corners)
        x = F.relu_(self.conv_4x(self.adapter_4x(shortcuts[-3]) + x))   # 1/4 hidden_dim/2

        mask_logit = self.conv_out(x)
        edge_logit = self.edge_predictor(x) if self.pred_edge else None

        if self.upsample_logits:
            # 上采样到原图的二分类logits
            mask_logit = F.interpolate(mask_logit, scale_factor=self.scale_factor, mode='bilinear', align_corners=self.align_corners)
            if edge_logit is not None:
                edge_logit = F.interpolate(edge_logit, scale_factor=self.scale_factor, mode='bilinear', align_corners=self.align_corners)
            
        if multi_object_fusion:
            multi_pred_prob, multi_pred_logit, binary_pred_prob = self.fusion_predict(mask_logit, valid)

            return multi_pred_prob, multi_pred_logit, binary_pred_prob, edge_logit
        return mask_logit, edge_logit

    def fusion_predict(self, binary_pred_logit, valid):
        #print(binary_pred_logit.shape)
        batch, no = valid.shape
        n_class, H, W = binary_pred_logit.shape[-3:]
        binary_pred_logit = binary_pred_logit.reshape(batch, no, n_class, H, W)
        binary_pred_prob = F.softmax(binary_pred_logit, dim=2) * valid[:, :, None, None, None]
        multi_pred_prob, multi_pred_logit = self.soft_aggregation(binary_pred_prob[:, :, 0, :, :])
        return multi_pred_prob, multi_pred_logit, binary_pred_prob.reshape(batch * no, n_class, H, W)

    def soft_aggregation(self, binary_pred_prob):
        '''aggregate all masks for each object
        args:
            binary_pred_prob (Tensor) - [batch, no, H, W] foreground probability for each object, size of [B, no, H, W]
        returns:
            prob (Tensor) - [batch, self.K, H, W], 第0维度表示未定义, 第1至self.K-2维度表示物体的概率, self.K-1维度表示背景概率
        '''
        B, no, H, W = binary_pred_prob.shape
        prob = ToCuda(torch.zeros(B, self.K, H, W))
        prob[:, -1] = torch.prod(1 - binary_pred_prob, dim=1) # background probability
        prob[:, 1: no+1] = binary_pred_prob # object probability
        prob = torch.clamp(prob, 1e-7, 1-1e-7)  # TODO: 这里为什么后面还会出现1
        multi_logit = torch.log((prob / (1-prob)))     # [-∞, 0]
        multi_prob = F.softmax(multi_logit, dim=1)
        if torch.any(torch.isnan(multi_prob)):
            raise Exception("Nan in multi_prob!")
        return multi_prob, multi_logit

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)




def build_segmentor(cfg):
    segmentor = Segmentor(cfg.MODEL.HIDDEN_DIM,
                          cfg.MODEL.DECODER.OUT_DIM,
                          hidden_dim=cfg.MODEL.HIDDEN_DIM,
                          shortcut_dims=cfg.MODEL.DECODER.SHORTCUT_DIMS,
                          align_corners=cfg.MODEL.DECODER.ALIGN_CORNER,
                          upsample_logits=cfg.MODEL.DECODER.UPSAMPLE_LOGITS,
                          norm=cfg.MODEL.DECODER.NORM,
                          pred_edge=cfg.MODEL.DECODER.PRED_EDGE)
    
    return segmentor


if __name__ =='__main__':
    segmentor = Segmentor(256, 2,
                        hidden_dim=256,
                        shortcut_dims=[256, 512, 1024],
                        align_corners=False,
                        norm='gn')

    input = torch.randn((5, 256, 24, 24))
    shortcuts = [
        torch.randn((5,256,96,96)),
        torch.randn((5,512,48,48)),
        torch.randn((5,1024,24,24)),
    ]

    out = segmentor(input, shortcuts)
    print(out.shape)