import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .segmentor import ConvN
from utils.helpers import ToCuda

class Scribble_Predictor(nn.Module):
    def __init__(self, in_dim,  # 从Transformer输入特征的维度
                       out_dim, # 输出mask的维度
                       hidden_dim=256,
                       shortcut_dims=[256, 512, 1024],   # Backbone特征的维度
                       align_corners=True,
                       scale_factor = 4,
                       norm='gn',
                       use_focal_loss=True):  # bn or gn
        super().__init__()
        self.K = 12
        self.num_class = out_dim
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.use_focal_loss = use_focal_loss

        self.conv_in = ConvN(in_dim, hidden_dim, 1, norm=norm)
        self.conv_16x = ConvN(hidden_dim, hidden_dim, 3, norm=norm)
        self.conv_8x = ConvN(hidden_dim, hidden_dim//2, 3, norm=norm)
        self.conv_4x = ConvN(hidden_dim//2, hidden_dim//2, 3, norm=norm)

        self.adapter_16x = nn.Conv2d(shortcut_dims[-1], hidden_dim, 1)
        self.adapter_8x = nn.Conv2d(shortcut_dims[-2], hidden_dim, 1)
        self.adapter_4x = nn.Conv2d(shortcut_dims[-3], hidden_dim//2, 1)

        self.conv_out = nn.Conv2d(hidden_dim//2, out_dim, 1)
        self._init_weight()
        #print(self.conv_out.bias.data)

    def forward(self, input, shortcuts, valid=None, multi_object_fusion=True):
        '''
        scribble 也需要做multi_object fusion, 按照现有的 memory 更新策略, 输出结果应该是既需要multi_prob, 又需要 binary logit
        这里只能暂时先忽略掉对背景涂鸦的预测, 对背景涂鸦预测的memory应该有mask memory吗? 或许只应该有scribble memory
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

        scribble_logit = self.conv_out(x)
        if multi_object_fusion:
            multi_scribble_prob, binary_scribble_logit = self.fusion_predict(scribble_logit, valid)
            return multi_scribble_prob, binary_scribble_logit
        return scribble_logit

    def fusion_predict(self, binary_pred_logit, valid):
        batch, no = valid.shape
        n_class, H, W = binary_pred_logit.shape[-3:]
        binary_pred_logit = binary_pred_logit.reshape(batch, no, n_class, H, W)[:, :, 0, :, :]
        binary_pred_prob = F.sigmoid(binary_pred_logit) * valid[:, :, None, None]
        multi_pred_prob, multi_pred_logit = self.soft_aggregation(binary_pred_prob[:, :, :, :])
        return multi_pred_prob, binary_pred_logit.reshape(batch * no, n_class, H, W)
    
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
        if self.use_focal_loss:
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.conv_out.bias.data = torch.ones(self.num_class) * bias_value

def build_scribble_predictor(cfg):
    scribble_predictor = Scribble_Predictor(in_dim=cfg.MODEL.HIDDEN_DIM,
                                            out_dim=cfg.MODEL.SCRIBBLE_PREDICTOR.OUT_DIM,
                                            hidden_dim=cfg.MODEL.HIDDEN_DIM,
                                            shortcut_dims=cfg.MODEL.DECODER.SHORTCUT_DIMS,# share segmentor参数
                                            align_corners=cfg.MODEL.DECODER.ALIGN_CORNER,# share segmentor参数
                                            norm=cfg.MODEL.SCRIBBLE_PREDICTOR.NORM,
                                            use_focal_loss=cfg.TRAIN.LOSS.SCRIBBLE_LOSS.USE_FOCAL_LOSS)
    return scribble_predictor


"""
class Scribble_Predictor(nn.Module):
    # TODO: 还是存在一个监督信息分辨率的问题, 要不然上采样预测结果, 要不然下采样ground-truth
    def __init__(self, in_dim, out_dim, hidden_dim=256, layers=3, norm='gn'):
        super().__init__()
        self.layers = layers
        conv_list = []
        self.conv_in = ConvN(in_dim, hidden_dim, 3, norm=norm)
        for i in range(self.layers-1):
            conv_list.append(ConvN(hidden_dim, hidden_dim, 3, norm))
        self.conv_list = nn.ModuleList(conv_list)
        self.conv_out = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)
        self._init_weight()

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input):
        x = F.relu_(self.conv_in(input))
        for i in range(self.layers-1):
            x = F.relu_(self.conv_list[i](x))

        scribble_logit = self.conv_out(x)
        return scribble_logit


def build_scribble_predictor(cfg):
    scribble_predictor = Scribble_Predictor(in_dim=cfg.MODEL.HIDDEN_DIM,
                                            out_dim=cfg.SCRIBBLE_PREDICTOR.OUT_DIM,
                                            hidden_dim=cfg.MODEL.HIDDEN_DIM,
                                            layers=cfg.MODEL.SCRIBBLE_PREDICTOR.LAYERS,
                                            norm=cfg.MODEL.SCRIBBLE_PREDICTOR.NORM)
    return scribble_predictor
"""
