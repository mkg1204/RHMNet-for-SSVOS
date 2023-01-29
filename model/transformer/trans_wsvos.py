import torch
import torch.nn as nn
import copy
from .backbone import build_backbone
from .position_embedding import build_position_embedding, build_frame_enbedding
from .transformer import build_transformer
from .scribble_pooling import build_scribble_pooling, build_softlabel_attaching
from .segmentor import build_segmentor
from .scribble_predictor import build_scribble_predictor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransWSVOS(nn.Module):
    def __init__(self, refine_layers=2, hidden_dim=256, stride=16, backbone=None, softlabel_attach=None,
                       position_embedding=None, frame_embedding=None, transformer=None, segmentor=None, scribble_predictor=None,
                       object_threshold=0.9, detach_inter_frame=False, use_pos_embedding=True, use_similarity=False, 
                       local_similarity=False, with_refine=True, use_hr_level_embedding=True):
        super(TransWSVOS, self).__init__()
        self.counting = 0
        self.K = 12
        self.refine_layers = refine_layers
        self.stride = stride
        self.object_threshold = object_threshold
        self.detach_inter_frame = detach_inter_frame
        self.use_pos_embedding = use_pos_embedding
        self.use_similarity = use_similarity
        self.local_similarity = local_similarity
        self.hr_level_embedding = use_hr_level_embedding
        self.with_refine = with_refine
        self.bottleneck = nn.Sequential(nn.Conv2d(1024, hidden_dim, kernel_size=1),
                                        nn.GroupNorm(32, hidden_dim))
        self.backbone = backbone
        self.softlabel_attach = softlabel_attach
        self.position_embedding = position_embedding
        #self.frame_embedding = frame_embedding
        self.transformer = transformer
        if self.with_refine:
            self.segmentors = _get_clones(segmentor, refine_layers)
        else:
            self.segmentors = nn.ModuleList([segmentor for _ in range(refine_layers)])
        self.scribble_predictor = scribble_predictor
        if self.hr_level_embedding:
            print("*********************** w/ hr_level_embedding ***************************")
            self.hr_embed_obj = nn.Embedding(2 + refine_layers, hidden_dim)# hierachical 0: init_scribble, 1: pred_scribble, others: pred_mask
            self.hr_embed_bg = nn.Embedding(2 + refine_layers, hidden_dim)
        else:
            print("*********************** w/o hr_level_embedding ***************************")
            self.hr_embed_obj = nn.Embedding(1, hidden_dim)
            self.hr_embed_bg = nn.Embedding(1, hidden_dim) 
        self.transformer.softlabel_attach = self.softlabel_attach
        self.transformer.segmentors = self.segmentors
        self.transformer.scribble_predictor = self.scribble_predictor
        self._reset_embedding()

    def _reset_embedding(self):
        nn.init.constant_(self.hr_embed_obj.weight, 1.0)
        nn.init.constant_(self.hr_embed_bg.weight, -1.0)
        nn.init.xavier_uniform_(self.bottleneck[0].weight, gain=1)
        nn.init.constant_(self.bottleneck[0].bias, 0)

    def forward(self, images, init_scribble, num_objects):
        """
        only used for training
        args:
            images (Tensor) - [B, clip_length, 3, im_h, im_w]
            init_scribble (Tensor) - [B, 12, im_h, im_w], self.K=12
            # soft_scirbbles (Tensor) - [[B, clip_length, 12, im_h/stride, im_w/stride]
            num_object (Tensot) - [B], num_objects in every sample
        returns:
            list of list of tensors - [B, K, im_h/4, im_w/4] or [B*no, 2, im_h/4, im_w/4]
            valid (Tensor) - [batch, no]
        """
        if self.counting == 500:
            #print('hr_embed_obj', self.hr_embed_obj.weight[0].mean())
            #print('hr_embed_bg', self.hr_embed_bg.weight[0].mean())
            self.counting = 0
        self.counting += 1
        memory = {}
        batch, clip_length, _, im_h, im_w = images.shape
        no = num_objects.max()
        valid = self.compute_valid(num_objects, no, batch, device=images.device)

        images = images.permute(1, 0, 2, 3, 4)# [clip_length, B, 3, im_h, im_w]
        middle_feats = self.backbone(images.reshape(clip_length * batch, 3, im_h, im_w))# (clip_length*B, C, H, H)
        feats = self.bottleneck(middle_feats[-1])
        
        learned_pos = self.position_embedding(feats)# 训练过程中, 这个位置编码并不会改变
        if not self.use_pos_embedding:
            learned_pos = torch.zeros_like(learned_pos)

        clip_feats, clip_middle_feats = self.parse_feature(feats, middle_feats, batch, clip_length)
        
        hr_embed_obj = self.hr_embed_obj.weight if self.hr_level_embedding else self.hr_embed_obj.weight.repeat(2 + self.refine_layers, 1)
        hr_embed_bg = self.hr_embed_bg.weight if self.hr_level_embedding else self.hr_embed_bg.weight.repeat(2 + self.refine_layers, 1)
        init_memory = self.transformer.init_memory(clip_feats[0], init_scribble, learned_pos, hr_embed_obj,
                                                   hr_embed_bg, self.stride, num_objects, self.use_similarity, 
                                                   self.local_similarity, clip_feats[0])
        memory = {'scribble': init_memory, 'pred_scribble': None, 'refined': [None] * self.refine_layers}
        multi_pred_prob_list, multi_pred_logit_list, binary_pred_prob_list, binary_edge_logit_list = [], [], [], []
        multi_scribble_prob_list, binary_scribble_logit_list = [], []
        for f_idx in range(clip_length):
            srcs, src_memories, multi_pred_probs, multi_pred_logits, binary_pred_probs, binary_edge_logits, \
                multi_scribble_prob, binary_scribble_logit, attached_scribble_memory = self.transformer(
                                                      clip_feats[f_idx], learned_pos, memory, 
                                                      hr_embed_obj, hr_embed_bg, 
                                                      clip_middle_feats[f_idx], num_objects, valid, f_idx)
            memory = self.update_memory(memory, src_memories, multi_pred_probs[-1], num_objects, attached_scribble_memory)# 只用上一帧的最终预测结果更新模型

            multi_pred_prob_list.append(multi_pred_probs)
            multi_pred_logit_list.append(multi_pred_logits)
            binary_pred_prob_list.append(binary_pred_probs)
            binary_edge_logit_list.append(binary_edge_logits)
            multi_scribble_prob_list.append(multi_scribble_prob)
            binary_scribble_logit_list.append(binary_scribble_logit)
        
        return multi_pred_prob_list, multi_pred_logit_list, binary_pred_prob_list, binary_edge_logit_list, valid, \
               multi_scribble_prob_list, binary_scribble_logit_list
        
    def update_memory(self, memory, src_memories, last_pred, num_objects, attached_scribble_memory):
        """
        用上一帧最终预测结果(最后一层预测结果)对上一帧各 decoder layer 计算的src_memory 进行embedding attach, 这个应该仅限refine layers
        对于直接预测 scribble 的layer, 不应该用 mask prediction 进行 attach, 而是应该用 scribble prediction 进行 attach, 初始涂鸦的memory另算, 先不做改动
        args:
            memory (Dict)
            src_memories (List[Tensor]) - src_memory of different decoder layers from last frame (only the layers for mask prediction)
            last_pred (Tensor) - predicted probability of the last frame by the last decoder layer
            attached_scribble_memory (Tensor) - scribble_memory, flattened tensor, [HW, B*no, C]
        """
        if memory['pred_scribble'] is not None:
            memory['pred_scribble'].append(attached_scribble_memory)
        else:
            memory['pred_scribble'] = [attached_scribble_memory]

        stride = int(last_pred.shape[-1] / src_memories[-1].shape[-1])
        for layer, src_m in enumerate(src_memories):# layer=0 denote the first mask prediction layer
            last_pred_clamp = last_pred * (last_pred > self.object_threshold)
            if self.detach_inter_frame:
                last_pred_clamp = last_pred_clamp.detach()
            hr_embed_obj = self.hr_embed_obj.weight if self.hr_level_embedding else self.hr_embed_obj.weight.repeat(2 + self.refine_layers, 1)
            hr_embed_bg = self.hr_embed_bg.weight if self.hr_level_embedding else self.hr_embed_bg.weight.repeat(2 + self.refine_layers, 1)
            attached_m = self.softlabel_attach(src_m, last_pred_clamp, hr_embed_obj[layer+2: layer+3],
                                  hr_embed_bg[layer+2: layer+3], stride, num_objects)
            attached_m_flatten = attached_m.flatten(-2).permute(2, 0, 1)
            # print('attached_m', attached_m.shape)
            if memory['refined'][layer] is not None:
                #memory['refined'][layer] = torch.cat([memory['refined'][layer], attached_m_flatten], dim=0)
                memory['refined'][layer].append(attached_m_flatten)
            else:
                memory['refined'][layer] = [attached_m_flatten]
        return memory
            

    def parse_feature(self, feats, middle_feats, batch, clip_length):
        """
        returns:
            clip_feats (Tensor) - [clip_length, batch, C, H, W]
            clip_middle_feats (List(List(Tensor))) - [batch, C, H, W], 第一层list是clip, 第二层list是
        """
        clip_feats = feats.reshape(clip_length, batch, *feats.shape[-3:])

        middle_feats = [feat.reshape(clip_length, batch, *feat.shape[-3:]) for feat in middle_feats]
        clip_middle_feats = [[feat[frame,...] for feat in middle_feats] for frame in range(clip_length)]
        return clip_feats, clip_middle_feats

    def compute_valid(self, num_objects, no, batch, device):
        valid = torch.zeros(batch, no, device=device)
        for i in range(batch):
            valid[i, : num_objects[i]] = 1.0
        return valid

def build_trans_wsvos(cfg):
    backbone = build_backbone(cfg)
    position_embedding = build_position_embedding(cfg)
    frame_embedding = build_frame_enbedding(cfg)
    transformer = build_transformer(cfg)
    softlabel_attach = build_softlabel_attaching(cfg)
    segmentor = build_segmentor(cfg)
    scribble_predictor = build_scribble_predictor(cfg)
    trans_wsvos = TransWSVOS(refine_layers=cfg.MODEL.REFINE_LAYERS,
                             hidden_dim=cfg.MODEL.HIDDEN_DIM,
                             stride=cfg.MODEL.TOTAL_STRIDE,
                             object_threshold=cfg.MODEL.OBJECT_THRESHOLD,
                             detach_inter_frame=cfg.MODEL.DETACH_INTER_FRAME,
                             use_pos_embedding=cfg.MODEL.TRANSFORMER.USE_POS_EMBEDDING,
                             use_similarity=cfg.MODEL.ATTACH.USE_SIMILARITY,
                             local_similarity=cfg.MODEL.ATTACH.LOCAL_SIMILARITY,
                             with_refine=cfg.MODEL.WITH_REFINE,
                             backbone=backbone, 
                             softlabel_attach=softlabel_attach, 
                             position_embedding=position_embedding, 
                             frame_embedding=None, 
                             transformer=transformer,
                             segmentor=segmentor,
                             scribble_predictor=scribble_predictor,
                             use_hr_level_embedding=cfg.MODEL.USE_HR_LEVEL_EMBEDDING)
    return trans_wsvos