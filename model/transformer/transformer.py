from weakref import ref
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

class WSVOSTransformer(nn.Module):
    def __init__(self, num_layers=2, d_model=256, nhead=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', refine_threshold=[0.95, 0.9],
                 detach_inter_layer=False, using_scribble_mem_for_mask=False):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.detach_inter_layer = detach_inter_layer
        self.using_scribble_mem_for_mask = using_scribble_mem_for_mask

        decoders = []
        for idx in range(num_layers):
            decoders.append(DecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
                                       dropout=dropout, activation=activation, ffn=True))
        self.decoders = nn.ModuleList(decoders)
        
        self.softlabel_attach = None
        self.segmentors = None
        self.scribble_predictor = None
        assert (len(refine_threshold) == num_layers), "Expecting to include the scribble prediction threshold!"
        self.refine_threshold = refine_threshold
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        #if not self.two_stage:
        #    xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        #    constant_(self.reference_points.bias.data, 0.)
        #normal_(self.level_embed)

    def init_memory(self, feat, scribble, pos, hr_embed_obj, hr_embed_bg, 
                    stride, num_objects, use_similarity, local_similarity, feat_l4, 
                    using_initial_bg=True):
        """
        args:
            feat (Tensor) - [B, C, H, W]
            scribble (Tensor) - [B, K, im_h, im_w]
            pos (Tensor) - [1, C, H, W]
        returns:
            attached_memories (List[Tensor]) - layers [HW, B*no, C]
        """
        B, C, H, W = feat.shape
        no = num_objects.max()
        feat_flatten = feat.flatten(-2).permute(2, 0, 1)
        pos_flatten = pos.flatten(-2).permute(2, 0, 1)
        
        output = feat_flatten
        src_memories = []
        for layer in range(self.num_layers):
            # 这个只是执行transformer各层模型的前向传播, 获取query_output & src_memory
            output, src_memory = self.decoders[layer](query=output, memory=None, query_pos=pos_flatten, 
                                                      num_objects=num_objects, layer=layer)
            src_memories.append(src_memory.permute(1, 2, 0).reshape(B * no, C, H, W))# src_memory from different decoder layer
        
        attached_memories = []
        for src_m in src_memories:
            if using_initial_bg:
                attached_m = self.softlabel_attach(src_m, scribble, hr_embed_obj[0:1], hr_embed_bg[0:1], 
                                               stride, num_objects, use_similarity=use_similarity, local_similarity=local_similarity, 
                                               feats_for_similarity=feat_l4)
            else:
                attached_m = self.softlabel_attach(src_m, scribble, hr_embed_obj[0:1], None, 
                                               stride, num_objects, use_similarity=use_similarity, local_similarity=local_similarity, 
                                               feats_for_similarity=feat_l4)
            attached_m_flatten = attached_m.flatten(-2).permute(2, 0, 1)# (HW, B*no, C)
            attached_memories.append(attached_m_flatten)
        return attached_memories

    def forward(self, feat, query_pos, memory, hr_embed_obj, hr_embed_bg, middle_feat,
                num_objects, valid, f_idx):
        """
        args:
            feat (Tensor) - [B, C, H, W]
            query_pos (Tensor) - [1, C, H, W]
            memory (Dict[List[Tensor]]) - (HW, B*no, C)
            hr_embed_obj (Tensor) - (num_layers+1, C)
            middle_feat (List[Tensor]) - feats from layer1, layer2, layer3, [B, C', H', W']
            valid (Tensor) - [B*no]
        returns:
            srcs (List(Tensor)) - length of 1+refine_layers
            src_memories (List(Tensor)) - length of refine_layers, we do not return the src_memory correspoding to the scribble prediction layer
            multi_pred_probs, multi_pred_logits, binary_pred_probs, binary_edge_logits (List(Tensor)) - length of refine_layers
            multi_scribble_prob, binary_scribble_logit (Tensor)
            attached_scribble_memory (Tensor) - flattened tensor [HW, B*no, C]
        """
        B, C, H, W = feat.shape
        feat_flatten = feat.flatten(-2).permute(2, 0, 1)
        pos_flatten = query_pos.flatten(-2).permute(2, 0, 1)
        #print(feat_flatten.shape, pos_flatten.shape)
        # memory 每层都不一样, 每帧也不一样, 有点复杂
        no = num_objects.max()
        middle_feat = [mfeat.unsqueeze(1).repeat(1, no, 1, 1, 1).reshape(B * no,
                       *mfeat.shape[-3:]) for mfeat in middle_feat]
        temporary_memory = [None] * self.num_layers
        output = feat_flatten
        srcs, src_memories, multi_pred_probs, multi_pred_logits, binary_pred_probs, binary_edge_logits = [], [], [], [], [], []
        # multi_scribble_probs, binary_scribble_logits, attached_scribble_memories = [], [], []
        for layer in range(self.num_layers):# 第一层为了降低计算量, query没有复制no次, 第二层及后续层复制no次
            # layer=0 denotes the scribble prediction layer, the other layers denote
            parsed_memory, memory_attn_mask = self.parse_memory(memory, temporary_memory, layer, device=feat.device)
            #output: [HW, B * no, C]
            output, src_memory = self.decoders[layer](query=output, memory=parsed_memory, query_pos=pos_flatten, 
                                                      num_objects=num_objects, layer=layer)
            
            src = output.permute(1, 2, 0).reshape(B * no, C, H, W)
            src_m = src_memory.permute(1, 2, 0).reshape(B * no, C, H, W)# src_m在decoder里面没有复制no次
            if layer==0:
                multi_scribble_prob, binary_scribble_logit = self.scribble_predictor(src, middle_feat, valid, multi_object_fusion=True)
                attached_m = self.attach_label(src_m, multi_scribble_prob, hr_embed_obj, None, layer, num_objects)
                attached_m = attached_m.flatten(-2).permute(2, 0, 1)
                attached_scribble_memory = attached_m
                # attached_scribble_memories.append(attached_m)
                # multi_scribble_probs.append(multi_scribble_prob)
                # binary_scribble_logits.append(binary_scribble_logit)
            else:
                multi_pred_prob, multi_pred_logit, binary_pred_prob, binary_edge_logit = self.segmentors[layer-1](src, middle_feat, 
                                                                                     valid, multi_object_fusion=True)
                attached_m = self.attach_label(src_m, multi_pred_prob, hr_embed_obj, hr_embed_bg, layer, num_objects)
                attached_m = attached_m.flatten(-2).permute(2, 0, 1)
                multi_pred_probs.append(multi_pred_prob)
                multi_pred_logits.append(multi_pred_logit)
                binary_pred_probs.append(binary_pred_prob)
                binary_edge_logits.append(binary_edge_logit)
                src_memories.append(src_m)
            #print('attached_m', attached_m.shape)
            temporary_memory[layer] = attached_m
            srcs.append(src)
            
        return srcs, src_memories, multi_pred_probs, multi_pred_logits, binary_pred_probs, binary_edge_logits, \
               multi_scribble_prob, binary_scribble_logit, attached_scribble_memory

    def attach_label(self, src_m, multi_pred_prob, hr_embed_obj, hr_embed_bg, layer, num_objects):
        """
        args:
            src_m (Tensor) - [B * no, C, H, W] output of self.attn in decoder layer
            multi_pred_prob (Tensor) - [B, K, im_h/4, im_w/4]
            hr_embed_obj (Tensor) - [1, C]
            hr_embed_bg (Tensor) - [1, C]
        returns:
            attached_src (Tensor) - [B*no, C, H, W]
        """
        stride = int(multi_pred_prob.shape[-1] / src_m.shape[-1])
        # TODO: 卡这个阈值应该是没啥用
        multi_pred_prob_clamp = multi_pred_prob * (multi_pred_prob > self.refine_threshold[layer]).to(torch.float32)
        if self.detach_inter_layer:
            multi_pred_prob_clamp = multi_pred_prob_clamp.detach()
        if layer == 0:
            assert hr_embed_bg is None
            attached_src = self.softlabel_attach(src_m, multi_pred_prob_clamp, hr_embed_obj[layer+1: layer+2], 
                                                 None, stride, num_objects)
        else:
            attached_src = self.softlabel_attach(src_m, multi_pred_prob_clamp, hr_embed_obj[layer+1: layer+2], 
                                                 hr_embed_bg[layer+1: layer+2], stride, num_objects)                   
        return attached_src

    def parse_memory(self, memory, temporary_memory, layer, device):
        """
        预测scribble时, memory仅有init&pred_scribble, 预测mask, memory既有scribble又有mask
        returns: memory_single_layer (Tensor) - [sum(HW), B*no, C] 
        """
        memory_single_layer = []
        memory_single_layer.append(memory['scribble'][layer])
        if layer == 0:# layer for predicting scribble
            if memory['pred_scribble'] is not None:
                memory_single_layer.extend(memory['pred_scribble'])
            if 'last_frame_scribble' in memory.keys() and memory['last_frame_scribble'] is not None:
                memory_single_layer.append(memory['last_frame_scribble'])
            memory_single_layer = torch.cat(memory_single_layer, dim=0)
        else:# layer for refinement 
            if self.using_scribble_mem_for_mask:
                if memory['pred_scribble'] is not None:
                    memory_single_layer.extend(memory['pred_scribble'])
            if memory['refined'][layer-1] is not None:
                memory_single_layer.extend(memory['refined'][layer-1])
                memory_single_layer.append(temporary_memory[layer-1])
            if 'last_frame' in memory.keys() and memory['last_frame'][layer-1] is not None:
                memory_single_layer.append(memory['last_frame'][layer-1])
            
            memory_single_layer = torch.cat(memory_single_layer, dim=0)
        return memory_single_layer, None        


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 activation="relu", ffn=True):
        super().__init__()
        self.ffn=ffn
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, query):
        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)
        return query

    def forward(self, query, memory=None, memory_key_padding_mask=None, 
                query_pos=None, num_objects=None, layer=0):
        """
        args:
            query (Tensor) - [HW, B, C] or [HW, B * no, C]
        returns:
            query (Tensor) - [HW, B * no, C] 经过self-attention层和cross-attention层
            current_memory (Tensor) - [HW, B * no, C] 只经过self-attention层
        """
        HW, B, C = query.shape
        #print('query', query.shape, 'query_pos', query_pos.shape)
        q = k = self.with_pos_embed(query, query_pos)
        #print('q', q.shape, 'k', k.shape, 'query', query.shape)
        query2 = self.self_attn(query=q, key=k, value=query)[0]
        #print(query2.shape)
        query = query + self.dropout1(query2)
        query = self.norm1(query)
        if layer == 0:
            no = num_objects.max()
            query = query.unsqueeze(2).repeat(1, 1, no, 1).reshape(HW, B * no, C)
        current_memory = self.with_pos_embed(query, query_pos)# 给memory加上空间位置编码
        if memory is not None:
            query=self.with_pos_embed(query, query_pos)
            #query_2 = query.permute(1,2,0).reshape(B, C, 24, 24).unsqueeze(1).repeat(1,no,1,1,1).reshape(B*no,C,24, 24).flatten(-2).permute(2, 0, 1)
            query2 = self.cross_attn(query=query, key=memory, value=memory, 
                                     key_padding_mask=memory_key_padding_mask)[0]
            query = query + self.dropout2(query2)
            query = self.norm2(query)
        
            if self.ffn:
                query = self.forward_ffn(query)
        return query, current_memory


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", ffn=True):
        super().__init__()
        self.ffn = ffn
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Implementation of Feedforward model
        if self.ffn:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.activation = _get_activation_fn(activation)
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if self.ffn:
            src = self.forward_ffn(src)
        return src


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_transformer(cfg):
    # TODO: 传入参数
    wsvos_transformer = WSVOSTransformer(num_layers=cfg.MODEL.REFINE_LAYERS + 1,
                                         d_model=cfg.MODEL.HIDDEN_DIM,
                                         nhead=cfg.MODEL.TRANSFORMER.NHEAD,
                                         dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
                                         dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
                                         activation=cfg.MODEL.TRANSFORMER.ACTIVATION,
                                         refine_threshold=cfg.MODEL.TRANSFORMER.REFINE_THRESHOLD,
                                         detach_inter_layer=cfg.MODEL.TRANSFORMER.DETACH_INTER_LAYER,
                                         using_scribble_mem_for_mask=cfg.MODEL.TRANSFORMER.USING_SCRIBBLE_MEM_FOR_MASK)
    return wsvos_transformer

if __name__ == '__main__':
    pass