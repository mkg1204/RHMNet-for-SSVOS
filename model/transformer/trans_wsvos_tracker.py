import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import ToCuda, pad_divide_by

class Trans_WSVOS_Tracker(nn.Module):
    def __init__(self, net, using_initial_bg=True, require_last=False, require_last_scribble=False, max_memory_length=100, 
                 update_frequency=12, scribble_update_frequency=12, upsample_logits=True):
        super(Trans_WSVOS_Tracker, self).__init__()
        self.net = net
        self.using_initial_bg = using_initial_bg

        self.require_last = require_last
        self.require_last_scribble = require_last_scribble
        self.max_memory_length = max_memory_length
        self.update_frequency = update_frequency
        self.scribble_update_frequency = scribble_update_frequency
        self.upsample_logits = upsample_logits
        self.return_binary_probs = False

    def initialize(self, image, scribble, num_objects, frame_num=0, test_mode='scribble'):
        """
        args:
            image (Tensor) - [batch or clip_length (1), 3, im_h, im_w]
            scribble (Tensor) - [batch or clip_length (1), K, im_h, im_w]

        """
        batch, _, self.im_h, self.im_w = image.shape
        (image, scribble), self.pad = pad_divide_by([image, scribble], self.net.stride, (self.im_h, self.im_w)) # padding
        self.valid = self.net.compute_valid(num_objects, num_objects.max(), batch, device=image.device)
        
        assert (batch==1), "Expecting the batch size for testing equals 1!"
        self.num_objects = num_objects.item()
        middle_feats = self.net.backbone(image)
        feat = self.net.bottleneck(middle_feats[-1])
        _, self.C, self.H, self.W = feat.shape
        learned_pos = self.net.position_embedding(feat)
        init_memory = self.net.transformer.init_memory(feat, scribble, learned_pos, self.net.hr_embed_obj.weight,
                                                       self.net.hr_embed_bg.weight, self.net.stride, num_objects, 
                                                       self.net.use_similarity, self.net.local_similarity, feat, 
                                                       using_initial_bg=self.using_initial_bg)

        self.memory = {'scribble': init_memory, 
                       'pred_scribble': None,
                       'last_frame_scribble': None,
                       'refined': [None] * self.net.refine_layers, 
                       'last_frame': [None] * self.net.refine_layers}
        self.next_recons_multi_prob = None

        
    def track(self, image, num_objects, frame_num):
        '''
        param:
            image:[batch, 3, im_h, im_w]
            frame_num: int
            init_info: [1, 1, 12, im_h, im_w] scribble or mask for first frame, only used when frame_num == 0
            prototypes_obj: [no, C, 1, 1] prototype of objects, only used when frame_num > 0
            prototypes_bg: [no, C, 1, 1] prototype of background, only used when frame_num > 0
            test_mode: str 'scribble' or 'mask', determine scribble pooling or mask pooling when init prototype
        return:

        '''
        batch = image.shape[0]
        (image,), self.pad = pad_divide_by([image,], self.net.stride, (self.im_h, self.im_w)) # padding
        middle_feats = self.net.backbone(image)
        feat = self.net.bottleneck(middle_feats[-1])
        learned_pos = self.net.position_embedding(feat)
        
        hr_embed_obj = self.net.hr_embed_obj.weight if self.net.hr_level_embedding else self.net.hr_embed_obj.weight.repeat(2 + self.net.refine_layers, 1)
        hr_embed_bg = self.net.hr_embed_bg.weight if self.net.hr_level_embedding else self.net.hr_embed_bg.weight.repeat(2 + self.net.refine_layers, 1)

        srcs, src_memories, multi_pred_probs, multi_pred_logits, binary_pred_probs, binary_edge_logits, \
        multi_scribble_prob, binary_scribble_logit, attached_scribble_memory = self.net.transformer(feat, learned_pos, self.memory, 
                                                      hr_embed_obj, hr_embed_bg, 
                                                      middle_feats, num_objects, self.valid, frame_num)
        
        
        if not self.upsample_logits:
            multi_pred_probs_up = [F.interpolate(prob, size=tuple(image.shape[-2:]), mode='bilinear', align_corners=True) for prob in multi_pred_probs]
            binary_pred_probs_up = [F.interpolate(prob, size=tuple(image.shape[-2:]), mode='bilinear', align_corners=True) for prob in binary_pred_probs]
        else:
            multi_pred_probs_up = multi_pred_probs
            binary_pred_probs_up = binary_pred_probs
        multi_pred_probs_depadding = [self.depadding((prob,), self.pad)[0] for prob in multi_pred_probs_up]
        binary_pred_probs_depadding = [self.depadding((prob,), self.pad)[0] for prob in binary_pred_probs_up]
        scores = torch.ones(num_objects)

        #up sample scribble prob
        multi_scribble_prob_up = F.interpolate(multi_scribble_prob, size=tuple(image.shape[-2:]), mode='bilinear', align_corners=True)
        multi_scribble_prob_depadding = self.depadding((multi_scribble_prob_up,), self.pad)[0]

        self.memory = self.update_memory(self.memory, src_memories, multi_pred_probs[-1], num_objects,
                                         attached_scribble_memory, frame_num, scores)

        if self.return_binary_probs:
            return multi_pred_probs_depadding[-1], binary_pred_probs_depadding[-1], multi_scribble_prob_depadding, scores
        else:
            return multi_pred_probs_depadding[-1], multi_scribble_prob_depadding, scores
        

    def update_memory(self, memory, src_memories, last_pred, num_objects, scribble_memory, frame_num, scores):
        # update scribble memory
        if frame_num % self.scribble_update_frequency == 0 and torch.all(scores >= 0.4):
            if memory['pred_scribble'] is not None:
                memory['pred_scribble'].append(scribble_memory)
            else:
                memory['pred_scribble'] = [scribble_memory]
        elif self.require_last_scribble:
            self.memory['last_frame_scribble'] = scribble_memory
        if len(memory['pred_scribble']) > int(self.max_memory_length):
        # if len(memory['pred_scribble']) > int(self.max_memory_length / num_objects.item()):
            memory['pred_scribble'].pop(1)

        hr_embed_obj = self.net.hr_embed_obj.weight if self.net.hr_level_embedding else self.net.hr_embed_obj.weight.repeat(2 + self.net.refine_layers, 1)
        hr_embed_bg = self.net.hr_embed_bg.weight if self.net.hr_level_embedding else self.net.hr_embed_bg.weight.repeat(2 + self.net.refine_layers, 1)

        # update mask memory
        stride = int(last_pred.shape[-1] / src_memories[-1].shape[-1])
        for layer, src_m in enumerate(src_memories):
            last_pred_clamp = last_pred * (last_pred > 0)
            attached_m = self.net.softlabel_attach(src_m, last_pred_clamp, hr_embed_obj[layer+1: layer+2],
                                  hr_embed_bg[layer+1: layer+2], stride, num_objects)
            attached_m_flatten = attached_m.flatten(-2).permute(2, 0, 1)
            if frame_num % self.update_frequency == 0 and torch.all(scores >= 0.4):
                if memory['refined'][layer] is not None:
                    memory['refined'][layer].append(attached_m_flatten)
                else:
                    memory['refined'][layer] = [attached_m_flatten]
            elif self.require_last:
                self.memory['last_frame'][layer] = attached_m_flatten
            if len(memory['refined'][layer]) > int(self.max_memory_length):
            # if len(memory['refined'][layer]) > int(self.max_memory_length / num_objects.item()):
                memory['refined'][layer].pop(1)
        return memory


    def depadding(self, inputs, pad):
        """inputs (List(Tensor))"""
        outputs = []
        for input in inputs:
            if pad[2]+pad[3] > 0:
                input = input[:,:,pad[2]:-pad[3],:]
            if pad[0]+pad[1] > 0:
                input= input[:,:,:,pad[0]:-pad[1]]
            outputs.append(input)
        return tuple(outputs)


def build_trans_wsvos_tracker(cfg, net):
    tracker = Trans_WSVOS_Tracker(net, using_initial_bg=cfg.TEST.USING_INITIAL_BG,
                                       require_last=cfg.TEST.REQUIRE_LAST,
                                       require_last_scribble=cfg.TEST.REQUIRE_LAST_SCRIBBLE,
                                       max_memory_length=cfg.TEST.MAX_MEMORY_LENGTH, 
                                       update_frequency=cfg.TEST.UPDATE_FREQUENCY,
                                       scribble_update_frequency=cfg.TEST.SCRIBBLE_UPDATE_FREQUENCY,
                                       upsample_logits=cfg.MODEL.DECODER.UPSAMPLE_LOGITS)
    return tracker