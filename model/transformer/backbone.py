import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet50, resnet101

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """
    def __init__(self, n, epsilon=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n) - epsilon)
        self.epsilon = epsilon

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
    def forward(self, x):
        """
        Refer to Detectron2 (https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/layers/batch_norm.py)
        """
        if x.requires_grad:
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            # move reshapes to the beginning to make it fuser-friendly
            w = self.weight.reshape(1, -1, 1, 1)
            b = self.bias.reshape(1, -1, 1, 1)
            rv = self.running_var.reshape(1, -1, 1, 1)
            rm = self.running_mean.reshape(1, -1, 1, 1)
            scale = w * (rv + self.epsilon).rsqrt()
            bias = b - rm * scale
            return x * scale + bias
        else:
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.epsilon,
            )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.epsilon})"


class Feature_Extractor(nn.Module):
    def __init__(self, name='resnet50', version='v2', pretrained=True, frozen=True, bn_frozen=False):
        super(Feature_Extractor, self).__init__()
        self.name = name
        self.version = version
        self.bn = nn.BatchNorm2d if bn_frozen == False else FrozenBatchNorm2d
        if name == 'resnet50':
            resnet = resnet50(pretrained=pretrained, version=version, bn=self.bn)
        elif name == 'resnet101':
            resnet = resnet101(pretrained=pretrained, version=version, bn=self.bn)
        else:
            raise Exception('Unknown backbone!')
        if self.version == 'v2':
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        elif self.version == 'v1':
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3 = resnet.layer1, resnet.layer2, resnet.layer3#, resnet.layer4

        if self.version == 'v2':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        
        self.frozen = frozen
        
    def forward(self, x):
        if self.frozen:
            with torch.no_grad():
                feat_layer0 = self.layer0(x)
                feat_layer1 = self.layer1(feat_layer0)
                feat_layer2 = self.layer2(feat_layer1)
                feat_layer3 = self.layer3(feat_layer2)
                #feat_layer4 = self.layer4(feat_layer3)
        else:
            feat_layer0 = self.layer0(x)
            feat_layer1 = self.layer1(feat_layer0)
            feat_layer2 = self.layer2(feat_layer1)
            feat_layer3 = self.layer3(feat_layer2)
            #feat_layer4 = self.layer4(feat_layer3)
        
        middle_feats = [feat_layer1, feat_layer2, feat_layer3]
                    
        return middle_feats

def build_backbone(cfg):
    backbone = Feature_Extractor(name = cfg.MODEL.BACKBONE.NAME,
                                  version = cfg.MODEL.BACKBONE.VERSION,
                                  frozen = cfg.MODEL.BACKBONE.FROZEN,
                                  bn_frozen = cfg.MODEL.BACKBONE.BN_FROZEN)
    return backbone


if __name__ == '__main__':
    backbone = Feature_Extractor(name='resnet50', version='v1', pretrained=True, hidden_dim=256, frozen=True, bn_frozen=True)
    print(backbone)
    img = torch.randn((10,3,384,384))
    feat, feats = backbone(img)
    print('out_feat', feat.shape)
    for k, v in feats.items():
        print(k, v.shape)