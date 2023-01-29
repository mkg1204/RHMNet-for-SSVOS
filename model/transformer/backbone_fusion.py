import torch
import torch.nn as nn
from .resnet import resnet50, resnet101

class Feature_Extractor(nn.Module):
    def __init__(self, name='resnet50', version='v2', pretrained=True, hidden_dim=256, freeze=True):
        super(Feature_Extractor, self).__init__()
        self.name = name
        self.version = version
        if name == 'resnet50':
            resnet = resnet50(pretrained=pretrained, version=version)
        elif name == 'resnet101':
            resnet = resnet101(pretrained=pretrained, version=version)
        else:
            raise Exception('Unknown backbone!')
        if self.version == 'v2':
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        elif self.version == 'v1':
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

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
        
        self.hidden_dim = hidden_dim
        if self.version == 'v1':
            self.bottleneck = nn.Conv2d(1024, hidden_dim, kernel_size=1)
        elif self.version == 'v2':
            self.bottleneck = nn.Conv2d(1024+512, hidden_dim, kernel_size=1)
        self.freeze = freeze

    
    def forward(self, x):
        if self.freeze:
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
        
        if self.version =='v1':
            out_feat = self.bottleneck(feat_layer3) # reduce dim
        elif self.version == 'v2':
            out_feat = self.bottleneck(torch.cat([feat_layer3, feat_layer2], dim=1))

        feat_dicts = {'layer0_feat': feat_layer0, 
                      'layer1_feat': feat_layer1, 
                      'layer2_feat': feat_layer2, 
                      'layer3_feat': feat_layer3,}
                      #'layer4_feat': feat_layer4,}
            
        return out_feat, feat_dicts