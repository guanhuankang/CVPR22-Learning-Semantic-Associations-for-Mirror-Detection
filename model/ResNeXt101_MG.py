from model.multiGrid import resnet101_atrous
import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNeXt101_MG(nn.Module):
    def __init__(self, num_classes, pretrained=True, os=16):
        super().__init__()
        self.model = resnet101_atrous(pretrained=pretrained, os=os)
        
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(1024, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(256, num_classes, 1)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, x):
        out = F.interpolate(
            self.classifier( self.model(x) ), 
            size=x.shape[2:4], mode='bilinear', align_corners=False
        )
        aux = F.interpolate(
            self.aux_classifier( self.model.layers[-2] ),
            size=x.shape[2:4], mode='bilinear', align_corners=False
        )
        return {
            "out": out,
            "aux": aux,
            "layers": self.model.layers
        }
        