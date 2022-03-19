from model.GNN import SpatialGNN
from typing import final
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.R50MGSeg25 import R50MGSeg
from model.ResNeXt101_MG import ResNeXt101_MG
from backbone.resnext.resnext101_regular import ResNeXt101

from model.CBAM import CBAM
from model.GNN import AttenMultiHead, GNN, LayerNorm

## QG Block
class MirrorAttention(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, pool=(1,1), size=(24,24), h=8, insize=24):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(pool)
        self.up = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        self.attngnn = AttenMultiHead(inter_channels, h=h)
        self.spatialgnn = SpatialGNN(inter_channels, insize, insize)
        self.back = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, reversed=False, residual=False):
        '''x: bs, f, h/2, w/2
            return: bs, out, h, w
        '''
        t = self.trans(x) ## bs, inter_channels, h, w
        y = self.pool(t)
        size = y.shape
        y = y.reshape(y.shape[0], y.shape[1], -1).permute(0, 2, 1) ## bs, n, k

        y = self.attngnn(y, y, y, reversed) + y
        y = self.spatialgnn(y) + y
        
        y = y.permute(0,2,1).reshape(size)
        y = self.back(self.up(y))
        if residual: y = y + self.up(x)
        return y

class GatingModule(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.gpool = nn.AdaptiveAvgPool2d((1,1))
        self.trans_x = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )
        self.trans_y = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )        
        self.fc = nn.Sequential(
            nn.Linear(channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.scores = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x, y):
        bs, k, h, w = x.shape
        x = self.trans_x(x)
        y = self.trans_y(y)

        fx = self.fc(self.gpool(x).reshape(bs, k))
        fy = self.fc(self.gpool(y).reshape(bs, k))
        scores = torch.softmax(self.scores( torch.cat([fx,fy],dim=1)), dim=-1) ## two weights
        score_x = scores[:, 0].reshape(bs, 1, 1, 1)
        score_y = scores[:, 1].reshape(bs, 1, 1, 1)
        # print("scores_x:",score_x.mean().item(), "scores_y:", score_y.mean().item())
        ret = score_x * x + score_y * y
        ret = self.conv(ret)
        return ret, nn.L1Loss()(score_x, score_y) * 0.1

class AE(nn.Module):
    def __init__(self, N, C, in_channels, inter_channels, out_channels, pool=(2,2), factor=2):
        super().__init__()
        self.C = C
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(pool)
        self.linearKC = nn.Linear(inter_channels, C)
        self.linearNC = nn.Linear(N, C)
        self.gnn = GNN(inter_channels)
        self.spatialgnn = SpatialGNN(inter_channels, 24, 24)
        self.up = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=True)
        self.back = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, SP):
        '''x: bs, f, h, w
            SP: bs, C, h, w
        '''
        SP = torch.softmax(SP, dim=1)
        SP = self.pool(SP) ## bs, C, h/2, w/2
        SP = SP.reshape(SP.shape[0], SP.shape[1], -1) ## bs, C, n

        t = self.trans(x) ## bs, k, h, w
        y = self.pool(t) ## bs, k, h/2, w/2
        size = y.shape
        y = y.reshape(y.shape[0], y.shape[1], -1) ## bs, k, n
        sigma = self.linearKC(self.linearNC(y).permute(0, 2, 1)) ## bs, c, c
        A = torch.matmul(SP.permute(0, 2, 1), torch.matmul(sigma, SP)) ## bs, n, n

        y = y.permute(0, 2, 1) ## bs, n, k
        y = self.gnn(A, y) + y 
        y = self.spatialgnn(y) + y

        y = self.dropout(self.up( y.permute(0, 2, 1).reshape(size) )) + t
        y = self.back(y)
        return self.dropout(y)

class Decoder(nn.Module):
    def __init__(self, num_convs=[2048, 1024, 512, 256]):
        super().__init__()        
        self.lateral_conv4 = nn.Sequential(nn.Conv2d(num_convs[0], 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.lateral_conv3 = nn.Sequential(nn.Conv2d(num_convs[1], 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.lateral_conv2 = nn.Sequential(nn.Conv2d(num_convs[2], 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.lateral_conv1 = nn.Sequential(nn.Conv2d(num_convs[3], 256, 1), nn.BatchNorm2d(256), nn.ReLU())

        self.fusion_4 = nn.Sequential(CBAM(256), nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256))
        self.fusion_3 = nn.Sequential(CBAM(256*2), nn.Conv2d(256*2, 256, 1), nn.BatchNorm2d(256))
        self.fusion_2 = nn.Sequential(CBAM(256*3), nn.Conv2d(256*3, 256, 1), nn.BatchNorm2d(256))
        self.fusion_1 = nn.Sequential(CBAM(256*4), nn.Conv2d(256*4, 256, 1), nn.BatchNorm2d(256))

        self.layer4_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.layer3_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.layer2_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.layer1_predict = nn.Conv2d(256, 1, 3, 1, 1)
        ## m1-4
        self.refinement = nn.Conv2d(4, 1, 1, 1, 0)

    def forward(self, x, layers):
        '''layers = [l4,l3,l2,l1]
        '''
        up4 = self.fusion_4( self.lateral_conv4( layers[0] ) ) ## 256, 24
        m4 = self.layer4_predict(up4)

        l3 = self.lateral_conv3(layers[1]) ## 256, 24
        up3 = self.fusion_3( torch.cat([
            l3,
            F.interpolate(up4, size=l3.shape[2:4], mode="bilinear", align_corners=True)
        ],dim=1) )
        m3 = self.layer3_predict(up3) ## 24

        l2 = self.lateral_conv2(layers[2])
        up2 = self.fusion_2( torch.cat([
            l2,
            F.interpolate(up4, size=l2.shape[2:4], mode="bilinear", align_corners=True),
            F.interpolate(up3, size=l2.shape[2:4], mode="bilinear", align_corners=True),
        ],dim=1) ) # 48
        m2 = self.layer2_predict(up2)

        l1 = self.lateral_conv1(layers[3])
        up1 = self.fusion_1( torch.cat([
            l1,
            F.interpolate(up4, size=l1.shape[2:4], mode="bilinear", align_corners=True),
            F.interpolate(up3, size=l1.shape[2:4], mode="bilinear", align_corners=True),
            F.interpolate(up2, size=l1.shape[2:4], mode="bilinear", align_corners=True),
        ],dim=1) ) # 96
        m1 = self.layer1_predict(up1)

        M1 = F.interpolate(m1, size=x.shape[2:4], mode="bilinear", align_corners=True)
        M2 = F.interpolate(m2, size=x.shape[2:4], mode="bilinear", align_corners=True)
        M3 = F.interpolate(m3, size=x.shape[2:4], mode="bilinear", align_corners=True)
        M4 = F.interpolate(m4, size=x.shape[2:4], mode="bilinear", align_corners=True)
        
        fuse_features = torch.cat([M1, M2, M3, M4],dim=1)
        M0 = self.refinement(fuse_features)

        return [M1, M2, M3, M4, M0]

class Architecture(nn.Module):
    def __init__(self, training = False):
        super().__init__()
        self.semantic_sidepath = R50MGSeg(num_classes = 25, pretrained=False, os=16)
        mirror_branch = ResNeXt101(None)
        if training:
            self.semantic_sidepath.load_state_dict(torch.load("weights/R50MGSeg-25_iter_30000.pt"))
            mirror_branch = ResNeXt101('weights/resnext_101_32x4d.pth')
        
        self.layer1 = nn.Sequential(mirror_branch.layer0, mirror_branch.layer1)
        self.layer2 = mirror_branch.layer2
        self.layer3 = mirror_branch.layer3
        self.layer4 = mirror_branch.layer4
        
        self.AE = AE(24*24, 25, 2048, 512, 512, pool=(1,1), factor=1)
        self.gating_module = GatingModule(512)

        self.l2_pred = nn.Conv2d(512, 1, 3, 1, 1)
        self.l3_pred = nn.Conv2d(1024, 1, 3, 1, 1)
        self.l4_pred = nn.Conv2d(2048, 1, 3, 1, 1)

        self.QG3_MHSA_Spatial = MirrorAttention(1024, 512, 1024, pool=(1,1), size=(24,24), h=8)
        self.QG3_MHSA_Spatial_reverse = MirrorAttention(1024, 512, 1024, pool=(1,1), size=(24,24), h=8)
        self.QG3_gating_module = GatingModule(channels=1024)
        self.QG4_MHRA_Spatial = MirrorAttention(2048, 512, 2048, pool=(1,1), size=(24,24), h=8, insize=12)
        self.QG4_MHRA_Spatial_reverse = MirrorAttention(2048, 512, 2048, pool=(1,1), size=(24,24), h=8, insize=12)
        self.QG4_gating_module = GatingModule(channels=2048)

        self.decoder = Decoder(num_convs=[2048, 1024, 512, 256])

    def resetSeg(self):
        self.semantic_sidepath.load_state_dict(torch.load("model/R50MGSeg-25_iter_30000.pt"))
        
    def frozenSeg(self):
        for param in self.semantic_sidepath.parameters():
            param.requires_grad = False

    def forward(self, x):
        layer1 = self.layer1(x) # 256 96 ## L0 and L1
        layer2 = self.layer2(layer1) # 512 48

        segout = self.semantic_sidepath(x)
        seg_context = self.AE(segout["layers"][-1], segout["SP"]) ## bs, 512, h, w (24*24)
        seg_context = F.interpolate(seg_context, size=layer2.shape[2:4], mode="bilinear")
        layer2, miniloss2 = self.gating_module(layer2, seg_context)

        layer3 = self.layer3(layer2) # 1024 24
        intra_relation = self.QG3_MHSA_Spatial(layer3, reversed=False, residual=True)
        inter_relation = self.QG3_MHSA_Spatial_reverse(layer3, reversed=True, residual=True)
        layer3, miniloss3 = self.QG3_gating_module(intra_relation, inter_relation)

        layer4 = self.layer4(layer3) # 2048 24
        intra_relation = self.QG4_MHRA_Spatial(layer4, reversed=False, residual=True)
        inter_relation = self.QG4_MHRA_Spatial_reverse(layer4, reversed=True, residual=True)
        layer4, miniloss4 = self.QG4_gating_module(intra_relation, inter_relation)

        ul2 = F.interpolate(self.l2_pred(layer2), size=x.shape[2:4], mode="bilinear", align_corners=True)
        ul3 = F.interpolate(self.l3_pred(layer3), size=x.shape[2:4], mode="bilinear", align_corners=True)
        ul4 = F.interpolate(self.l4_pred(layer4), size=x.shape[2:4], mode="bilinear", align_corners=True)
        
        preds = self.decoder(x, [layer4, layer3, layer2, layer1])
        final_pred = preds[-1]

        return {
            "final": final_pred,
            # "preds":preds,
            # "top": [ul2, ul3, ul4],
            # "deeppreds": [],
            # "losses": [], #miniloss2, miniloss3, miniloss4
            # "finals":finals,
            # "segout": segout["out"],
            # "SP": segout["SP"]
        }
        
# archi = Architecture()
# archi(torch.rand(2,3,384,384))
# torch.save( archi.state_dict(), "random.pt")