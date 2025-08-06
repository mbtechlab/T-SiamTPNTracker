"""
Basic STARK Model (Spatial-only).
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import sparse_
from typing import Dict, List

from .backbone.backbone_shufflenet_v2 import build_backbone
# from .backbone.backbone_resnet18 import build_backbone
# from .backbone.backbone_resnet50_pruned import build_backbone
from .head import build_head
from .fpn import build_fpn
#torch.autograd.set_detect_anomaly(True)

class SiamTPN(nn.Module):
    def __init__(self, backbone, fpn, head, cfg):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self.cfg = cfg

    def forward(self, train_imgs_list, test_img, run_cls_head=False, run_box_head=True):
        # print("////////////////",len(train_imgs_list))
        train_feats = []
        for img in train_imgs_list:
            train_feat = self.backbone(img)
            train_feat_fpn = self.fpn(train_feat)
            train_feats.append(train_feat_fpn)
        
        # tmp = torch.cat(train_imgs_list, dim=0)
        # train_feats_ = self.fpn(self.backbone(tmp))
        # train_feats = [
        #     train_feats_[:len(train_imgs_list[0])],
        #     train_feats_[len(train_imgs_list[0]):]
        # ]
            
        # print("train_feats[0]:" ,train_feats[0].shape)
        # print("train_feats[1]:" ,train_feats[1].shape)

        # Concatenate all the features in train_feats along the channel dimension (dim=1)
        combined_train_feats = torch.cat(train_feats, dim=1)

        # add all the features in train_feats along the channel dimension (dim=1)
        # combined_train_feats = torch.sum(torch.stack(train_feats), dim=0)

        # mean all the features in train_feats along the channel dimension (dim=1)
        # combined_train_feats = torch.mean(torch.stack(train_feats), dim=0)

        # print("combined_train_feats:",combined_train_feats.shape)
        # print("Combined train_feats shape:", combined_train_feats.shape)

        test_feat = self.backbone(test_img)
        test_feat = self.fpn(test_feat)
        
        # فراخوانی head با توجه به مقداردهی run_cls_head و run_box_head
        return self.head(test_feat, combined_train_feats, run_cls_head=run_cls_head, run_box_head=run_box_head)


def build_network(cfg):
    backbone = build_backbone(cfg)
    fpn = build_fpn(cfg)
    head = build_head(cfg)
    
    model = SiamTPN(
        backbone,
        fpn,
        head=head,
        cfg=cfg
    )
    return model
