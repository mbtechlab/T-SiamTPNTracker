from sympy import im
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .util import xcorr_depthwise, conv
from .fpn.tan import Block

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output layer for IOU prediction

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for IOU output
        return x
    
class PatchMLP(nn.Module):
    def __init__(self, cfg):
        super(PatchMLP, self).__init__()
        self.cfg = cfg

        '''clssification head'''
        # MLP(input_dim, hidden_dim, output_dim, num_layers)
        # print("hiden dim: ", cfg.MODEL.HEAD.IN_DIM)
        self.cls_head = MLP(cfg.MODEL.HEAD.IN_DIM, cfg.MODEL.HEAD.IN_DIM, 2, 2)
        self.cen_head = MLP(cfg.MODEL.HEAD.IN_DIM, cfg.MODEL.HEAD.IN_DIM, 1, 2)
        self.box_head = MLP(cfg.MODEL.HEAD.IN_DIM, cfg.MODEL.HEAD.IN_DIM, 4, 2)     

    def forward(self, feat):
        B, L, C = feat.shape
        pred_cls = self.cls_head(feat)
        pred_cen = self.cen_head(feat)
        pred_box = self.box_head(feat)
        results = {'box': pred_box, 'cls': pred_cls, 'cen': pred_cen}    
        return results

class PatchHead(nn.Module):
    def __init__(self, cfg):
        super(PatchHead, self).__init__()
        self.cfg = cfg
        hidden = cfg['MODEL']['HEAD']['IN_DIM']
        num_heads = cfg['MODEL']['FPN']['NHEADS']
        mlp_ratio = cfg['MODEL']['FPN']['MLP_RATIOS']

        '''block for attention-based correlation'''
        # The kernel has more channels, so we'll allow Block to handle this.
        self.block = Block(dim_q=hidden, dim_kv=hidden * cfg['DATA']['TEMPLATE']['NUMBER'], cross=True, stride=1 ,num_heads=num_heads, mlp_ratio=mlp_ratio)

        '''classification head'''
        self.cls_head = nn.Sequential(
            conv(hidden, hidden // 2),
            conv(hidden // 2, hidden // 2),
            nn.Conv2d(hidden // 2, 1, kernel_size=1)  # 1 output neuron for class prediction
        )

        '''box head'''
        self.box_head = nn.Sequential(
            conv(hidden, hidden // 2),
            conv(hidden // 2, hidden // 2),
            nn.Conv2d(hidden // 2, 4, kernel_size=1)  # 4 output neurons for bounding box coordinates
        )

        # '''MLP classification head score '''
        # self.cls_head_score = nn.Sequential(
        #     conv(hidden, hidden // 2),
        #     conv(hidden // 2, hidden // 2),
        #     nn.Conv2d(hidden // 2, 1, kernel_size=1),  # 1 output neuron for classification score
        #     nn.AdaptiveAvgPool2d(1)  # Output shape: [batch_size, 1, 1, 1]
        # )

    def forward(self, feat, kernel, run_box_head=True, run_cls_head=False):
        # Flatten and permute to match Block input requirements
        feat = feat.flatten(2).permute(0, 2, 1)  # Shape: [B, L, C]
        kernel = kernel.flatten(2).permute(0, 2, 1)  # Shape: [B, L, C]

        # Print shapes for debugging
        # print("feat.shape", feat.shape)
        # print("kernel.shape", kernel.shape)

        # Pass through the Block (attention mechanism)
        feat = self.block(feat, kernel)  # Using Block instead of correlation

        # Restore feat back to the original spatial dimensions
        B, L, C = feat.shape
        H = W = int(math.sqrt(L))
        feat = feat.permute(0, 2, 1).view(B, C, H, W)  # Shape: [B, C, H, W]

        outputs = {}

        # if run_cls_head:
        #     # Classification head score output
        #     cls_head_score = self.cls_head_score(feat).view(B, -1)  # Reshape to [B, 1]
        #     outputs['cls_head_score'] = cls_head_score

        if run_box_head:
            pred_cls = self.cls_head(feat).permute(0, 2, 3, 1).reshape(-1, self.cls_head(feat).size(-1))
            pred_cls = pred_cls.view(-1, 1)
            pred_box = self.box_head(feat).permute(0, 2, 3, 1).reshape(B, -1, 4)
            outputs['pred_cls'] = pred_cls
            outputs['pred_box'] = F.relu(pred_box)

        if not outputs:
            raise ValueError("Either run_cls_head or run_box_head must be True")
        
        return outputs
    
def build_head(cfg):
    if cfg['MODEL']['HEAD']['TYPE'] == 'MLP':
        print("//////////////////MLP/////////////")
        return PatchMLP(cfg)  # Ensure PatchMLP is designed to take the entire cfg as an argument or adjust accordingly.
    elif cfg['MODEL']['HEAD']['TYPE'] == 'CONV':
        # print("//////////////////CONV/////////////")
        return PatchHead(cfg)  # ارسال کل دیکشنری cfg به جای فقط یک مقدار
    else:
        print('Head not implemented')
