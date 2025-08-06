# Modified from
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/facebookresearch/LeViT/blob/main/levit.py
# Copyright 2020 Daitao Xing, Apache-2.0 License

# Necessary imports for neural network construction
import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math


class Linear(nn.Module):
    """
    A custom linear module that performs a two-step transformation:
    An initial linear transformation, followed by an activation, and another linear transformation.
    """
    def __init__(self, in_dim, hid_dim=None, act_layer=nn.GELU):
        """
        Initializes the Linear module.
        Parameters:
            in_dim (int): Dimensionality of the input feature space.
            hid_dim (int): Dimensionality of the hidden layer space.
            act_layer (nn.Module): Activation function to use after the first linear transformation.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)  # First linear transformation
        self.act = act_layer()  # Activation function, default is GELU
        self.fc2 = nn.Linear(hid_dim, in_dim)  # Second linear transformation to project back to the original dimension

    def forward(self, x):
        """
        Forward pass of the Linear module.
        Parameters:
            x (Tensor): Input tensor
        Returns:
            Tensor: Transformed tensor after two linear transformations and an activation.
        """
        x = self.fc1(x)  # Apply first linear transformation
        x = self.act(x)  # Apply activation
        x = self.fc2(x)  # Apply second linear transformation
        return x
    
class Cattention(nn.Module):
    def __init__(self, in_dim):
        super(Cattention, self).__init__()
        self.chanel_in = in_dim
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim * 2, in_dim, kernel_size=1, stride=1),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.linear2 = nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

    def forward(self, x, y):
        # فرض این است که x و y دارای شکل (B, C, H, W) هستند
        ww = self.linear2(self.dropout(self.activation(self.linear1(self.avg_pool(y)))))
        # اطمینان حاصل می‌کنیم که x و y دارای ابعاد (B, C, H, W) باشند
        weight = self.conv1(torch.cat((x, y), dim=1)) * ww
        return x + self.gamma * weight * x

class Attention(nn.Module):
    """
    Implements a multi-head attention mechanism, optional pooling for reducing dimensionality.
    """
    def __init__(self, dim_q, dim_kv, num_heads=4, qkv_bias=False, stride=1):
        """
        Initializes the Attention module.
        Parameters:
            dim_q (int): Dimensionality of the query vectors.
            dim_kv (int): Dimensionality of the key/value vectors.
            num_heads (int): Number of attention heads.
            qkv_bias (bool): Whether to include bias in the query/key/value transformations.
            stride (int): Stride for optional pooling operation. If stride > 1, pooling is applied.
        """
        super().__init__()
        self.dim = dim_q
        self.num_heads = num_heads
        head_dim = dim_q // num_heads
        self.scale = head_dim ** -0.5  # Scale factor for stabilizing the dot products

        # Linear transformations for query, key, and value
        self.q = nn.Linear(dim_q, dim_q, bias=qkv_bias)
        self.kv = nn.Linear(dim_kv, dim_q * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim_q, dim_q)

        # Optional pooling and additional transformations if stride is greater than 1
        self.stride = stride
        if stride > 1:
            self.pool = nn.AvgPool2d(stride, stride=stride)
            self.sr = nn.Conv2d(dim_kv, dim_kv, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim_kv)
            self.act = nn.GELU()

    def forward(self, x, y):
        """
        Forward pass of the Attention module.
        Parameters:
            x (Tensor): Query tensor
            y (Tensor): Key/Value tensor
        Returns:
            Tensor: Output tensor after applying attention.
        """
        B, N, C = x.shape  # Batch size, number of queries, dimension of each query
        B, L, C2 = y.shape  # Batch size, number of keys/values, dimension of each key/value
        H = W = int(math.sqrt(L))  # Assuming square dimensionality for pooling

        # Prepare query matrix (multi head)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.stride > 1:
            # Apply pooling and convolution if stride is greater than 1
            y_ = y.permute(0, 2, 1).contiguous().reshape(B, C2, H, W)
            y_ = self.sr(self.pool(y_)).reshape(B, C2, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            y_ = self.act(y_)

            # Prepare kv matrix (multi head)
            kv = self.kv(y_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]  # Split the combined key/value tensor into separate keys and values

        # Compute attention using scaled dot-product
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # Apply softmax to get probabilities

        # Apply attention to the values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)  # Project back to the dimension of the input
        return x



class Block(nn.Module):
    def __init__(self, dim_q, dim_kv, cross=True, num_heads=4, mlp_ratio=2., qkv_bias=False,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm, stride=1):
        super().__init__()
        self.norm1 = norm_layer(dim_q)
        self.attn = Attention(
            dim_q, dim_kv,
            num_heads=num_heads, qkv_bias=qkv_bias, stride=stride)
        self.norm2 = norm_layer(dim_q)
        mlp_hidden_dim = int(dim_q * mlp_ratio)
        self.mlp = Linear(in_dim=dim_q, hid_dim=mlp_hidden_dim, act_layer=act_layer)

        if cross:
            self.norm3 = norm_layer(dim_kv)
            self.modulation_layer = Cattention(dim_q)  # Modulation layer اضافه شده
        self.cross = cross

    def forward(self, x, y):
        if self.cross:
            return self.forward_cross(x, y)
        else:
            return self.forward_self(x)

    def forward_self(self, x):
        norm_x = self.norm1(x)
        x = x + self.attn(norm_x, norm_x)
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_cross(self, x, y):
        # اجرای cross-attention
        attn_out = self.attn(self.norm1(x), self.norm3(y))
        # تغییر شکل برای سازگاری با Cattention
        B, N, C = attn_out.shape
        H = W = int(N ** 0.5)  # فرض می‌کنیم ورودی 2D است
        attn_out_reshaped = attn_out.permute(0, 2, 1).view(B, C, H, W)
        x_reshaped = x.permute(0, 2, 1).view(B, C, H, W)

        # استفاده از لایه‌ی modulation layer برای تنظیم x و y
        modulated_output = self.modulation_layer(x_reshaped, attn_out_reshaped)

        # بازگرداندن شکل به حالت اولیه
        modulated_output = modulated_output.view(B, C, N).permute(0, 2, 1)
        
        # اعمال MLP پس از modulation
        x = modulated_output + self.mlp(self.norm2(modulated_output))
        return x


class ReFPNBlock(nn.Module):

    def __init__(self, dim, **kwargs):
        super().__init__()
        self.block_l4_1 = Block(dim, dim, cross=False, stride=2, **kwargs)
        self.block_l4_2 = Block(dim, dim, cross=False, stride=2, **kwargs)
        self.block_l4_3 = Block(dim, dim, cross=False, stride=2, **kwargs)
        self.block_l5_l4 = Block(dim, dim, stride=1, **kwargs)
        self.block_l3_l4 = Block(dim, dim, stride=4, **kwargs)

    def forward(self, feat_l3, feat_l4, feat_l5):
        # print("\n/////////////////////////////////////////////")
        # print("Input feat_l3 shape:", feat_l3.shape)
        # print("Input feat_l4 shape:", feat_l4.shape)
        # print("Input feat_l5 shape:", feat_l5.shape)
        
        feat = self.block_l4_1(feat_l4, feat_l4) + \
                self.block_l5_l4(feat_l4, feat_l5) + \
                    self.block_l3_l4(feat_l4, feat_l3)
        
        feat = self.block_l4_2(feat, feat)
        feat = self.block_l4_3(feat, feat)
        

        return feat


class TReFPN(nn.Module):

    def __init__(self, in_dim,  num_blocks=None, pre_conv=None, **kwargs):
        super().__init__()
        assert pre_conv is not None
        self.num_blocks = num_blocks
        num_l = len(in_dim)
        self.num_layers = num_l
        if pre_conv is not None:
            self.bottleneck = nn.ModuleList([nn.Conv2d(in_dim[i], pre_conv[i], kernel_size=1) for i in range(num_l)])
        # print("in_dim:", in_dim)
        # print("pre_conv:", pre_conv)
        # print("num_blocks:", num_blocks)
        # print("bottleneck:", self.bottleneck)

        hidden_dim = pre_conv[0]
        self.rpn = nn.ModuleList([ReFPNBlock(hidden_dim, **kwargs) for i in range(num_blocks)])
        self.apply(self._init_weights)
        # print("hidden_dim:", hidden_dim)
        # print("rpn:", self.rpn)
 
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, feat):
        """
        tfeat_l3, tfeat_l4: B, C, 8, 8  B, C2, 4, 4
        sfeat_l3, sfeat_l4: B, C, 16, 16  B, C2, 8, 8
        """
        feat = [self.bottleneck[i](feat[i]) for i in range(self.num_layers)]
        feat_l3, feat_l4, feat_l5 = feat    
        B, C, H4, W4 = feat_l4.shape

        feat_l3 = feat_l3.flatten(2).permute(0,2,1)
        feat_l4 = feat_l4.flatten(2).permute(0,2,1)
        feat_l5 = feat_l5.flatten(2).permute(0,2,1)

        for i in range(self.num_blocks):
            feat_l4 = self.rpn[i](feat_l3, feat_l4, feat_l5)
        feat_l4 = feat_l4.permute(0,2,1).reshape(B, C, H4, W4)
        return feat_l4


