# -*- coding: utf-8 -*-
"""
author: shanzha
WeChat: shanzhan09
create_time: 2021/12/29 18:57
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class MlpBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(in_features, out_features)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        
        # [B, HW, C]
        return x


class PatchEmbedding(nn.Module):
    """Patch Embedding Block"""
    
    def __init__(self, image_size=224, patch_size=4, in_channel=3, embed_dim=96):
        super().__init__()
        assert image_size % patch_size == 0, f"img_size {image_size} should be divided by patch_size {patch_size}."
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        
        self.imgSize = image_size
        self.patchSize = patch_size
        self.H = self.imgSize[0] // self.patchSize[0]
        self.W = self.imgSize[1] // self.patchSize[1]
        self.patchNum = self.H * self.W
        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x, H, W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)
        
        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
        
        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]
        
        return x


class DropPath(nn.Module):
    def __init__(self, drop_rate=None):
        super().__init__()
        self.drop_rate = drop_rate
    
    def forward(self, x):
        x = self.dropPath(x, self.drop_rate, self.training)
        return x
    
    @staticmethod
    def dropPath(x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


def window_partition(x, window_size: int):
    """将feature map按照window_size划分成一个个没有重叠的window"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """将一个个window还原成一个feature map"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class MutilHeadAttention(nn.Module):
    """Swin Transformer attention"""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attention_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.window_size = (window_size, window_size)  # M
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # table shape [2M-1*2M-1, numHeaders]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )
        
        # from mmseg
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attention_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        # [num_windows*B, N, C]
        B, N, C = x.shape
        # [B, N, 3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class TransformerEncoder(nn.Module):
    def __init__(self, dim, window_size, num_heads, hidden_features=None, qkv_bias=True, drop_path=0.,
                 attention_drop=0., proj_drop=0., mlp_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attention_drop = attention_drop
        self.proj_drop = proj_drop
        self.hidden_features = hidden_features
        self.mlp_drop = mlp_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        shift_size = window_size // 2
        self.shift_size = shift_size
        self.norm = nn.LayerNorm(dim)
        self.attn = MutilHeadAttention(self.dim, self.window_size, self.num_heads, self.qkv_bias, self.attention_drop,
                                       self.proj_drop)
        self.mlp = MlpBlock(self.dim, act_layer=nn.GELU, drop=self.mlp_drop)
    
    def create_mask(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        
        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    
    def forward(self, x, H, W):
        self.H, self.W = H, W
        attn_mask = self.create_mask(x, H, W)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm(x)
        x = x.view(B, H, W, C)
        
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]
        
        # SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]
        
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]
        
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm(x)))
        
        return x


class ConnectExpand(nn.Module):
    """From level L to L-1"""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upSample = nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=2,
                                           stride=2)
        self.norm = nn.LayerNorm(self.out_channel)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.upSample(x)
        x = x.flatten(2).transpose(2, 1)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, self.out_channel, 2 * H, 2 * W)
        
        return x


class ConnectBranchLeft(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=2, stride=2)
        self.upSample = nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=2,
                                           stride=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.upSample(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class ConnectBranchRight(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.GELU()
    
    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = shortcut + x
        x = self.relu(x)
        
        return x


class ConnectBlock(nn.Module):
    """Merge left and right branch"""
    
    def __init__(self, in_channel_encoder, in_channel_connect, out_channel):
        super().__init__()
        self.in_channel_encoder = in_channel_encoder
        self.in_channel_connect = in_channel_connect
        self.out_channel = out_channel
        self.expand = ConnectExpand(in_channel=self.in_channel_encoder, out_channel=self.out_channel)
        self.connectleft = ConnectBranchLeft(in_channel=self.in_channel_connect, out_channel=self.out_channel)
        self.connectright = ConnectBranchRight(in_channel=self.in_channel_connect, out_channel=self.out_channel)
    
    def forward(self, x, y):
        Bx, Lx, Cx = x.shape
        Hx = Wx = int(Lx ** 0.5)
        x = x.transpose(1, 2).view(Bx, Cx, Hx, Wx)
        By, Ly, Cy = y.shape
        Hy = Wy = int(Ly ** 0.5)
        y = y.transpose(1, 2).view(By, Cy, Hy, Wy)
        x = self.expand(x)
        x = self.connectleft(x)
        y = self.connectright(y)
        x = x.flatten(2).transpose(1, 2)
        y = y.flatten(2).transpose(1, 2)
        out = x + y
        
        return out


class PosCNN(nn.Module):
    """PEG  from https://arxiv.org/abs/2102.10882"""
    
    def __init__(self, in_channel, embed_dim, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channel, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x
    
    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]
    

class DecoderUpSampling(nn.Module):
    # def __init__(self, in_channel):
    #     super().__init__()
    #     self.in_channel = in_channel
    #     self.out_channel = 2 * in_channel
    #     self.linear = nn.Linear(self.in_channel, self.out_channel)
    #
    # def forward(self, x, H, W):
    #     B, L, C = x.shape
    #     x = self.linear(x)
    #     x = x.view(B, 4 * H * W, int(C / 2))
    #     return x

    def __init__(self,  in_channel, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = in_channel
        self.expand = nn.Linear(in_channel, 2 * in_channel, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(in_channel // dim_scale)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class VitEncoderBlock(nn.Module):
    """One stage block"""
    def __init__(self,
                 embed_dim=96,
                 window_size=7,
                 num_heads=3,
                 hidden_features=None,
                 qkv_bias=True,
                 attention_drop=0.,
                 proj_drop=0.,
                 mlp_drop=0.):
        super().__init__()
        self.mhattn = TransformerEncoder(dim=embed_dim, window_size=window_size, num_heads=num_heads,
                                         hidden_features=hidden_features, qkv_bias=qkv_bias,
                                         attention_drop=attention_drop, proj_drop=proj_drop, mlp_drop=mlp_drop)
        self.peg = PosCNN(in_channel=embed_dim, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, H, W):
        x = self.mhattn(x, H, W)
        shortcut = x
        x = self.peg(x, H, W)
        x = x + shortcut
        x = self.mhattn(x, H, W)
        x = self.mhattn(x, H, W)
        return x
    

class VitDecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()


class Model(nn.Module):
    def __init__(self, num_classes=21,
                 image_size=224,
                 patch_size=4,
                 in_channel=3,
                 embed_dim=[96, 192, 384, 768],
                 window_size=7,
                 num_heads=[3, 6, 12, 24],
                 hidden_features=None,
                 qkv_bias=True,
                 attention_drop=0.2,
                 proj_drop=0.2,
                 mlp_drop=0.2,
                 drop_path=0.2):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(image_size=image_size,
                                          patch_size=patch_size,
                                          in_channel=in_channel,
                                          embed_dim=embed_dim[0])
        self.vit1 = VitEncoderBlock(
                            embed_dim=embed_dim[0],
                            window_size=window_size,
                            num_heads=num_heads[0],
                            hidden_features=hidden_features,
                            qkv_bias=qkv_bias,
                            attention_drop=attention_drop,
                            proj_drop=proj_drop,
                            mlp_drop=mlp_drop)
        self.pame1 = PatchMerging(embed_dim[0])
        self.vit2 = VitEncoderBlock(
            embed_dim=embed_dim[1],
            window_size=window_size,
            num_heads=num_heads[1],
            hidden_features=hidden_features,
            qkv_bias=qkv_bias,
            attention_drop=attention_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop)
        self.pame2 = PatchMerging(embed_dim[1])
        self.vit3 = VitEncoderBlock(
            embed_dim=embed_dim[2],
            window_size=window_size,
            num_heads=num_heads[2],
            hidden_features=hidden_features,
            qkv_bias=qkv_bias,
            attention_drop=attention_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop)
        self.pame3 = PatchMerging(embed_dim[2])
        self.vit4 = VitEncoderBlock(
            embed_dim=embed_dim[3],
            window_size=window_size,
            num_heads=num_heads[3],
            hidden_features=hidden_features,
            qkv_bias=qkv_bias,
            attention_drop=attention_drop,
            proj_drop=proj_drop,
            mlp_drop=mlp_drop)
        self.connect1 = ConnectBlock(in_channel_encoder=embed_dim[1],
                                     in_channel_connect=embed_dim[0],
                                     out_channel=embed_dim[0])
        self.connect2 = ConnectBlock(in_channel_encoder=embed_dim[2],
                                     in_channel_connect=embed_dim[1],
                                     out_channel=embed_dim[1])
        self.connect3 = ConnectBlock(in_channel_encoder=embed_dim[3],
                                     in_channel_connect=embed_dim[2],
                                     out_channel=embed_dim[2])
        self.decoderUpSampling0 = DecoderUpSampling(in_channel=int(embed_dim[0] / 2))
        self.decoderUpSampling = DecoderUpSampling(in_channel=embed_dim[0])
        self.decoderUpSampling1 = DecoderUpSampling(in_channel=embed_dim[1])
        self.decoderUpSampling2 = DecoderUpSampling(in_channel=embed_dim[2])
        self.decoderUpSampling3 = DecoderUpSampling(in_channel=embed_dim[3])
        self.softmax = nn.Softmax(dim=-1)
        self.conv = nn.Conv2d(8 * in_channel, self.num_classes, 1)
        self.norm = nn.LayerNorm(8 * in_channel)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(in_features=8 * in_channel, out_features=num_classes)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
      
    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.vit1(x, H, W)
        encoderOut1 = x
        x = self.pame1(x, H, W)
        x = self.vit2(x, int(H / 2), int(W / 2))
        encoderOut2 = x
        x = self.pame2(x, int(H / 2), int(W / 2))
        x = self.vit3(x, int(H / 4), int(W / 4))
        encoderOut3 = x
        x = self.pame3(x, int(H / 4), int(W / 4))
        x = self.vit4(x, int(H / 8), int(W / 8))
        encoderOut4 = x
        connectout3 = self.connect3(encoderOut4, encoderOut3)
        connectout2 = self.connect2(connectout3, encoderOut2)
        connectout1 = self.connect1(connectout2, encoderOut1)
        x = self.vit4(x, int(H / 8), int(W / 8))
        decoderOut4 = x
        x = self.decoderUpSampling3(x, int(H / 8), int(W / 8))
        x = x + connectout3
        x = self.vit3(x, int(H / 4), int(W / 4))
        decoderOut3 = x
        x = self.decoderUpSampling2(x, int(H / 4), int(W / 4))
        x = x + connectout2
        x = self.vit2(x, int(H / 2), int(W / 2))
        decoderOut2 = x
        x = self.decoderUpSampling1(x, int(H / 2), int(W / 2))
        x = x + connectout1
        x = self.vit1(x, H, W)
        decoderOut1 = x
        x = self.decoderUpSampling(x, H, W)
        x = self.decoderUpSampling0(x, 2 * H, 2 * W)
        x = self.norm(x)
        
        # x = self.conv(x)
        x = self.linear(x)
        x = self.softmax(x)
        B, L, C = x.shape
        x = x.transpose(2, 1).view(B, C, self.image_size, self.image_size)
        # print(x)
        return x
    

