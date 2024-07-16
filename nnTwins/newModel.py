
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_


class MlpBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
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
    
    def __init__(self, image_size=224, patch_size=4, in_channel=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        assert image_size % patch_size == 0, f"img_size {image_size} should be divided by patch_size {patch_size}."
        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.imgSize = image_size
        self.patchSize = patch_size
        self.H = self.imgSize[0] // self.patchSize[0]
        self.W = self.imgSize[1] // self.patchSize[1]
        self.patchNum = self.H * self.W
        self.patches_resolution = patches_resolution
        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        
        self.in_channel = in_channel
        self.embed_dim = embed_dim
    
    def forward(self, x):
        z = self.patches_resolution
        x = self.proj(x)
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    
    def __init__(self, dim, input_resolution, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution[0], self.input_resolution[1]
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)
        
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
        
        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]
        
        return x


class PatchExpand(nn.Module):
    """implement of swin unet"""
    
    def __init__(self, dim, input_resolution, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)
    
    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, input_resolution, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
    
    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        
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
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attention_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.window_size = (window_size, window_size)  # M
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
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
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
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


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim,
                 input_resolution,
                 window_size,
                 num_heads,
                 mlp_ratio=4.,
                 hidden_features=None,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 attn_drop=0.,
                 proj_drop=0.,
                 mlp_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.attention_drop = attn_drop
        self.proj_drop = proj_drop
        self.hidden_features = hidden_features
        self.mlp_drop = mlp_drop
        self.act_layer = act_layer
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        shift_size = window_size // 2
        self.shift_size = shift_size
        self.norm = norm_layer(dim)
        self.attn = MutilHeadAttention(dim=self.dim,
                                       window_size=self.window_size,
                                       num_heads=self.num_heads,
                                       qkv_bias=self.qkv_bias,
                                       qk_scale=qk_scale,
                                       attention_drop=self.attention_drop,
                                       proj_drop=self.proj_drop)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpBlock(in_features=self.dim,
                            hidden_features=mlp_hidden_dim,
                            out_features=None,
                            act_layer=self.act_layer,
                            drop=self.mlp_drop)
        
        if self.shift_size > 0:
            H, W = self.input_resolution[0], self.input_resolution[1]
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size
            # 拥有和feature map一样的通道排列顺序，方便后续window_partition
            img_mask = torch.zeros((1, Hp, Wp, 1))  # [1, Hp, Wp, 1]
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
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask)
    
    def forward(self, x):
        H, W = self.input_resolution[0], self.input_resolution[1]
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm(x)
        x = x.view(B, H, W, C)
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]
        
        # SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # [nW*B, Mh*Mw, C]
        
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # [B, H', W', C]
        
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm(x)))
        
        return x


class PosCNN(nn.Module):
    """PEG  from https://arxiv.org/abs/2102.10882"""
    
    def __init__(self, in_channel, input_resolution, embed_dim, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channel, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s
        self.H = input_resolution[0]
        self.W = input_resolution[0]
    
    def forward(self, x):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, self.H, self.W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x
    
    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


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
        self.out_channel = in_channel
        self.conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(self.out_channel)
        self.relu = nn.GELU()
    
    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        t, u = self.in_channel, self.out_channel
        x = x.transpose(2, 1).view(B, C, H, W)
        shortcut = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = shortcut + x
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        
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


class TwinsBlock(TransformerEncoderBlock):
    def __init__(self, dim,
                 input_resolution,
                 window_size,
                 num_heads,
                 stage_deepth,
                 mlp_ratio=4.,
                 hidden_features=None,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_path=0.,
                 attn_drop=0.,
                 proj_drop=0.,
                 mlp_drop=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(TwinsBlock, self).__init__(dim, input_resolution, window_size, num_heads,
                                         mlp_ratio, hidden_features, qkv_bias, qk_scale,
                                         drop_path, attn_drop, proj_drop, mlp_drop,
                                         act_layer, norm_layer)
        self.poscnn = PosCNN(dim, input_resolution, dim)
        self.block = TransformerEncoderBlock(dim=dim, input_resolution=input_resolution,
                                             num_heads=num_heads, window_size=window_size,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                             proj_drop=proj_drop, attn_drop=attn_drop,
                                             drop_path=drop_path,
                                             norm_layer=norm_layer)
        self.stage = nn.ModuleList([
            TransformerEncoderBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    proj_drop=proj_drop, attn_drop=attn_drop,
                                    drop_path=drop_path,
                                    norm_layer=norm_layer)
            for i in range(stage_deepth)])
    
    def forward(self, x):
        x = self.block(x)
        shortcut = x
        x = shortcut + self.drop_path(x)
        for blk in self.stage:
            x = blk(x)
        
        return x


class BasicLayerDown(nn.Module):
    
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, stage_deepth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, proj_drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # build blocks
        self.blocks = nn.ModuleList([
            TwinsBlock(dim=dim, input_resolution=input_resolution, stage_deepth=stage_deepth,
                       num_heads=num_heads, window_size=window_size,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                       proj_drop=proj_drop, attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer)
            for i in range(depth)])
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, input_resolution=input_resolution, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None
    
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, stage_deepth,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, proj_drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # build blocks
        self.blocks = nn.ModuleList([
            TwinsBlock(dim=dim, input_resolution=input_resolution, stage_deepth=stage_deepth,
                       num_heads=num_heads, window_size=window_size,
                       mlp_ratio=mlp_ratio,
                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                       proj_drop=proj_drop, attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer)
            for i in range(depth)])
        
        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(dim=dim, input_resolution=input_resolution, dim_scale=2,
                                        norm_layer=norm_layer)
        else:
            self.upsample = None
    
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class ModelBase(nn.Module):
    def __init__(self, image_size=224, patch_size=4, in_channel=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], stage_deepth=[2, 2, 17, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        
        # split to patches
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channel=in_channel,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        # build encoder and bottleneck layers
        self.layers_down = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerDown(dim=int(embed_dim * (2 ** i_layer)),
                                   input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                     patches_resolution[1] // (2 ** i_layer)),
                                   depth=depths[i_layer],
                                   stage_deepth=stage_deepth[i_layer],
                                   num_heads=num_heads[i_layer],
                                   window_size=window_size,
                                   mlp_ratio=self.mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   proj_drop=drop_rate, attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                   use_checkpoint=use_checkpoint)
            self.layers_down.append(layer)
            
            # build decoder layers
            self.layers_up = nn.ModuleList()
            self.concat_back_dim = nn.ModuleList()  # skip
            for i_layer in range(self.num_layers):
                concat_linear = ConnectBranchRight(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                                   int(embed_dim * 2 ** (
                                                           self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
                if i_layer == 0:
                    layer_up = PatchExpand(
                        input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                          patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                        dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
                else:
                    layer_up = BasicLayerUp(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                            input_resolution=(
                                                patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                            depth=depths[(self.num_layers - 1 - i_layer)],
                                            stage_deepth=stage_deepth[i_layer],
                                            num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                            window_size=window_size,
                                            mlp_ratio=self.mlp_ratio,
                                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            proj_drop=drop_rate, attn_drop=attn_drop_rate,
                                            drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                                depths[:(self.num_layers - 1 - i_layer) + 1])],
                                            norm_layer=norm_layer,
                                            upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                            use_checkpoint=use_checkpoint)
                self.layers_up.append(layer_up)
                self.concat_back_dim.append(concat_linear)
                
                self.norm_down = norm_layer(self.num_features)
                self.norm_up = norm_layer(self.embed_dim)
                
                if self.final_upsample == "expand_first":
                    print("---final upsample expand_first---")
                    self.up = FinalPatchExpand_X4(dim=embed_dim,
                                                  input_resolution=(image_size // patch_size, image_size // patch_size),
                                                  dim_scale=4)
                    self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1,
                                            bias=False)
                
                self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    
    # Encoder and Bottleneck
    def forward_down_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x_downsample = []  # skip
        for layer in self.layers_down:
            x_downsample.append(x)
            x = layer(x)
        
        x = self.norm_down(x)  # B L C
        
        return x, x_downsample
    
    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                """skip"""
                # x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = x + x_downsample[3 - inx]
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
        
        x = self.norm_up(x)  # B L C
        
        return x
    
    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"
        
        if self.final_upsample == "expand_first":
            x = self.up(x)
            x = x.view(B, 4 * H, 4 * W, -1)
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)
        
        return x
    
    def forward(self, x):
        x, x_downsample = self.forward_down_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        
        return x
