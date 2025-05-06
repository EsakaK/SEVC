import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_
from functools import lru_cache

p_s = 4
w_s = 16


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2],
                                                                  C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attn_Attn(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)  # nW*B, num_heads, L, L

        return attn


class SwinTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(2, 7, 7),
                 shift_size=(0, 0, 0),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        b, t, h, w, c = x.shape  # n t h w c

        shortcut = x

        x = self.norm1(x)

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - t % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x,
                                     self.window_size)  # nw*b, window_size[0]*window_size[1]*window_size[2], c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if any(i > 0 for i in self.shift_size):
            attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        else:
            attn_windows = self.attn(x_windows, mask=None)

        # merge windows
        attn_windows = attn_windows.view(-1, *(self.window_size + (c,)))
        shifted_x = window_reverse(attn_windows, self.window_size, b, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in self.shift_size):
            x = torch.roll(
                shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :t, :h, :w, :].contiguous()

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # b,t,h,w,c
        return x

    def get_attn_map_SWIN(self, x, attn_mask):
        b, t, h, w, c = x.shape  # n t h w c

        shortcut = x

        x = self.norm1(x)

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.window_size[0] - t % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        pad_r = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in self.shift_size):
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x,
                                     self.window_size)  # nw*b, window_size[0]*window_size[1]*window_size[2], c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if any(i > 0 for i in self.shift_size):
            attn_windows = self.attn.get_attn_Attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        else:
            attn_windows = self.attn.get_attn_Attn(x_windows, mask=None)

        # b,t,h,w,c
        return attn_windows


class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0], window_size[1] // 2, window_size[2] // 2),
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

    def forward(self, x, attn_mask):
        # n c t h w
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # n t h w c
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        x = x.permute(0, 4, 1, 2, 3).contiguous()  # b,c,t,h,w
        return x

    def get_attn_map_BasicLayer(self, x, attn_mask):
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # n t h w c
        attn_map = self.blocks[0].get_attn_map_SWIN(x, attn_mask)
        return attn_map


class RSTB(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 patch_size=1,
                 resi_connection='1conv',
                 num_frames=5):
        super(RSTB, self).__init__()

        self.dim = dim
        self.num_frames = num_frames
        self.residual_group = BasicLayer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, attn_mask):
        n, c, t, h, w = x.shape  # n c t h w
        x_ori = x
        x = self.residual_group(x, attn_mask)  # n c t h w
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)  # (n t) c h w
        x = self.conv(x)  # (n t), c, h, w
        x = x.view(n, t, -1, h, w)  # n, t, c, h, w
        x = self.patch_embed(x)  # n, t, c, h, w  -->  n, c, t, h, w, fake patch_emb
        x = x + x_ori
        return x

    def get_attn_map_RSTB(self, x, attn_mask):
        attn_map = self.residual_group.get_attn_map_BasicLayer(x, attn_mask)  # n c t h w
        return attn_map


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=1, in_chans=48, embed_dim=48, num_frames=2, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        n, t, c, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (n, c, t, h ,w)
        if self.norm is not None:
            h, w = h // self.patch_size, w // self.patch_size
            x = x.flatten(2).transpose(1, 2)  # (n, L, c)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, t, h, w)
        return x  # (n c t h w)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x


@lru_cache()
def compute_mask(t, x_size, window_size, shift_size, device):
    h, w = x_size
    Dp = int(np.ceil(t / window_size[0])) * window_size[0]
    Hp = int(np.ceil(h / window_size[1])) * window_size[1]
    Wp = int(np.ceil(w / window_size[2])) * window_size[2]
    img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=device)  # 1 h w 1

    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    # mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    mask_windows = mask_windows.view(-1, window_size[0] * window_size[1] * window_size[2])
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class SwinIRFM(nn.Module):
    def __init__(self,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 resi_connection='1conv',
                 num_frames=2,
                 **kwargs):
        super(SwinIRFM, self).__init__()
        num_in_ch = in_chans  # 3
        num_feat = 64
        self.window_size = window_size
        self.patch_size = patch_size
        self.shift_size = (window_size[0], window_size[1] // 2, window_size[2] // 2)
        self.num_frames = num_frames

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_first_feat = nn.Conv2d(num_feat, embed_dim, 3, 1, 1)
        # self.feature_extraction = make_layer(ResidualBlockNoBN, num_blocks_extraction, mid_channels=embed_dim)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        # Ture patch_emb
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            num_frames=num_frames,
            norm_layer=norm_layer if self.patch_norm else None)

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build RSTB blocks
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                resi_connection=resi_connection,
                num_frames=num_frames)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

    def forward_features(self, x):
        x = self.patch_embed(x)  # (n,t,c,h,w) --> # (n c t h w), h,w ↓ 2
        x = self.pos_drop(x)
        x_size = (x.shape[3], x.shape[4])
        attn_mask = compute_mask(self.num_frames, x_size, tuple(self.window_size), self.shift_size, x.device)
        for layer in self.layers:
            x = layer(x.contiguous(), attn_mask)

        x = x.permute(0, 2, 3, 4, 1).contiguous()  # n t h w c
        x = self.norm(x)  # n t h w c

        x = x.permute(0, 1, 4, 2, 3).contiguous()  # n t c h w

        return x

    def forward(self, x):
        x_cur = x[:, 0, :, :, :]
        feats = self.forward_features(x)  # (n,t,c,h,w) --> (n,t,c,h,w)

        x = self.conv_after_body(feats[:, 0, :, :, :])  # h,w ↑ 2
        return x + x_cur

    def forward_attn_map(self, x):
        x = self.patch_embed(x)  # (n,t,c,h,w) --> # (n c t h w)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_size = (x.shape[3], x.shape[4])
        attn_mask = compute_mask(self.num_frames, x_size, tuple(self.window_size), self.shift_size, x.device)
        # for layer in self.layers:
        #     x = layer(x.contiguous(), attn_mask)
        attn_map = self.layers[0].get_attn_map_RSTB(x.contiguous(), attn_mask)

        return attn_map

    def get_attn_map(self, x):
        attn_map = self.forward_attn_map(x)  # (n,t,c,h,w) --> (n,t,c,h,w)
        return attn_map


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x
