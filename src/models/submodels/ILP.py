import torch
from torch import nn

from src.layers.layers import subpel_conv3x3, subpel_conv1x1, DepthConvBlock
from src.utils.stream_helper import pad_for_x, get_padded_size, slice_to_x
from src.models.video_net import ResBlock, bilinearupsacling, flow_warp
from src.models.submodels.RSTB import SwinIRFM

g_ch_1x = 48
g_ch_2x = 64
g_ch_4x = 96
g_ch_8x = 96
g_ch_16x = 128


class RefineUnit(nn.Module):
    def __init__(self, in_ch, it_ch, out_ch, inplace=True):
        super().__init__()
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.activate = nn.LeakyReLU(inplace=inplace)
        self.conv0 = nn.Conv2d(in_ch, it_ch, 3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(it_ch, it_ch * 2, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(it_ch * 2, it_ch * 4, 3, stride=1, padding=1)
        self.up2 = subpel_conv1x1(it_ch * 4, it_ch * 2, r=2)
        self.up1 = subpel_conv1x1(it_ch * 3, it_ch, r=2)
        self.up0 = nn.Conv2d(it_ch + in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        # down
        d0 = self.conv0(x)  # 2x

        d1 = self.activate(d0)
        d1 = self.conv1(d1)  # 4x

        d2 = self.activate(d1)
        d2 = self.conv2(d2)  # 4x
        # up
        u2 = self.up2(d2)  # 2x
        u1 = self.up1(torch.cat([self.activate(u2), d0], dim=1))
        u0 = self.up0(torch.cat([self.activate(u1), x], dim=1))
        return u0


class MultiRoundEnhancement(nn.Module):
    def __init__(self, iter_num=2, out_ch=g_ch_1x):
        super().__init__()
        self.iter_num = iter_num
        self.motion_layers = nn.ModuleList([])
        self.texture_layers = nn.ModuleList([])
        for i in range(iter_num):
            self.motion_layers.append(RefineUnit(in_ch=out_ch * 2 + 2, it_ch=out_ch // 2, out_ch=2))
            self.texture_layers.append(RefineUnit(in_ch=out_ch * 2, it_ch=out_ch * 3 // 4, out_ch=out_ch))

    def forward(self, f, t, v):
        for i in range(self.iter_num):
            v = self.motion_layers[i](torch.cat([f, t, v], dim=1)) + v
            f_align = flow_warp(f, v)
            t = self.texture_layers[i](torch.cat([f_align, t], dim=1)) + t
        return t, v

    def get_mv_list(self, f, t, v):
        v_list = [v]
        for i in range(self.iter_num):
            v = self.motion_layers[i](torch.cat([f, t, v], dim=1)) + v
            f_align = flow_warp(f, v)
            t = self.texture_layers[i](torch.cat([f_align, t], dim=1)) + t
            v_list.append(v)
        return v_list

    def get_ctx_list(self, f, t, v):
        t_list = [t]
        for i in range(self.iter_num):
            v = self.motion_layers[i](torch.cat([f, t, v], dim=1)) + v
            f_align = flow_warp(f, v)
            t = self.texture_layers[i](torch.cat([f_align, t], dim=1)) + t
            t_list.append(t)
        return t_list


class OffsetDiversity(nn.Module):
    def __init__(self, in_channel=g_ch_1x, aux_feature_num=g_ch_1x + 3 + 2,
                 offset_num=2, group_num=16, max_residue_magnitude=40, inplace=False):
        super().__init__()
        self.in_channel = in_channel
        self.offset_num = offset_num
        self.group_num = group_num
        self.max_residue_magnitude = max_residue_magnitude
        self.conv_offset = nn.Sequential(
            nn.Conv2d(aux_feature_num, g_ch_2x, 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, g_ch_2x, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=inplace),
            nn.Conv2d(g_ch_2x, 3 * group_num * offset_num, 3, 1, 1),
        )
        self.fusion = nn.Conv2d(in_channel * offset_num, in_channel, 1, 1, groups=group_num)

    def forward(self, x, aux_feature, flow):
        B, C, H, W = x.shape
        out = self.conv_offset(aux_feature)
        out = bilinearupsacling(out)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        mask = torch.sigmoid(mask)
        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow.repeat(1, self.group_num * self.offset_num, 1, 1)

        # warp
        offset = offset.view(B * self.group_num * self.offset_num, 2, H, W)
        mask = mask.view(B * self.group_num * self.offset_num, 1, H, W)
        x = x.view(B * self.group_num, C // self.group_num, H, W)
        x = x.repeat(self.offset_num, 1, 1, 1)
        x = flow_warp(x, offset)
        x = x * mask
        x = x.view(B, C * self.offset_num, H, W)
        x = self.fusion(x)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_1x, g_ch_2x, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_2x, g_ch_4x, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(g_ch_4x, inplace=inplace)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv3_up = subpel_conv3x3(g_ch_4x, g_ch_2x, 2)
        self.res_block3_up = ResBlock(g_ch_2x, inplace=inplace)
        self.conv3_out = nn.Conv2d(g_ch_4x, g_ch_4x, 3, padding=1)
        self.res_block3_out = ResBlock(g_ch_4x, inplace=inplace)
        self.conv2_up = subpel_conv3x3(g_ch_2x * 2, g_ch_1x, 2)
        self.res_block2_up = ResBlock(g_ch_1x, inplace=inplace)
        self.conv2_out = nn.Conv2d(g_ch_2x * 2, g_ch_2x, 3, padding=1)
        self.res_block2_out = ResBlock(g_ch_2x, inplace=inplace)
        self.conv1_out = nn.Conv2d(g_ch_1x * 2, g_ch_1x, 3, padding=1)
        self.res_block1_out = ResBlock(g_ch_1x, inplace=inplace)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out

        return context1, context2, context3


class InterLayerPrediction(nn.Module):
    def __init__(self, iter_num=2, inplace=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.texture_adaptor = nn.Conv2d(g_ch_1x, g_ch_4x, 3, stride=1, padding=1)
        self.feature_extractor = FeatureExtractor(inplace=inplace)
        self.up2 = nn.Sequential(
            subpel_conv3x3(g_ch_4x, g_ch_2x, r=2),
            ResBlock(g_ch_2x)
        )
        self.up1 = nn.Sequential(
            subpel_conv3x3(g_ch_2x, g_ch_1x, r=2),
            ResBlock(g_ch_1x)
        )

        self.mre1 = MultiRoundEnhancement(iter_num=iter_num, out_ch=g_ch_1x)
        self.mre2 = MultiRoundEnhancement(iter_num=iter_num, out_ch=g_ch_2x)
        self.mre3 = MultiRoundEnhancement(iter_num=iter_num, out_ch=g_ch_4x)

        self.align = OffsetDiversity(inplace=inplace)
        self.context_fusion_net = MultiScaleContextFusion(inplace=inplace)

        self.fuse1 = DepthConvBlock(g_ch_1x * 2, g_ch_1x, inplace=inplace)
        self.fuse2 = DepthConvBlock(g_ch_2x * 2, g_ch_2x, inplace=inplace)
        self.fuse3 = DepthConvBlock(g_ch_4x * 2, g_ch_4x, inplace=inplace)

    def forward(self, BL_feature, BL_flow, ref_feature, ref_frame):
        ref_texture3 = self.texture_adaptor(BL_feature)
        ref_feature1, ref_feature2, ref_feature3 = self.feature_extractor(ref_feature)

        ref_texture3, mv3 = self.mre3(ref_feature3, ref_texture3, BL_flow)

        mv2 = bilinearupsacling(mv3) * 2.0
        ref_texture2 = self.up2(ref_texture3)
        ref_texture2, mv2 = self.mre2(ref_feature2, ref_texture2, mv2)

        mv1 = bilinearupsacling(mv2) * 2.0
        ref_texture1 = self.up1(ref_texture2)
        ref_texture1, mv1 = self.mre1(ref_feature1, ref_texture1, mv1)

        warpframe = flow_warp(ref_frame, mv1)
        context1_init = flow_warp(ref_feature1, mv1)
        context1 = self.align(ref_feature1, torch.cat(
            (context1_init, warpframe, mv1), dim=1), mv1)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)

        context1 = self.fuse1(torch.cat([context1, ref_texture1], dim=1))
        context2 = self.fuse2(torch.cat([context2, ref_texture2], dim=1))
        context3 = self.fuse3(torch.cat([context3, ref_texture3], dim=1))
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe


class LatentInterLayerPrediction(nn.Module):
    def __init__(self, window_size=8, inplace=True):
        super().__init__()
        self.window_size = window_size
        self.upsampler = nn.Sequential(
            subpel_conv3x3(g_ch_16x, g_ch_16x, r=2),
            nn.LeakyReLU(inplace),
            subpel_conv3x3(g_ch_16x, g_ch_16x, r=2)
        )
        self.fusion = SwinIRFM(
            patch_size=1,
            in_chans=g_ch_16x,
            embed_dim=g_ch_16x,
            depths=(4, 4, 4, 4),
            num_heads=(8, 8, 8, 8),
            window_size=(4, 8, 8),
            mlp_ratio=2.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            resi_connection='1conv')

    def forward(self, y_hat_BL, ref_ys, slice_shape=None):
        y_hat_BL = self.upsampler(y_hat_BL)
        if slice_shape is not None:
            slice_shape = tuple(item // 4 for item in slice_shape)
            y_hat_BL = slice_to_x(y_hat_BL, slice_shape)

        y_hat_BL, slice_shape = pad_for_x(y_hat_BL, p=self.window_size, mode='replicate')  # query
        ref_ys_cp = []
        for frame_idx in range(len(ref_ys)):
            if ref_ys[frame_idx] is None:
                ref_ys_cp.append(y_hat_BL)
            else:
                ref_ys_cp.append(pad_for_x(ref_ys[frame_idx], p=self.window_size, mode='replicate')[0])  # key-value
        y_fusion = self.fusion(torch.stack([y_hat_BL, *ref_ys_cp], dim=1))
        y_fusion = slice_to_x(y_fusion, slice_shape)
        return y_fusion
