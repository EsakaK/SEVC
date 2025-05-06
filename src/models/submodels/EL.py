import time

import torch
from torch import nn
import numpy as np

from src.models.common_model import CompressionModel
from src.models.video_net import ResBlock, UNet
from src.layers.layers import subpel_conv3x3, DepthConvBlock
from src.utils.stream_helper import encode_p, decode_p, filesize, \
    get_state_dict

g_ch_1x = 48
g_ch_2x = 64
g_ch_4x = 96
g_ch_8x = 96
g_ch_16x = 128


def shift_and_add(ref_ys: list, new_y):
    ref_ys.pop(0)
    ref_ys.append(new_y)
    return ref_ys


class ContextualEncoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_1x + 3, g_ch_2x, 3, stride=2, padding=1)
        self.res1 = ResBlock(g_ch_2x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv2 = nn.Conv2d(g_ch_2x * 2, g_ch_4x, 3, stride=2, padding=1)
        self.res2 = ResBlock(g_ch_4x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.conv3 = nn.Conv2d(g_ch_4x * 2, g_ch_8x, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3, quant_step):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = feature * quant_step
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.up1 = subpel_conv3x3(g_ch_16x, g_ch_8x, 2)
        self.up2 = subpel_conv3x3(g_ch_8x, g_ch_4x, 2)
        self.res1 = ResBlock(g_ch_4x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up3 = subpel_conv3x3(g_ch_4x * 2, g_ch_2x, 2)
        self.res2 = ResBlock(g_ch_2x * 2, bottleneck=True, slope=0.1,
                             end_with_relu=True, inplace=inplace)
        self.up4 = subpel_conv3x3(g_ch_2x * 2, 32, 2)

    def forward(self, x, context2, context3, quant_step):
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = feature * quant_step
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=g_ch_1x, res_channel=32, inplace=False):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, g_ch_1x, 3, stride=1, padding=1)
        self.unet_1 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.unet_2 = UNet(g_ch_1x, g_ch_1x, inplace=inplace)
        self.recon_conv = nn.Conv2d(g_ch_1x, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon


class ConetxtEncoder(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.activate = nn.LeakyReLU(0.1, inplace=inplace)
        self.enc1 = nn.Conv2d(g_ch_1x, g_ch_2x, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(g_ch_2x * 2, g_ch_4x, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(g_ch_4x * 2, g_ch_8x, 3, stride=2, padding=1)
        self.enc4 = nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1)

    def forward(self, ctx1, ctx2, ctx3):
        f = self.activate(self.enc1(ctx1))
        f = self.activate(self.enc2(torch.cat([ctx2, f], dim=1)))
        f = self.activate(self.enc3(torch.cat([ctx3, f], dim=1)))
        f = self.enc4(f)
        return f


class EL(CompressionModel):
    def __init__(self, anchor_num=4, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='laplace', z_channel=64, ec_thread=ec_thread, stream_part=stream_part)
        self.anchor_num = int(anchor_num)
        self.noise_level = 0.4

        self.contextual_encoder = ContextualEncoder(inplace=inplace)

        self.temporal_prior_encoder = ConetxtEncoder(inplace=inplace)

        self.y_prior_fusion_adaptor_0 = DepthConvBlock(g_ch_16x * 2, g_ch_16x * 2,
                                                       inplace=inplace)
        self.y_prior_fusion_adaptor_1 = DepthConvBlock(g_ch_16x * 2, g_ch_16x * 2,
                                                       inplace=inplace)
        self.y_prior_fusion_adaptor_2 = DepthConvBlock(g_ch_16x * 2, g_ch_16x * 2,
                                                       inplace=inplace)
        self.y_prior_fusion_adaptor_3 = DepthConvBlock(g_ch_16x * 2, g_ch_16x * 2,
                                                       inplace=inplace)

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(g_ch_16x * 2, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
        )

        self.y_spatial_prior_adaptor_1 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(g_ch_16x * 4, g_ch_16x * 3, 1)

        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=inplace),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 2, inplace=inplace),
        )

        self.contextual_decoder = ContextualDecoder(inplace=inplace)
        self.recon_generation_net = ReconGeneration(inplace=inplace)

        self.y_q_basic_enc = nn.Parameter(torch.ones((1, g_ch_2x * 2, 1, 1)))
        self.y_q_scale_enc = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_enc_fine = None
        self.y_q_basic_dec = nn.Parameter(torch.ones((1, g_ch_2x, 1, 1)))
        self.y_q_scale_dec = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_scale_dec_fine = None

        self.previous_frame_recon = None
        self.previous_frame_feature = None
        self.previous_frame_y_hat = [None, None, None]

    def load_fine_q(self):
        with torch.no_grad():
            y_q_scale_enc_fine = np.linspace(np.log(self.y_q_scale_enc[0, 0, 0, 0]),
                                             np.log(self.y_q_scale_enc[3, 0, 0, 0]), 64)
            self.y_q_scale_enc_fine = np.exp(y_q_scale_enc_fine)
            y_q_scale_dec_fine = np.linspace(np.log(self.y_q_scale_dec[0, 0, 0, 0]),
                                             np.log(self.y_q_scale_dec[3, 0, 0, 0]), 64)
            self.y_q_scale_dec_fine = np.exp(y_q_scale_dec_fine)

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_q_scale_enc = ckpt["y_q_scale_enc"].reshape(-1)
        y_q_scale_dec = ckpt["y_q_scale_dec"].reshape(-1)
        return y_q_scale_enc, y_q_scale_dec

    def res_prior_param_decoder(self, dpb, contexts):
        temporal_params = self.temporal_prior_encoder(*contexts)
        params = torch.cat((temporal_params, dpb['ref_latent']), dim=1)
        if dpb["ref_ys"][-1] is None:
            params = self.y_prior_fusion_adaptor_0(params)
        elif dpb["ref_ys"][-2] is None:
            params = self.y_prior_fusion_adaptor_1(params)
        elif dpb["ref_ys"][-3] is None:
            params = self.y_prior_fusion_adaptor_2(params)
        else:
            params = self.y_prior_fusion_adaptor_3(params)
        params = self.y_prior_fusion(params)
        return params

    def get_recon_and_feature(self, y_hat, context1, context2, context3, y_q_dec):
        recon_image_feature = self.contextual_decoder(y_hat, context2, context3, y_q_dec)
        feature, x_hat = self.recon_generation_net(recon_image_feature, context1)
        # x_hat = x_hat.clamp_(0, 1)
        return x_hat, feature

    def get_q_for_inference(self, q_in_ckpt, q_index):
        y_q_scale_enc = self.y_q_scale_enc if q_in_ckpt else self.y_q_scale_enc_fine
        y_q_enc = self.get_curr_q(y_q_scale_enc, self.y_q_basic_enc, q_index=q_index)
        y_q_scale_dec = self.y_q_scale_dec if q_in_ckpt else self.y_q_scale_dec_fine
        y_q_dec = self.get_curr_q(y_q_scale_dec, self.y_q_basic_dec, q_index=q_index)
        return y_q_enc, y_q_dec

    def forward_one_frame(self, x, dpb, q_in_ckpt=False, q_index=None):
        y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        context1, context2, context3 = dpb['ref_feature']
        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)
        params = self.res_prior_param_decoder(dpb, [context1, context2, context3])

        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec)

        B, _, H, W = x.size()
        pixel_num = H * W
        mse = self.mse(x, x_hat)
        mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num

        y_for_bit = y_q
        bits_y = self.get_y_laplace_bits(y_for_bit, scales_hat)

        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num

        # storage multi-frame latent
        ref_ys = shift_and_add(dpb['ref_ys'], y_hat)

        return {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_ys": ref_ys,
            },
            "bit": bit,
        }

    def evaluate(self, x, dpb, q_in_ckpt=False, q_index=None):
        return self.forward_one_frame(x, dpb, q_in_ckpt, q_index)

    def encode_decode(self, x, dpb, q_in_ckpt, q_index, output_path=None,
                      pic_width=None, pic_height=None, frame_idx=0):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        if output_path is not None:
            device = x.device
            torch.cuda.synchronize(device=device)
            t0 = time.time()
            encoded = self.compress(x, dpb, q_in_ckpt, q_index, frame_idx)
            encode_p(encoded['bit_stream'], q_in_ckpt, q_index, output_path)
            bits = filesize(output_path) * 8
            torch.cuda.synchronize(device=device)
            t1 = time.time()
            q_in_ckpt, q_index, string = decode_p(output_path)

            decoded = self.decompress(dpb, string, pic_height, pic_width,
                                      q_in_ckpt, q_index, frame_idx)
            torch.cuda.synchronize(device=device)
            t2 = time.time()
            result = {
                "dpb": decoded["dpb"],
                "bit": bits,
                "encoding_time": t1 - t0,
                "decoding_time": t2 - t1,
            }
            return result

        encoded = self.forward_one_frame(x, dpb, q_in_ckpt=q_in_ckpt, q_index=q_index)
        result = {
            "dpb": encoded['dpb'],
            "bit": encoded['bit'].item(),
            "encoding_time": 0,
            "decoding_time": 0,
        }
        return result

    def compress(self, x, dpb, q_in_ckpt, q_index):
        # pic_width and pic_height may be different from x's size. x here is after padding
        y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)
        context1, context2, context3 = dpb['ref_feature']
        y = self.contextual_encoder(x, context1, context2, context3, y_q_enc)
        params = self.res_prior_param_decoder(dpb, [context1, context2, context3])
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = \
            self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        self.entropy_coder.reset()
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        self.entropy_coder.flush()

        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec)
        bit_stream = self.entropy_coder.get_encoded_stream()
        # storage multi-frame latent
        ref_ys = shift_and_add(dpb['ref_ys'], y_hat)

        result = {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_ys": ref_ys,
            },
            "bit_stream": bit_stream,
        }
        return result

    def decompress(self, dpb, string, q_in_ckpt, q_index):
        y_q_enc, y_q_dec = self.get_q_for_inference(q_in_ckpt, q_index)

        self.entropy_coder.set_stream(string)
        context1, context2, context3 = dpb['ref_feature']

        params = self.res_prior_param_decoder(dpb, [context1, context2, context3])
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)
        x_hat, feature = self.get_recon_and_feature(y_hat, context1, context2, context3, y_q_dec)
        ref_ys = shift_and_add(dpb['ref_ys'], y_hat)

        result = {
            "dpb": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_ys": ref_ys,
            }
        }
        return result
