import torch
from torch import nn
from timm.models.layers import trunc_normal_
import math
import time

from src.utils.core import imresize
from src.utils.stream_helper import get_state_dict, pad_for_x, slice_to_x, \
    get_downsampled_shape, get_slice_shape, encode_p_two_layer, decode_p_two_layer, filesize

from src.models.submodels.BL import BL
from src.models.submodels.ILP import InterLayerPrediction, LatentInterLayerPrediction
from src.models.submodels.EL import EL

g_ch_1x = 48
g_ch_2x = 64
g_ch_4x = 96
g_ch_8x = 96
g_ch_16x = 128


class DMC(nn.Module):
    def __init__(self, anchor_num=4, r=4.0, ec_thread=False, stream_part=1, inplace=False):
        super().__init__()
        self.anchor_num = anchor_num

        self.BL_codec = BL(anchor_num=anchor_num, ec_thread=ec_thread, stream_part=stream_part, inplace=inplace)
        self.ILP = InterLayerPrediction(iter_num=2, inplace=inplace)
        self.latent_ILP = LatentInterLayerPrediction(inplace=inplace)
        self.EL_codec = EL(anchor_num=anchor_num, ec_thread=ec_thread, stream_part=stream_part, inplace=inplace)

        self.feature_adaptor_I = nn.Conv2d(3, g_ch_1x, 3, stride=1, padding=1)
        self.feature_adaptor = nn.ModuleList([nn.Conv2d(g_ch_1x, g_ch_1x, 1) for _ in range(3)])

        self.r = r

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict)
        self.BL_codec.load_fine_q()
        self.EL_codec.load_fine_q()

    def feature_extract(self, dpb, index):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        else:
            index = index % 4
            index_map = [0, 1, 0, 2]
            index = index_map[index]
            feature = self.feature_adaptor[index](dpb["ref_feature"])
        return feature

    def forward_one_frame(self, x, dpb_BL, dpb_EL, q_in_ckpt=False, q_index=None, frame_index=None):
        B, _, H, W = x.size()
        x_EL, flag_shape = pad_for_x(x, p=16, mode='replicate')

        if flag_shape == (0, 0, 0, 0):
            x_BL = imresize(x, scale=1 / self.r)
            _, _, h, w = x_BL.size()
            x_BL, slice_shape = pad_for_x(x_BL, p=16)
        else:
            x_BL = imresize(x_EL, scale=1 / self.r)

        BL_res = self.BL_codec.evaluate(x_BL, dpb_BL, q_in_ckpt=q_in_ckpt, q_index=q_index, frame_idx=frame_index)

        ref_frame = dpb_EL['ref_frame']
        anchor_num = self.anchor_num // 2
        if q_index is None:
            if B == ref_frame.size(0):
                ref_frame = dpb_EL['ref_frame'].repeat(anchor_num, 1, 1, 1)
                dpb_EL["ref_frame"] = ref_frame
            x_EL = x_EL.repeat(anchor_num, 1, 1, 1)
        # ILP
        if flag_shape == (0, 0, 0, 0):  # No padding for layer1
            feature_hat_BL = slice_to_x(BL_res['dpb']['ref_feature'], slice_shape)
            mv_hat_BL = slice_to_x(BL_res['dpb']['ref_mv'], slice_shape)
        else:
            feature_hat_BL = BL_res['dpb']['ref_feature']  # padding for layer1
            mv_hat_BL = BL_res['dpb']['ref_mv']
            slice_shape = None
        y_hat_BL = BL_res['dpb']['ref_y']
        ref_feature = self.feature_extract(dpb_EL, index=frame_index)
        context1, context2, context3, warp_frame = self.ILP(feature_hat_BL, mv_hat_BL, ref_feature, ref_frame)
        latent_prior = self.latent_ILP(y_hat_BL, dpb_EL['ref_ys'], slice_shape)
        # EL
        dpb_EL['ref_latent'] = latent_prior
        dpb_EL['ref_feature'] = [context1, context2, context3]
        EL_res = self.EL_codec.evaluate(x_EL, dpb_EL, q_in_ckpt=q_in_ckpt, q_index=q_index)
        all_res = {
            'dpb_BL': BL_res['dpb'],
            'dpb_EL': EL_res['dpb'],
            'bit': BL_res['bit'] + EL_res['bit']
        }
        return all_res

    def evaluate(self, x, base_dpb, dpb, q_in_ckpt, q_index, frame_idx=0):
        return self.forward_one_frame(x, base_dpb, dpb, q_in_ckpt=q_in_ckpt, q_index=q_index, frame_index=frame_idx)

    def encode_decode(self, x, base_dpb, dpb, q_in_ckpt, q_index, output_path=None,
                      pic_width=None, pic_height=None, frame_idx=0):
        if output_path is not None:
            device = x.device
            dpb_copy = dpb.copy()
            # generate base input
            x_padded, tmp_shape = pad_for_x(x, p=16, mode='replicate')  # 1080p uses replicate

            if tmp_shape == (0, 0, 0, 0):
                base_x = imresize(x, scale=1 / self.r)
                _, _, h, w = base_x.size()
                base_x, slice_shape = pad_for_x(base_x, p=16)
            else:
                base_x = imresize(x_padded, scale=1 / self.r)  # direct downsampling fo 1080p

            # Encode
            torch.cuda.synchronize(device=device)
            t0 = time.time()
            encoded = self.compress(base_x, x_padded, base_dpb, dpb, q_in_ckpt, q_index, frame_idx, tmp_shape)
            encode_p_two_layer(encoded['base_bit_stream'], encoded['bit_stream'], q_in_ckpt, q_index, output_path)

            bits = filesize(output_path) * 8

            # Decode
            torch.cuda.synchronize(device=device)
            t1 = time.time()
            q_in_ckpt, q_index, string1, string2 = decode_p_two_layer(output_path)
            decoded = self.decompress(base_dpb, dpb_copy, string1, string2, pic_height // self.r, pic_width // self.r,
                                      q_in_ckpt, q_index, frame_idx)
            torch.cuda.synchronize(device=device)
            t2 = time.time()
            result = {
                "dpb_BL": decoded['base_dpb'],
                "dpb_EL": decoded['dpb'],
                "bit_BL": 0,
                "bit_EL": 0,
                "bit": bits,
                "encoding_time": t1 - t0,
                "decoding_time": t2 - t1,
            }
            return result
        else:
            encoded = self.forward_one_frame(x, base_dpb, dpb, q_in_ckpt=q_in_ckpt, q_index=q_index,
                                             frame_index=frame_idx, forward_end='MF')
            result = {
                "dpb_BL": encoded['dpb_BL'],
                "dpb_EL": encoded['dpb_EL'],
                "bit": encoded['bit'].item(),
                "bit_BL": encoded['bit_mv'].item(),
                "bit_EL": encoded['bit_res'].item(),
                "encoding_time": 0,
                "decoding_time": 0,
            }
            return result

    def compress(self, base_x, x, base_dpb, dpb, q_in_ckpt, q_index, frame_idx, slice_shape):
        base_result = self.BL_codec.compress(base_x, base_dpb, q_in_ckpt, q_index, frame_idx)
        if slice_shape == (0, 0, 0, 0):  # train or CDE !!!!!!!!
            feature_hat_BL = slice_to_x(base_result['dpb']['ref_feature'], slice_shape)
            mv_hat_BL = slice_to_x(base_result['dpb']['ref_mv'], slice_shape)
        else:
            feature_hat_BL = base_result['dpb']['ref_feature']
            mv_hat_BL = base_result['dpb']['ref_mv']
            slice_shape = None
        y_hat_BL = base_result['dpb']['ref_y']
        ref_feature = self.feature_extract(dpb, index=frame_idx)
        context1, context2, context3, warp_frame = self.ILP(feature_hat_BL, mv_hat_BL, ref_feature, dpb['ref_frame'])
        latent_prior = self.latent_ILP(y_hat_BL, dpb['ref_ys'], slice_shape)
        # EL
        dpb['ref_latent'] = latent_prior
        dpb['ref_feature'] = [context1, context2, context3]

        result = self.EL_codec.compress(x, dpb, q_in_ckpt, q_index)
        all_res = {
            'base_dpb': base_result['dpb'],
            'dpb': result['dpb'],
            'base_bit_stream': base_result['bit_stream'],
            'bit_stream': result['bit_stream']
        }
        return all_res

    def decompress(self, base_dpb, dpb, string1, string2, height, width, q_in_ckpt, q_index, frame_idx):
        base_result = self.BL_codec.decompress(base_dpb, string1, height, width, q_in_ckpt, q_index, frame_idx)
        slice_shape = get_slice_shape(height, width)
        if slice_shape == (0, 0, 0, 0):  # train or CDE !!!!!!!!
            feature_hat_BL = slice_to_x(base_result['dpb']['ref_feature'], slice_shape)
            mv_hat_BL = slice_to_x(base_result['dpb']['ref_mv'], slice_shape)
        else:
            feature_hat_BL = base_result['dpb']['ref_feature']
            mv_hat_BL = base_result['dpb']['ref_mv']
            slice_shape = None
        y_hat_BL = base_result['dpb']['ref_y']
        ref_feature = self.feature_extract(dpb, index=frame_idx)
        context1, context2, context3, warp_frame = self.ILP(feature_hat_BL, mv_hat_BL, ref_feature, dpb['ref_frame'])
        latent_prior = self.latent_ILP(y_hat_BL, dpb['ref_ys'], slice_shape)
        # EL
        dpb['ref_latent'] = latent_prior
        dpb['ref_feature'] = [context1, context2, context3]
        result = self.EL_codec.decompress(dpb, string2, q_in_ckpt, q_index)
        all_res = {
            'base_dpb': base_result['dpb'],
            'dpb': result['dpb']
        }
        return all_res
