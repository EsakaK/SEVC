import json
import os
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np


def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def get_latest_checkpoint_path(dir_cur):
    files = os.listdir(dir_cur)
    all_best_checkpoints = []
    for file in files:
        if file[-4:] == '.tar' and 'ckpt' in file:
            all_best_checkpoints.append(os.path.join(dir_cur, file))
    if len(all_best_checkpoints) > 0:
        return max(all_best_checkpoints, key=os.path.getmtime)

    return 'not_exist'


def get_latest_status_path(dir_cur):
    files = os.listdir(dir_cur)
    all_status_files = []
    for file in files:
        if 'status_epo' in file in file:
            all_status_files.append(os.path.join(dir_cur, file))
    all_status_files.sort(key=lambda x: os.path.getmtime(x))
    if len(all_status_files) > 2:
        return [all_status_files[-2], all_status_files[-1]]
    return all_status_files


def ddp_sync_state_dict(task_id, to_sync, rank, device, device_ids):
    # to_sync is model or optimizer, which supports state_dict() and load_state_dict()
    ckpt_path = os.path.join("/dev", "shm", f"train_{task_id}_tmp", "tmp_error.ckpt")
    if rank == 0:
        print(f"sync model with {ckpt_path}")
        torch.save(to_sync.state_dict(), ckpt_path)
    dist.barrier(device_ids=device_ids)
    to_sync.load_state_dict(torch.load(ckpt_path, map_location=device))
    dist.barrier(device_ids=device_ids)
    if rank == 0:
        os.remove(ckpt_path)
    dist.barrier(device_ids=device_ids)


def interpolate_log(min_val, max_val, num, decending=True):
    assert max_val > min_val
    assert min_val > 0
    if decending:
        values = np.linspace(np.log(max_val), np.log(min_val), num)
    else:
        values = np.linspace(np.log(min_val), np.log(max_val), num)
    values = np.exp(values)
    return values


def scale_list_to_str(scales):
    s = ''
    for scale in scales:
        s += f'{scale:.2f} '

    return s


def avg_per_rate(result, B, anchor_num, weight=None, key=None):
    if key not in result or result[key] is None:
        return None
    if weight is not None:
        y = result[key] * weight
    else:
        y = result[key]
    y = y.reshape((anchor_num, B))
    return torch.sum(y, dim=1) / B


def avg_layer_weight(result, anchor_num, key='l_w'):
    l_w = result[key].reshape((anchor_num, 1))
    return l_w


def generate_str(x):
    if x is None:
        return 'None'
    # print(x)
    if x.numel() == 1:
        return f'{x.item():.5f} '
    s = ''
    for a in x:
        s += f'{a.item():.5f} '
    return s


def create_folder(path, print_if_create=False):
    if not os.path.exists(path):
        os.makedirs(path)
        if print_if_create:
            print(f"created folder: {path}")


def remove_nan_grad(parameters):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.nan_to_num_(0.0, 0.0, 0.0)


@patch('json.encoder.c_make_encoder', None)
def dump_json(obj, fid, float_digits=-1, **kwargs):
    of = json.encoder._make_iterencode  # pylint: disable=W0212

    def inner(*args, **kwargs):
        args = list(args)
        # fifth argument is float formater which we will replace
        args[4] = lambda o: format(o, '.%df' % float_digits)
        return of(*args, **kwargs)

    with patch('json.encoder._make_iterencode', wraps=inner):
        json.dump(obj, fid, **kwargs)


def generate_log_json(frame_num, frame_pixel_num, test_time, frame_types, bits, psnrs, ssims,
                      psnrs_y=None, psnrs_u=None, psnrs_v=None,
                      ssims_y=None, ssims_u=None, ssims_v=None, verbose=False):
    include_yuv = psnrs_y is not None
    if include_yuv:
        assert psnrs_u is not None
        assert psnrs_v is not None
        assert ssims_y is not None
        assert ssims_u is not None
        assert ssims_v is not None
    i_bits = 0
    i_psnr = 0
    i_psnr_y = 0
    i_psnr_u = 0
    i_psnr_v = 0
    i_ssim = 0
    i_ssim_y = 0
    i_ssim_u = 0
    i_ssim_v = 0
    p_bits = 0
    p_psnr = 0
    p_psnr_y = 0
    p_psnr_u = 0
    p_psnr_v = 0
    p_ssim = 0
    p_ssim_y = 0
    p_ssim_u = 0
    p_ssim_v = 0
    i_num = 0
    p_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            i_bits += bits[idx]
            i_psnr += psnrs[idx]
            i_ssim += ssims[idx]
            i_num += 1
            if include_yuv:
                i_psnr_y += psnrs_y[idx]
                i_psnr_u += psnrs_u[idx]
                i_psnr_v += psnrs_v[idx]
                i_ssim_y += ssims_y[idx]
                i_ssim_u += ssims_u[idx]
                i_ssim_v += ssims_v[idx]
        else:
            p_bits += bits[idx]
            p_psnr += psnrs[idx]
            p_ssim += ssims[idx]
            p_num += 1
            if include_yuv:
                p_psnr_y += psnrs_y[idx]
                p_psnr_u += psnrs_u[idx]
                p_psnr_v += psnrs_v[idx]
                p_ssim_y += ssims_y[idx]
                p_ssim_u += ssims_u[idx]
                p_ssim_v += ssims_v[idx]

    log_result = {}
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['i_frame_num'] = i_num
    log_result['p_frame_num'] = p_num
    log_result['ave_i_frame_bpp'] = i_bits / i_num / frame_pixel_num
    log_result['ave_i_frame_psnr'] = i_psnr / i_num
    log_result['ave_i_frame_msssim'] = i_ssim / i_num
    if include_yuv:
        log_result['ave_i_frame_psnr_y'] = i_psnr_y / i_num
        log_result['ave_i_frame_psnr_u'] = i_psnr_u / i_num
        log_result['ave_i_frame_psnr_v'] = i_psnr_v / i_num
        log_result['ave_i_frame_msssim_y'] = i_ssim_y / i_num
        log_result['ave_i_frame_msssim_u'] = i_ssim_u / i_num
        log_result['ave_i_frame_msssim_v'] = i_ssim_v / i_num
    if verbose:
        log_result['frame_bpp'] = list(np.array(bits) / frame_pixel_num)
        log_result['frame_psnr'] = psnrs
        log_result['frame_msssim'] = ssims
        log_result['frame_type'] = frame_types
        if include_yuv:
            log_result['frame_psnr_y'] = psnrs_y
            log_result['frame_psnr_u'] = psnrs_u
            log_result['frame_psnr_v'] = psnrs_v
            log_result['frame_msssim_y'] = ssims_y
            log_result['frame_msssim_u'] = ssims_u
            log_result['frame_msssim_v'] = ssims_v
    # log_result['test_time'] = test_time['test_time']
    # log_result['encoding_time'] = test_time['encoding_time']
    # log_result['decoding_time'] = test_time['decoding_time']
    if p_num > 0:
        total_p_pixel_num = p_num * frame_pixel_num
        log_result['ave_p_frame_bpp'] = p_bits / total_p_pixel_num
        log_result['ave_p_frame_psnr'] = p_psnr / p_num
        log_result['ave_p_frame_msssim'] = p_ssim / p_num
        if include_yuv:
            log_result['ave_p_frame_psnr_y'] = p_psnr_y / p_num
            log_result['ave_p_frame_psnr_u'] = p_psnr_u / p_num
            log_result['ave_p_frame_psnr_v'] = p_psnr_v / p_num
            log_result['ave_p_frame_msssim_y'] = p_ssim_y / p_num
            log_result['ave_p_frame_msssim_u'] = p_ssim_u / p_num
            log_result['ave_p_frame_msssim_v'] = p_ssim_v / p_num
    else:
        log_result['ave_p_frame_bpp'] = 0
        log_result['ave_p_frame_psnr'] = 0
        log_result['ave_p_frame_msssim'] = 0
        if include_yuv:
            log_result['ave_p_frame_psnr_y'] = 0
            log_result['ave_p_frame_psnr_u'] = 0
            log_result['ave_p_frame_psnr_v'] = 0
            log_result['ave_p_frame_msssim_y'] = 0
            log_result['ave_p_frame_msssim_u'] = 0
            log_result['ave_p_frame_msssim_v'] = 0
    log_result['ave_all_frame_bpp'] = (i_bits + p_bits) / (frame_num * frame_pixel_num)
    log_result['ave_all_frame_psnr'] = (i_psnr + p_psnr) / frame_num
    log_result['ave_all_frame_msssim'] = (i_ssim + p_ssim) / frame_num
    if include_yuv:
        log_result['ave_all_frame_psnr_y'] = (i_psnr_y + p_psnr_y) / frame_num
        log_result['ave_all_frame_psnr_u'] = (i_psnr_u + p_psnr_u) / frame_num
        log_result['ave_all_frame_psnr_v'] = (i_psnr_v + p_psnr_v) / frame_num
        log_result['ave_all_frame_msssim_y'] = (i_ssim_y + p_ssim_y) / frame_num
        log_result['ave_all_frame_msssim_u'] = (i_ssim_u + p_ssim_u) / frame_num
        log_result['ave_all_frame_msssim_v'] = (i_ssim_v + p_ssim_v) / frame_num

    return log_result
