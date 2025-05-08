# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import concurrent.futures
import json
import multiprocessing
import time

import torch
import numpy as np
from src.models.SEVC_main_model import DMC
from src.models.image_model import IntraNoAR
from src.utils.common import str2bool, create_folder, generate_log_json, dump_json
from src.utils.stream_helper import get_state_dict, pad_for_x, get_slice_shape, slice_to_x
from src.utils.video_reader import PNGReader
from pytorch_msssim import ms_ssim
from src.utils.core import imresize


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument("--ec_thread", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--stream_part_i", type=int, default=1)
    parser.add_argument("--stream_part_p", type=int, default=1)
    parser.add_argument('--i_frame_model_path', type=str)
    parser.add_argument('--p_frame_model_path', type=str)
    parser.add_argument('--rate_num', type=int, default=4)
    parser.add_argument('--i_frame_q_indexes', type=int, nargs="+")
    parser.add_argument('--p_frame_q_indexes', type=int, nargs="+")
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--ratio', type=float, default=4.0)
    parser.add_argument('--refresh_interval', type=int, default=32)

    args = parser.parse_args()
    return args


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


def run_test(p_frame_net, i_frame_net, args):
    frame_num = args['frame_num']
    gop_size = args['gop_size']
    refresh_interval = args['refresh_interval']
    device = next(i_frame_net.parameters()).device

    src_reader = PNGReader(args['src_path'], args['src_width'], args['src_height'])

    frame_types = []
    psnrs = []
    msssims = []

    bits = []
    frame_pixel_num = 0

    start_time = time.time()
    p_frame_number = 0
    with torch.no_grad():
        for frame_idx in range(frame_num):
            rgb = src_reader.read_one_frame(dst_format="rgb")
            x = np_image_to_tensor(rgb)
            x = x.to(device)
            pic_height = x.shape[2]
            pic_width = x.shape[3]

            if frame_pixel_num == 0:
                frame_pixel_num = x.shape[2] * x.shape[3]
            else:
                assert frame_pixel_num == x.shape[2] * x.shape[3]

            # pad if necessary
            slice_shape = get_slice_shape(pic_height, pic_width, p=16)

            if frame_idx == 0 or (gop_size > 0 and frame_idx % gop_size == 0):
                x_padded, _ = pad_for_x(x, p=16, mode='replicate')
                result = i_frame_net.evaluate(x_padded, args['q_in_ckpt'], args['i_frame_q_index'])
                if slice_shape == (0, 0, 0, 0):
                    ref_BL, _ = pad_for_x(imresize(result['x_hat'], scale=1 / args['ratio']), p=16)
                else:
                    ref_BL = imresize(result['x_hat'], scale=1 / args['ratio'])  # 1080p direct resize
                dpb_BL = {
                    "ref_frame": ref_BL,
                    "ref_feature": None,
                    "ref_mv_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                }
                dpb_EL = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_mv_feature": None,
                    "ref_ys": [None, None, None],
                    "ref_mv_y": None,
                }
                recon_frame = result["x_hat"]
                frame_types.append(0)
                bits.append(result["bit"])
            else:
                if frame_idx % refresh_interval == 1:
                    dpb_BL['ref_feature'] = None
                    dpb_EL['ref_feature'] = None
                result = p_frame_net.evaluate(x, dpb_BL, dpb_EL, args['q_in_ckpt'], args['i_frame_q_index'], frame_idx=(frame_idx % refresh_interval) % 4)
                dpb_BL = result["dpb_BL"]
                dpb_EL = result["dpb_EL"]
                recon_frame = dpb_EL["ref_frame"]
                frame_types.append(1)
                bits.append(result['bit'])
                p_frame_number += 1

            recon_frame = recon_frame.clamp_(0, 1)
            x_hat = slice_to_x(recon_frame, slice_shape)

            psnr = PSNR(x_hat, x)
            msssim = ms_ssim(x_hat, x, data_range=1).item()  # cal msssim in psnr model
            psnrs.append(psnr)
            msssims.append(msssim)

    print('sequence name:', args['video_path'], '    q:', args['rate_idx'], 'Finished')

    test_time = {}
    test_time['test_time'] = time.time() - start_time
    log_result = generate_log_json(frame_num, frame_pixel_num, frame_types, bits, psnrs, msssims)
    return log_result


i_frame_net = None  # the model is initialized after each process is spawn, thus OK for multiprocess
p_frame_net = None


def evaluate_one(args):
    global i_frame_net
    global p_frame_net

    sub_dir_name = args['video_path']

    args['src_path'] = os.path.join(args['dataset_path'], sub_dir_name)
    result = run_test(p_frame_net, i_frame_net, args)

    result['ds_name'] = args['ds_name']
    result['video_path'] = args['video_path']
    result['rate_idx'] = args['rate_idx']

    return result


def worker(args):
    return evaluate_one(args)


def init_func(args):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    gpu_num = 0
    if args.cuda:
        gpu_num = torch.cuda.device_count()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    global i_frame_net
    i_state_dict = get_state_dict(args.i_frame_model_path)
    i_frame_net = IntraNoAR(ec_thread=args.ec_thread, stream_part=args.stream_part_i,
                            inplace=True)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    global p_frame_net
    p_state_dict = get_state_dict(args.p_frame_model_path)
    p_frame_net = DMC(ec_thread=args.ec_thread, stream_part=args.stream_part_p,
                      inplace=True)
    p_frame_net.load_state_dict(p_state_dict)
    p_frame_net = p_frame_net.to(device)
    p_frame_net.eval()


def main():
    begin_time = time.time()

    torch.backends.cudnn.enabled = True
    args = parse_args()

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num,
                                                                 initializer=init_func,
                                                                 initargs=(args,))
    objs = []

    count_frames = 0
    count_sequences = 0

    rate_num = args.rate_num
    i_frame_q_scale_enc, i_frame_q_scale_dec = \
        IntraNoAR.get_q_scales_from_ckpt(args.i_frame_model_path)
    i_frame_q_indexes = []
    q_in_ckpt = False
    if args.i_frame_q_indexes is not None:
        assert len(args.i_frame_q_indexes) == rate_num
        i_frame_q_indexes = args.i_frame_q_indexes
    elif len(i_frame_q_scale_enc) == rate_num:
        assert rate_num == 4
        q_in_ckpt = True
        i_frame_q_indexes = [0, 1, 2, 3]
    else:
        assert rate_num >= 2 and rate_num <= 64
        for i in np.linspace(0, 63, num=rate_num):
            i_frame_q_indexes.append(int(i + 0.5))

    if args.p_frame_q_indexes is None:
        p_frame_q_indexes = i_frame_q_indexes

    print(f"testing {rate_num} rates, using q_indexes: ", end='')
    for q in i_frame_q_indexes:
        print(f"{q}, ", end='')
    print()

    root_path = config['root_path']
    config = config['test_classes']
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for seq_name in config[ds_name]['sequences']:
            count_sequences += 1
            for rate_idx in range(rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['q_in_ckpt'] = q_in_ckpt
                cur_args['i_frame_q_index'] = i_frame_q_indexes[rate_idx]
                cur_args['p_frame_q_index'] = p_frame_q_indexes[rate_idx]
                cur_args['video_path'] = seq_name
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq_name]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq_name]['width']
                cur_args['gop_size'] = config[ds_name]['sequences'][seq_name]['gop']
                cur_args['frame_num'] = config[ds_name]['sequences'][seq_name]['frames']
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['ds_name'] = ds_name
                cur_args['ratio'] = args.ratio
                cur_args['refresh_interval'] = args.refresh_interval

                count_frames += cur_args['frame_num']

                obj = threadpool_executor.submit(worker, cur_args)
                objs.append(obj)

    results = []
    for obj in objs:
        result = obj.result()
        results.append(result)

    log_result = {}
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        log_result[ds_name] = {}
        for seq in config[ds_name]['sequences']:
            log_result[ds_name][seq] = {}
            for rate in range(rate_num):
                for res in results:
                    if res['rate_idx'] == rate and ds_name == res['ds_name'] \
                            and seq == res['video_path']:
                        log_result[ds_name][seq][f"{rate:03d}"] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)

    total_minutes = (time.time() - begin_time) / 60
    print('Test finished')
    print(f'Tested {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
