import os
import torch
import argparse
import numpy as np

from pathlib import Path
from src.utils.common import str2bool
from src.models.SEVC_main_model import DMC
from src.models.image_model import IntraNoAR
from src.utils.stream_helper import get_state_dict, slice_to_x, read_uints, read_ints, get_slice_shape, decode_i, decode_p, decode_p_two_layer
from src.utils.video_writer import PNGWriter
import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release")

def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument("--ec_thread", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--stream_part_i", type=int, default=1)
    parser.add_argument("--stream_part_p", type=int, default=1)
    parser.add_argument('--i_frame_model_path', type=str)
    parser.add_argument('--p_frame_model_path', type=str)
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--refresh_interval', type=int, default=32)
    parser.add_argument('-b', '--bin_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)

    args = parser.parse_args()
    return args


def init_func(args):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    if args.cuda:
        device = "cuda:0"
    else:
        device = "cpu"

    i_state_dict = get_state_dict(args.i_frame_model_path)
    i_frame_net = IntraNoAR(ec_thread=args.ec_thread, stream_part=args.stream_part_i)
    i_frame_net.load_state_dict(i_state_dict)
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    p_state_dict = get_state_dict(args.p_frame_model_path)
    p_frame_net = DMC(ec_thread=args.ec_thread, stream_part=args.stream_part_p,
                      inplace=True)
    p_frame_net.load_state_dict(p_state_dict)
    p_frame_net = p_frame_net.to(device)
    p_frame_net.eval()

    i_frame_net.update(force=True)
    p_frame_net.update(force=True)

    return i_frame_net, p_frame_net


def read_header(output_path):
    with Path(output_path).open("rb") as f:
        ip = read_ints(f, 1)[0]
        height, width, qp, fast_flag = read_uints(f, 4)
    return ip, height, width, qp, fast_flag


def decode():
    torch.backends.cudnn.enabled = True
    args = parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    i_net, p_net = init_func(args)
    os.makedirs(os.path.join(args.output_path, 'full'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'skim'), exist_ok=True)
    header_path = os.path.join(args.bin_path, f"headers.bin")
    ip, height, width, qp, fast_flag = read_header(header_path)
    rec_writer_full = PNGWriter(os.path.join(args.output_path, 'full'), width, height)
    rec_writer_skim = PNGWriter(os.path.join(args.output_path, 'skim'), width // 4.0, height // 4.0)

    count_frame = 0
    dpb_BL = None
    dpb_EL = None
    while True:
        bin_path = os.path.join(args.bin_path, f"{count_frame}.bin")
        if not os.path.exists(bin_path):
            break
        if count_frame == 0 or (ip > 0 and count_frame % ip == 0):
            bitstream = decode_i(bin_path)
            dpb_BL, dpb_EL = i_net.decode_one_frame(bitstream, height, width, qp)
            dpb_EL = None if fast_flag else dpb_EL
        else:
            if count_frame % args.refresh_interval == 1:
                dpb_BL['ref_feature'] = None
                if dpb_EL is not None:
                    dpb_EL['ref_feature'] = None
            bitstream = decode_p(bin_path) if fast_flag else decode_p_two_layer(bin_path)
            dpb_BL, dpb_EL = p_net.decode_one_frame(bitstream, height, width, dpb_BL, dpb_EL, qp, count_frame)
        # slice
        ss_EL = get_slice_shape(height, width, p=16)
        ss_BL = get_slice_shape(height // 4, width // 4, p=16)
        rec_writer_skim.write_one_frame(slice_to_x(dpb_BL['ref_frame'], ss_BL).clamp_(0, 1).squeeze(0).cpu().numpy())
        if not fast_flag:
            rec_writer_full.write_one_frame(slice_to_x(dpb_EL['ref_frame'], ss_EL).clamp_(0, 1).squeeze(0).cpu().numpy())
        count_frame += 1
    rec_writer_skim.close()
    rec_writer_full.close()


if __name__ == '__main__':
    with torch.no_grad():
        decode()
