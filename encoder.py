import os
import torch
import argparse
import numpy as np

from pathlib import Path
from src.utils.common import str2bool
from src.models.SEVC_main_model import DMC
from src.models.image_model import IntraNoAR
from src.utils.stream_helper import get_state_dict, pad_for_x, slice_to_x, write_uints, write_ints, get_slice_shape, encode_i, encode_p, encode_p_two_layer
from src.utils.video_reader import PNGReader


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument("--ec_thread", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--stream_part_i", type=int, default=1)
    parser.add_argument("--stream_part_p", type=int, default=1)
    parser.add_argument('--i_frame_model_path', type=str)
    parser.add_argument('--p_frame_model_path', type=str)
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--refresh_interval', type=int, default=32)
    parser.add_argument('-b', '--bin_path', type=str, default='out_bin')
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('-q', '--qp', type=int, required=True)
    parser.add_argument('-f', '--frames', type=int, default=-1)
    parser.add_argument('--fast', type=str2bool, default=False)
    parser.add_argument('--ip', type=int, default=-1)

    args = parser.parse_args()
    return args


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def init_func(args):
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    if args.cuda:
        device = f"cuda:{0}"
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


def write_header(n_headers, output_path):
    with Path(output_path).open("wb") as f:
        write_ints(f, (n_headers[0],))
        write_uints(f, n_headers[1:])


def encode():
    torch.backends.cudnn.enabled = True
    args = parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    i_net, p_net = init_func(args)
    device = next(i_net.parameters()).device
    src_reader = PNGReader(args.input_path, args.width, args.height)
    os.makedirs(args.bin_path, exist_ok=True)
    ip = args.ip
    height = args.height
    width = args.width
    qp = args.qp
    fast_flag = args.fast
    header_path = os.path.join(args.bin_path, f"headers.bin")
    write_header((ip, height, width, qp, fast_flag), header_path)

    count_frame = 0
    dpb_BL = None
    dpb_EL = None
    while True:
        x = src_reader.read_one_frame()
        if x is None or (args.frames >= 0 and count_frame >= args.frames):
            break
        bin_path = os.path.join(args.bin_path, f"{count_frame}.bin")
        x = np_image_to_tensor(x)
        x = x.to(device)
        if count_frame == 0 or (ip > 0 and count_frame % ip == 0):
            dpb_BL, dpb_EL, bitstream = i_net.encode_one_frame(x, qp)
            encode_i(True, qp, bitstream, bin_path)  # i bin
            dpb_EL = None if fast_flag else dpb_EL
        else:
            if count_frame % args.refresh_interval == 1:
                dpb_BL['ref_feature'] = None
                if dpb_EL is not None:
                    dpb_EL['ref_feature'] = None
            dpb_BL, dpb_EL, bitstream = p_net.encode_one_frame(x, dpb_BL, dpb_EL, qp, count_frame)
            encode_p(True, qp, bitstream[0], bin_path) if fast_flag else encode_p_two_layer(True, qp, bitstream, bin_path)
        count_frame += 1


if __name__ == '__main__':
    encode()
