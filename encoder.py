import os
import torch
import argparse
import numpy as np

from src.utils.common import str2bool
from src.models.SEVC_main_model import DMC
from src.models.image_model import IntraNoAR
from src.utils.stream_helper import get_state_dict, pad_for_x, slice_to_x, write_uints, get_slice_shape, encode_i, encode_p, encode_p_two_layer
from src.utils.video_reader import PNGReader
from src.utils.core import imresize


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
    parser.add_argument('-w', '--width', type=int, required=True)
    parser.add_argument('-h', '--height', type=int, required=True)
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
    i_frame_net = IntraNoAR(ec_thread=args.ec_thread, stream_part=args.stream_part_i,
                            inplace=True)
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


def write_header(n_heads):
    pass


def encode():
    torch.backends.cudnn.enabled = True
    args = parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    i_net, p_net = init_func(args)
    device = next(i_net.parameters()).device
    src_reader = PNGReader(args['input_path'], args['width'], args['height'])
    os.makedirs(args.b, exist_ok=True)
    ip = args.ip
    height = args.h
    width = args.w
    qp = args.q
    fast_flag = args.fast

    count_frame = 0
    dpb_BL = None
    dpb_EL = None
    while True:
        x = src_reader.read_one_frame()
        if x is None:
            break
        bin_path = os.path.join(args.b, f"{count_frame}.bin")
        x = np_image_to_tensor(x)
        x = x.to(device)
        if count_frame == 0 or (ip > 0 and count_frame % ip == 0):
            dpb_BL, dpb_EL, bitstream = i_net.encode(x, qp)
            encode_i(True, qp, bitstream, bin_path)  # i bin
        else:
            if count_frame % args.refresh_interval == 1:
                dpb_BL['ref_feature'] = None
                dpb_EL['ref_feature'] = None
            dpb_BL, dpb_EL, bitstream = p_net.encode(x, dpb_BL, dpb_EL, qp, count_frame, fast_flag)
