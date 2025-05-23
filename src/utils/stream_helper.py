# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import struct
from pathlib import Path

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def pad_for_x(x, p=16, mode='constant'):
    _, _, H, W = x.size()
    padding_l, padding_r, padding_t, padding_b = get_padding_size(H, W, p)
    y_pad = torch.nn.functional.pad(
        x,
        (padding_l, padding_r, padding_t, padding_b),
        mode=mode,
    )
    return y_pad, (-padding_l, -padding_r, -padding_t, -padding_b)


def slice_to_x(x, slice_shape):
    return torch.nn.functional.pad(x, slice_shape)


def get_padding_size(height, width, p=16):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom


def get_padded_size(height, width, p=16):
    padding_left, padding_right, padding_top, padding_bottom = get_padding_size(height, width, p=p)
    return (height + padding_top + padding_bottom, width + padding_left + padding_right)


def get_multi_scale_padding_size(height, width, p=16):
    p1 = get_padding_size(height, width, p=16)
    pass


def get_slice_shape(height, width, p=16):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return (int(-padding_left), int(-padding_right), int(-padding_top), int(-padding_bottom))


def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


def get_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    if "state_dict" in ckpt:
        ckpt = ckpt['state_dict']
    if "net" in ckpt:
        ckpt = ckpt["net"]
    consume_prefix_in_state_dict_if_present(ckpt, prefix="module.")
    return ckpt


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def write_ints(fd, values, fmt=">{:d}i"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_ints(fd, n, fmt=">{:d}i"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def write_ushorts(fd, values, fmt=">{:d}H"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_ushorts(fd, n, fmt=">{:d}H"):
    sz = struct.calcsize("H")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def encode_i(bit_stream, output):
    with Path(output).open("wb") as f:
        stream_length = len(bit_stream)
        write_uints(f, (stream_length,))
        write_bytes(f, bit_stream)


def decode_i(inputpath):
    with Path(inputpath).open("rb") as f:
        stream_length = read_uints(f, 1)[0]
        bit_stream = read_bytes(f, stream_length)

    return bit_stream


def encode_p(string, output):
    with Path(output).open("wb") as f:
        string_length = len(string)
        write_uints(f, (string_length,))
        write_bytes(f, string)


def decode_p(inputpath):
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 1)
        string_length = header[0]
        string = read_bytes(f, string_length)

    return [string]


def encode_p_two_layer(string, output):
    string1 = string[0]
    string2 = string[1]
    with Path(output).open("wb") as f:
        string_length = len(string1)
        write_uints(f, (string_length,))
        write_bytes(f, string1)

        string_length = len(string2)
        write_uints(f, (string_length,))
        write_bytes(f, string2)


def decode_p_two_layer(inputpath):
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 1)
        string_length = header[0]
        string1 = read_bytes(f, string_length)

        header = read_uints(f, 1)
        string_length = header[0]
        string2 = read_bytes(f, string_length)

    return [string1, string2]
