# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image


class VideoWriter():
    def __init__(self, dst_path, width, height):
        self.dst_path = dst_path
        self.width = width
        self.height = height

    def write_one_frame(self, rgb=None, src_format="rgb"):
        raise NotImplementedError


class PNGWriter(VideoWriter):
    def __init__(self, dst_path, width, height):
        super().__init__(dst_path, width, height)
        self.padding = 5
        self.current_frame_index = 1
        os.makedirs(dst_path, exist_ok=True)

    def write_one_frame(self, rgb=None, src_format="rgb"):
        rgb = rgb.transpose(1, 2, 0)
        png_path = os.path.join(self.dst_path, f"im{str(self.current_frame_index).zfill(self.padding)}.png")
        img = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
        Image.fromarray(img).save(png_path)

        self.current_frame_index += 1

    def close(self):
        self.current_frame_index = 1
