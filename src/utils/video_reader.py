# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image


class VideoReader():
    def __init__(self, src_path, width, height):
        self.src_path = src_path
        self.width = width
        self.height = height
        self.eof = False

    @staticmethod
    def _none_exist_frame(dst_format):
        assert dst_format == "rgb"
        return None


class PNGReader(VideoReader):
    def __init__(self, src_path, width, height, start_num=1):
        super().__init__(src_path, width, height)

        pngs = os.listdir(self.src_path)
        if 'im1.png' in pngs:
            self.padding = 1
        elif 'im00001.png' in pngs:
            self.padding = 5
        else:
            raise ValueError('unknown image naming convention; please specify')
        self.current_frame_index = start_num

    def read_one_frame(self, dst_format="rgb"):
        if self.eof:
            return self._none_exist_frame(dst_format)

        png_path = os.path.join(self.src_path,
                                f"im{str(self.current_frame_index).zfill(self.padding)}.png"
                                )
        if not os.path.exists(png_path):
            self.eof = True
            return self._none_exist_frame(dst_format)

        rgb = Image.open(png_path).convert('RGB')
        rgb = np.asarray(rgb).astype('float32').transpose(2, 0, 1)
        rgb = rgb / 255.
        _, height, width = rgb.shape
        assert height == self.height
        assert width == self.width

        self.current_frame_index += 1
        return rgb

    def close(self):
        self.current_frame_index = 1
