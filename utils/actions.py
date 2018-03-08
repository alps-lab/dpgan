#!/usr/bin/env python
import numpy as np
from PIL import Image

from utils.data_utils import scale, center_crop


class OpenImage(object):

    def __int__(self):
        pass

    def __call__(self, x):
        return Image.open(x)


class Scale(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, x):
        return scale(x, self.size, self.interpolation)


class CenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        return center_crop(x, self.size)


class ToArray(object):

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, x):
        if self.dtype is not None:
            arr = np.asarray(x, dtype=self.dtype)
        else:
            arr = np.asarray(x)
        x.close()
        return arr
