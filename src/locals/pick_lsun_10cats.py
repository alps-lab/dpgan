#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 20:56:01 2017

@author: zhangxinyang
"""

from six.moves import xrange


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


PATH = "/Users/zhangxinyang/PycharmProjects/dpgannew/lsun_10cats.npy"


def generate_images_and_show(arr, mode):
    m, h, w, channels = arr.shape
    size = int(np.ceil(np.sqrt(m)))

    new_shape = [size * h, size * w, channels]
    new_arr = np.zeros(new_shape, dtype=arr.dtype)

    for i in xrange(m):
        r = int(i / size)
        c = i % size
        new_arr[r*h:(r+1)*h, c*w:(c+1)*w, :] = arr[i]

    if mode.lower() == "rgb":
        new_arr = np.asarray((new_arr + 1) * 127.5, np.uint8)
        image = Image.fromarray(new_arr, "RGB")
    elif mode.lower() == "gray":
        new_arr = np.asarray((new_arr + 1) * 127.5, np.uint8)[:, :, 0]
        image = Image.fromarray(new_arr, "L")
    else:
        raise Exception("Unsupported mode %s" % mode)

    image.show()
    plt.show()
    
def generate_images_and_save(arr, mode, output_path):
    m, h, w, channels = arr.shape
    size_h = 5
    size_w = 10

    new_shape = [size_h * h, size_w * w, channels]
    new_arr = np.zeros(new_shape, dtype=arr.dtype)

    for i in xrange(m):
        r = int(i / size_w)
        c = i % size_w
        new_arr[r*h:(r+1)*h, c*w:(c+1)*w, :] = arr[i]

    if mode.lower() == "rgb":
        new_arr = np.asarray((new_arr + 1) * 127.5, np.uint8)
        image = Image.fromarray(new_arr, "RGB")
    elif mode.lower() == "gray":
        new_arr = np.asarray((new_arr + 1) * 127.5, np.uint8)[:, :, 0]
        image = Image.fromarray(new_arr, "L")
    else:
        raise Exception("Unsupported mode %s" % mode)

    image.save(output_path)

imgs = np.load(PATH)
indices = []

# generate_images_and_show(imgs[:64], mode="rgb")
to_pick = [
        (6, 3),
        (6, 4),
        (6, 5),
        (6, 6),
        (6, 7),
        (0, 5),
        (2, 0),
        (2, 1),
        (2, 5),
        (2, 7),
        (3, 0),
        (4, 0),
        (4, 3),
        (4, 5),
   ]
indices.extend([i * 8 + j for (i, j) in to_pick])
total = 14

# generate_images_and_show(imgs[64:128], mode="rgb")
to_pick = [
        (7, 0),
        (7, 2),
        (7, 6),
        (5, 0), 
        (5, 2),
        (5, 3),
        (5, 7),
        (3, 4),
        (4, 0),
        (2, 5),
        (2, 7),
   ]
indices.extend([64 + i * 8 + j for (i, j) in to_pick])
total = 11

# generate_images_and_show(imgs[128:192], mode="rgb")
to_pick = [
        (0, 3),
        (0, 4),
        (0, 6),
        (0, 7), 
        (1, 5),
        (1, 6),
        (1, 7),
        (2, 0),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 1),
        (3, 3),
        (3, 6),
        (6, 0),
        (6, 6)
        
   ]
indices.extend([128 + i * 8 + j for (i, j) in to_pick])
total = 16

# generate_images_and_show(imgs[192:256], mode="rgb")
to_pick = [
        (0, 1),
        (0, 7),
        (1, 0),
        (0, 7), 
        (1, 5),
        (1, 6),
        (1, 7),
        (4, 2),
        (4, 5),
   ]
indices.extend([192 + i * 8 + j for (i, j) in to_pick])
total = 9

for i, index in enumerate(indices, start=1):
    arr = np.asarray(127.5 * (imgs[index] + 1), np.uint8)
    img = Image.fromarray(arr, "RGB")
    img.save("gen_%d.png" % i)

generate_images_and_save(imgs[indices], "rgb", "merged.png")