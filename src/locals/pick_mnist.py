#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 23:03:11 2017

@author: zhangxinyang
"""

from six.moves import xrange

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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
    size_h = 4
    size_w = 16

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
    
PATH = "/Users/zhangxinyang/PycharmProjects/dpgannew/.results/mnist_new.npy"
imgs = np.load(PATH)

# generate_images_and_show(imgs[:64], mode="gray")

indices = []
to_pick = [
    (0, 1),
    (0, 4),
    (0, 5),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7), 
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (2, 7),
    (3, 0),
    (3, 1),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 0),
    (4, 2),
    (4, 4),
    (4, 5),
    (4, 6),
    (6, 0),
    (6, 3),
    (7, 1),
    (7, 2),
    (7, 3),
    (7, 4)
]
indices.extend([i * 8 + j for (i, j) in to_pick])
total = 33

# generate_images_and_show(imgs[128:192], mode="gray")
to_pick = [
           (7, 0),
           (7, 2),
           (7, 5),
           (6, 3),
           (6, 4),
           (5, 0),
           (5, 1),
           (5, 3),
           (5, 4),
           (5, 7),
           (4, 0),
           (4, 1),
           (4, 3), 
           (4, 7),
           (3, 6),
           (3, 7),
           (3, 2),
           (2, 0),
           (2, 2),
           (2, 3),
           (1, 1),
           (1, 3),
           (1, 4),
           (0, 2),
           (0, 4)
           ]
indices.extend([128 + i * 8 + j for (i, j) in to_pick])
total = 25

# generate_images_and_show(imgs[192:256], mode="gray")
to_pick = [
           (7, 5),
           (1, 6),
           (2, 0),
           (2, 1),
           (5, 1),
           (7, 4)
           ]
indices.extend([192 + i * 8 + j for (i, j) in to_pick])


print(len(indices))

for i, index in enumerate(indices[:64], start=1):
    arr = np.asarray(127.5 * (imgs[index] + 1), np.uint8)
    img = Image.fromarray(arr[:, :, 0], "L")
    img.save("gen_%d.png" % i)

generate_images_and_save(imgs[indices][:64], "gray", "merged.png")