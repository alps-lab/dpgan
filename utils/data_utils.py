#!/usr/bin/env python
from __future__ import print_function

from six.moves import xrange, cPickle
from io import open

import os

from PIL import Image
import numpy as np
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets


class DataLoader(object):

    def num_steps(self, batch_size):
        pass

    def next_batch(self, batch_size):
        pass

    def shuffle(self, *args, **kwargs):
        pass

    def mode(self):
        pass

    def shape(self):
        pass


class Cifar10Loader(DataLoader):
    filenames = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]

    def __init__(self, data_dir=".", normalize=True, one_hot=True,
                 include_train=True, include_test=True,
                 first=None, last=None, seed=None):
        data, labels = [], []
        names = []
        if include_train:
            names.extend(Cifar10Loader.filenames[:-1])
        if include_test:
            names.extend([Cifar10Loader.filenames[-1]])

        for name in names:
            with open(os.path.join(data_dir, name), "rb") as f:
                dataobj = cPickle.load(f, encoding="latin1")
                data.append(dataobj["data"])
                labels.append(dataobj["labels"])

        data = np.concatenate(data, axis=0).reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1]).astype(np.float32)
        labels = np.concatenate(labels, axis=0)[:, None].astype(np.int64)

        if one_hot:
            one_hot_labels = np.zeros((len(data), 20), np.int64)
            one_hot_labels[np.arange(0, len(data), dtype=np.int64), labels[:, 0]] = 1
            labels = one_hot_labels
        labels = np.asarray(labels, np.float32)

        if normalize:
            data = -1 + data / 127.5

        assert first is None or last is None
        if first is not None:
            indices = np.random.RandomState(seed).permutation(np.arange(0, len(data), dtype=np.int64))
            data = data[indices[:first]]
            labels = labels[indices[:first]]
        if last is not None:
            indices = np.random.RandomState(seed).permutation(np.arange(0, len(data), dtype=np.int64))
            data = data[indices[-last:]]
            labels = labels[indices[-last:]]

        self.data = data
        self.labels = labels
        self.n = len(data)
        self.pos = 0

    def num_steps(self, batch_size):
        return int(self.n / batch_size)

    def next_batch(self, batch_size):
        if self.pos + batch_size > self.n:
            self.pos = 0
            self.shuffle()
        s = slice(self.pos, self.pos + batch_size)
        ret = (self.data[s], self.labels[s])
        self.pos += batch_size
        return ret

    def shuffle(self):
        new_indices = np.random.permutation(np.arange(self.n, dtype=np.int32))
        self.data = self.data[new_indices]
        self.labels = self.labels[new_indices]

    def mode(self):
        return "rgb"

    def shape(self):
        return [32, 32, 3]


class STL10Loader(DataLoader):

    def __init__(self, data_dir=".", normalize=True):
        data, labels = [], []
        for name in Cifar10Loader.filenames:
            with open(os.path.join(data_dir, name), "rb") as f:
                dataobj = cPickle.load(f, encoding="latin1")
                data.append(dataobj["data"])
                labels.append(dataobj["labels"])

        data = np.concatenate(data, axis=0).reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1]).astype(np.float32)
        labels = np.concatenate(labels, axis=0)[:, None].astype(np.float32)

        if normalize:
            data = -1 + data / 127.5

        self.data = data
        self.labels = labels
        self.n = len(data)
        self.pos = 0

    def num_steps(self, batch_size):
        return int(self.n / batch_size)

    def next_batch(self, batch_size):
        if self.pos + batch_size > self.n:
            self.pos = 0
            self.shuffle()
        s = slice(self.pos, self.pos + batch_size)
        ret = (self.data[s], self.labels[s])
        self.pos += batch_size
        return ret

    def shuffle(self):
        new_indices = np.random.permutation(np.arange(self.n, dtype=np.int32))
        self.data = self.data[new_indices]
        self.labels = self.labels[new_indices]


class MNISTLoader(DataLoader):
    filenames = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]

    def __init__(self, data_dir="/tmp/mnist", normalize=True, one_hot=True,
                 include_train=True, include_test=True,
                 first=None, last=None, seed=None):
        mnist = read_data_sets(train_dir=data_dir)

        data, labels = [], []
        train = mnist.train
        test = mnist.test

        if include_train:
            data.append(train.images)
            labels.append(train.labels)
        if include_test:
            data.append(test.images)
            labels.append(test.labels)

        data = np.concatenate(data, axis=0).reshape((-1, 28, 28, 1)).astype(np.float32)
        labels = np.concatenate(labels, axis=0)[:, None].astype(np.int64)
        assert first is None or last is None
        if first is not None:
            n = min(first, len(data))
            indices = np.random.RandomState(seed).permutation(np.arange(0, len(data), dtype=np.int64))[:n]
            data = data[indices]
            labels = labels[indices]
        if last is not None:
            n = min(last, len(data))
            indices = np.random.RandomState(seed).permutation(np.arange(0, len(data), dtype=np.int64))[-n:]
            data = data[indices]
            labels = labels[indices]

        if one_hot:
            one_hot_labels = np.zeros((len(data), 10), np.float32)
            one_hot_labels[np.arange(0, len(data), dtype=np.int64), labels[:, 0]] = 1.
            labels = one_hot_labels
        else:
            labels = np.asarray(labels, np.float32)

        if normalize:
            data = -1.0 + data * 2
        else:
            data = data * 255

        self.data = data
        self.labels = labels
        self.n = len(data)
        self.pos = 0

    def num_steps(self, batch_size):
        return int(self.n / batch_size)

    def next_batch(self, batch_size):
        if self.pos + batch_size > self.n:
            self.pos = 0
            self.shuffle()
        s = slice(self.pos, self.pos + batch_size)
        ret = (self.data[s], self.labels[s])
        self.pos += batch_size
        return ret

    def shuffle(self):
        new_indices = np.random.permutation(np.arange(self.n, dtype=np.int32))
        self.data = self.data[new_indices]
        self.labels = self.labels[new_indices]

    def mode(self):
        return "gray"

    def shape(self):
        return [28, 28, 1]

    def classes(self):
        return 10


def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


class FashionMNISTLoader(DataLoader):
    filenames = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]

    def __init__(self, data_dir="/tmp/mnist", normalize=True):
        mnist = read_data_sets(train_dir=data_dir)
        train = mnist.train
        test = mnist.test

        data, labels = [train.images, test.images], [train.labels, test.labels]
        data = np.concatenate(data, axis=0).reshape((-1, 28, 28, 1)).astype(np.float32)
        labels = np.concatenate(labels, axis=0)[:, None].astype(np.float32)

        if normalize:
            data = -1.0 + data * 2
        else:
            data = data * 255

        self.data = data
        self.labels = labels
        self.n = len(data)
        self.pos = 0

    def num_steps(self, batch_size):
        return int(self.n / batch_size)

    def next_batch(self, batch_size):
        if self.pos + batch_size > self.n:
            self.pos = 0
            self.shuffle()
        s = slice(self.pos, self.pos + batch_size)
        ret = (self.data[s], self.labels[s])
        self.pos += batch_size
        return ret

    def shuffle(self):
        new_indices = np.random.permutation(np.arange(self.n, dtype=np.int32))
        self.data = self.data[new_indices]
        self.labels = self.labels[new_indices]

    def mode(self):
        return "gray"

    def shape(self):
        return [28, 28, 1]


class LSUNLoader(DataLoader):

    def __init__(self, datadir):
        self.names = [os.path.join(datadir, name) for name in os.listdir(datadir)]
        self.n = len(self.names)
        self.pos = 0
        self.shuffle()

    def num_steps(self, batch_size):
        return int(self.n / batch_size)

    def next_batch(self, batch_size):
        if self.pos + batch_size > self.n:
            self.pos = 0
            self.shuffle()
        ret = self._read_images(batch_size)
        self.pos += batch_size
        return (ret, None)

    def shuffle(self):
        new_indices = np.random.permutation(np.arange(self.n, dtype=np.int32))
        self.names = [self.names[i] for i in new_indices]

    def _read_images(self, batch_size):
        s = slice(self.pos, self.pos + batch_size)
        names = self.names[s]
        arrs = []

        for name in names:
            img = Image.open(name)
            img = scale(img, 64, Image.ANTIALIAS)
            img = center_crop(img, 64)
            arrs.append(np.asarray(img)[None])
            img.close()

        images = np.concatenate(arrs, axis=0).astype(np.float32)
        images = -1 + images / 127.5

        return images

    def mode(self):
        return "rgb"

    def shape(self):
        return [64, 64, 3]


class LSUN32Loader(DataLoader):

    def __init__(self, datadir):
        self.names = [os.path.join(datadir, name) for name in os.listdir(datadir)]
        self.n = len(self.names)
        self.pos = 0
        self.shuffle()

    def num_steps(self, batch_size):
        return int(self.n / batch_size)

    def next_batch(self, batch_size):
        if self.pos + batch_size > self.n:
            self.pos = 0
            self.shuffle()
        ret = self._read_images(batch_size)
        self.pos += batch_size
        return (ret, None)

    def shuffle(self):
        new_indices = np.random.permutation(np.arange(self.n, dtype=np.int32))
        self.names = [self.names[i] for i in new_indices]

    def _read_images(self, batch_size):
        s = slice(self.pos, self.pos + batch_size)
        names = self.names[s]
        arrs = []

        for name in names:
            img = Image.open(name)
            img = scale(img, 32, Image.ANTIALIAS)
            img = center_crop(img, 32)
            arrs.append(np.asarray(img)[None])
            img.close()

        images = np.concatenate(arrs, axis=0).astype(np.float32)
        images = -1 + images / 127.5

        return images

    def mode(self):
        return "rgb"

    def shape(self):
        return [32, 32, 3]


def scale(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size, interpolation)


def center_crop(img, size):
    size = (size, size)
    w, h = img.size
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return img.crop((x1, y1, x1 + tw, y1 + th))


def generate_images(arr, mode, output_path):
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

    image.save(output_path)
