#!/usr/bin/env python
from __future__ import print_function
from six.moves import xrange

import glob
import os
from multiprocessing import Process, Queue, Value
from time import sleep

import numpy as np

from utils.actions import *


def fetch(patterns, max_numbers, block_size, queue, n_value, actions=None, worker_id=-1,
          one_hot=True, public_num=None, public_seed=None):
    np.random.seed()
    path_labels = []
    pattern_labels = [(pattern.strip(), label) for label, pattern in enumerate(patterns.split(";"))]
    for pattern, label in pattern_labels:
        paths = glob.glob(pattern)
        if max_numbers is None or max_numbers[label] == -1:
            path_labels.extend([(path, label) for path in paths])
        else:
            indices = np.random.choice(len(paths), min(max_numbers[label], len(paths)), False).astype(np.int32)
            path_labels.extend([(paths[i], label) for i in indices])

    paths = [path for path, _ in path_labels]
    labels = [label for _, label in path_labels]

    if public_num is not None:
        indices = np.random.RandomState(public_seed).permutation(
            np.arange(0, len(paths), dtype=np.int64)
        )[:public_num]
        paths = [paths[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]

    n = len(labels)
    if worker_id == 0:
        assert n >= block_size
        n_value.value = n

    while True:
        indices = np.random.permutation(np.arange(n, dtype=np.int32))
        paths = [paths[i] for i in indices]
        labels = [labels[i] for i in indices]

        pos = 0
        while pos + block_size <= n:
            s = slice(pos, pos + block_size)
            X, y = [], []
            for path, label in zip(paths[s], labels[s]):
                item = path
                for action in actions:
                    item = action(item)
                X.append(item)
                y.append(label)

            X = np.asarray(X)
            y = np.asarray(y, np.int32)
            if one_hot:
                one_hot_y = np.zeros((len(y), len(pattern_labels)), np.float32)
                one_hot_y[np.arange(0, len(y), dtype=np.int64), y] = 1.
                y = one_hot_y

            X = (X - 127.5) / 127.5
            X = X.astype(np.float32)

            queue.put((X, y))
            sleep(0.00)

            pos += block_size


def fetch_lsun(data_dir, block_size, queue, n_value, actions=None, worker_id=-1):
    np.random.seed()
    paths = [path for path in glob.glob(os.path.join(data_dir, "*.jpg"))]
    n = len(paths)
    if worker_id == 0:
        assert n >= block_size
        n_value.value = n

    while True:
        indices = np.random.permutation(np.arange(n, dtype=np.int32))
        paths = [paths[i] for i in indices]

        pos = 0
        while pos + block_size <= n:
            s = slice(pos, pos + block_size)
            X = []
            for path in paths[s]:
                item = path
                for action in actions:
                    item = action(item)
                X.append(item)

            X = np.asarray(X)
            X = (X - 127.5) / 127.5
            X = X.astype(np.float32)

            queue.put((X, None))
            sleep(0.00)

            pos += block_size


class LSUNLoader(object):

    def __init__(self, data_dir, dim=64, block_size=18, max_blocks=2048,
                 actions=None, num_workers=6):
        self.data_dir = data_dir
        self.dim = dim

        self.num_workers = num_workers
        self.actions= actions

        self.block_size = block_size
        self.max_blocks = max_blocks

        self._processes = []
        self._queue = None
        self._n = None

    def start_fetch(self):
        assert len(self._processes) == 0, "fetcher is already running."
        self._n = Value("i", -1)
        self._queue = Queue(self.max_blocks)
        for i in xrange(self.num_workers):
            self._processes.append(Process(group=None, target=fetch_lsun, args=(self.data_dir,
                                                 self.block_size, self._queue, self._n
                                                 ), kwargs={"actions": self.actions, "worker_id": i}))

        print("data loader %r is starting..." % self)
        for i in xrange(self.num_workers):
            self._processes[i].start()

    def stop_fetch(self):
        self._queue.close()
        for process in self._processes:
            process.terminate()
        print("data laoder %r is closed" % self)

        self._queue = None
        self._processes = None

    def get(self):
        return self._queue.get()

    def next_batch(self, batch_size):
        if self.block_size >= batch_size:
            X, _ = self.get()
        else:
            times = int((batch_size + self.block_size - 1) / self.block_size)
            data = [self.get() for _ in xrange(times)]
            X = np.concatenate([X for X, _ in data])
        return X[:batch_size], None

    def num_steps(self, batch_size):
        while self._n.value == -1:
            sleep(0.01)
        return int(self._n.value / batch_size)

    def mode(self):
        return "rgb"

    def shape(self):
        return [self.dim, self.dim, 3]

    def classes(self):
        return -1


class LSUNCatLoader(object):

    def __init__(self, patterns, max_numbers=None, block_size=32, max_blocks=128,
                 actions=None, num_workers=6, one_hot=True, public_num=None,
                 public_seed=None):
        self.patterns = patterns
        self.max_numbers = max_numbers
        self.num_workers = num_workers
        self.actions= actions
        self.one_hot = one_hot
        self.public_num = public_num
        self.public_seed = public_seed

        self.block_size = block_size
        self.max_blocks = max_blocks

        self._processes = []
        self._queue = None
        self._n = None
        self._classes = len(patterns.split(";"))

    def start_fetch(self):
        assert len(self._processes) == 0, "fetcher is already running."
        self._n = Value("i", -1)
        self._queue = Queue(self.max_blocks)
        for i in xrange(self.num_workers):
            self._processes.append(Process(group=None, target=fetch, args=(self.patterns, self.max_numbers,
                                                 self.block_size, self._queue, self._n
                                                 ), kwargs={"actions": self.actions, "worker_id": i,
                                                            "one_hot": self.one_hot,
                                                            "public_num": self.public_num,
                                                            "public_seed": self.public_seed}))

        print("data loader %r is starting..." % self)
        for i in xrange(self.num_workers):
            self._processes[i].start()

    def stop_fetch(self):
        self._queue.close()
        for process in self._processes:
            process.terminate()
        print("data laoder %r is closed" % self)

        self._queue = None
        self._processes = None

    def get(self):
        return self._queue.get()

    def next_batch(self, batch_size):
        if self.block_size >= batch_size:
            X, y = self.get()
        else:
            times = int((batch_size + self.block_size - 1) / self.block_size)
            data = [self.get() for _ in xrange(times)]
            X = np.concatenate([X for X, _ in data])
            y = np.concatenate([y for _, y in data])
        return X[:batch_size], y[:batch_size]

    def num_steps(self, batch_size):
        while self._n.value == -1:
            sleep(0.01)
        return int(self._n.value / batch_size)

    def shape(self):
        return [64, 64, 3]

    def mode(self):
        return "rgb"

    def classes(self):
        return self._classes


def get_lsun_patterns(base_dir):
    names = [name for name in sorted(os.listdir(base_dir)) if not name.startswith(".")]
    fullnames = [os.path.join(base_dir, name) for name in sorted(os.listdir(base_dir)) if not name.startswith(".")]
    patterns = []
    for i, (name, fullname) in enumerate(zip(names, fullnames)):
        if os.path.isdir(fullname):
            print("category %s is labeled with %d." % (name, i))
            patterns.append(os.path.join(fullname, "*.jpg"))
    return ";".join(patterns)


def lsun_process_actions(dim=64):
    return [OpenImage(), Scale(dim, Image.ANTIALIAS), CenterCrop(dim), ToArray()]


def celeba_process_actions(dim=64):
    return [OpenImage(), Scale(dim, Image.ANTIALIAS), CenterCrop(dim), ToArray()]


def fetch_celeba(data_dir, block_size, queue, n_value, actions=None, worker_id=-1):
    np.random.seed()
    paths = [path for path in glob.glob(os.path.join(data_dir, "*.png"))]
    n = len(paths)
    if worker_id == 0:
        assert n >= block_size
        n_value.value = n

    while True:
        indices = np.random.permutation(np.arange(n, dtype=np.int32))
        paths = [paths[i] for i in indices]

        pos = 0
        while pos + block_size <= n:
            s = slice(pos, pos + block_size)
            X = []
            for path in paths[s]:
                item = path
                for action in actions:
                    item = action(item)
                X.append(item)

            X = np.asarray(X)
            X = (X - 127.5) / 127.5
            X = X.astype(np.float32)

            queue.put((X, None))
            sleep(0.00)

            pos += block_size


class CelebALoader(object):

    def __init__(self, data_dir, dim=64, block_size=18, max_blocks=2048,
                 actions=None, num_workers=6):
        self.data_dir = data_dir
        self.dim = dim

        self.num_workers = num_workers
        self.actions= actions

        self.block_size = block_size
        self.max_blocks = max_blocks

        self._processes = []
        self._queue = None
        self._n = None

    def start_fetch(self):
        assert len(self._processes) == 0, "fetcher is already running."
        self._n = Value("i", -1)
        self._queue = Queue(self.max_blocks)
        for i in xrange(self.num_workers):
            self._processes.append(Process(group=None, target=fetch_celeba, args=(self.data_dir,
                                                 self.block_size, self._queue, self._n
                                                 ), kwargs={"actions": self.actions, "worker_id": i}))

        print("data loader %r is starting..." % self)
        for i in xrange(self.num_workers):
            self._processes[i].start()

    def stop_fetch(self):
        self._queue.close()
        for process in self._processes:
            process.terminate()
        print("data laoder %r is closed" % self)

        self._queue = None
        self._processes = None

    def get(self):
        return self._queue.get()

    def next_batch(self, batch_size):
        if self.block_size >= batch_size:
            X, _ = self.get()
        else:
            times = int((batch_size + self.block_size - 1) / self.block_size)
            data = [self.get() for _ in xrange(times)]
            X = np.concatenate([X for X, _ in data])
        return X[:batch_size], None

    def num_steps(self, batch_size):
        while self._n.value == -1:
            sleep(0.01)
        return int(self._n.value / batch_size)

    def mode(self):
        return "rgb"

    def shape(self):
        return [self.dim, self.dim, 3]

    def classes(self):
        return -1
