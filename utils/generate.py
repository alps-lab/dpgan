#!/usr/bin/env python
from six.moves import xrange

import os

import tensorflow as tf
import tflearn
import numpy as np
from PIL import Image
from tqdm import trange


def generate_steps(config, generator_forward):
    with tf.device("/cpu:0"):
        fake_data = generator_forward(config)
    saver = tf.train.Saver()
    sess = tf.Session()

    saver.restore(sess, config.load_path)
    print("loaded model from %s." % config.load_path)

    tflearn.is_training(False, sess)
    to_stack = []
    for _ in xrange(config.times):
        generated = sess.run(fake_data)
        to_stack.append(generated)
    stacked = np.concatenate(to_stack, axis=0)

    if config.save_path:
        np.save(config.save_path, stacked, allow_pickle=False)


def generate_steps_png(config, generator_forward):
    with tf.device("/cpu:0"):
        fake_data = generator_forward(config)
    saver = tf.train.Saver()
    sess = tf.Session()

    saver.restore(sess, config.load_path)
    print("loaded model from %s." % config.load_path)

    tflearn.is_training(False, sess)
    iid = 0
    os.makedirs(config.save_dir, exist_ok=True)

    for _ in trange(config.times):
        generated = sess.run(fake_data)
        for arr in generated:
            arr = (127.5 * (arr + 1)).astype(np.uint8)
            img = Image.fromarray(arr, "RGB")
            img.save(os.path.join(config.save_dir, "%d.png" % iid))
            iid += 1



