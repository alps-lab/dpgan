#!/usr/bin/env python
from six.moves import xrange

import numpy as np
import tensorflow as tf

from tflearn.layers.conv import conv_2d
from tflearn.activations import leaky_relu
from tflearn.layers import fully_connected
from tflearn.layers.normalization import batch_normalization

def discriminator_forward(config, incoming, labels, scope="discriminator",
                          name=None, reuse=False):

    with tf.variable_scope(scope, name, reuse=reuse):
        output = leaky_relu(batch_normalization(
            conv_2d(incoming, config.dim, 5, 2, name="conv1"))
            , 0.2)

        output = leaky_relu(batch_normalization(
            conv_2d(output, 2 * config.dim, 5, 2, name="conv2")),
            0.2)

        output_shared = conv_2d(output, 2 * config.dim, 5, 2, name="conv3_shared")
        output_cs = [conv_2d(output, 2 * config.dim, 5, 2, name="conv3_cs")
                     for _ in xrange(5)]

        output = tf.concat([output_cs, output_shared])

        output = tf.reshape(output, [-1, 4 * 4 * 4 * config.dim])
        output = tf.reshape(fully_connected(output, 1, bias=False), [-10])
