#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
from tflearn.layers import batch_normalization
from tflearn.layers.core import dropout, flatten
from tflearn.activations import relu

from utils.ops import conv_2d, fully_connected


def classifier_forward(config, incoming, name=None, reuse=False,
                       scope="classifier"):
    with tf.variable_scope(scope, name, reuse=reuse):
        network = incoming
        network = relu(batch_normalization(
            conv_2d(network, 32, 5, activation='relu', regularizer="L2", strides=2)))
        network = relu(batch_normalization(
            conv_2d(network, 64, 5, activation='relu', regularizer="L2", strides=2)))
        network = flatten(network)

        network = relu(batch_normalization(fully_connected(network, 1024)))
        network = dropout(network, 0.5)

        network = fully_connected(network, 10)

    return network


