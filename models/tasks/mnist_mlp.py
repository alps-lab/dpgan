#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
from tflearn.layers.core import dropout, flatten
from tflearn.activations import leaky_relu, relu

from utils.ops import conv_2d, fully_connected


def classifier_forward(config, incoming, name=None, reuse=False,
                       scope="classifier"):
    with tf.variable_scope(scope, name, reuse=reuse):
        network = incoming
        network = relu(
            conv_2d(network, 32, 5, strides=2))
        network = relu(
            conv_2d(network, 64, 5, strides=2))
        network = flatten(network)

        network = relu(fully_connected(network, 1024))
        network = dropout(network, 0.7)

        network = fully_connected(network, 10)

    return network

