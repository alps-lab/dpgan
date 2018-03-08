#!/usr/bin/env python
import tensorflow as tf
from tflearn.layers import conv_2d, fully_connected, residual_block, batch_normalization, dropout
from tflearn.activations import relu


def res18_forward(incoming, scope=None, name="resnet_18", reuse=False):
    with tf.variable_scope(scope, default_name=name, reuse=reuse):
        network = conv_2d(incoming, 32, 5, 2, name="conv1",)
        network = residual_block(network, 2, 32, downsample=True, batch_norm=True, name="rb1")
        network = residual_block(network, 2, 64, downsample=True, batch_norm=True, name="rb2")
        network = residual_block(network, 2, 128, downsample=True, batch_norm=True, name="rb3")
        network = residual_block(network, 2, 256, downsample=True, batch_norm=True, name="rb4")
        network = dropout(network, 0.6)
        network = relu(batch_normalization(fully_connected(network, 128, name="fc1")))
        network = fully_connected(network, 1, name="fc2")

    return network


def classifier_forward(config, incoming, name=None, reuse=False,
                       scope="classifier"):
    return res18_forward(incoming, scope=scope, name=name, reuse=reuse)
