#!/usr/bin/env python3
import tensorflow as tf
from tflearn.layers import batch_normalization
from tflearn.layers.conv import conv_2d_transpose
from tflearn.activations import leaky_relu, relu

from utils.ops import fully_connected, conv_2d, layer_norm

# We use DCGAN here


def generator_forward(config, noise=None, scope="generator", name=None, reuse=False, num_samples=-1):
    with tf.variable_scope(scope, name, reuse=reuse):
        if noise is None:
            noise = tf.random_normal([config.batch_size if num_samples == -1 else num_samples, 128], name="noise")

        output = fully_connected(noise, 4 * 4 * 8 * config.gen_dim, name="input")
        output = tf.reshape(output, [-1, 4, 4, 8 * config.gen_dim])
        output = batch_normalization(output)
        output = relu(output)

        output = conv_2d_transpose(output, 4 * config.gen_dim, 5, [8, 8], name="conv1", strides=2)
        output = batch_normalization(output)
        output = relu(output)

        output = conv_2d_transpose(output, 2 * config.gen_dim, 5, [16, 16], name="conv2", strides=2)
        output = batch_normalization(output)
        output = relu(output)

        output = conv_2d_transpose(output, config.gen_dim, 5, [32, 32], name="conv3", strides=2)
        output = batch_normalization(output)
        output = relu(output)

        output = conv_2d_transpose(output, 3, 5, [64, 64], name="conv4", strides=2)
        output = tf.tanh(output)

    return output


def discriminator_forward(config, incoming,
                      scope="discriminator", name=None, reuse=False):
    with tf.variable_scope(scope, name, reuse=reuse):
        output = conv_2d(incoming, config.gen_dim, 5, strides=2, name="conv1")
        output = leaky_relu(output, 0.2)
        output = conv_2d(output, 2 * config.gen_dim, 5, strides=2, name="conv2")

        output = leaky_relu(output, 0.2)
        output = conv_2d(output, 4 * config.gen_dim, 5, strides=2, name="conv3")

        output = leaky_relu(output, 0.2)
        output = conv_2d(output, 8 * config.gen_dim, 5, strides=2, name="conv4")

        output = leaky_relu(output, 0.2)
        output = tf.reshape(output, [-1, 4 * 4 * 8 * config.gen_dim])
        output = fully_connected(output, 1, bias=False)

    return output