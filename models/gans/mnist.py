import tensorflow as tf
from tflearn.layers import activation, batch_normalization
from tflearn.layers.conv import conv_2d_transpose
from tflearn.activations import leaky_relu

from utils.ops import conv_2d, fully_connected


def generator_forward(config, noise=None,
                      scope="generator", name=None, reuse=False, num_samples=-1):
    with tf.variable_scope(scope, name, reuse=reuse):
        if noise is None:
            noise = tf.random_normal([config.batch_size if num_samples == -1 else num_samples, 128], name="noise")

        output = fully_connected(noise, 4*4*4*config.dim)
        output = batch_normalization(output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4, 4, 4*config.dim])

        output = conv_2d_transpose(output, 2 * config.dim, 5, [8, 8], strides=2)
        output = output[:, :7, :7, :]

        output = conv_2d_transpose(output, config.dim, 5, [14, 14], strides=2)
        output = tf.nn.relu(output)

        output = conv_2d_transpose(output, 1, 5, [28, 28], strides=2)

        output = tf.tanh(output)

    return output


def discriminator_forward(config, incoming,
                      scope="discriminator", name=None, reuse=False):
    with tf.variable_scope(scope, name, reuse=reuse):
        output = leaky_relu(conv_2d(incoming, config.dim, 5, 2), 0.2)
        output = leaky_relu(conv_2d(output, 2 * config.dim, 5, 2), 0.2)
        output = leaky_relu(conv_2d(output, 4 * config.dim, 5, 2), 0.2)

        output = tf.reshape(output, [-1, 4 * 4 * 4 * config.dim])
        output = tf.reshape(fully_connected(output, 1, bias=False), [-1])

    return output