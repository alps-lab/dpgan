import tensorflow as tf
from tflearn.layers.normalization import batch_normalization

from utils.ops import conv_2d, fully_connected
from utils.ops import residual_block_upsample, residual_block_downsample


def generator_forward(config, noise=None, scope="generator", name=None, reuse=False, num_samples=-1):
    with tf.variable_scope(scope, name, reuse=reuse):
        if noise is None:
            noise = tf.random_normal([config.batch_size if num_samples == -1 else num_samples, 128], name="noise")

        output = fully_connected(noise, 4 * 4 * 8 * config.gen_dim, name="input")
        output = tf.reshape(output, [-1, 4, 4, 8 * config.gen_dim])

        output = residual_block_upsample(output, 8 * config.gen_dim, 3, name="rb1")
        output = residual_block_upsample(output, 4 * config.gen_dim, 3, name="rb2")
        output = residual_block_upsample(output, 2 * config.gen_dim, 3, name="rb3")
        output = residual_block_upsample(output, config.gen_dim, 3, name="rb4")

        output = batch_normalization(output)
        output = tf.nn.relu(output)
        output = conv_2d(output, 3, 3, name="output")
        output = tf.tanh(output)

    return output


def discriminator_forward(config, incoming,
                      scope="discriminator", name=None, reuse=False):
    with tf.variable_scope(scope, name, reuse=reuse):
        output = conv_2d(incoming, config.disc_dim, 3, name="input")

        output = residual_block_downsample(output, 2 * config.disc_dim, 3, name="rb1")
        output = residual_block_downsample(output, 4 * config.disc_dim, 3, name="rb2")
        output = residual_block_downsample(output, 8 * config.disc_dim, 3, name="rb3")
        output = residual_block_downsample(output, 8 * config.disc_dim, 3, name="rb4")

        output = tf.reshape(output, [-1, 4 * 4 * 8 * config.disc_dim])
        output = fully_connected(output, 1, name="output")

    return tf.reshape(output, [-1])