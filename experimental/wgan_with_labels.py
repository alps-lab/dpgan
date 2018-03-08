#!/usr/bin/env python
from six.moves import xrange

import argparse
import os

import numpy as np
import tensorflow as tf
import tflearn
from tflearn.activations import leaky_relu, relu
from tflearn.layers.conv import conv_2d_transpose, conv_2d
from tflearn.layers import fully_connected, batch_normalization
from tqdm import trange

from utils.data_utils import MNISTLoader, generate_images


class DataLoader(object):

    def __init__(self):
        pass


def sample_labels(batch_size):
    counts = np.random.multinomial(batch_size, [0.1] * 10)
    indices = [[i] * count for i, count in enumerate(counts)]
    indices = [j for js in indices for j in js]
    indices = np.random.permutation(indices)

    y = np.zeros((len(indices), 10), np.int64)
    y[np.arange(0, len(indices), dtype=np.int64), indices] = 1.
    return np.asarray(y, np.float32)


def regular_labels():
    indices = [[i] * 10 for i in xrange(10)]
    indices = [j for js in indices for j in js]

    y = np.zeros((len(indices), 10), np.int64)
    y[np.arange(0, len(indices), dtype=np.int64), indices] = 1
    return np.asarray(y, np.float32)


def generator_forward(config, labels, noise=None,
                      scope="generator", name=None, reuse=False, num_samples=-1):
    with tf.variable_scope(scope, name, reuse=reuse):
        if noise is None:
            noise = tf.random_normal([config.batch_size if num_samples == -1 else num_samples, 128], name="noise")
        embed = fully_connected(labels, 8 * config.dim)
        noise = fully_connected(noise, 56 * config.dim)
        cat = relu(batch_normalization(tf.concat([embed, noise], axis=-1)))
        output = fully_connected(cat, 4 * 4 * 4 * config.dim)
        output = batch_normalization(output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4, 4, 4*config.dim])

        output = conv_2d_transpose(output, 2 * config.dim, 5, [8, 8], strides=2)
        output = output[:, :7, :7, :]
        output = batch_normalization(output)
        output = relu(output)

        output = conv_2d_transpose(output, config.dim, 5, [14, 14], strides=2)
        output = batch_normalization(output)
        output = tf.nn.relu(output)

        output = conv_2d_transpose(output, 1, 5, [28, 28], strides=2)

        output = tf.tanh(output)

    return output


def discriminator_forward(config, labels, incoming,
                      scope="discriminator", name=None, reuse=False):
    with tf.variable_scope(scope, name, reuse=reuse):
        output = leaky_relu(batch_normalization(conv_2d(incoming, config.dim, 5, 2, name="conv1"), 0.2))
        output = leaky_relu(batch_normalization(conv_2d(output, 2 * config.dim, 5, 2, name="conv2"), 0.2))
        output = leaky_relu(batch_normalization(conv_2d(output, 4 * config.dim, 5, 2, name="Conv3"), 0.2))
        output = tf.reshape(output, [-1, 4 * 4 * 4 * config.dim])

        output = fully_connected(output, 56 * config.dim, name="fc1_1")
        embed = fully_connected(labels, 8 * config.dim, name="fc1_2")

        output = leaky_relu(batch_normalization(tf.concat([output, embed], axis=-1)), 0.2)
        output = fully_connected(output, 8 * config.dim, name="fc2")
        output = batch_normalization(output)
        output = leaky_relu(output, 0.2)
        output = tf.reshape(fully_connected(output, 1, bias=False, name="fc3"), [-1])

    return output


def build_graph(config):
    real_labels = tf.placeholder(tf.float32, [None, 10], name="real_labels")
    fake_labels = tf.placeholder(tf.float32, [None, 10], name="fake_labels")
    real_inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name="real_images")
    fake_inputs = generator_forward(config, fake_labels, name="fake_images")

    return real_labels, fake_labels, real_inputs, fake_inputs


def create_train_ops(config, global_step, real_labels, fake_labels, real_inputs, fake_inputs):
    fake_output = discriminator_forward(config, fake_labels, fake_inputs, name="discriminator")
    real_output = discriminator_forward(config, real_labels, real_inputs, reuse=True, name="discriminator")

    disc_cost = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
    gen_cost = -tf.reduce_mean(fake_output)

    alphas = tf.random_uniform([config.batch_size], minval=0, maxval=1)
    differences = real_inputs - fake_inputs
    interploted_images = fake_inputs + alphas[:, None, None, None] * differences
    interploted_labels = fake_labels + alphas[:, None] * (real_labels - fake_labels)
    interploted_output = discriminator_forward(config, interploted_labels, interploted_images,
                                               reuse=True, name="discriminator")
    gradients = tf.gradients(interploted_output, [interploted_images, interploted_labels],
                             colocate_gradients_with_ops=True)
    slopes = tf.sqrt(tf.reduce_sum(gradients[0] ** 2, axis=[1, 2, 3]))
    penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += config.lambd * penalty

    gen_train_op = tf.train.AdamOptimizer(learning_rate=3e-4, beta1=0.5, beta2=0.9).minimize(
        gen_cost,
        var_list=[var for var in tf.trainable_variables() if var.name.startswith("generator")],
        global_step=global_step
    )
    disc_train_op = tf.train.AdamOptimizer(learning_rate=3e-4, beta1=0.5, beta2=0.9).minimize(
        disc_cost,
        var_list=[var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
    )

    return gen_train_op, disc_train_op, gen_cost, disc_cost


def train(config):
    data_loader = MNISTLoader(config.data_dir)
    real_labels, fake_labels, real_inputs, fake_inputs = build_graph(config)
    global_step = tf.Variable(0, False)
    gen_train_ops, disc_train_ops, gen_loss, disc_loss = create_train_ops(config,
                                    global_step, real_labels, fake_labels,
                                    real_inputs, fake_inputs)
    saver = tf.train.Saver(max_to_keep=20)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    num_steps = data_loader.num_steps(config.batch_size)

    if config.save_dir:
        os.makedirs(config.save_dir, exist_ok=True)
    if config.image_dir:
        os.makedirs(config.image_dir, exist_ok=True)

    total_step = 0
    for epoch in xrange(config.epoch):
        bar = trange(num_steps, leave=False)
        for _ in bar:
            disc_loss_value, gen_loss_value = 0.0, 0.0
            tflearn.is_training(True, sess)
            if total_step == 0:
                sess.run([], feed_dict={global_step: 1})
            else:
                gen_loss_value, _ = sess.run([gen_loss, gen_train_ops],
                         feed_dict={fake_labels: sample_labels(config.batch_size)})
            for i in xrange(5):
                bx, by = data_loader.next_batch(config.batch_size)
                disc_loss_value, _ = sess.run(
                    [disc_loss, disc_train_ops],
                    feed_dict={real_labels: by,
                               fake_labels: by,
                               real_inputs: bx}
                )
            bar.set_description("epoch %d, gen loss %.4f, disc loss %.4f"
                                % (epoch, gen_loss_value, disc_loss_value))
            tflearn.is_training(False, sess)
            if total_step % 20 == 0 and config.image_dir:
                sampled_labels = regular_labels()
                generated = sess.run(fake_inputs, feed_dict={fake_labels: sampled_labels})
                generate_images(generated, data_loader.mode(),
                                os.path.join(config.image_dir, "gen_step_%d.jpg" % total_step))
                generate_images(
                    data_loader.next_batch(config.batch_size)[0],
                    data_loader.mode(),
                    os.path.join(config.image_dir, "real_step_%d.jpg" % total_step))

            total_step += 1
        bar.close()
        if config.save_dir is not None:
            saver.save(sess, os.path.join(config.save_dir, "model"), global_step=global_step,
                       write_meta_graph=False)
    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dim", dest="dim", type=int, default=64)
    parser.add_argument("--data-dir", dest="data_dir")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=100)
    parser.add_argument("-e", "--epoch", dest="epoch", type=int, default=5)
    parser.add_argument("-l", "--lambd", dest="lambd", type=float, default=10.0)

    parser.add_argument("--image-dir", dest="image_dir")
    parser.add_argument("--save-dir", dest="save_dir")

    train(parser.parse_args())


