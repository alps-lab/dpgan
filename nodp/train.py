from six.moves import xrange

import os

import numpy as np
import tensorflow as tf
from tqdm import trange
import tflearn

from utils.data_utils import generate_images
from .flow import train_graph_per_tower, aggregate_flow


def get_train_ops(config, real_data, fake_data, global_step, discriminator_forward,
                  gen_optimizer, disc_optimizer):
    with tf.device("/cpu:0"):
        discriminator_forward(config, tf.zeros(shape=[1] + [d.value for d in real_data.shape[1:]]), scope="discriminator")

    real_data_splits = tf.split(real_data, config.num_gpu, axis=0)
    fake_data_splits = tf.split(fake_data, config.num_gpu, axis=0)
    disc_grads = []
    gen_costs = []
    disc_costs = []

    for g, (real_data, fake_data) in enumerate(zip(real_data_splits, fake_data_splits)):
        with tf.device("/gpu:%d" % g):
            disc_cost, gen_cost, disc_grad = train_graph_per_tower(
                config, discriminator_forward, real_data, fake_data)
            disc_costs.append(disc_cost)
            gen_costs.append(gen_cost)
            disc_grads.append(disc_grad)

    return aggregate_flow(config, disc_costs, gen_costs, disc_grads,
                          gen_optimizer=gen_optimizer,
                          disc_optimizer=disc_optimizer,
                          global_step=global_step)


def train_steps(config, data_loader, real_data, fake_data, global_step,
                gen_train_op, disc_train_op, gen_cost, disc_cost, callback_before_train=None):
    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=15)
    sess = tf.Session()

    if config.load_path:
        saver.restore(sess, config.load_path)
        total_step = sess.run(global_step)
        print("continue training at step %d..." % total_step)
    else:
        sess.run(init)
        total_step = 0

    if callback_before_train is not None:
        callback_before_train(sess, total_step)

    for epoch in xrange(config.num_epoch):
        gen_losses = []
        disc_losses = []
        bar = trange(data_loader.num_steps(config.batch_size * config.num_gpu), leave=False)
        for step in bar:
            if config.total_step is not None and total_step > config.total_step:
                break

            tflearn.is_training(True, sess)
            gen_cost_value = 0.0
            if total_step > 0:
                gen_cost_value, _ = sess.run([gen_cost, gen_train_op])
                gen_losses.append(gen_cost_value)
            else:
                sess.run([], feed_dict={global_step: 1})

            for i in xrange(config.critic_iters):
                disc_cost_value, _ = sess.run([disc_cost, disc_train_op],
                                              feed_dict=
                                              {real_data: data_loader.next_batch(config.num_gpu * config.batch_size)[0]})
                if i == config.critic_iters - 1:
                    disc_losses.append(disc_cost_value)
            bar.set_description("gen loss: %.4f, disc loss: %.4f" % (gen_cost_value, disc_cost_value))

            tflearn.is_training(False, sess)
            if total_step % config.image_every == 0 and config.image_dir:
                generated = sess.run(fake_data)
                generate_images(generated, data_loader.mode(),
                                os.path.join(config.image_dir, "gen_step_%d.jpg" % total_step))
                generate_images(np.concatenate(
                    [data_loader.next_batch(config.batch_size)[0] for _ in xrange(config.num_gpu)], axis=0),
                    data_loader.mode(),
                    os.path.join(config.image_dir, "real_step_%d.jpg" % total_step))

            if total_step % config.save_every == 0 and config.save_dir and total_step > 0:
                saver.save(sess, os.path.join(config.save_dir, "model"), write_meta_graph=False,
                           global_step=global_step)

            total_step += 1
        bar.close()

    if config.save_dir:
        saver.save(sess, os.path.join(config.save_dir, "model"), write_meta_graph=False,
                   global_step=global_step)


def train(config, data_loader, generator_forward, discriminator_forward,
          gen_optimizer, disc_optimizer, callback_before_train=None):
    print("parameters:", config)

    if config.image_dir:
        os.makedirs(config.image_dir, exist_ok=True)
    if config.save_dir:
        os.makedirs(config.save_dir, exist_ok=True)

    print("building graph...")
    global_step = tf.Variable(0, trainable=False)
    real_data = tf.placeholder(tf.float32, shape=[config.num_gpu * config.batch_size] + data_loader.shape())
    fake_data = generator_forward(config, num_samples=config.num_gpu * config.batch_size)

    (gen_train_op, disc_train_op), (gen_cost, disc_cost) = get_train_ops(config, real_data, fake_data, global_step,
                                                                         discriminator_forward,
                                                                         gen_optimizer,
                                                                         disc_optimizer)
    print("graph built.")

    train_steps(config, data_loader, real_data,
                fake_data, global_step, gen_train_op, disc_train_op, gen_cost, disc_cost,
                callback_before_train=callback_before_train)
    print("done with parameters:", config)