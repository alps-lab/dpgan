from six.moves import xrange

import os
from functools import partial

import tflearn
import tensorflow as tf
import numpy as np
from tqdm import trange

from .per_example_flow import train_graph_per_tower, aggregate_flow, gradient_norms_estimate_tower
from utils.data_utils import generate_images


def get_train_ops(config, real_data, fake_data, global_step, discriminator_forward,
                  gen_optimizer, disc_optimizer,
                  supervisor, accountant=None):
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
                config, discriminator_forward, real_data, fake_data, supervisor)
            disc_costs.append(disc_cost)
            gen_costs.append(gen_cost)
            disc_grads.append(disc_grad)

    if supervisor.sampler is not None:
        supervisor.sampler.set_forward_function(partial(gradient_norms_estimate_tower,
                                                        config, discriminator_forward,
                                                        real_data_splits[0], fake_data_splits[0],
                                                        supervisor))

    return aggregate_flow(config, disc_costs, gen_costs, disc_grads,
                          disc_optimizer=disc_optimizer,
                          gen_optimizer=gen_optimizer,
                          global_step=global_step, supervisor=supervisor, accountant=accountant)


def train_steps(config, data_loader, real_data, fake_data, global_step,
                gen_train_op, gen_cost, supervisor,
                accountant=None):
    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=25)
    gan_saver = tf.train.Saver(var_list=
                               [var for var in tf.global_variables()
                                if var.name.startswith(("generator", "discriminator"))
                                and not var.name.endswith("is_training:0")])
    sess = tf.Session()
    if config.load_path:
        saver.restore(sess, config.load_path)
        total_step = sess.run(global_step)
        print("continue training at step %d..." % total_step)
    elif config.gan_load_path:
        sess.run(init)
        gan_saver.restore(sess, config.gan_load_path)
        total_step = 0
        print("continue training at model in %s..." % config.gan_load_path)
    else:
        sess.run(init)
        total_step = 0

    supervisor.callback_before_train(sess, total_step)

    early_stop = False
    for epoch in xrange(config.num_epoch):
        gen_losses = []
        disc_losses = []
        bar = trange(data_loader.num_steps(config.batch_size * config.num_gpu), leave=False)
        for _ in bar:
            if early_stop:
                break
            if config.total_step is not None and total_step > config.total_step:
                break
            tflearn.is_training(True, sess)
            gen_cost_value = 0.0
            # if total_step > 0:
            #     gen_cost_value, _ = sess.run([gen_cost, gen_train_op])
            #     gen_losses.append(gen_cost_value)
            # else:
            #     sess.run([], feed_dict={global_step: 1})

            ret = supervisor.callback_before_iter(sess, total_step)
            num_critic = ret["num_critic"]
            for i in xrange(num_critic):
                disc_cost_value = supervisor.callback_disc_iter(sess, total_step, i,
                           real_data, data_loader,
                           accountant=accountant)
            #     if i == num_critic - 1:
            #         disc_losses.append(disc_cost_value)
            # bar.set_description("gen loss: %.4f, disc loss: %.4f" % (gen_cost_value, disc_cost_value))

            tflearn.is_training(False, sess)
            # if total_step % config.image_every == 0 and config.image_dir:
            #     generated = sess.run(fake_data)
            #     generate_images(generated, data_loader.mode(),
            #                     os.path.join(config.image_dir, "gen_step_%d.jpg" % total_step))
            #     generate_images(np.concatenate(
            #         [data_loader.next_batch(config.batch_size)[0] for _ in xrange(config.num_gpu)], axis=0),
            #         data_loader.mode(),
            #         os.path.join(config.image_dir, "real_step_%d.jpg" % total_step))

            # if total_step % config.save_every == 0 and config.save_dir and total_step > 0:
            #     saver.save(sess, os.path.join(config.save_dir, "model"), write_meta_graph=False,
            #                global_step=global_step)

            # if total_step % config.log_every == 0 and accountant and config.log_path:
            #     spent_eps_deltas = accountant.get_privacy_spent(
            #         sess, target_eps=config.target_epsilons)
            #
            #     with open(config.log_path, "a") as log_file:
            #         log_file.write("privacy log at step: %d\n" % total_step)
            #         for spent_eps, spent_delta in spent_eps_deltas:
            #             to_print = "spent privacy: eps %.4f delta %.5g" % (spent_eps, spent_delta)
            #             log_file.write(to_print + "\n")
            #         log_file.write("\n")

            if total_step % 10 == 0 and accountant and config.terminate:
                spent_eps_deltas = accountant.get_privacy_spent(
                    sess, target_eps=config.target_epsilons)

                for (spent_eps, spent_delta), target_delta in zip(
                        spent_eps_deltas, config.target_deltas):
                    # print(spent_delta, target_delta, type(target_delta), type(spent_delta))
                    # print(spent_delta, target_delta)
                    if spent_delta > target_delta:
                        early_stop = True
                        print("terminate at %d." % total_step)
                        break

            total_step += 1
        bar.close()

    if config.save_dir:
        saver.save(sess, os.path.join(config.save_dir, "model"), write_meta_graph=False, global_step=global_step)


def train(config, data_loader, generator_forward, discriminator_forward,
          disc_optimizer, gen_optimizer,
          supervisor, accountant=None):
    print("parameters:", config)

    if config.image_dir:
        os.makedirs(config.image_dir, exist_ok=True)
    if config.save_dir:
        os.makedirs(config.save_dir, exist_ok=True)

    print("building graph...")
    global_step = tf.Variable(0, trainable=False)
    real_data = tf.placeholder(tf.float32, shape=[config.num_gpu * config.batch_size] + data_loader.shape())
    fake_data = generator_forward(config, num_samples=config.num_gpu * config.batch_size)

    gen_train_op, gen_cost = get_train_ops(config, real_data, fake_data, global_step,
                                         discriminator_forward,
                                         disc_optimizer=disc_optimizer,
                                         gen_optimizer=gen_optimizer,
                                         accountant=accountant,
                                         supervisor=supervisor)
    print("graph built.")

    train_steps(config, data_loader, real_data,
                fake_data, global_step, gen_train_op, gen_cost, accountant=accountant,
                supervisor=supervisor)
    print("done with parameters:", config)