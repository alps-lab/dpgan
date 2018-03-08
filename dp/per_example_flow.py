import numpy as np
import tensorflow as tf

from dp.per_example_forward import discriminator_forward_per_examples, discriminator_forward_with_lookups


def gradient_norms_estimate_tower(config, discriminator_forward, real_data, fake_data,
                         supervisor):
    disc_real_outputs, lookups = discriminator_forward_per_examples(discriminator_forward, config, real_data)
    disc_fake_outputs = discriminator_forward_with_lookups(discriminator_forward,
                                                           config, fake_data, lookups
                                                           )

    disc_cost = tf.reshape((tf.add_n(disc_fake_outputs) - tf.add_n(disc_real_outputs)), []) / config.batch_size
    alphas = tf.random_uniform(shape=[config.batch_size], minval=0., maxval=1.)
    differences = fake_data - real_data
    interpolates = real_data + (alphas[:, tf.newaxis, tf.newaxis, tf.newaxis] * differences)
    disc_interploated_outputs = tf.concat(discriminator_forward_with_lookups(
        discriminator_forward, config, interpolates, lookups), axis=0)

    gradients = tf.gradients(disc_interploated_outputs, [interpolates], colocate_gradients_with_ops=True)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += config.lambd * gradient_penalty

    disc_weights = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

    disc_final_norms = {weight: [] for weight in disc_weights}
    for lookup in lookups:
        grads = tf.gradients(disc_cost, [lookup[weight] for weight in disc_weights])
        m = [(weight, grad) for weight, grad in zip(disc_weights, grads)]
        grads = [grad for weight, grad in m]
        for weight, grad in zip(disc_weights, grads):
            disc_final_norms[weight].append(tf.norm(grad))

    return real_data, disc_final_norms, config.batch_size, config.batch_size * config.num_gpu


def train_graph_per_tower(config, discriminator_forward, real_data, fake_data,
                          supervisor):
    disc_real_outputs, lookups = discriminator_forward_per_examples(discriminator_forward, config, real_data)
    disc_fake_outputs = discriminator_forward_with_lookups(discriminator_forward,
                                                           config, fake_data, lookups
                                                           )

    gen_cost = -tf.reshape(tf.add_n(disc_fake_outputs) / config.batch_size, [])
    disc_cost = tf.reshape((tf.add_n(disc_fake_outputs) - tf.add_n(disc_real_outputs)), []) / config.batch_size
    alphas = tf.random_uniform(shape=[config.batch_size], minval=0., maxval=1.)
    differences = fake_data - real_data
    interpolates = real_data + (alphas[:, tf.newaxis, tf.newaxis, tf.newaxis] * differences)
    disc_interploated_outputs = tf.concat(discriminator_forward_with_lookups(
        discriminator_forward, config, interpolates, lookups), axis=0)

    gradients = tf.gradients(disc_interploated_outputs, [interpolates], colocate_gradients_with_ops=True)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += config.lambd * gradient_penalty

    disc_weights = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

    disc_final_grad = {weight: 0.0 for weight in disc_weights}
    for lookup in lookups:
        grads = tf.gradients(disc_cost, [lookup[weight] for weight in disc_weights])
        m = [(weight, grad) for weight, grad in zip(disc_weights, grads)]
        clipped_grads = supervisor.callback_clip_grads(m)
        for weight, clipped_grad in zip(disc_weights, [config.batch_size * grad
                                                       for grad in clipped_grads]):
            disc_final_grad[weight] += clipped_grad / config.batch_size

    return disc_cost, gen_cost, [(g, w) for w, g in disc_final_grad.items()]


def aggregate_flow(config, disc_costs, gen_costs, disc_grads, gen_optimizer,
                   disc_optimizer, global_step,
                    supervisor, accountant=None):
    gen_weights = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    disc_weights = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

    final_disc_cost = tf.add_n(disc_costs) / config.num_gpu
    final_gen_cost = tf.add_n(gen_costs) / config.num_gpu

    final_gen_grads = {w: g for g, w in tf.train.GradientDescentOptimizer(
            learning_rate=1e-3).compute_gradients(final_gen_cost,
                                                  var_list=gen_weights, colocate_gradients_with_ops=True)}
    final_disc_grads = {w: 0.0 for w in disc_weights}

    for ws in disc_grads:
        for g, w in ws:
            final_disc_grads[w] += g
    for w in final_disc_grads:
        final_disc_grads[w] /= config.num_gpu
    final_disc_grads = supervisor.callback_noise_grads(final_disc_grads,
                                                       config.num_gpu * config.batch_size)

    gen_train_op = gen_optimizer.apply_gradients(
        [(g, w) for w, g in final_gen_grads.items()], global_step)

    if accountant:
        accountant_sigma = supervisor._accountant_sigma
        accountant_n = tf.placeholder(tf.int32, shape=())
        accountant_op = accountant.accumulate_privacy_spending(None, accountant_sigma, accountant_n)
        supervisor.register_accountant_ops(accountant_op, accountant_n)

    disc_train_ops = supervisor.callback_create_disc_train_ops(
        final_disc_grads, disc_optimizer, global_step)
    supervisor.register_disc_train_ops([final_disc_cost] + disc_train_ops, 0)

    return gen_train_op, final_gen_cost
