import tensorflow as tf


def train_graph_per_tower(config, discriminator_forward, real_data, fake_data):
    disc_real_outputs = discriminator_forward(config, real_data, reuse=True)
    disc_fake_outputs = discriminator_forward(config, fake_data, reuse=True)

    gen_cost = -tf.reduce_mean(disc_fake_outputs)
    disc_cost = tf.reduce_mean(disc_fake_outputs) - tf.reduce_mean(disc_real_outputs)
    alphas = tf.random_uniform(shape=[config.batch_size], minval=0., maxval=1.)
    differences = fake_data - real_data
    interpolates = real_data + (alphas[:, tf.newaxis, tf.newaxis, tf.newaxis] * differences)
    disc_interploated_outputs = discriminator_forward(config, interpolates, reuse=True)

    gradients = tf.gradients(disc_interploated_outputs, [interpolates], colocate_gradients_with_ops=True)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    disc_cost += config.lambd * gradient_penalty

    disc_weights = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

    return disc_cost, gen_cost, tf.train.GradientDescentOptimizer(1e-3).compute_gradients(disc_cost, disc_weights)


def aggregate_flow(config, disc_costs, gen_costs, disc_grads,
                   gen_optimizer, disc_optimizer, global_step):
    gen_weights = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    disc_weights = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]

    final_disc_cost = tf.add_n(disc_costs) / config.num_gpu
    final_gen_cost = tf.add_n(gen_costs) / config.num_gpu

    final_gen_grads = {w: g for g, w in tf.train.GradientDescentOptimizer(
            learning_rate=1e-3).compute_gradients(final_gen_cost, var_list=gen_weights,
                                                  colocate_gradients_with_ops=True)}
    final_disc_grads = {w: 0.0 for w in disc_weights}

    for w in disc_weights:
        final_disc_grads[w] = tf.add_n([tup[0] for ws in disc_grads for tup in ws if tup[1] == w])

    for w in final_disc_grads:
        final_disc_grads[w] /= config.num_gpu

    gen_train_op = gen_optimizer.apply_gradients(
        [(g, w) for w, g in final_gen_grads.items()], global_step)

    disc_train_op = disc_optimizer.apply_gradients([(g, w) for w, g in final_disc_grads.items()])

    return (gen_train_op, disc_train_op), (final_gen_cost, final_disc_cost)