from six.moves import xrange

import os

import tensorflow as tf
import tflearn
import numpy as np
from tqdm import trange
from scipy.stats import entropy


EPS = 1e-6


def run_task(config,
               train_data_loader,
               eval_data_loader,
               generator_forward,
               code_classifier_forward,
               image_classifier_forward,
               image_classifier_optimizer,
               code_classifier_optimizer,
               model_path):
    print("building graph...")
    # generator
    code = tf.random_normal([config.batch_size, 128], name="code")
    fake_data = generator_forward(config, noise=code, name="generator")
    generator_variables = {var.name.split(":")[0]: var for var in tf.global_variables() if var.name.startswith("generator")
                           and not var.name.endswith("is_training:0")}

    # image classifier
    image_classifier_inputs = tf.placeholder(tf.float32, shape=[None] + train_data_loader.shape(), name="input")
    image_classifier_label_inputs = tf.placeholder(tf.int32,
                            shape=[None, train_data_loader.classes()], name="labels")
    image_classifier_logits = image_classifier_forward(config, image_classifier_inputs, name="image_classifier")
    image_classifier_variables = [var for var in tf.all_variables() if var.name.startswith("image_classifier")]

    global_step = tf.Variable(0, False)

    image_classifier_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=image_classifier_label_inputs,
        logits=image_classifier_logits))
    image_classifier_labels = tf.cast(tf.argmax(tf.nn.softmax(image_classifier_logits), axis=-1), tf.int32)
    image_classifier_step = image_classifier_optimizer.minimize(image_classifier_loss, global_step=global_step,
                    var_list=[var for var in image_classifier_variables if var in tf.trainable_variables()])
    image_classifier_accuracy = tf.reduce_mean(tf.cast(tf.equal(
        image_classifier_labels,
        tf.cast(tf.argmax(image_classifier_label_inputs, axis=-1), tf.int32)),
        tf.float32))

    # code classifier
    code_classifier_truth = tf.nn.softmax(
        image_classifier_forward(config, fake_data, name="code_classifier", reuse=True))
    code_classifier_logits = code_classifier_forward(config, code, fake_data)
    code_classifier_probs = tf.nn.softmax(code_classifier_logits)
    code_classifier_labels = tf.cast(tf.argmax(code_classifier_probs, axis=-1), tf.int32)
    code_classifier_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=code_classifier_truth, logits=code_classifier_logits))
    code_classifier_accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.cast(tf.argmax(code_classifier_truth, axis=-1), tf.int32),
                         tf.cast(tf.argmax(code_classifier_logits, axis=-1), tf.int32)), tf.float64)
    )
    code_classifier_variables = [var for var in tf.all_variables()
                                 if var.name.startswith("code_classifier")]
    code_classifier_step = code_classifier_optimizer.minimize(code_classifier_loss,
                    var_list=[var for var in code_classifier_variables if var in tf.trainable_variables()])

    saver_generator = tf.train.Saver(generator_variables)
    saver_task_classifier = tf.train.Saver(image_classifier_variables + code_classifier_variables, max_to_keep=15)
    print("graph built.")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("loading generator weights from %s..." % model_path)
    saver_generator.restore(sess, model_path)
    print("weights loaded.")

    if config.log_path is not None:
        log_fobj = open(config.log_path, "w")
    else:
        log_fobj = None

    total_step = 0
    for epoch in xrange(config.num_epoch):
        num_steps = int(config.num_example / config.batch_size)
        bar = trange(num_steps, leave=False)
        for step in bar:
            # update image classifier
            tflearn.is_training(False, sess)
            if total_step < config.gen_start:
                current_frac = 0.0
            else:
                current_frac = config.gen_frac_final if total_step >= config.gen_frac_step + config.gen_start else (
                    config.gen_frac_init + (total_step - config.gen_start) *
                    (config.gen_frac_final - config.gen_frac_init) /
                    config.gen_frac_step
                )

            # keep_prob = 0.9993 ** total_step
            # if np.random.uniform(0, 1) > keep_prob:
            #     current_frac = 0.0

            if current_frac <= EPS:
                images, labels = train_data_loader.next_batch(config.batch_size)
            else:
                fake_images, fake_probs, fake_labels = sess.run([fake_data,
                                                                 code_classifier_probs,
                                                                 code_classifier_labels])
                ent = entropy(np.transpose(fake_probs))
                sample_prob = np.exp(-4 * ent)
                sample_prob = sample_prob / np.sum(sample_prob)

                # print(sample_prob, sample_prob.shape, fake_probs, fake_probs.shape)
                picked = np.random.choice(np.arange(
                    0, len(fake_probs), dtype=np.int64),
                    int(config.batch_size * current_frac),
                    replace=False,
                    p=sample_prob)
                fake_images = fake_images[picked]
                fake_labels = fake_labels[picked]

                fake_labels_one_hot = np.zeros(
                    (len(fake_labels), train_data_loader.classes()), np.int32)
                fake_labels_one_hot[np.arange(0, len(fake_labels), dtype=np.int64),
                    fake_labels] = 1
                fake_labels = fake_labels_one_hot

                if current_frac < 1 - EPS:
                    real_images, real_labels = train_data_loader.next_batch(config.batch_size -
                                                                      int(config.batch_size * current_frac))
                    images, labels = (np.concatenate([fake_images, real_images], axis=0),
                                     np.concatenate([fake_labels, real_labels], axis=0))
                else:
                    images = fake_images
                    labels = fake_labels

            eval_images, eval_labels = eval_data_loader.next_batch(config.batch_size)
            feed_dict = {image_classifier_inputs: images,
                         image_classifier_label_inputs: labels.astype(np.int32)}
            tflearn.is_training(True, sess)
            train_acc, _ = sess.run([image_classifier_accuracy, image_classifier_step], feed_dict=feed_dict)
            tflearn.is_training(False, sess)
            eval_acc = sess.run(image_classifier_accuracy, feed_dict={
                image_classifier_inputs: eval_images,
                image_classifier_label_inputs: eval_labels.astype(np.int32)})

            # update code classifier
            tflearn.is_training(True, sess)
            code_acc, _ = sess.run([code_classifier_accuracy, code_classifier_step])
            bar.set_description("(I)train accuracy %.4f, (I)eval accuracy %.4f, (C)train accuracy %.4f"
                                % (train_acc, eval_acc, code_acc))
            total_step += 1

            if log_fobj is not None:
                log_fobj.write("(I)train accuracy %.4f, (I)eval accuracy %.4f, (C)train accuracy %.4f\n"
                                % (train_acc, eval_acc, code_acc))

        if config.save_dir:
            saver_task_classifier.save(sess, os.path.join(config.save_dir, "task-model"))

    if log_fobj is not None:
        log_fobj.close()

    sess.close()