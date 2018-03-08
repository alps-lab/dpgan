#!/usr/bin/env python
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import tflearn
from six.moves import xrange
from tqdm import trange


def run_task_eval(config, eval_data_loader,
                  classifier_forward, model_dir):
    print("building graph...")

    classifier_inputs = tf.placeholder(tf.float32, shape=[None] +
                                                         eval_data_loader.shape(), name="input")
    classifier_label_inputs = tf.placeholder(tf.int32,
                                             shape=[None, eval_data_loader.classes()], name="labels")
    classifier_logits = classifier_forward(config, classifier_inputs, name="classifier")
    classifier_variables = [var for var in tf.all_variables() if var.name.startswith("classifier")
                            and not var.name.endswith("is_training:0")]

    # global_step = tf.Variable(0, False)
    # classifier_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=classifier_label_inputs,
    #                                                                      logits=classifier_logits))
    classifier_labels = tf.cast(tf.argmax(tf.nn.softmax(classifier_logits), axis=-1), tf.int32)
    classifier_accuracy = tf.reduce_mean(tf.cast(tf.equal(
        classifier_labels,
        tf.cast(tf.argmax(classifier_label_inputs, axis=1), tf.int32)), tf.float32))

    saver_classifier = tf.train.Saver(classifier_variables, max_to_keep=10)
    print("graph built.")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("loading classifier weights from %s..." % model_dir)
    saver_classifier.restore(sess, tf.train.latest_checkpoint(model_dir))
    print("weights loaded.")

    total_step = 0
    eval_accuracies = []
    for epoch in xrange(1):
        num_steps = eval_data_loader.num_steps(config.batch_size)
        bar = trange(num_steps, leave=False)
        for step in bar:
            eval_images, eval_labels = eval_data_loader.next_batch(config.batch_size)

            tflearn.is_training(False, sess)
            eval_accuracy = sess.run(classifier_accuracy, feed_dict={
                classifier_inputs: eval_images,
                classifier_label_inputs: eval_labels.astype(np.int32)})
            eval_accuracies.append(eval_accuracy)
            total_step += 1

    sess.close()
    return np.mean(eval_accuracies)
    # print("mean accuracy: %.4f" % np.mean(eval_accuracies))


def run_task(config, train_data_loader,
               eval_data_loader,
               generator_forward,
               classifier_forward,
               optimizer,
               model_path):
    print("building graph...")
    fake_data = generator_forward(config, name="generator")
    generator_variables = {var.name.split(":")[0]: var for var in tf.global_variables() if var.name.startswith("generator")
                           and not var.name.endswith("is_training:0")}

    classifier_inputs = tf.placeholder(tf.float32, shape=[None] + train_data_loader.shape(), name="input")
    classifier_label_inputs = tf.placeholder(tf.int32,
                            shape=[None, train_data_loader.classes()], name="labels")
    classifier_logits = classifier_forward(config, classifier_inputs, name="classifier")
    classifier_variables = [var for var in tf.all_variables() if var.name.startswith("classifier")]

    global_step = tf.Variable(0, False)

    classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=classifier_label_inputs,
                                                                         logits=classifier_logits))
    classifier_labels = tf.cast(tf.argmax(tf.nn.softmax(classifier_logits), axis=-1), tf.int32)
    classifier_step = optimizer.minimize(classifier_loss, global_step=global_step,
                    var_list=[var for var in classifier_variables if var in tf.trainable_variables()])
    classifier_accuracy = tf.reduce_mean(tf.cast(tf.equal(
        classifier_labels,
        tf.cast(tf.argmax(classifier_label_inputs, axis=-1), tf.int32)),
        tf.float32))

    saver_generator = tf.train.Saver(generator_variables)
    saver_classifier = tf.train.Saver(classifier_variables, max_to_keep=10)
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
            current_frac = config.gen_frac_final if total_step >= config.gen_frac_step else (
                config.gen_frac_init + total_step * (config.gen_frac_final - config.gen_frac_init) /
                config.gen_frac_step
            )
            if current_frac == 0:
                images, labels = train_data_loader.next_batch(config.batch_size)
                # labels = labels[:, 0]
            elif current_frac < 1:
                tflearn.is_training(False, sess)
                fake_images = sess.run(fake_data)
                fake_labels = sess.run(classifier_labels, feed_dict={classifier_inputs: fake_images})
                fake_images = fake_images[:int(config.batch_size * current_frac)]
                fake_labels = fake_labels[:int(config.batch_size * current_frac)]

                fake_labels_one_hot = np.zeros(
                    (len(fake_labels), train_data_loader.classes()), np.int32)
                fake_labels_one_hot[np.arange(0, len(fake_labels), dtype=np.int64),
                    fake_labels] = 1
                fake_labels = fake_labels_one_hot

                real_images, real_labels = train_data_loader.next_batch(config.batch_size -
                                                                  int(config.batch_size * current_frac))
                images, labels = (np.concatenate([fake_images, real_images], axis=0),
                                 np.concatenate([fake_labels, real_labels], axis=0))
            else:
                tflearn.is_training(False, sess)
                images = sess.run(fake_data)
                labels = sess.run(classifier_labels, feed_dict={classifier_inputs: images})

                labels_one_hot = np.zeros(
                    (len(labels), train_data_loader.classes()), np.int32)
                labels_one_hot[np.arange(0, len(labels), dtype=np.int64),
                    labels] = 1
                labels = labels_one_hot

            eval_images, eval_labels = eval_data_loader.next_batch(config.batch_size)
            # eval_labels = eval_labels

            feed_dict = {classifier_inputs: images, classifier_label_inputs: labels.astype(np.int32)}
            tflearn.is_training(True, sess)
            loss, _ = sess.run([classifier_accuracy, classifier_step], feed_dict=feed_dict)
            tflearn.is_training(False, sess)
            eval_loss = sess.run(classifier_accuracy, feed_dict={
                classifier_inputs: eval_images, classifier_label_inputs: eval_labels.astype(np.int32)})
            bar.set_description("train accuracy %.4f, eval accuracy %.4f" % (loss, eval_loss))
            total_step += 1

            if log_fobj is not None:
                log_fobj.write("train accuracy: %.4f; eval accuracy: %.4f\n" %
                               (loss, eval_loss))

        if config.save_dir:
            saver_classifier.save(sess, os.path.join(config.save_dir, "mnist_task"))

    if log_fobj is not None:
        log_fobj.close()

    sess.close()

    # parser.add_argument("--data-dir", dest="data_dir")
    # parser.add_argument("--gen-frac-init", dest="gen_frac_init", type=float, default=0.00)
    # parser.add_argument("--gen-frac-final", dest="gen_frac_final", type=float, default=0.35)
    # parser.add_argument("--gen-frac-step", dest="gen_frac_step", type=int, default=250)
    # parser.add_argument("--gen-dim", dest="gen_dim", type=int, default=64)
    # parser.add_argument("gen_model_path", metavar="GENMODELPATH")
    # parser.add_argument("--num-labeled", dest="num_labeled", type=int, default=1000)
    # parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=128)
    # parser.add_argument("-e", "--num-epoch", dest="num_epoch", type=int, default=25)
    # parser.add_argument("--save-dir", dest="save_dir")