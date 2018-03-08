#!/usr/bin/env python
from six.moves import xrange

import argparse
import os

import numpy as np
import tflearn
from tqdm import trange
import tensorflow as tf

from models.tasks.lsun_5cat import classifier_forward
from utils.data_mp import LSUNCatLoader, get_lsun_patterns, lsun_process_actions


def run_task(config, train_data_loader,
               eval_data_loader,
               classifier_forward,
               optimizer):
    classifier_inputs = tf.placeholder(tf.float32, shape=[None] + train_data_loader.shape(), name="input")
    classifier_label_inputs = tf.placeholder(tf.int32, shape=[None, 5], name="labels")
    classifier_logits = classifier_forward(config, classifier_inputs, name="classifier")
    classifier_variables = [var for var in tf.all_variables() if var.name.startswith("classifier")]

    global_step = tf.Variable(0, False)

    classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=classifier_label_inputs,
                                                                         logits=classifier_logits))
    classifier_labels = tf.cast(tf.argmax(tf.nn.softmax(classifier_logits), axis=-1), tf.int64)
    classifier_step = optimizer.minimize(classifier_loss, global_step=global_step,
                    var_list=[var for var in classifier_variables if var in tf.trainable_variables()])
    classifier_accuracy = tf.reduce_mean(tf.cast(tf.equal(classifier_labels,
                                        tf.argmax(classifier_label_inputs, axis=-1)),
                                                          tf.float32))

    print("graph built.")

    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    total_step = 0
    for epoch in xrange(config.num_epoch):
        num_steps = int(config.num_example / config.batch_size)
        bar = trange(num_steps, leave=False)
        for _ in bar:
            images, labels = train_data_loader.next_batch(config.batch_size)

            eval_images, eval_labels = eval_data_loader.next_batch(config.batch_size)

            feed_dict = {classifier_inputs: images, classifier_label_inputs: labels.astype(np.int32)}
            tflearn.is_training(True, sess)
            loss, _ = sess.run([classifier_accuracy, classifier_step], feed_dict=feed_dict)
            tflearn.is_training(False, sess)
            eval_loss = sess.run(classifier_accuracy, feed_dict={
                classifier_inputs: eval_images, classifier_label_inputs: eval_labels.astype(np.int32)})
            bar.set_description("train accuracy %.4f, eval accuracy %.4f" % (loss, eval_loss))
            total_step += 1

        if config.save_dir is not None:
            saver.save(sess, os.path.join(config.save_dir, "model"),
                       global_step=global_step)

    sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", metavar="PUBDATADIR")
    parser.add_argument("val_data_dir", metavar="VALDATADIR")
    parser.add_argument("--save-dir", dest="save_dir")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("-e", "--num-epoch", dest="num_epoch", type=int, default=50)
    parser.add_argument("--dim", dest="dim", default=64, type=int)
    parser.add_argument("-n", "--num-example", dest="num_example", type=int,
                        default=2500000)

    config = parser.parse_args()

    print("config: %r" % config)
    os.makedirs(config.save_dir, exist_ok=True)

    train_data_loader = LSUNCatLoader(get_lsun_patterns(config.train_data_dir),
                                      actions=lsun_process_actions())
    eval_data_loader = LSUNCatLoader(get_lsun_patterns(config.val_data_dir),
                                     num_workers=2, actions=lsun_process_actions())

    try:
        train_data_loader.start_fetch()
        eval_data_loader.start_fetch()
        run_task(config, train_data_loader, eval_data_loader,
                 classifier_forward, tf.train.AdamOptimizer())
    finally:
        train_data_loader.stop_fetch()
        eval_data_loader.stop_fetch()

