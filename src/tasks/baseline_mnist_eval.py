#!/usr/bin/env python
from six.moves import xrange

import argparse

import numpy as np
import tflearn
from tqdm import trange
import tensorflow as tf

from models.tasks.mnist import classifier_forward
from utils.data_utils import MNISTLoader


def run_task(config, eval_data_loader,
               classifier_forward,
               optimizer):
    classifier_inputs = tf.placeholder(tf.float32,
                                       shape=[None] + eval_data_loader.shape(), name="input")
    classifier_label_inputs = tf.placeholder(tf.int32, shape=[None, 10], name="labels")
    classifier_logits = classifier_forward(config, classifier_inputs, name="classifier")

    global_step = tf.Variable(0, False)
    classifier_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=classifier_label_inputs,
                                                                         logits=classifier_logits))
    classifier_labels = tf.cast(tf.argmax(tf.nn.softmax(classifier_logits), axis=-1), tf.int64)
    classifier_accuracy = tf.reduce_mean(tf.cast(tf.equal(classifier_labels,
                                        tf.argmax(classifier_label_inputs, axis=-1)),
                                                          tf.float32))

    print("graph built.")

    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, config.model_path)

    tflearn.is_training(False, sess)

    eval_losses = []

    num_steps = eval_data_loader.num_steps(config.batch_size)
    bar = trange(num_steps, leave=False)
    for _ in bar:
        eval_images, eval_labels = eval_data_loader.next_batch(config.batch_size)
        eval_loss = sess.run(classifier_accuracy, feed_dict={
            classifier_inputs: eval_images, classifier_label_inputs: eval_labels.astype(np.int32)})
        eval_losses.append(eval_loss)

    sess.close()
    print("accuracy:", np.mean(eval_losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", dest="data_dir")
    parser.add_argument("model_path", metavar="MODELPATH")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=100)
    parser.add_argument("--dim", dest="dim", default=64, type=int)

    config = parser.parse_args()

    print("config: %r" % config)

    eval_data_loader = MNISTLoader(config.data_dir, include_train=False)

    run_task(config, eval_data_loader,
             classifier_forward, tf.train.AdamOptimizer())

