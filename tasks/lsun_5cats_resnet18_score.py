#!/usr/bin/env python
from __future__ import division
from six.moves import xrange

import tensorflow as tf
import numpy as np
import tflearn

from models.tasks.lsun_5cat import classifier_forward


def next_batch(images_iter, batch_size=100):
    count = 0
    elements = []
    for img in images_iter:
        elements.append(img[None])
        count += 1
        if count == batch_size:
            yield np.concatenate(elements, axis=0)
            elements.clear()
            count = 0

    if count > 0:
        yield np.concatenate(elements, axis=0)


def get_resnet18_score(images_iter, model_path, batch_size=100, split=10):
    tf.reset_default_graph()

    incoming = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name="input")
    logits = classifier_forward(None, incoming, name="classifier")
    probs = tf.nn.softmax(logits)
    saver = tf.train.Saver([var for var in tf.global_variables()
                            if var.name.startswith("classifier")
                            and not var.name.endswith("is_training:0")])

    preds, scores = [], []

    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    tflearn.is_training(False, sess)
    saver.restore(sess, model_path)

    for images in next_batch(images_iter, batch_size):
        pred = sess.run(probs, feed_dict={incoming: images})
        preds.append(pred)

        # print(images)

    sess.close()

    preds = np.concatenate(preds, 0)

    for i in xrange(split):
        part = preds[i * len(preds) // split: (i + 1) * len(preds) // split]
        kl = part * (np.log(np.maximum(part, 1e-12)) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


