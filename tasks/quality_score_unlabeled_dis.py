#!/usr/bin/env python
from __future__ import division
from six.moves import xrange

import numpy as np
import tflearn
from scipy.stats import entropy


def next_batch(images_iter, batch_size=100):
    count = 0
    images = []
    for img in images_iter:
        images.append(img[None])
        count += 1

        if count == batch_size:
            yield np.concatenate(images, axis=0)
            images.clear()
            count = 0

    if count > 0:
        yield np.concatenate(images, axis=0)


def get_quality_score(sess,
                      incoming, probs,
                      image_iter, batch_size=100, split=10):
    preds, scores = [], []

    tflearn.is_training(False, sess)

    for images in next_batch(image_iter, batch_size):
        pred = sess.run(probs, feed_dict={incoming: images})
        preds.append(pred)

    sess.close()
    preds = np.concatenate(preds, 0)

    for i in xrange(split):
        part = preds[i * len(preds) // split: (i + 1) * len(preds) // split]
        p = np.concatenate([part, 1.0 - part], axis=1)
        q = np.full(p.shape, 0.5, np.float64)
        s = 0.5 * (p + q)
        ent = 0.5 * (entropy(np.transpose(p), np.transpose(s), base=2) +
                     entropy(np.transpose(q), np.transpose(s), base=2))
        kl = np.mean(ent)
        scores.append(kl)

    return np.mean(scores), np.std(scores)
