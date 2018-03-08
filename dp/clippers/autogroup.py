#!/usr/bin/env python
from itertools import combinations

import numpy as np
import tensorflow as tf

from .base import Clipper


class AutoGroupClipper(Clipper):

    def __init__(self, num_groups, get_bounds_callback, no_noise=False):
        super(Clipper, self).__init__()
        self.num_groups = num_groups
        self.get_bounds_callback = get_bounds_callback

        self._keys = set()
        self._clip_tensors = {}

        self.no_noise = no_noise

    def update_feed_dict(self, sess, steps):
        current_var_bounds = self.get_bounds_callback(sess, steps)
        current_var_group, current_group_bounds = self._get_groups(current_var_bounds)
        feed_dict = {}
        for k, v in self._clip_tensors.items():
            feed_dict[v] = current_group_bounds[current_var_group[k]]

        return feed_dict

    def _get_groups(self, current_var_bounds):
        groups = [(key,) for key in self._keys]
        bounds = {(key,): current_var_bounds[key] for key in self._keys}

        while len(groups) > self.num_groups:
            values = []
            for g1, g2 in combinations(groups, r=2):
                u, v = bounds[g1], bounds[g2]
                values.append((np.sqrt(1 + max(u, v) / min(u, v)), (g1, g2)))
            min_value = min(values, key=lambda x: x[0])
            _, (g1, g2) = min_value
            groups = [g for g in groups if g != g1 and g != g2] + [g1 + g2]
            bounds[g1 + g2] = np.sqrt(np.square(bounds[g1]) + np.square(bounds[g2]))
            del bounds[g1]
            del bounds[g2]

        var_groups = {}
        for g in groups:
            for k in g:
                var_groups[k] = g
        group_bounds = bounds
        return var_groups, group_bounds

    def clip_grads(self, m):
        clipped = []
        for k, v in m:
            if k.name not in self._keys:
                self._clip_tensors[k.name] = tf.placeholder(tf.float32, shape=())
                self._keys.add(k.name)
            clipped.append(tf.clip_by_norm(v, self._clip_tensors[k.name]))
        return clipped

    def noise_grads(self, m, batch_size, sigma):
        noised = {k: 0 for k in m}
        for k, v in m.items():
            if not self.no_noise:
                noised[k] = v + (tf.random_normal(shape=k.shape, mean=0.0,
                                                  stddev=self._clip_tensors[k.name]
                                                         * sigma) / np.sqrt(batch_size))
            else:
                noised[k] = v
        return noised

    def num_accountant_terms(self, steps):
        return self.num_groups
