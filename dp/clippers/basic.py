import tensorflow as tf
import numpy as np

from dp.clippers.base import Clipper


class BasicClipper(Clipper):

    def __init__(self, bound, specials=None):
        super(BasicClipper, self).__init__()
        self.bound = bound
        self.specials = set(specials) if specials is not None else {}
        self.keys = set()
        self._bounds = {}

    def clip_grads(self, m):
        clipped = []
        for k, v in m:
            self.keys.add(k)
            if k in self.specials:
                self._bounds[k] = self.specials[k]
                clipped.append(tf.clip_by_norm(v, self.specials[k].get_bound_tensor()))
            else:
                self._bounds[None] = self.bound
                clipped.append(tf.clip_by_norm(v, self._bounds[None].get_bound_tensor()))
        return clipped

    def num_accountant_terms(self, step):
        return len(self.keys)

    def noise_grads(self, m, batch_size, sigma):
        noised = {k: 0 for k in m}
        for k, v in m.items():
            assert k in self.keys
            if k in self.specials:
                c_value = self.specials[k].get_bound_tensor()
            else:
                c_value = self.bound.get_bound_tensor()
            noised[k] = v + (tf.random_normal(shape=k.shape, mean=0.0, stddev=c_value * sigma) /
                           np.sqrt(batch_size))
        return noised

    def info(self):
        return "Basic clipper with bound: %r" % self.bound

    def update_feed_dict(self, sess, steps):
        d = {}
        for k, b in self._bounds.items():
            d.update(b.update_feed_dict(sess, steps))
        return d

