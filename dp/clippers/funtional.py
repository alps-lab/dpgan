import tensorflow as tf
import numpy as np

from .basic import Clipper


class FunctionalClipper(Clipper):

    def __init__(self, callback):
        super(FunctionalClipper, self).__init__()
        self.callback = callback
        self.keys = set()

    def clip_grads(self, m):
        clipped = []
        for k, v in m:
            self.keys.add(k)
            clipped.append(tf.clip_by_norm(v, self.callback(k)))
        return clipped

    def num_accountant_terms(self, steps):
        return len(self.keys)

    def noise_grads(self, m, batch_size, sigma):
        noised = {k: 0 for k in m}
        for k, v in m.items():
            c_value = self.callback(k)
            noised[k] = v + (tf.random_normal(shape=k.shape, mean=0.0, stddev=c_value * sigma) /
                           np.sqrt(batch_size))
        return noised

    def info(self):
        f = "Functional clipper\n"
        r = []
        for key in self.keys:
            value = self.callback(key)
            r.append("(%s, %.4f)" % (key, value))
        return f + "\n".join(r)


