import tensorflow as tf
import numpy as np


class Sampler(object):

    def __init__(self, update_every=5, keep_memory=True):
        self.update_every = update_every
        self.keep_memory = keep_memory

        self._bounds = None
        self._last_step = -1
        self._bound_tensors = {}

    def get_bound_tensor(self, key):
        if key not in self._bound_tensors:
            self._bound_tensors[key] = tf.placeholder(tf.float32, shape=())
        return self._bound_tensors[key]

    def update_feed_dict(self, sess, total_step):
        if (self.keep_memory and
            total_step % self.update_every == 0 and self._last_step != total_step) \
                or self._bounds is None:
            self._last_step = total_step
            self._update_bounds(sess, total_step)
        else:
            self._update_bounds(sess, total_step)
        d = {}
        for key, tensor in self._bound_tensors.items():
            d[tensor] = self._bounds[key]
        return d

    def _update_bounds(self, sess, total_step):
        self._bounds = {}
        grad_norms = sess.run(self.grad_norms,
                              feed_dict={self.real_input:
                              self.data_loader.next_batch(self.est_batch_size)[0]})

        for key, bounds in grad_norms.items():
            m, v = np.mean(bounds), np.var(bounds)
            self._bounds[key.name] = m

    def set_forward_function(self, forward_function):
        (self.real_input, self.grad_norms, self.est_batch_size,
            self.tot_batch_size) = forward_function()

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader