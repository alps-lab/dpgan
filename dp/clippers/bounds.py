from abc import ABCMeta, abstractmethod
import numpy as np


class Bound(object):

    __meta__ = ABCMeta

    def __init__(self):
        self._keys = {}

    @abstractmethod
    def get_bound_tensor(self, global_step=None, **kwargs):
        pass

    @abstractmethod
    def update_feed_dict(self, sess, total_step, **kwargs):
        pass

    def put_key(self, key, value):
        self._keys[key] = value

    def get_key(self, key):
        return self._keys[key]

    def update_keys(self, d):
        self._keys.update(d)


class ConstantBound(Bound):

    def __init__(self, value):
        super(ConstantBound, self).__init__()
        self.value = value

    def get_bound_tensor(self, global_step=None, **kwargs):
        return np.float32(self.value)

    def update_feed_dict(self, sess, total_step, **kwargs):
        return {}


class TensorBound(Bound):

    def __init__(self, tensor, update_callback=None):
        super(TensorBound, self).__init__()
        self.tensor = tensor
        self.update_callback = update_callback

    def get_bound_tensor(self, global_step=None, **kwargs):
        return self.tensor

    def update_feed_dict(self, sess, total_step, **kwargs):
        if self.update_callback is not None:
            return self.update_callback(self, sess, total_step, **kwargs)
        return {}

    def __repr__(self):
        return "TensorBound with tensor: %r" % self.tensor
