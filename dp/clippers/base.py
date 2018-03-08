from abc import ABCMeta, abstractmethod


class Clipper(object):

    __meta__ = ABCMeta

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def num_accountant_terms(self, steps):
        pass

    @abstractmethod
    def clip_grads(self, m):
        pass

    @abstractmethod
    def noise_grads(self, m, batch_size, sigma):
        pass

    def update_feed_dict(self, sess, steps):
        return {}

    def info(self):
        return repr(self)