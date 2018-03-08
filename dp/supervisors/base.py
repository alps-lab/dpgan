#!/usr/bin/env python
from abc import ABCMeta, abstractmethod


class Supervisor(object):

    __meta__ = ABCMeta

    def __init__(self, config):
        self.config = config

        self._disc_train_cost_index = None
        self._disc_train_tensors = None

        self._accountant_op = None
        self._accountant_sigma = config.sigma
        self._accountant_n = None

        self._keys = {}

    @abstractmethod
    def callback_before_train(self, sess, total_step, **kwargs):
        pass

    @abstractmethod
    def callback_before_iter(self, sess, total_step, **kwargs):
        # {num_critic: ..., }
        pass

    @abstractmethod
    def callback_disc_iter(self, sess, total_step, i,
                           real_input, data_loader, accountant=None,
                           **kwargs):
        pass

    @abstractmethod
    def callback_clip_grads(self, weights_grads, **kwargs):
        # clipped gradients
        pass

    @abstractmethod
    def callback_create_disc_train_ops(self, weight_grads, optimizer,
                                       global_step, **kwargs):
        # train tensors and cost tensor index
        pass

    @abstractmethod
    def callback_noise_grads(self, weights_grads, batch_size, **kwargs):
        pass

    def register_disc_train_ops(self, tensors, cost_index=0,
                                **kwargs):
        self._disc_train_tensors = tensors
        self._disc_train_cost_index = cost_index

    def register_accountant_ops(self,
                                accountant_op,
                                n):
        self._accountant_op = accountant_op
        self._accountant_n = n

    def put_key(self, key, value):
        self._keys[key] = value

    def get_key(self, key):
        return self._keys[key]

    def update_keys(self, d):
        self._keys.update(d)