#!/usr/bin/env python
from six.moves import xrange

from .base import Supervisor


class BasicSupervisor(Supervisor):

    def __init__(self, config, clipper, scheduler, sampler=None,
                 callback_before_train=None):
        super(BasicSupervisor, self).__init__(config)
        self.clipper = clipper
        self.scheduler = scheduler
        self.sampler = sampler
        self._callback_before_train = callback_before_train
        self.num_critic = config.critic_iters

    def callback_before_train(self, sess, total_step, **kwargs):
        if self._callback_before_train is not None:
            self._callback_before_train(self, sess, total_step, **kwargs)

    def callback_before_iter(self, sess, total_step, **kwargs):
        return dict(num_critic=self.scheduler.get_critic_steps(total_step))

    def callback_disc_iter(self, sess, total_step, i,
                           real_input, data_loader,
                           accountant=None,
                           **kwargs):
        if self.sampler is not None:
            feed_dict = self.sampler.update_feed_dict(sess, total_step)
        else:
            feed_dict = {}
        feed_dict.update(self.clipper.update_feed_dict(sess, total_step))
        feed_dict.update(
            {
                real_input: data_loader.next_batch(self.config.num_gpu *
                                                   self.config.batch_size)[0],
            }
        )
        values = sess.run(
                    self._disc_train_tensors,
                    feed_dict=feed_dict)

        if accountant is not None:
            for _ in xrange(self.clipper.num_accountant_terms(total_step)):
                sess.run(self._accountant_op, feed_dict={
                    self._accountant_n: self.config.num_gpu * self.config.batch_size,
                })
        return values[self._disc_train_cost_index]

    def callback_clip_grads(self, weights_grads, **kwargs):
        return self.clipper.clip_grads(weights_grads)

    def callback_noise_grads(self, weights_grads, batch_size, **kwargs):
        return self.clipper.noise_grads(weights_grads, batch_size, self._accountant_sigma)

    def callback_create_disc_train_ops(self, weight_grads, optimizer,
                                       global_step, **kwargs):
        return [optimizer.apply_gradients([(g, w) for w, g in weight_grads.items()],
                                         global_step=None)]
