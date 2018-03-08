#!/usr/bin/env python
from six.moves import xrange

from .basic import BasicSupervisor


class BasicSupervisorMNIST(BasicSupervisor):

    def __init__(self, config, clipper, scheduler, sampler=None,
                 callback_before_train=None):
        super(BasicSupervisorMNIST, self).__init__(config, clipper,
                                                   scheduler, sampler, callback_before_train)

    @staticmethod
    def get_lr(total_step):
        if total_step > 500:
            lr = 3e-4
        else:
            lr = (3e-4 - 2e-4) / 500 * total_step + 2e-4
        return lr

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
                                                   self.config.batch_size)[0]
            }
        )
        if "lr" in self._keys:
            feed_dict.update({self._keys["lr"]: self.get_lr(total_step)})

        values = sess.run(
                    self._disc_train_tensors,
                    feed_dict=feed_dict)

        if accountant is not None:
            for _ in xrange(self.clipper.num_accountant_terms(total_step)):
                sess.run(self._accountant_op, feed_dict={
                    self._accountant_n: self.config.num_gpu * self.config.batch_size,
                })
        return values[self._disc_train_cost_index]

