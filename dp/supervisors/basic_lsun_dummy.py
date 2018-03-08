#!/usr/bin/env python
from six.moves import xrange

from .basic import BasicSupervisor


class BasicSupervisorLSUN(BasicSupervisor):

    def __init__(self, config, clipper, scheduler,
                 sampler=None, callback_before_train=None):
        super(BasicSupervisorLSUN, self).__init__(config, clipper,
                                                   scheduler,
                                                  sampler, callback_before_train)

    @staticmethod
    def get_lr(total_step):
        # if total_step > 5000:
        #     lr = 1e-3
        # elif total_step > 1500:
        #     lr = 5e-4 + (1e-3 - 5e-4) / 3500 * (total_step - 1500)
        if total_step > 500:
            lr = 3e-4
        else:
            lr = (3e-4 - 2e-4) / 500 * total_step + 2e-4
        return lr

    def callback_disc_iter(self, sess, total_step, i,
                           real_input, data_loader,
                           accountant=None,
                           **kwargs):
        if accountant is not None:
            for _ in xrange(self.clipper.num_accountant_terms(total_step)):
                sess.run(self._accountant_op, feed_dict={
                    self._accountant_n: self.config.num_gpu * self.config.batch_size,
                })
        return 0.0

