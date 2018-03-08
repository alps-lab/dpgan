#!/usr/bin/env python
"""
Source: https://github.com/tensorflow/models/blob/master/differential_privacy/privacy_accountant/tf/accountant.py
(Apache License, Version 2.0)

Reference: https://github.com/tensorflow/models/blob/master/differential_privacy/privacy_accountant/tf/accountant.py
"""
from __future__ import print_function

import collections
import math
import sys

import numpy as np
import tensorflow as tf
from six.moves import xrange


EpsDelta = collections.namedtuple("EpsDelta", ["spent_eps", "spent_delta"])


def GenerateBinomialTable(m):
    """Generate binomial table.
    Args:
    m: the size of the table.
    Returns:
    A two dimensional array T where T[i][j] = (i choose j),
    for 0<= i, j <=m.
    """

    table = np.zeros((m + 1, m + 1), dtype=np.float64)
    for i in range(m + 1):
        table[i, 0] = 1
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            v = table[i - 1, j] + table[i - 1, j -1]
            assert not math.isnan(v) and not math.isinf(v)
            table[i, j] = v
    return tf.convert_to_tensor(table)


class GaussianMomentsAccountant(object):

    def __init__(self, total_examples, moment_order=32):
        assert total_examples > 0
        self._total_examples = total_examples
        self._moment_orders = (
            moment_order
            if isinstance(moment_order, (list, tuple))
            else [1 + i for i in xrange(moment_order)]
        )
        self._max_moment_order = max(self._moment_orders)
        assert self._max_moment_order < 100, "The moment order is too large."

        self._log_moments = [tf.Variable(np.float64(0.0),
                             trainable=False,
                             name=("log_moments-%d" % moment_order))
                             for moment_order in self._moment_orders]

        self._binomial_table = GenerateBinomialTable(self._max_moment_order)


    def _differential_moments(self, sigma, s, t):
        """Compute 0 to t-th differential moments for Gaussian variable.
            E[(P(x+s)/P(x+s-1)-1)^t]
          = sum_{i=0}^t (t choose i) (-1)^{t-i} E[(P(x+s)/P(x+s-1))^i]
          = sum_{i=0}^t (t choose i) (-1)^{t-i} E[exp(-i*(2*x+2*s-1)/(2*sigma^2))]
          = sum_{i=0}^t (t choose i) (-1)^{t-i} exp(i(i+1-2*s)/(2 sigma^2))
        Args:
          sigma: the noise sigma, in the multiples of the sensitivity.
          s: the shift.
          t: 0 to t-th moment.
        Returns:
          0 to t-th moment as a tensor of shape [t+1].
        """
        assert t <= self._max_moment_order, ("The order of %d is out "
                                             "of the upper bound %d."
                                             % (t, self._max_moment_order))
        binomial = tf.slice(self._binomial_table, [0, 0],
                            [t + 1, t + 1])
        signs = np.zeros((t + 1, t + 1), dtype=np.float64)
        for i in range(t + 1):
            for j in range(t + 1):
                signs[i, j] = 1.0 - 2 * ((i - j) % 2)
        exponents = tf.constant([j * (j + 1.0 - 2.0 * s) / (2.0 * sigma * sigma)
                                 for j in range(t + 1)], dtype=tf.float64)
        # x[i, j] = binomial[i, j] * signs[i, j] = (i choose j) * (-1)^{i-j}
        x = tf.multiply(binomial, signs)
        # y[i, j] = x[i, j] * exp(exponents[j])
        #         = (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))
        # Note: this computation is done by broadcasting pointwise multiplication
        # between [t+1, t+1] tensor and [t+1] tensor.
        y = tf.multiply(x, tf.exp(exponents))
        # z[i] = sum_j y[i, j]
        #      = sum_j (i choose j) * (-1)^{i-j} * exp(j(j-1)/(2 sigma^2))
        z = tf.reduce_sum(y, 1)
        return z

    def _compute_log_moment(self, sigma, q, moment_order):
        """Compute high moment of privacy loss.
        Args:
          sigma: the noise sigma, in the multiples of the sensitivity.
          q: the sampling ratio.
          moment_order: the order of moment.
        Returns:
          log E[exp(moment_order * X)]
        """
        assert moment_order <= self._max_moment_order, ("The order of %d is out "
                                                        "of the upper bound %d."
                                                        % (moment_order,
                                                           self._max_moment_order))
        binomial_table = tf.slice(self._binomial_table, [moment_order, 0],
                                  [1, moment_order + 1])
        # qs = [1 q q^2 ... q^L] = exp([0 1 2 ... L] * log(q))
        qs = tf.exp(tf.constant([i * 1.0 for i in range(moment_order + 1)],
                                dtype=tf.float64) * tf.cast(
                                    tf.log(q), dtype=tf.float64))
        moments0 = self._differential_moments(sigma, 0.0, moment_order)
        term0 = tf.reduce_sum(binomial_table * qs * moments0)
        moments1 = self._differential_moments(sigma, 1.0, moment_order)
        term1 = tf.reduce_sum(binomial_table * qs * moments1)
        return tf.squeeze(tf.log(tf.cast(q * term0 + (1.0 - q) * term1,
                                         tf.float64)))


    def _compute_delta(self, log_moments, eps):
        """Compute delta for given log_moments and eps.
        Args:
          log_moments: the log moments of privacy loss, in the form of pairs
            of (moment_order, log_moment)
          eps: the target epsilon.
        Returns:
          delta
        """
        min_delta = 1.0
        for moment_order, log_moment in log_moments:
            if math.isinf(log_moment) or math.isnan(log_moment):
                sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
                continue
            if log_moment < moment_order * eps:
                min_delta = min(min_delta,
                            math.exp(log_moment - moment_order * eps))
        return min_delta


    def _compute_eps(self, log_moments, delta):
        min_eps = float("inf")
        for moment_order, log_moment in log_moments:
            if math.isinf(log_moment) or math.isnan(log_moment):
                sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
                continue
            min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
        return min_eps


    def accumulate_privacy_spending(self, unused_eps_delta,
                                  sigma, num_examples):
        """Accumulate privacy spending.
        In particular, accounts for privacy spending when we assume there
        are num_examples, and we are releasing the vector
        (sum_{i=1}^{num_examples} x_i) + Normal(0, stddev=l2norm_bound*sigma)
        where l2norm_bound is the maximum l2_norm of each example x_i, and
        the num_examples have been randomly selected out of a pool of
        self.total_examples.
        Args:
          unused_eps_delta: EpsDelta pair which can be tensors. Unused
            in this accountant.
          sigma: the noise sigma, in the multiples of the sensitivity (that is,
            if the l2norm sensitivity is k, then the caller must have added
            Gaussian noise with stddev=k*sigma to the result of the query).
          num_examples: the number of examples involved.
        Returns:
          a TensorFlow operation for updating the privacy spending.
        """
        q = tf.cast(num_examples, tf.float64) * 1.0 / self._total_examples

        moments_accum_ops = []
        for i in range(len(self._log_moments)):
            moment = self._compute_log_moment(sigma, q, self._moment_orders[i])
            moments_accum_ops.append(tf.assign_add(self._log_moments[i], moment))
        return tf.group(*moments_accum_ops)

    def get_privacy_spent(self, sess, target_eps=None, target_deltas=None):
        """Compute privacy spending in (e, d)-DP form for a single or list of eps.
        Args:
          sess: the session to run the tensor.
          target_eps: a list of target epsilon's for which we would like to
            compute corresponding delta value.
          target_deltas: a list of target deltas for which we would like to
            compute the corresponding eps value. Caller must specify
            either target_eps or target_delta.
        Returns:
          A list of EpsDelta pairs.
        """
        assert (target_eps is None) ^ (target_deltas is None)
        eps_deltas = []
        log_moments = sess.run(self._log_moments)
        log_moments_with_order = zip(self._moment_orders, log_moments)
        if target_eps is not None:
            for eps in target_eps:
                eps_deltas.append(
                    EpsDelta(eps, self._compute_delta(log_moments_with_order, eps)))
        else:
            assert target_deltas
            for delta in target_deltas:
                eps_deltas.append(
                    EpsDelta(self._compute_eps(log_moments_with_order, delta), delta))
        return eps_deltas


class DummyAccountant(object):
  """An accountant that does no accounting."""

  def accumulate_privacy_spending(self, *unused_args):
    return tf.no_op()

  def get_privacy_spent(self, unused_sess, **unused_kwargs):
    return [EpsDelta(np.inf, 1.0)]