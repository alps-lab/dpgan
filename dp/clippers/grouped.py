from six.moves import xrange

import tensorflow as tf
import numpy as np

from dp.clippers.base import Clipper


class GroupedClipper(Clipper):

    def __init__(self, groups, no_noise=False):
        super(GroupedClipper, self).__init__()
        self.no_noise = no_noise
        self.group_vars = {}
        self.group_bounds = {}
        self.var_groups = {}

        for i, (variables, bound) in enumerate(groups):
            self.group_vars[i] = list(variables)
            self.group_bounds[i] = bound
            for var in variables:
                self.var_groups[var] = i

    def clip_grads(self, m):
        clipped = []
        groups = {i: [] for i in xrange(len(self.group_vars))}
        for k, v in m:
            assert k.name in self.var_groups
            groups[self.var_groups[k.name]].append((k.name, v))

        ret = {}
        for group, name_grads in groups.items():
            names = [name for name, _ in name_grads]
            grads = [grad for _, grad in name_grads]
            shapes = [grad.shape for grad in grads]
            reshaped = [tf.reshape(grad, [-1]) for grad in grads]
            sizes = [grad.shape[0].value for grad in reshaped]
            cat = tf.concat(reshaped, axis=0)
            clip = tf.clip_by_norm(cat, self.group_bounds[group].get_bound_tensor())
            split = tf.split(clip, sizes)
            for shape, name, grad in zip(shapes, names, split):
                ret[name] = tf.reshape(grad, shape)

        for k, v in m:
            clipped.append(ret[k.name])
        return clipped

    def num_accountant_terms(self, step):
        return len(self.group_vars)

    def noise_grads(self, m, batch_size, sigma):
        noised = {k: 0 for k in m}
        for k, v in m.items():
            assert k.name in self.var_groups
            c_value = self.group_bounds[self.var_groups[k.name]].get_bound_tensor()
            if not self.no_noise:
                noised[k] = v + (tf.random_normal(shape=k.shape, mean=0.0, stddev=c_value * sigma) /
                               np.sqrt(batch_size))
            else:
                noised[k] = v
        return noised

    def info(self):
        f = "Basic clipper\n"
        r = []
        for group, vars in sorted(self.group_vars.items(), key=lambda x: x[0]):
            names = ",".join(vars)
            r.append("(%s, %r)" % (names, self.group_bounds[group]))
        return f + "\n".join(r)

    def update_feed_dict(self, sess, steps):
        d = {}
        for k, b in self.group_bounds.items():
            d.update(b.update_feed_dict(sess, steps))
        return d