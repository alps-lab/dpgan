import tensorflow as tf

from utils.ops import get_variable_hook, get_variable_hook_replace


def discriminator_forward_per_examples(forward_func, flags, incoming,
                      scope="discriminator", name=None, reuse=True):
    examples = tf.split(incoming, flags.batch_size, axis=0)
    lookups = []
    outputs = []
    for example in examples:
        lookup = {}
        output = get_variable_hook(lookup)(forward_func)(flags, example,
                      scope=scope, name=name, reuse=reuse)
        lookups.append(lookup)
        outputs.append(output)

    return outputs, lookups


def discriminator_forward_with_lookups(forward_func, flags, incoming, lookups,
                                       scope="discriminator", name=None, reuse=True):
    examples = tf.split(incoming, flags.batch_size, axis=0)
    outputs = []
    for example, lookup in zip(examples, lookups):
        output = get_variable_hook_replace(lookup)(forward_func)(flags, example,
                                                                  scope=scope, name=name, reuse=reuse)
        outputs.append(output)

    return outputs