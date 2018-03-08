from six.moves import xrange

import numpy as np
import tensorflow as tf
import tflearn
from tflearn import utils, initializations, losses, activations
import tflearn.variables as vs


def fully_connected(incoming, n_units, activation='linear', bias=True,
                    weights_init='truncated_normal', bias_init='zeros',
                    regularizer=None, weight_decay=0.001, trainable=True,
                    restore=True, reuse=False, scope=None,
                    name="FullyConnected"):
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) > 1, "Incoming Tensor shape must be at least 2-D"
    n_inputs = int(np.prod(input_shape[1:]))

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer is not None:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=[n_inputs, n_units], regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = vs.variable('b', shape=[n_units], initializer=bias_init,
                            trainable=trainable, restore=restore)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        inference = incoming
        # If input is not 2d, flatten it.
        if len(input_shape) > 2:
            inference = tf.reshape(inference, [-1, n_inputs])

        inference = tf.matmul(inference, W)
        if b is not None: inference = tf.nn.bias_add(inference, b)
        if activation:
            if isinstance(activation, str):
                inference = activations.get(activation)(inference)
            elif hasattr(activation, '__call__'):
                inference = activation(inference)
            else:
                raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def conv_2d(incoming, nb_filter, filter_size, strides=1, padding='same',
            activation='linear', bias=True, weights_init='uniform_scaling',
            bias_init='zeros', regularizer=None, weight_decay=0.001,
            trainable=True, restore=True, reuse=False, scope=None,
            name="Conv2D"):
    input_shape = utils.get_incoming_shape(incoming)
    assert len(input_shape) == 4, "Incoming Tensor shape must be 4-D"
    filter_size = utils.autoformat_filter_conv2d(filter_size,
                                                 input_shape[-1],
                                                 nb_filter)
    strides = utils.autoformat_kernel_2d(strides)
    padding = utils.autoformat_padding(padding)

    with tf.variable_scope(scope, default_name=name, values=[incoming],
                           reuse=reuse) as scope:
        name = scope.name

        W_init = weights_init
        if isinstance(weights_init, str):
            W_init = initializations.get(weights_init)()
        W_regul = None
        if regularizer is not None:
            W_regul = lambda x: losses.get(regularizer)(x, weight_decay)
        W = vs.variable('W', shape=filter_size, regularizer=W_regul,
                        initializer=W_init, trainable=trainable,
                        restore=restore)

        # Track per layer variables
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, W)

        b = None
        if bias:
            if isinstance(bias_init, str):
                bias_init = initializations.get(bias_init)()
            b = vs.variable('b', shape=nb_filter, initializer=bias_init,
                            trainable=trainable, restore=restore)
            # Track per layer variables
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, b)

        inference = tf.nn.conv2d(incoming, W, strides, padding)
        if b is not None: inference = tf.nn.bias_add(inference, b)

        if activation:
            if isinstance(activation, str):
                inference = activations.get(activation)(inference)
            elif hasattr(activation, '__call__'):
                inference = activation(inference)
            else:
                raise ValueError("Invalid Activation.")

        # Track activations.
        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, inference)

    # Add attributes to Tensor to easy access weights.
    inference.scope = scope
    inference.W = W
    inference.b = b

    # Track output tensor.
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, inference)

    return inference


def layer_norm(inputs, reuse=False, scope=None, name="LayerNorm", ):
    with tf.variable_scope(scope, default_name=name, reuse=reuse):
        input_shape = inputs.get_shape().as_list()
        norm_axes = [i for i in xrange(1, len(input_shape))]
        mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)

        n_neurons = inputs.get_shape().as_list()[norm_axes[-1]]

        offset = tf.get_variable(name="offset", shape=[n_neurons], initializer=
                                tf.zeros_initializer())
        scale = tf.get_variable(name="scale", shape=[n_neurons], initializer=
                                tf.constant_initializer(1))

        # Add broadcasting dims to offset and scale (e.g. BCHW conv data)
        offset = tf.reshape(offset, [1 for i in xrange(len(norm_axes)-1)] + [-1])
        scale = tf.reshape(scale, [1 for i in xrange(len(norm_axes)-1)] + [-1])

        result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

    return result


def mean_pool_conv(incoming, nb_filter, filter_size,
                   scope=None, name="mean_pool_conv", reuse=False, bias=True):
    with tf.variable_scope(scope, name, reuse=reuse):
        output = incoming
        output = tf.add_n(
            [output[:, ::2, ::2, :], output[:, 1::2, ::2, :],
             output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]]) / 4.
        output = conv_2d(output, nb_filter, filter_size, name="conv", padding="same", bias=bias)
    return output


def conv_mean_pool(incoming, nb_filter, filter_size,
                   scope=None, name="conv_mean_pool", reuse=False, bias=True):
    with tf.variable_scope(scope, name, reuse=reuse):
        output = incoming
        output = conv_2d(output, nb_filter, filter_size, bias=bias)
        output = tf.add_n([output[:, ::2, ::2, :], output[:, 1::2, ::2, :],
                           output[:, ::2, 1::2, :], output[:, 1::2, 1::2, :]]) / 4.
    return output


def upsample_conv(incoming, nb_filter, filter_size,
                   scope=None, name="upsample_conv", reuse=False, bias=True):
    with tf.variable_scope(scope, name, reuse=reuse):
        output = incoming
        output = tf.concat([output, output, output, output], axis=-1)
        output = tf.depth_to_space(output, 2)
        output = conv_2d(output, nb_filter, filter_size, padding="same", bias=bias)
        return output


def residual_block_upsample(incoming, out_channels, filter_size, reuse=False,
                            scope=None, name="ResidualBlockUp"):
    input_dim = incoming.shape[-1]
    with tf.variable_scope(scope, default_name=name, reuse=reuse):
        shortcut = upsample_conv(incoming, out_channels, 1, name="shortcut")

        output = incoming
        output = tflearn.layers.batch_normalization(output)
        output = tf.nn.relu(output)
        output = upsample_conv(output, out_channels, filter_size, name="conv1", bias=False)
        output = tflearn.layers.batch_normalization(output)
        output = tf.nn.relu(output)
        output = conv_2d(output, out_channels, filter_size, name="conv2")

    return output + shortcut


def residual_block_downsample(incoming, out_channels, filter_size, reuse=False,
                            scope=None, name="ResidualBlockDown"):
    input_dim = incoming.shape[-1]
    with tf.variable_scope(scope, default_name=name, reuse=reuse):
        shortcut = mean_pool_conv(incoming, out_channels, 1, name="shortcut")

        output = incoming
        output = layer_norm(output)
        output = tf.nn.relu(output)
        output = conv_2d(output, input_dim, filter_size, name="conv1", bias=False)
        output = layer_norm(output)
        output = tf.nn.relu(output)
        output = conv_mean_pool(output, out_channels, filter_size, name="conv2")

    return output + shortcut


def residual_block(incoming, out_channels, filter_size, reuse=False,
                            scope=None, name="ResidualBlock"):
    input_dim = incoming.shape[-1]
    with tf.variable_scope(scope, default_name=name, reuse=reuse):
        if input_dim == out_channels:
            shortcut = incoming
        else:
            shortcut = conv_2d(incoming, out_channels, 1, name="shortcut")

        output = incoming
        output = layer_norm(output)
        output = tf.nn.relu(output)
        output = conv_2d(output, input_dim, filter_size, name="conv1")
        output = layer_norm(output)
        output = tf.nn.relu(output)
        output = conv_2d(output, out_channels, filter_size, name="conv2")

    return output + shortcut


def optimized_residual_block(incoming, out_channels, filter_size,
                             name="OptimizedResidualBlock", scope=None, reuse=False):
    with tf.variable_scope(scope, name, reuse=reuse):
        shortcut = mean_pool_conv(incoming, nb_filter=out_channels, filter_size=1, bias=True,
                                  name="shortcut")

        output = incoming
        output = conv_2d(output, out_channels, filter_size, name="conv1")
        output = tf.nn.relu(output)
        output = conv_mean_pool(output, out_channels, filter_size, name="conv2")

    return shortcut + output


def get_variable_hook(to_add):
    """
    To replace the variable with identity of variable for per example gradients.
    :param to_add:
    :return:
    """
    def overall(f):
        origin = tf.VariableScope.get_variable

        def get_variable(scope, name, *args, **kwargs):
            ret = origin(scope, name, *args, **kwargs)
            ret_id = tf.identity(ret)
            to_add[ret] = ret_id
            return ret_id

        def func(*args, **kwargs):
            origin = tf.VariableScope.get_variable
            tf.VariableScope.get_variable = get_variable
            try:
                return f(*args, **kwargs)
            finally:
                tf.VariableScope.get_variable = origin
        return func

    return overall


def get_variable_hook_replace(lookup):
    """
    To replace the variable with specific tensor via lookup table
    :param lookup:
    :return:
    """
    def overall(f):
        origin = tf.VariableScope.get_variable

        def get_variable(scope, name, *args, **kwargs):
            ret = origin(scope, name, *args, **kwargs)
            ret_id = lookup[ret]
            return ret_id

        def func(*args, **kwargs):
            origin = tf.VariableScope.get_variable
            tf.VariableScope.get_variable = get_variable
            try:
                return f(*args, **kwargs)
            finally:
                tf.VariableScope.get_variable = origin
        return func

    return overall


def group_varible_hook(l):

    def overall(f):
        origin = tf.VariableScope.get_variable

        def get_varible(scope, name, *args, **kwargs):
            ret = origin