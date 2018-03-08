import tensorflow as tf
from tflearn.layers import fully_connected, dropout
from tflearn.activations import relu, leaky_relu

from .mnist import classifier_forward


def code_classifier_forward(config, incoming=None, image=None,
                            scope="code_classifier", name=None, reuse=False):
    with tf.variable_scope(scope, name, reuse=reuse):
        code_output = leaky_relu(fully_connected(incoming, 512))

        output = leaky_relu(fully_connected(tf.reshape(image, [config.batch_size, 28 * 28]), 512))
        prod = tf.matmul(code_output[:, :, None], output[:, None, :])

        prob = tf.nn.softmax(prod)
        prob2 = tf.nn.softmax(tf.transpose(prod, perm=[0, 2, 1]))

        output = tf.concat([code_output,
                            tf.matmul(prob, output[:, :, None])[:, :, 0],
                            tf.matmul(prob2, code_output[:, :, None])[:, :, 0]], axis=-1)
        output = relu(fully_connected(output, 1024))
        output = dropout(output, 0.6)

        output = relu(fully_connected(output, 512))
        output = dropout(output, 0.6)

        output = relu(fully_connected(output, 256))
        output = dropout(output, 0.8)

        output = fully_connected(output, 10)

    return output


def image_classifier_forward(config, incoming,
                             scope="image_classifier", name=None, reuse=False):
    return classifier_forward(config, incoming, scope=scope, name=name, reuse=reuse)
