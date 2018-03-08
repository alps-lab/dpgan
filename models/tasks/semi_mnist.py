import tensorflow as tf
from tflearn.layers import fully_connected, dropout
from tflearn.activations import relu

from .mnist import classifier_forward


def code_classifier_forward(config, incoming=None, image=None,
                            scope="code_classifier", name=None, reuse=False):
    with tf.variable_scope(scope, name, reuse=reuse):
        output = relu(fully_connected(incoming, 512))
        output1 = dropout(output, 0.8)

        print(config.batch_size, image.shape)
        output = relu(fully_connected(tf.reshape(image, [config.batch_size, 28 * 28]), 512))
        output2 = dropout(output, 0.8)

        output = tf.concat([output1, output2], axis=-1)

        output = relu(fully_connected(output, 1024))
        output = dropout(output, 0.5)

        output = relu(fully_connected(output, 512))
        output = dropout(output, 0.8)

        output = fully_connected(output, 10)

    return output


def image_classifier_forward(config, incoming,
                             scope="image_classifier", name=None, reuse=False):
    return classifier_forward(config, incoming, scope=scope, name=name, reuse=reuse)
