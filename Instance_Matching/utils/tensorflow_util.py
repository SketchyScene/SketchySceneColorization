import tensorflow as tf
import os
import scipy.io


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def conv2d_strided(x, W, b, stride=1):
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride=2):
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def add_gradient_summary(grads):
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)


def get_model_data(model_path):
    if not os.path.exists(model_path):
        raise IOError("Model %s not found!" % model_path)
    data = scipy.io.loadmat(model_path)
    return data


def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")