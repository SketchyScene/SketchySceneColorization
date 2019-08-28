import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
from time import gmtime, strftime
from PIL import Image
import sys


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv


def conv_ex(batch_input, out_channels, stride, filter_size=4):
    with tf.variable_scope("conv_ex"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [filter_size, filter_size, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding="SAME")
        return conv


def nchw_conv_ex(batch_input_, out_channels, stride, filter_size=4):
    batch_input = tf.transpose(batch_input_, [0, 2, 3, 1])
    output = conv_ex(batch_input, out_channels, stride, filter_size)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels],
                                      [1, 2, 2, 1], padding="SAME")
        return conv


def bottleneck_residual_en(x_, out_filter, stride):
    """ encoder bottleneck_residual """
    x = tf.transpose(x_, [0, 2, 3, 1])  # [NCHW] => [NHWC]
    orig_x = x

    with tf.variable_scope('block_1'):
        x = conv(x, int(round(out_filter / 4)), stride=stride)
        x = batchnorm(x)
        x = lrelu(x, 0.2)

    with tf.variable_scope('block_2'):
        x = conv_ex(x, int(round(out_filter / 4)), stride=1, filter_size=3)
        x = batchnorm(x)
        x = lrelu(x, 0.2)

    with tf.variable_scope('block_3'):
        x = conv_ex(x, out_filter, stride=1, filter_size=1)
        x = batchnorm(x)

    with tf.variable_scope('block_add'):
        if stride != 1:
            orig_x = conv(orig_x, out_filter, stride=stride)
            orig_x = batchnorm(orig_x)
        x += orig_x
        x = lrelu(x, 0.2)

    # tf.logging.info('image after unit %s', x.get_shape())
    x = tf.transpose(x, [0, 3, 1, 2])
    return x


def bottleneck_residual_de(x_, out_filter, need_relu=True):
    """ decoder bottleneck_residual """
    x = tf.transpose(x_, [0, 2, 3, 1])
    orig_x = x

    with tf.variable_scope('block_1'):
        x = deconv(x, int(round(out_filter / 4)))
        x = batchnorm(x)
        x = tf.nn.relu(x)

    with tf.variable_scope('block_2'):
        x = conv_ex(x, int(round(out_filter / 4)), stride=1, filter_size=3)
        x = batchnorm(x)
        x = tf.nn.relu(x)

    with tf.variable_scope('block_3'):
        x = conv_ex(x, out_filter, stride=1, filter_size=1)
        x = batchnorm(x)

    with tf.variable_scope('block_add'):
        orig_x = deconv(orig_x, out_filter)
        orig_x = batchnorm(orig_x)

        x += orig_x

        if need_relu:
            x = tf.nn.relu(x)

    # tf.logging.info('image after unit %s', x.get_shape())
    x = tf.transpose(x, [0, 3, 1, 2])
    return x


def bottleneck_residual_pu(x_, out_filter, is_encoder):
    """ public bottleneck_residual """
    x = tf.transpose(x_, [0, 2, 3, 1])
    orig_x = x

    with tf.variable_scope('block_1'):
        x = conv_ex(x, int(round(out_filter / 4)), stride=1)
        x = batchnorm(x)
        x = lrelu(x, 0.2) if is_encoder else tf.nn.relu(x)

    with tf.variable_scope('block_2'):
        x = conv_ex(x, int(round(out_filter / 4)), stride=1, filter_size=3)
        x = batchnorm(x)
        x = lrelu(x, 0.2) if is_encoder else tf.nn.relu(x)

    with tf.variable_scope('block_3'):
        x = conv_ex(x, out_filter, stride=1, filter_size=1)
        x = batchnorm(x)

    with tf.variable_scope('block_add'):
        x += orig_x
        x = lrelu(x, 0.2) if is_encoder else tf.nn.relu(x)

    x = tf.transpose(x, [0, 3, 1, 2])
    return x
