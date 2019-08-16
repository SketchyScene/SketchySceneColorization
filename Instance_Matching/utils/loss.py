from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np


def weighed_logistic_loss(scores, labels, pos_loss_mult=1.0, neg_loss_mult=1.0):
    # Apply different weights to loss of positive samples and negative samples
    # positive samples have label 1 while negative samples have label 0
    loss_mult = tf.add(tf.multiply(labels, tf.subtract(pos_loss_mult, neg_loss_mult)), neg_loss_mult)

    # Classification loss as the average of weighed per-score loss
    cls_loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=labels),
        loss_mult)))

    return cls_loss


def logistic_loss_cond(scores, labels):
    # Classification loss as the average of weighed per-score loss
    cond = tf.select(tf.equal(labels, tf.zeros(tf.shape(labels))),
                     tf.zeros(tf.shape(labels)),
                     tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=labels)
                     )
    cls_loss = tf.reduce_mean(tf.reduce_sum(cond, [1, 2, 3]))

    return cls_loss


def l2_regularization_loss(variables, weight_decay):
    l2_losses = [tf.nn.l2_loss(var) for var in variables]
    total_l2_loss = weight_decay * tf.add_n(l2_losses)

    return total_l2_loss


def dsc_loss(scores, labels):
    scores = tf.sigmoid(scores)
    inter = tf.scalar_mul(2., tf.reduce_sum(tf.multiply(scores, labels), [1, 2, 3]))
    union = tf.add(tf.reduce_sum(scores, [1, 2, 3]), tf.reduce_sum(labels, [1, 2, 3]))
    dsc_loss = tf.reduce_mean(tf.sub(1., tf.div(inter, union)))

    return dsc_loss


def iou_loss(scores, labels):
    scores = tf.sigmoid(scores)
    inter = tf.reduce_sum(tf.multiply(scores, labels), [1, 2, 3])
    union = tf.add(tf.reduce_sum(scores, [1, 2, 3]), tf.reduce_sum(labels, [1, 2, 3]))
    union = tf.sub(union, inter)
    iou_loss = tf.reduce_mean(tf.sub(1., tf.div(inter, union)))

    return iou_loss


def smooth_l1_loss(scores, labels, ld=1.0):
    box_diff = scores - labels
    abs_box_diff = tf.abs(box_diff)
    smooth_l1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_box_diff, 1.)))
    loss_box_raw = tf.pow(box_diff, 2) * 0.5 * smooth_l1_sign \
                   + (abs_box_diff - 0.5) * (1.0 - smooth_l1_sign)
    loss_box = ld * tf.reduce_mean(tf.reduce_sum(loss_box_raw, [1]))

    return loss_box
