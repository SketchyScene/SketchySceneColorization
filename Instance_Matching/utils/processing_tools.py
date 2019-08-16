from __future__ import absolute_import, division, print_function

import numpy as np

def generate_spatial_batch(N, featmap_H, featmap_W):
    spatial_batch_val = np.zeros((N, featmap_H, featmap_W, 8), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w+1) / featmap_W * 2 - 1
            xctr = (xmin+xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h+1) / featmap_H * 2 - 1
            yctr = (ymin+ymax) / 2
            spatial_batch_val[:, h, w, :] = \
                [xmin, ymin, xmax, ymax, xctr, yctr, 1/featmap_W, 1/featmap_H]
    return spatial_batch_val

def generate_bilinear_filter(stride):
    # Bilinear upsampling filter
    f = np.concatenate((np.arange(0, stride), np.arange(stride, 0, -1))) / stride
    return np.outer(f, f).astype(np.float32)[:, :, np.newaxis, np.newaxis]

def compute_accuracy(scores, labels):
    is_pos = (labels != 0)
    is_neg = np.logical_not(is_pos)
    num_pos = np.sum(is_pos)
    num_neg = np.sum(is_neg)
    num_all = num_pos + num_neg

    is_correct = np.logical_xor(scores < 0, is_pos)
    accuracy_all = np.sum(is_correct) / num_all
    accuracy_pos = np.sum(is_correct[is_pos]) / (num_pos + 1)
    accuracy_neg = np.sum(is_correct[is_neg]) / num_neg
    return accuracy_all, accuracy_pos, accuracy_neg

def spatial_feature_from_bbox(bboxes, imsize):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))
    im_w, im_h = imsize
    assert(np.all(bboxes[:, 0] < im_w) and np.all(bboxes[:, 2] < im_w))
    assert(np.all(bboxes[:, 1] < im_h) and np.all(bboxes[:, 3] < im_h))

    feats = np.zeros((bboxes.shape[0], 8))
    feats[:, 0] = bboxes[:, 0] * 2.0 / im_w - 1  # x1
    feats[:, 1] = bboxes[:, 1] * 2.0 / im_h - 1  # y1
    feats[:, 2] = bboxes[:, 2] * 2.0 / im_w - 1  # x2
    feats[:, 3] = bboxes[:, 3] * 2.0 / im_h - 1  # y2
    feats[:, 4] = (feats[:, 0] + feats[:, 2]) / 2  # x0
    feats[:, 5] = (feats[:, 1] + feats[:, 3]) / 2  # y0
    feats[:, 6] = feats[:, 2] - feats[:, 0]  # w
    feats[:, 7] = feats[:, 3] - feats[:, 1]  # h
    return feats
