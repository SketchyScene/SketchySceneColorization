import tensorflow as tf
import numpy as np
from utils import tensorflow_util as util

VGG_MODEL_PATH = 'models/imagenet-vgg-verydeep-19.mat'


def vgg_net(weights, image, use_vgg_weight):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]

            with tf.variable_scope(name):
                if not use_vgg_weight:
                    filter_size, in_filters, out_filters = kernels.shape[0], kernels.shape[2], kernels.shape[3]
                    n = filter_size * filter_size * out_filters
                    kernels = tf.get_variable(
                        'DW', [filter_size, filter_size, in_filters, out_filters],
                        tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
                    bias = tf.get_variable('bias', bias.reshape(-1).shape, initializer=tf.constant_initializer())
                else:
                    kernels = util.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name="DW")
                    bias = util.get_variable(bias.reshape(-1), name="bias")
                current = util.conv2d_strided(current, kernels, bias, stride=1)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = util.avg_pool_2x2(current)
        net[name] = current

    return net


class FCN_8s(object):
    """DeepLab model."""

    def __init__(self, batch_size=1,
                 num_classes=47,
                 lrn_rate=0.0001,
                 lr_decay_step=70000,
                 lrn_rate_end=0.00001,
                 weight_decay_rate=0.0001,
                 optimizer='adam',  # 'sgd' or 'mom' or 'adam'
                 images=tf.placeholder(tf.float32),
                 labels=tf.placeholder(tf.int32),
                 keep_prob=1.0,
                 ignore_class_bg=True,
                 use_vgg_weight=True,
                 mode='test',
                 is_intermediate=False):
        """FCN-8s constructor.

    Args:
      : Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, image_size, image_size]
    """
        self.images = images
        self.labels = labels
        self.H = tf.shape(self.images)[1]
        self.W = tf.shape(self.images)[2]
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.lrn_rate = lrn_rate
        self.lr_decay_step = lr_decay_step
        self.lrn_rate_end = lrn_rate_end
        self.weight_decay_rate = weight_decay_rate
        self.optimizer = optimizer
        self.keep_prob = keep_prob
        self.ignore_class_bg = ignore_class_bg
        self.use_vgg_weight = use_vgg_weight
        self.mode = mode
        self.is_intermediate = is_intermediate
        self._extra_train_ops = []

        with tf.variable_scope("FCN_8s"):
            self.build_graph()

    def build_graph(self):
        """Build a whole graph for the model."""
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()

    def _build_model(self):
        vgg_model_data = util.get_model_data(VGG_MODEL_PATH)
        vgg_weights = np.squeeze(vgg_model_data['layers'])

        image_net = vgg_net(vgg_weights, self.images, use_vgg_weight=self.use_vgg_weight)
        conv_final_layer = image_net["conv5_3"]

        pool5 = util.max_pool_2x2(conv_final_layer)

        with tf.variable_scope('fc6'):
            W6 = util.weight_variable([7, 7, 512, 4096], name="DW")
            b6 = util.bias_variable([4096], name="bias")
            conv6 = util.conv2d_strided(pool5, W6, b6)
            relu6 = tf.nn.relu(conv6, name="relu6")
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=self.keep_prob)

        with tf.variable_scope('fc7'):
            W7 = util.weight_variable([1, 1, 4096, 4096], name="DW")
            b7 = util.bias_variable([4096], name="bias")
            conv7 = util.conv2d_strided(relu_dropout6, W7, b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=self.keep_prob)

        with tf.variable_scope('fc8'):
            W8 = util.weight_variable([1, 1, 4096, self.num_classes], name="DW")
            b8 = util.bias_variable([self.num_classes], name="bias")
            conv8 = util.conv2d_strided(relu_dropout7, W8, b8)

        # now to upscale to actual image size
        with tf.variable_scope('deconv1'):
            deconv_shape1 = image_net["pool4"].get_shape()
            W_t1 = util.weight_variable([4, 4, deconv_shape1[3].value, self.num_classes], name="DW")
            b_t1 = util.bias_variable([deconv_shape1[3].value], name="bias")
            conv_t1 = util.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
            fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        with tf.variable_scope('deconv2'):
            deconv_shape2 = image_net["pool3"].get_shape()
            W_t2 = util.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="DW")
            b_t2 = util.bias_variable([deconv_shape2[3].value], name="bias")
            conv_t2 = util.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
            fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        if self.is_intermediate:
            self.intermediate_feat = fuse_2
            return

        with tf.variable_scope('deconv3'):
            shape = tf.shape(self.images)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self.num_classes])
            W_t3 = util.weight_variable([16, 16, self.num_classes, deconv_shape2[3].value], name="DW")
            b_t3 = util.bias_variable([self.num_classes], name="bias")
            logits_up = util.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        # below is similar to Deeplab-v2

        self.logits_up = logits_up  # (N, H, W, num_classes)
        logits_flat = tf.reshape(self.logits_up, [-1, self.num_classes])
        pred = tf.nn.softmax(logits_flat)
        self.pred = tf.reshape(pred, tf.shape(self.logits_up))  # shape = [1, H, W, nClasses]

        pred_label = tf.argmax(self.pred, 3)  # shape = [1, H, W]
        pred_label = tf.expand_dims(pred_label, axis=3)
        self.pred_label = pred_label  # shape = [1, H, W, 1], contains [0, nClasses)

    def _build_train_op(self):
        """Build training specific ops for the graph."""

        logits_flatten = tf.reshape(self.logits_up, [-1, self.num_classes])
        pred_flatten = tf.reshape(self.pred, [-1, self.num_classes])

        labels_gt = self.labels

        if self.ignore_class_bg:
            # ignore background labels: 255
            gt_labels_flatten = tf.reshape(labels_gt, [-1, ])
            indices = tf.squeeze(tf.where(tf.less_equal(gt_labels_flatten, self.num_classes - 1)), 1)
            remain_logits = tf.gather(logits_flatten, indices)
            remain_pred = tf.gather(pred_flatten, indices)
            remain_labels = tf.gather(gt_labels_flatten, indices)
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=remain_logits, labels=remain_labels)
        else:
            xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_up, labels=labels_gt)

        self.cls_loss = tf.reduce_mean(xent, name='xent')  # xent.shape=[nIgnoredBgPixels]
        self.cost = self.cls_loss + self._decay()
        tf.summary.scalar('cost', self.cost)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.polynomial_decay(self.lrn_rate,
                                                       self.global_step,
                                                       self.lr_decay_step,
                                                       end_learning_rate=self.lrn_rate_end,
                                                       power=0.9)
        tf.summary.scalar('learning rate', self.learning_rate)

        tvars = tf.trainable_variables()

        if self.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.optimizer == 'mom':
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        elif self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise NameError("Unknown optimizer type %s!" % self.optimizer)

        grads_and_vars = optimizer.compute_gradients(self.cost, var_list=tvars)
        var_lr_mult = {}
        for var in tvars:
            if var.op.name.find(r'fc_final_sketch46') > 0 and var.op.name.find(r'biases') > 0:
                var_lr_mult[var] = 20.
            elif var.op.name.find(r'fc_final_sketch46') > 0:
                var_lr_mult[var] = 10.
            else:
                var_lr_mult[var] = 1.
        grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v)
                          for g, v in grads_and_vars]

        ## summary grads
        # for grad, grad_var in grads_and_vars:
        #     if grad is not None:
        #         tf.summary.histogram(grad_var.op.name + "/gradient", grad)

        apply_op = optimizer.apply_gradients(grads_and_vars,
                                             global_step=self.global_step, name='train_step')

        train_ops = [apply_op] + self._extra_train_ops
        self.train_step = tf.group(*train_ops)

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.histogram_summary(var.op.name, var)

        return tf.multiply(self.weight_decay_rate, tf.add_n(costs))
