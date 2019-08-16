import tensorflow as tf
import numpy as np


class SegNet(object):
    """DeepLab model."""

    def __init__(self, batch_size=1,
                 num_classes=47,
                 lrn_rate=0.0001,
                 lr_decay_step=70000,
                 lrn_rate_end=0.00001,
                 weight_decay_rate=0.0001,
                 optimizer='adam',  # 'sgd' or 'mom' or 'adam'
                 images=tf.placeholder(tf.float32, [None, 750, 750, 3]),
                 labels=tf.placeholder(tf.int32),
                 ignore_class_bg=True,
                 mode='test',
                 is_intermediate=False):
        """SegNet constructor.

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
        self.ignore_class_bg = ignore_class_bg
        self.mode = mode
        self.is_intermediate = is_intermediate
        self._extra_train_ops = []

        with tf.variable_scope("SegNet"):
            self.build_graph()

    def build_graph(self):
        """Build a whole graph for the model."""
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()

    def _build_model(self):
        x = self.images

        # encoders
        with tf.variable_scope('enc_1'):
            x = self.conv_bn_relu('conv1', x, 3, 64)
            x = self.conv_bn_relu('conv2', x, 3, 64)
            x, ind_1 = tf.nn.max_pool_with_argmax(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")  # (N, 384, 384, 64)

        with tf.variable_scope('enc_2'):
            x = self.conv_bn_relu('conv1', x, 3, 128)
            x = self.conv_bn_relu('conv2', x, 3, 128)
            x, ind_2 = tf.nn.max_pool_with_argmax(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")  # (N, 192, 192, 128)

        with tf.variable_scope('enc_3'):
            x = self.conv_bn_relu('conv1', x, 3, 256)
            x = self.conv_bn_relu('conv2', x, 3, 256)
            x = self.conv_bn_relu('conv3', x, 3, 256)
            x, ind_3 = tf.nn.max_pool_with_argmax(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")  # (N, 96, 96, 256)

        with tf.variable_scope('enc_4'):
            x = self.conv_bn_relu('conv1', x, 3, 512)
            x = self.conv_bn_relu('conv2', x, 3, 512)
            x = self.conv_bn_relu('conv3', x, 3, 512)
            x, ind_4 = tf.nn.max_pool_with_argmax(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")  # (N, 48, 48, 512)

        with tf.variable_scope('enc_5'):
            x = self.conv_bn_relu('conv1', x, 3, 512)
            x = self.conv_bn_relu('conv2', x, 3, 512)
            x = self.conv_bn_relu('conv3', x, 3, 512)
            x, ind_5 = tf.nn.max_pool_with_argmax(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")  # (N, 24, 24, 512)

        # decoders
        with tf.variable_scope('dec_5'):
            x = self._unpool_2d(x, ind_5, out_size=[48, 48])
            x = self.conv_bn_relu('conv1', x, 3, 512)
            x = self.conv_bn_relu('conv2', x, 3, 512)
            x = self.conv_bn_relu('conv3', x, 3, 512)

        with tf.variable_scope('dec_4'):
            x = self._unpool_2d(x, ind_4, out_size=[96, 96])
            x = self.conv_bn_relu('conv1', x, 3, 512)
            x = self.conv_bn_relu('conv2', x, 3, 512)
            # x = self.conv_bn_relu('conv3', x, 3, 256)

        if self.is_intermediate:
            self.intermediate_feat = x
            return

        with tf.variable_scope('dec_3'):
            x = self._unpool_2d(x, ind_3, out_size=[188, 188])
            x = self.conv_bn_relu('conv1', x, 3, 256)
            x = self.conv_bn_relu('conv2', x, 3, 256)
            x = self.conv_bn_relu('conv3', x, 3, 128)

        with tf.variable_scope('dec_2'):
            x = self._unpool_2d(x, ind_2, out_size=[375, 375])
            x = self.conv_bn_relu('conv1', x, 3, 128)
            x = self.conv_bn_relu('conv2', x, 3, 64)

        with tf.variable_scope('dec_1'):
            x = self._unpool_2d(x, ind_1, out_size=[750, 750])
            x = self.conv_bn_relu('conv1', x, 3, 64)
            x = self.conv_bn_relu('conv2', x, 3, self.num_classes)

        logits_up = x

        # below is similar to Deeplab-v2

        self.logits_up = logits_up  # (N, H, W, num_classes)
        logits_flat = tf.reshape(self.logits_up, [-1, self.num_classes])
        pred = tf.nn.softmax(logits_flat)
        self.pred = tf.reshape(pred, tf.shape(self.logits_up))  # shape = [1, H, W, nClasses]

        pred_label = tf.argmax(self.pred, 3)  # shape = [1, H, W]
        pred_label = tf.expand_dims(pred_label, axis=3)
        self.pred_label = pred_label  # shape = [1, H, W, 1], contains [0, nClasses)

    def conv_bn_relu(self, name, input, ksize, out_size, stride=1):
        in_size = input.shape[3]
        rst = self._conv(name, input, ksize, in_size, out_size, self._stride_arr(stride))
        rst = tf.contrib.layers.batch_norm(rst)
        rst = tf.nn.relu(rst)
        return rst

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            conv = tf.nn.conv2d(x, w, strides, padding='SAME')
            b = tf.get_variable('biases', [out_filters], initializer=tf.constant_initializer())
            return conv + b

    def _unpool_2d(self,
                   pool,
                   ind,
                   out_size,
                   scope='unpool_2d'):
        """Adds a 2D unpooling op.
        https://arxiv.org/abs/1505.04366
        Unpooling layer after max_pool_with_argmax.
           Args:
               pool:        max pooled output tensor
               ind:         argmax indices
               stride:      stride is the same as for the pool
           Return:
               unpool:    unpooling tensor
        """
        with tf.variable_scope(scope):
            input_shape = tf.shape(pool)
            output_shape = [input_shape[0], out_size[0], out_size[1], input_shape[3]]

            flat_input_size = tf.reduce_prod(input_shape)
            flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

            pool_ = tf.reshape(pool, [flat_input_size])
            batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                     shape=[input_shape[0], 1, 1, 1])
            b = tf.ones_like(ind) * batch_range
            b1 = tf.reshape(b, [flat_input_size, 1])
            ind_ = tf.reshape(ind, [flat_input_size, 1])
            ind_ = tf.concat([b1, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
            ret = tf.reshape(ret, output_shape)

            set_input_shape = pool.get_shape()
            set_output_shape = [set_input_shape[0], out_size[0], out_size[1],
                                set_input_shape[3]]
            ret.set_shape(set_output_shape)
            return ret

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
        #     print('>>>', grad_var.op.name)
        #     if grad is None:
        #         print('None grad')
        #     # if grad is not None:
        #     #     tf.summary.histogram(grad_var.op.name + "/gradient", grad)

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
