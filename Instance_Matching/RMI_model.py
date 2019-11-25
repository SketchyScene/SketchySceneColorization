import tensorflow as tf

import sys
sys.path.append('Instance_Matching')

import deeplab_model, fcn8s_model, segnet_model, deeplab_v3plus_model
from utils.processing_tools import generate_spatial_batch
from utils import loss


class RMI_model(object):

    def __init__(self, batch_size=1,
                 max_len=15,
                 vf_h=96,
                 vf_w=96,
                 H=768,
                 W=768,
                 vf_dim=2048,
                 vocab_size=59,
                 w_emb_dim=1000,
                 v_emb_dim=1000,
                 m_rnn_size=500,
                 w_rnn_size=1000,
                 start_lr=0.00025,
                 end_lr=0.00001,
                 lr_decay_step=75000,
                 lr_decay_rate=1.0,
                 keep_prob_rnn=1.0,
                 keep_prob_emb=1.0,
                 keep_prob_mlp=1.0,
                 num_rnn_layers=1,
                 optimizer='adam',
                 weight_decay=0.0005,
                 mode='eval',
                 weights='deeplab',
                 training_ignore_bg=True,
                 use_attn=False,
                 train_fusion_var_only=True,
                 fusion_type='RMI'):
        assert fusion_type in ['RMI', 'RecurAttn']
        self.batch_size = batch_size
        self.max_len = max_len
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.H = H
        self.W = W
        self.vf_dim = vf_dim
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.vocab_size = vocab_size

        if fusion_type == 'RecurAttn':
            self.m_rnn_size = 256
        else:
            self.m_rnn_size = m_rnn_size

        self.w_emb_dim = w_emb_dim if fusion_type != 'RecurAttn' else self.m_rnn_size
        self.v_emb_dim = v_emb_dim if fusion_type != 'RecurAttn' else self.m_rnn_size
        self.w_rnn_size = w_rnn_size if fusion_type != 'RecurAttn' else self.m_rnn_size
        self.keep_prob_rnn = keep_prob_rnn
        self.keep_prob_emb = keep_prob_emb
        self.keep_prob_mlp = keep_prob_mlp
        self.num_rnn_layers = num_rnn_layers
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.mode = mode
        self.weights = weights
        self.training_ignore_bg = training_ignore_bg
        self.use_attn = use_attn
        self.train_fusion_var_only = train_fusion_var_only  # Train fusion + CNN variables if False.
        self.fusion_type = fusion_type  # Whether to use the recurrent attention module from LBIE

        self.words = tf.placeholder(tf.int32, [self.batch_size, self.max_len])
        self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.im = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 3])
        self.target_mask = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 1])

        if self.weights == 'deeplab':
            deeplabmodel = deeplab_model.DeepLab(batch_size=self.batch_size,
                                                 images=self.im,
                                                 labels=tf.constant(0.),
                                                 is_intermediate=True)
            self.visual_feat = deeplabmodel.intermediate_feat  # (1, 96, 96, 2048)

        elif self.weights == 'fcn_8s':
            fcn8smodel = fcn8s_model.FCN_8s(batch_size=self.batch_size,
                                            num_classes=46,
                                            images=self.im,
                                            labels=tf.constant(0.),
                                            use_vgg_weight=False,
                                            is_intermediate=True)
            self.visual_feat = fcn8smodel.intermediate_feat  # (1, 96, 96, 256)

        elif self.weights == 'segnet':
            segnetmodel = segnet_model.SegNet(batch_size=self.batch_size,
                                              images=self.im,
                                              labels=tf.constant(0.),
                                              is_intermediate=True)
            self.visual_feat = segnetmodel.intermediate_feat  # (1, 96, 96, 512)

        elif self.weights == 'deeplab_v3plus':
            deeplabv3plusmodel = deeplab_v3plus_model.DeepLab_v3plus(batch_size=self.batch_size,
                                                                     images=self.im,
                                                                     labels=tf.constant(0.),
                                                                     is_intermediate=True)
            self.visual_feat = deeplabv3plusmodel.intermediate_feat  # (1, 96, 96, 2048)
        else:
            raise Exception('Unknown backbone:', self.weights)

        with tf.variable_scope("text_sketchyscene"):
            self.build_graph()
            if self.mode == 'eval':
                return
            self.train_op()

    def build_graph(self):
        visual_feat = self._conv("visual_feat_projection", self.visual_feat, 1,
                                 self.visual_feat.shape[-1], self.v_emb_dim, [1, 1, 1, 1])

        visual_feat_norm = tf.nn.l2_normalize(visual_feat, 3)  # [N, h, w, v_emb_dim(1000)]

        # spatial coordinate feature: [N, h, w, 8]
        spatial = tf.convert_to_tensor(generate_spatial_batch(self.batch_size, self.vf_h, self.vf_w))

        embedding_mat = tf.get_variable("embedding", [self.vocab_size, self.w_emb_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
        words_embed = tf.nn.embedding_lookup(embedding_mat, self.words)  # [N, max_len, w_emb_dim(1000)]

        self.rnn_cell_w = tf.nn.rnn_cell.LSTMCell(self.w_rnn_size, state_is_tuple=False)

        self.rnn_cell_m = tf.nn.rnn_cell.LSTMCell(self.m_rnn_size, state_is_tuple=False)

        self.w_lstm_output, self.w_lstm_last_h = self.build_text_encoder(words_embed, self.sequence_lengths)

        if self.fusion_type == 'RMI':
            print('Using fusion module from RMI without recurrent attention')
            m_last_h = self.build_RMI_fusion_module(words_embed, self.sequence_lengths, self.w_lstm_output,
                                                    self.w_lstm_last_h, visual_feat_norm, spatial)
        elif self.fusion_type == 'RecurAttn':
            print('Using fusion module with recurrent attention')
            m_last_h = self.build_recurrent_attn_fusion_module(self.w_lstm_output, visual_feat_norm)
        else:
            raise Exception('Unknown fusion_type:', self.fusion_type)

        m_lstm_output_proj = self.build_fusion_out_processing(m_last_h)
        self.pred = m_lstm_output_proj  # shape = [1, 96, 96, 1]
        self.up = tf.image.resize_bilinear(self.pred, [self.H, self.W])  # shape = [1, 768, 768, 1]
        self.sigm = tf.sigmoid(self.up)

    def build_text_encoder(self, word_embed, sequence_lengths, reuse=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            ## word LSTM
            w_output, w_last_state = tf.nn.dynamic_rnn(
                self.rnn_cell_w,
                word_embed,
                sequence_length=sequence_lengths,
                dtype=tf.float32,
                time_major=False,
                swap_memory=True,
                scope='wLSTM'
            )  # output: [N, max_len, w_rnn_size(1000)], state: [batch_size, w_rnn_size(1000) * 2]
            _, w_last_h = tf.split(w_last_state, 2, 1)  # each: [N, 1000]

            return w_output, w_last_h

    def build_RMI_fusion_module(self, word_embed, sequence_lengths, w_output, w_last_h, visual_feat, spatial,
                                reuse=False):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            ## mLSTM
            lang_feat_tile = tf.nn.l2_normalize(w_output, 2)
            lang_feat_tile = tf.reshape(lang_feat_tile, [self.batch_size, 1, 1, self.max_len, self.w_rnn_size])
            lang_feat_tile = tf.tile(lang_feat_tile,
                                     [1, self.vf_h, self.vf_w, 1, 1])  # [N, h, w, max_len, w_rnn_size]
            w_feat_tile = tf.reshape(word_embed, [self.batch_size, 1, 1, self.max_len, self.w_emb_dim])
            w_feat_tile = tf.tile(w_feat_tile, [1, self.vf_h, self.vf_w, 1, 1])  # [N, h, w, max_len, w_emb_dim]
            visual_feat_tile = tf.reshape(visual_feat, [self.batch_size, self.vf_h, self.vf_w, 1, self.v_emb_dim])
            visual_feat_tile = tf.tile(visual_feat_tile, [1, 1, 1, self.max_len, 1])  # [N, h, w, max_len, v_emb_dim]
            spatial_tile = tf.reshape(spatial, [self.batch_size, self.vf_h, self.vf_w, 1, 8])
            spatial_tile = tf.tile(spatial_tile, [1, 1, 1, self.max_len, 1])  # [N, h, w, max_len, 8]

            feat_concat = tf.concat([visual_feat_tile, w_feat_tile, lang_feat_tile, spatial_tile], 4)
            feat_concat = tf.reshape(feat_concat, [self.batch_size * self.vf_h * self.vf_w, self.max_len, -1])
            # [N * h * w, max_len, w_rnn_size + w_emb_dim + v_emb_dim + 8]

            sequence_lengths_tile = tf.reshape(sequence_lengths, [self.batch_size, 1, 1])
            sequence_lengths_tile = tf.tile(sequence_lengths_tile, [1, self.vf_h, self.vf_w])
            sequence_lengths_tile = tf.reshape(sequence_lengths_tile, [-1])  # [N * h * w]

            m_output, m_last_state = tf.nn.dynamic_rnn(
                self.rnn_cell_m,
                feat_concat,
                sequence_length=sequence_lengths_tile,
                dtype=tf.float32,
                time_major=False,
                swap_memory=True,
                scope='mLSTM'
            )  # output: [N * h * w, max_len, m_rnn_size(500)], state: [N * h * w, m_rnn_size(500) * 2]

            ## Attention mechanism
            if self.use_attn:
                print('Using attention mechanism')
                w_output_flat = tf.reshape(w_output, (self.batch_size * self.max_len, self.w_rnn_size))
                attn = self._fully_connected(w_output_flat,
                                             in_dim=self.w_rnn_size, out_dim=1, name="attn_fc")  # [N * max_len, 1]
                attn = tf.reshape(attn, (self.batch_size, self.max_len))  # [N, max_len]
                attn = tf.nn.softmax(attn)  # [N, max_len]
                self.attn = attn

                attn_tile = tf.reshape(attn, (self.batch_size, 1, 1, self.max_len))
                attn_tile = tf.tile(attn_tile, [1, self.vf_h, self.vf_w, 1])  # [N, h, w, max_len]
                attn_tile = tf.reshape(attn_tile, (-1, 1, self.max_len))  # [N * h * w, 1, max_len]
                # [N * h * w, 1, max_len] * [N * h * w, max_len, 500] = [N * h * w, 1, 500]
                weighted_m_output = tf.matmul(attn_tile, m_output)
                m_last_h = tf.squeeze(weighted_m_output, axis=1)  # [N * h * w, 500]
            else:
                print('Not using attention mechanism')
                unused_c, m_last_h = tf.split(m_last_state, 2, 1)  # each: [N * h * w, 500]

            return m_last_h

    def build_recurrent_attn_fusion_module(self, w_output, visual_feat):
        """
        use recurrent attention similar to LBIE
        :param w_output: [N, max_len, w_rnn_size]
        :param visual_feat: [N, h, w, v_emb_dim]
        :return:
        """
        cell_m = tf.nn.rnn_cell.MultiRNNCell([self.rnn_cell_m] * self.num_rnn_layers, state_is_tuple=False)

        # Convolutional LSTM
        state_m = cell_m.zero_state(self.batch_size * self.vf_h * self.vf_w, tf.float32)
        state_m_shape = state_m.get_shape().as_list()
        state_m_shape[0] = self.batch_size * self.vf_h * self.vf_w
        state_m.set_shape(state_m_shape)

        h_m = tf.reshape(visual_feat, (-1, self.v_emb_dim))  # [N * h * w, v_emb_dim]

        def f1():
            return state_m, h_m

        def f2():
            h_proj = self._fully_connected(h_m, in_dim=self.m_rnn_size, out_dim=self.w_rnn_size,
                                           name="h_proj")  # [N * h * w, w_rnn_size]
            h_proj = tf.reshape(h_proj, [-1, 1, self.w_rnn_size])  # [N * h * w, 1, w_rnn_size]
            w_output_trans = tf.transpose(w_output, (0, 2, 1))  # [N, w_rnn_size, max_len]
            w_output_trans = tf.reshape(w_output_trans, (self.batch_size, 1, 1, self.w_rnn_size, self.max_len))
            w_output_trans = tf.tile(w_output_trans, (1, self.vf_h, self.vf_w, 1, 1))
            w_output_trans = tf.reshape(w_output_trans,
                                        (-1, self.w_rnn_size, self.max_len))  # [N * h * w, w_rnn_size, max_len]

            attn_map = tf.matmul(h_proj, w_output_trans)  # [N * h * w, 1, max_len]
            attn_map = tf.nn.softmax(attn_map)  # [N * h * w, 1, max_len]

            attn_feat = tf.matmul(attn_map, tf.transpose(w_output_trans, (0, 2, 1)))  # [N * h * w, 1, w_rnn_size]
            attn_feat = tf.squeeze(attn_feat, axis=1)  # [N * h * w, w_rnn_size]

            # Convolutional LSTM
            with tf.variable_scope("mLSTM"):
                h_m_flatten, cell_state_m = cell_m(attn_feat, state_m)

            return cell_state_m, h_m_flatten

        with tf.variable_scope("Recurrent_Attn"):
            for n in range(self.max_len):
                if n > 0:
                    tf.get_variable_scope().reuse_variables()

                state_m, h_m = tf.cond(tf.equal(self.words[0, n], tf.constant(0)), f1, f2)

        return h_m

    def build_fusion_out_processing(self, m_last_h, reuse=False):
        m_lstm_output = tf.reshape(m_last_h, [self.batch_size, self.vf_h, self.vf_w, -1])  # [N, h, w, 500]
        m_lstm_output = tf.multiply(tf.subtract(tf.log(tf.add(1.0 + 1e-3, m_lstm_output)),
                                                tf.log(tf.subtract(1.0 + 1e-3, m_lstm_output))), 0.5)
        m_lstm_output = tf.nn.relu(m_lstm_output)
        if self.mode == 'train' and self.keep_prob_mlp < 1:
            m_lstm_output = tf.nn.dropout(m_lstm_output, self.keep_prob_mlp)

        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            m_lstm_output_proj = self._conv("m_lstm_output_projection", m_lstm_output, 1,
                                            self.m_rnn_size, 1, [1, 1, 1, 1])
            return m_lstm_output_proj

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.conv2d(x, w, strides, padding='SAME') + b

    def _atrous_conv(self, name, x, filter_size, in_filters, out_filters, rate):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                                initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.atrous_conv2d(x, w, rate=rate, padding='SAME') + b

    def _fully_connected(self, x, in_dim, out_dim, name):
        """FullyConnected layer for final output."""
        with tf.variable_scope(name):
            w = tf.get_variable(
                'DW', [in_dim, out_dim],
                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable('biases', [out_dim],
                                initializer=tf.constant_initializer())
            return tf.nn.xw_plus_b(x, w, b)

    def train_op(self):
        # define loss, loss function ignore bg
        target_bin_drawings = tf.expand_dims(self.im[:, :, :, 0], axis=3)  # [1, 768, 768, 1], {0-mu ~ 255-mu}

        pred_for_loss = self.up
        target_for_loss = self.target_mask
        bin_drawings_for_loss = target_bin_drawings

        if self.train_fusion_var_only:
            # Fixed the CNN backbone.
            print('Fixing the CNN variables when training.')
            tvars = [var for var in tf.trainable_variables() if
                     var.op.name.startswith('text_sketchyscene')]
        else:
            print('Training all the variables.')
            tvars = [var for var in tf.trainable_variables()]

        reg_var_list = [var for var in tvars if var.op.name.find(r'DW') > 0]
        self.optim_params = tvars

        ## ignore background
        pred_flatten = tf.reshape(pred_for_loss, (-1,))  # shape = [1 * h * w]
        target_flatten = tf.reshape(target_for_loss, (-1,))  # shape = [1 * h * w]
        target_bin_drawings_flatten = tf.reshape(bin_drawings_for_loss, (-1,))  # shape = [1 * h * w]
        non_bg_indices = tf.where(target_bin_drawings_flatten < 0)[:, 0]  # [nIndices]
        self.pred_remain = tf.gather(pred_flatten, non_bg_indices)  # [nIndices]
        self.target_remain = tf.gather(target_flatten, non_bg_indices)  # [nIndices]

        if self.training_ignore_bg:
            print('Training with the ignore BG strategy.')
            self.cls_loss = loss.weighed_logistic_loss(self.pred_remain, self.target_remain)
        else:
            print('Training without the ignore BG strategy.')
            self.cls_loss = loss.weighed_logistic_loss(pred_for_loss, target_for_loss)
        self.reg_loss = loss.l2_regularization_loss(reg_var_list, self.weight_decay)
        self.cost = self.cls_loss + self.reg_loss

        ## summaries
        tf.summary.scalar('class_loss_current', self.cls_loss)
        tf.summary.scalar('cost', self.cost)

        # learning rate
        self.global_step = tf.Variable(0.0, trainable=False)
        # self.global_step = tf.Variable(0.0, name='global_step', trainable=False)
        self.learning_rate = tf.train.polynomial_decay(self.start_lr, self.global_step, self.lr_decay_step,
                                                       end_learning_rate=self.end_lr, power=0.9)

        # optimizer
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError("Unknown optimizer type %s!" % self.optimizer)

        # learning rate multiplier
        grads_and_vars = optimizer.compute_gradients(self.cost, var_list=tvars)
        var_lr_mult = {var: (2.0 if var.op.name.find(r'biases') > 0 else 1.0) for var in tvars}
        grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v) for g, v in
                          grads_and_vars]

        # training step
        self.train_step = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
