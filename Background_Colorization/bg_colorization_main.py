import tensorflow as tf
import numpy as np
import argparse
import os
import json
import random
import collections
import time
from time import gmtime, strftime
from PIL import Image
import sys

sys.path.append('data_processing')
from text_processing import preprocess_sentence, load_vocab_dict_from_file
from image_processing import load_image, load_region_mask

os.environ['CUDA_VISIBLE_DEVICES'] = "1"


EPS = 1e-12

Model = collections.namedtuple("Model",
                               "outputs, output_region_segment, predict_real, predict_fake, "
                               "discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, "
                               "region_mask_loss, gen_loss, gen_grads_and_vars, train")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


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


def preprocess_examples(inputs_, targets_):
    """
    :param inputs_: [1, H, W, 3], [0-255], uint8
    :param targets_:  [1, H, W, 3], [0-255], uint8
    :return: [1, H, W, 3], [-1., 1.], float32
    """
    with tf.name_scope("process_images"):
        inputs = tf.image.convert_image_dtype(inputs_, dtype=tf.float32)  # [0., 1.]
        targets = tf.image.convert_image_dtype(targets_, dtype=tf.float32)  # [0., 1.]

        inputs = preprocess(inputs)  # [-1., 1.]
        targets = preprocess(targets)  # [-1., 1.]

    return inputs, targets


def encode_feat_with_text(visual_encoded, vocab_indices, vocab_size, input_e_dims, scope_name):
    """
    :param visual_encoded:   # [batch, 512, 24, 24]
    :param vocab_indices:   # [batch, T]
    :return:
    """
    assert visual_encoded.get_shape().as_list()[0] == vocab_indices.get_shape().as_list()[0]
    print('# Use text to control color')

    # some params
    num_rnn_layers = 1
    batch_size = input_e_dims[0]
    w_emb_dim = input_e_dims[1]
    rnn_size = input_e_dims[1]
    mlp_dim = input_e_dims[1]
    vf_h = input_e_dims[2]
    vf_w = input_e_dims[3]
    num_steps = vocab_indices.get_shape().as_list()[1]

    lstm_output = []

    with tf.variable_scope(scope_name):
        for i in range(batch_size):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            visual_encoded_s = tf.expand_dims(visual_encoded[i], axis=0)  # [1, 512, 24, 24]
            vocab_indices_s = tf.expand_dims(vocab_indices[i], axis=0)  # [1, T]

            embedding_mat = tf.get_variable("embedding", [vocab_size, w_emb_dim],
                                            initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
            embedded_seq = tf.nn.embedding_lookup(embedding_mat, tf.transpose(vocab_indices_s))

            rnn_cell_w = tf.nn.rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=False)
            cell_w = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_w] * num_rnn_layers, state_is_tuple=False)
            rnn_cell_a = tf.nn.rnn_cell.BasicLSTMCell(mlp_dim, state_is_tuple=False)
            cell_a = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_a] * num_rnn_layers, state_is_tuple=False)

            # Word LSTM
            state_w = cell_w.zero_state(1, tf.float32)  # batch_size -> 1
            state_w_shape = state_w.get_shape().as_list()
            state_w_shape[0] = 1  # batch_size -> 1
            state_w.set_shape(state_w_shape)

            # Convolutional LSTM
            state_a = cell_a.zero_state(1 * vf_h * vf_w, tf.float32)  # batch_size -> 1
            state_a_shape = state_a.get_shape().as_list()
            state_a_shape[0] = 1 * vf_h * vf_w  # batch_size -> 1
            state_a.set_shape(state_a_shape)

            visual_feat = tf.transpose(visual_encoded_s, [0, 2, 3, 1])  # [1, 24, 24, 512]
            visual_feat = tf.nn.l2_normalize(visual_feat, 3)

            h_a = tf.zeros([1 * vf_h * vf_w, mlp_dim])  # batch_size -> 1

            def f1():
                return state_w, state_a, h_a

            def f2():
                # Word input to embedding layer
                w_emb = embedded_seq[n, :, :]
                with tf.variable_scope("WLSTM"):
                    h_w, state_w_ret = cell_w(w_emb, state_w)

                lang_feat = tf.reshape(h_w, [1, 1, 1, rnn_size])  # batch_size -> 1
                lang_feat = tf.nn.l2_normalize(lang_feat, 3)
                lang_feat = tf.tile(lang_feat, [1, vf_h, vf_w, 1])
                w_feat = tf.reshape(w_emb, [1, 1, 1, w_emb_dim])  # batch_size -> 1
                w_feat = tf.tile(w_feat, [1, vf_h, vf_w, 1])

                feat_all = tf.concat([visual_feat, w_feat, lang_feat], 3)

                feat_all_flatten = tf.reshape(feat_all, [1 * vf_h * vf_w, -1])  # batch_size -> 1
                # Convolutional LSTM
                with tf.variable_scope("ALSTM"):
                    h_a_flatten, state_a_ret = cell_a(feat_all_flatten, state_a)

                return state_w_ret, state_a_ret, h_a_flatten

            with tf.variable_scope("RNN"):
                for n in range(num_steps):
                    if n > 0:
                        tf.get_variable_scope().reuse_variables()

                    state_w, state_a, h_a = tf.cond(tf.equal(vocab_indices_s[0, n], tf.constant(0, dtype=tf.int32)), f1,
                                                    f2)

            lstm_output_s = tf.reshape(h_a, [1, vf_h, vf_w, -1])  # batch_size -> 1
            lstm_output_s = tf.multiply(tf.subtract(tf.log(tf.add(1.0 + 1e-3, lstm_output_s)),
                                                    tf.log(tf.subtract(1.0 + 1e-3, lstm_output_s))), 0.5)
            lstm_output_s = tf.nn.relu(lstm_output_s)  # [1, 24, 24, 512]
            lstm_output_s = tf.transpose(lstm_output_s, [0, 3, 1, 2])  # [1, 512, 24, 24]

            lstm_output.append(lstm_output_s)

    lstm_output = tf.concat(lstm_output, axis=0)

    return lstm_output


def bottleneck_residual_en(x, out_filter, stride):
    """ encoder bottleneck_residual """
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
    return x


def bottleneck_residual_de(x, out_filter, need_relu=True):
    """ decoder bottleneck_residual """
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
    return x


def bottleneck_residual_pu(x, out_filter, is_encoder):
    """ public bottleneck_residual """
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

    return x


def create_residual_generator(generator_inputs, generator_outputs_channels, vocab_indices,
                              ngf, vocab_size, seg_classes=3, multi_residual=True):
    """
    the generator with residual blocks
    :param generator_inputs: [batch, 768, 768, 3]
    :param generator_outputs_channels: 3
    :param vocab_indices: [batch, T]
    :return:
    """
    print('# Residual Generator')
    layers = []

    num_residual_units = [3, 4, 6, 3]

    # encoder_1: [batch, 768, 768, 3] => [batch, 384, 384, 64]
    with tf.variable_scope("encoder_1"):
        # output = bottleneck_residual_en(generator_inputs, ngf, stride=2)

        output = conv_ex(generator_inputs, ngf, stride=2, filter_size=7)
        output = batchnorm(output)
        output = lrelu(output, 0.2)
        layers.append(output)

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 384, 384, 64] => [batch, 192, 192, 128]
        ngf * 4,  # encoder_3: [batch, 192, 192, 128] => [batch, 96, 96, 256]
        ngf * 8,  # encoder_4: [batch, 96, 96, 256] => [batch, 48, 48, 512]
        ngf * 16,  # encoder_5: [batch, 48, 48, 512] => [batch, 24, 24, 1024]
    ]
    for encoder_layer, (out_channels) in enumerate(layer_specs):
        if not multi_residual:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                output = bottleneck_residual_en(layers[-1], out_channels, stride=2)
        else:
            with tf.variable_scope("encoder_%d_0" % (len(layers) + 1)):
                output = bottleneck_residual_en(layers[-1], out_channels, stride=2)
            for uId in range(1, num_residual_units[encoder_layer]):
                with tf.variable_scope("encoder_%d_%d" % (len(layers) + 1, uId)):
                    output = bottleneck_residual_pu(output, out_channels, True)
        layers.append(output)

    # layers[-1].shape = [batch, 24, 24, 1024], text_vocab_indices.shape = [batch, T]
    ## Add text LSTM
    visual_encoded = tf.transpose(layers[-1], [0, 3, 1, 2])  # [batch, 1024, 24, 24]
    input_e_dims = visual_encoded.get_shape().as_list()
    lstm_output = encode_feat_with_text(visual_encoded, vocab_indices, vocab_size, input_e_dims,
                                        'mLSTM_G')  # [batch, 1024, 24, 24]
    lstm_output = tf.transpose(lstm_output, [0, 2, 3, 1])
    feat_encoded_final = lstm_output  # [batch, 24, 24, 1024]

    ## region branch
    encoded_image_feat = layers[-1]
    region_br_layers = []

    # region_br_1: [batch, 24, 24, 1024] => [batch, 24, 24, 3]
    with tf.variable_scope("region_br_projection"):
        reg_feat = conv_ex(encoded_image_feat, seg_classes, stride=1, filter_size=1)
        reg_feat = batchnorm(reg_feat)
        reg_feat = tf.nn.relu(reg_feat)
        region_br_layers.append(reg_feat)

    layer_specs = [
        (ngf * 8, 0.0),  # decoder_5: [batch, 24, 24, 1024 * 2] => [batch, 48, 48, 512]
        (ngf * 4, 0.0),  # decoder_4: [batch, 48, 48, 512 * 2] => [batch, 96, 96, 256]
        (ngf * 2, 0.0),  # decoder_3: [batch, 96, 96, 256 * 2] => [batch, 192, 192, 128]
        (ngf, 0.0),  # decoder_2: [batch, 192, 192, 128 * 2] => [batch, 384, 384, 64]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        if not multi_residual:
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = feat_encoded_final
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                output = bottleneck_residual_de(input, out_channels)

        else:
            with tf.variable_scope("decoder_%d_0" % (skip_layer + 1)):
                if decoder_layer == 0:
                    input = feat_encoded_final
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
                output = bottleneck_residual_de(input, out_channels)
            for uId in range(1, num_residual_units[skip_layer - 1]):
                with tf.variable_scope("decoder_%d_%d" % (skip_layer + 1, uId)):
                    output = bottleneck_residual_pu(output, out_channels, False)

        layers.append(output)

        with tf.variable_scope("region_br_%d" % (skip_layer + 1)):
            former_region_mask = region_br_layers[-1]
            former_region_mask_up = deconv(former_region_mask, seg_classes)
            former_region_mask_up = batchnorm(former_region_mask_up)
            former_region_mask_up = tf.nn.relu(former_region_mask_up)
            region_br_layers.append(former_region_mask_up)

    # decoder_1: [batch, 384, 384, 64 * 2] => [batch, 768, 768, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        # output = bottleneck_residual_de(input, generator_outputs_channels, need_relu=False)

        output = deconv(input, generator_outputs_channels)
        output = batchnorm(output)
        output = tf.tanh(output)
        layers.append(output)

    with tf.variable_scope("region_br_1"):
        former_region_mask = region_br_layers[-1]
        former_region_mask_up = deconv(former_region_mask, seg_classes)
        former_region_mask_up = batchnorm(former_region_mask_up)
        former_region_mask_up = tf.nn.relu(former_region_mask_up)
        region_br_layers.append(former_region_mask_up)

    return layers[-1], region_br_layers[-1]


def create_generator(generator_inputs, generator_outputs_channels, vocab_indices, ngf, vocab_size):
    """
    the pix2pix original generator
    :param generator_inputs: [batch, 768, 768, 3]
    :param generator_outputs_channels: 3
    :param vocab_indices: [batch, T]
    :return:
    """
    print('# Pix2pix Generator')
    layers = []

    # encoder_1: [batch, 768, 768, 3] => [batch, 384, 384, 64]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, ngf, stride=2)
        layers.append(output)

    layer_specs = [
        ngf * 2,  # encoder_2: [batch, 384, 384, 64] => [batch, 192, 192, 128]
        ngf * 4,  # encoder_3: [batch, 192, 192, 128] => [batch, 96, 96, 256]
        ngf * 8,  # encoder_4: [batch, 96, 96, 256] => [batch, 48, 48, 512]
        ngf * 8,  # encoder_5: [batch, 48, 48, 512] => [batch, 24, 24, 512]

        # ngf * 8, # encoder_6: [batch, 24, 24, 512] => [batch, 12, 12, 512]
        # ngf * 8, # encoder_7: [batch, 12, 12, 512] => [batch, 6, 6, 512]
        # ngf * 8, # encoder_8: [batch, 6, 6, 512] => [batch, 3, 3, 512]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
            layers.append(output)

    # layers[-1].shape = [batch, 24, 24, 512], text_vocab_indices.shape = [batch, T]
    ## Add text LSTM
    visual_encoded = tf.transpose(layers[-1], [0, 3, 1, 2])  # [batch, 512, 24, 24]
    input_e_dims = visual_encoded.get_shape().as_list()
    lstm_output = encode_feat_with_text(visual_encoded, vocab_indices, vocab_size, input_e_dims,
                                        'mLSTM_G')  # [batch, 512, 24, 24]
    lstm_output = tf.transpose(lstm_output, [0, 2, 3, 1])
    feat_encoded_final = lstm_output  # [batch, 24, 24, 512]

    layer_specs = [
        # (ngf * 8, 0.5),   # decoder_8: [batch, 3, 3, 512] => [batch, 6, 6, 512 * 2]
        # (ngf * 8, 0.5),   # decoder_7: [batch, 6, 6, 512 * 2] => [batch, 12, 12, 512 * 2]
        # (ngf * 8, 0.5),   # decoder_6: [batch, 12, 12, 512 * 2] => [batch, 24, 24, 512 * 2]

        (ngf * 8, 0.0),  # decoder_5: [batch, 24, 24, 512 * 2] => [batch, 48, 48, 512 * 2]
        (ngf * 4, 0.0),  # decoder_4: [batch, 48, 48, 512 * 2] => [batch, 96, 96, 256 * 2]
        (ngf * 2, 0.0),  # decoder_3: [batch, 96, 96, 256 * 2] => [batch, 192, 192, 128 * 2]
        (ngf, 0.0),  # decoder_2: [batch, 192, 192, 128 * 2] => [batch, 384, 384, 64 * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = feat_encoded_final
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 384, 384, 64 * 2] => [batch, 768, 768, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_model(inputs, targets, text_vocab_indices, labels_gt, mode_type, 
                 ndf, ngf, gan_weight, l1_weight, seg_weight, lr, max_steps, vocab_size,
                 residual_enc_g=True, residual_enc_d=True, multi_residual=True,
                 seg_classes=3, beta1=0.5):
    def create_discriminator(discrim_inputs, discrim_targets, vocab_indices):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, 3] => [batch, height, width, 3 * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 768, 768, 6] => [batch, 384, 384, 64]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 384, 384, 64] => [batch, 192, 192, 128]
        # layer_3: [batch, 192, 192, 128] => [batch, 96, 96, 256]
        # layer_4: [batch, 96, 96, 256] => [batch, 95, 95, 512]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 95, 95, 512] => [batch, 94, 94, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)

            # convolved.shape = [batch, 94, 94, 1], text_vocab_indices.shape = [batch, T]
            feat_encoded_final_d = convolved

            output = tf.sigmoid(feat_encoded_final_d)
            layers.append(output)

        return layers[-1]

    def create_residual_discriminator(discrim_inputs, discrim_targets, vocab_indices):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, 3] => [batch, height, width, 3 * 2]
        input_fusion = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 768, 768, 6] => [batch, 384, 384, 64]
        with tf.variable_scope("layer_1"):
            rectified = bottleneck_residual_en(input_fusion, ndf, stride=2)
            layers.append(rectified)

        # layer_2: [batch, 384, 384, 64] => [batch, 192, 192, 128]
        # layer_3: [batch, 192, 192, 128] => [batch, 96, 96, 256]
        # layer_4: [batch, 96, 96, 256] => [batch, 48, 48, 512]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channel = ndf * min(2 ** (i + 1), 8)
                rectified = bottleneck_residual_en(layers[-1], out_channel, stride=2)
                layers.append(rectified)

        # layer_5: [batch, 48, 48, 512] => [batch, 24, 24, 1024]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = bottleneck_residual_en(rectified, 1024, stride=2)

            # convolved.shape = [batch, 24, 24, 1], text_vocab_indices.shape = [batch, T]
            feat_encoded_final_d = convolved

            # feat_encoded_final_d = conv(feat_encoded_final_d, out_channels=1, stride=1)
            output = tf.sigmoid(feat_encoded_final_d)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        if residual_enc_g:
            outputs, region_mask_logits = create_residual_generator(inputs, out_channels, text_vocab_indices,
                                                                    ngf, vocab_size, seg_classes, multi_residual)
        else:
            outputs = create_generator(inputs, out_channels, text_vocab_indices, ngf, vocab_size)

    with tf.name_scope("region_mask_branch"):
        region_mask_logits_flat = tf.reshape(region_mask_logits, [-1, seg_classes])
        region_mask_pred = tf.nn.softmax(region_mask_logits_flat)
        region_mask_pred = tf.reshape(region_mask_pred, tf.shape(region_mask_logits))  # shape = [1, H, W, 3]
        region_mask_pred_label = tf.argmax(region_mask_pred, 3)  # shape = [1, H, W]

        region_mask_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=region_mask_logits, labels=labels_gt),
            name='region_mask_loss')

    if mode_type != "train":
        return Model(
            predict_real=None,
            predict_fake=None,
            discrim_loss=None,
            discrim_grads_and_vars=None,
            gen_loss_GAN=None,
            gen_loss_L1=None,
            region_mask_loss=None,
            gen_loss=None,
            gen_grads_and_vars=None,
            outputs=outputs,
            output_region_segment=region_mask_pred_label,
            train=None,
        )

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 94, 94, 1]
            if residual_enc_d:
                predict_real = create_residual_discriminator(inputs, targets, text_vocab_indices)
            else:
                predict_real = create_discriminator(inputs, targets, text_vocab_indices)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 94, 94, 1]
            if residual_enc_d:
                predict_fake = create_residual_discriminator(inputs, outputs, text_vocab_indices)
            else:
                predict_fake = create_discriminator(inputs, outputs, text_vocab_indices)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))

        # only penalize non-fg area (labels_gt != 0)
        original_abs = tf.abs(targets - outputs)
        original_abs_flatten = tf.reshape(original_abs, [-1, original_abs.get_shape()[3]])
        gt_labels_flatten = tf.reshape(labels_gt, [-1, ])
        indices = tf.squeeze(tf.where(tf.not_equal(gt_labels_flatten, 0)), 1)
        remain_abs = tf.gather(original_abs_flatten, indices)
        gen_loss_L1 = tf.reduce_mean(remain_abs)

        gen_loss = gen_loss_GAN * gan_weight + gen_loss_L1 * l1_weight + region_mask_loss * seg_weight

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.name_scope("lr_decay"):
        learning_rate = tf.train.polynomial_decay(lr,
                                                  global_step,
                                                  decay_steps=int(round(max_steps * 0.75)),
                                                  end_learning_rate=lr / 10.,
                                                  power=0.9)
        # learning_rate = tf.train.exponential_decay(lr, global_step,
        #                                            20000, 0.75, staircase=True)
        tf.summary.scalar('learning rate', learning_rate)

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(learning_rate, beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(learning_rate, beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1, region_mask_loss, gen_loss])

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        region_mask_loss=ema.average(region_mask_loss),
        gen_loss=ema.average(gen_loss),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        output_region_segment=region_mask_pred_label,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def bg_colorization(**kwargs):
    mode = kwargs['mode']
    resume_from = kwargs['resume_from']
    data_base_dir = kwargs['data_base_dir']
    image_size = kwargs['image_size']
    lr = kwargs['lr']
    max_steps = kwargs['max_steps']
    batch_size = kwargs['batch_size']

    ndf = kwargs['ndf']
    ngf = kwargs['ngf']

    gan_weight = kwargs['gan_weight']
    l1_weight = kwargs['l1_weight']
    seg_weight = kwargs['seg_weight']
    seg_classes = kwargs['seg_classes']

    T = kwargs['text_len']
    vocab_size = kwargs['vocab_size']
    vocab_file = kwargs['vocab_file']

    summary_freq = kwargs['summary_freq']
    progress_freq = kwargs['progress_freq']
    save_freq = kwargs['save_freq']

    cur_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    if resume_from == '':
        output_dir = os.path.join("outputs", cur_time)
    else:
        output_dir = os.path.join("outputs", resume_from)

    residual_enc_g = True  # 'True' to ues residual encoder in G, 'False' to use pix2pix's
    residual_enc_d = True  # 'True' to ues residual encoder in D, 'False' to use PatchGAN's D
    multi_residual = True  # 'True' to use [1, 3, 4, 6, 3] res' blocks in each encoder/decoder,
                           # 'False' to use only 1 blocks each

    inputs_base_dir = os.path.join(data_base_dir, 'foreground', mode)
    targets_base_dir = os.path.join(data_base_dir, 'background', mode)
    segment_base_dir = os.path.join(data_base_dir, 'segment', mode)

    seed = random.randint(0, 2 ** 31 - 1)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    save_model_base_dir = os.path.join(output_dir, "snapshot")
    os.makedirs(save_model_base_dir, exist_ok=True)

    # read json data
    image_data_json = os.path.join(data_base_dir, 'captions', mode + '.json')
    fp = open(image_data_json, "r")
    json_data = fp.read()
    json_data = json.loads(json_data)
    nImgs = len(json_data)
    print('## nImgs =', nImgs, '\n')

    if mode == "test":
        if resume_from == '':
            raise Exception("checkpoint required for test mode")

    vocab_dict = load_vocab_dict_from_file(vocab_file)

    input_images = tf.placeholder(tf.uint8, shape=[1, image_size, image_size, 3])  # [1, H, W, 3], [0-255], uint8
    target_images = tf.placeholder(tf.uint8, shape=[1, image_size, image_size, 3])  # [1, H, W, 3], [0-255], uint8
    text_vocab_indiceses = tf.placeholder(tf.int32, shape=[1, T])  # [1, T]
    region_labels = tf.placeholder(tf.int32, shape=[1, image_size, image_size])

    # [1, H, W, 3], [-1., 1.], float32
    input_images_pro, target_images_pro = preprocess_examples(input_images, target_images)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(input_images_pro, target_images_pro, text_vocab_indiceses, region_labels, mode,
                         ndf, ngf, gan_weight, l1_weight, seg_weight, lr, max_steps, vocab_size,
                         residual_enc_g=residual_enc_g, residual_enc_d=residual_enc_d, multi_residual=multi_residual,
                         seg_classes=seg_classes)

    # [-1., 1.] -> [0., 1.], float32
    inputs_depro = deprocess(input_images_pro)
    targets_depro = deprocess(target_images_pro)
    outputs_depro = deprocess(model.outputs)

    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # [0., 1.] -> [0, 255], uint8
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs_depro)
    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets_depro)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs_depro)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    if mode == "train":
        # summaries
        tf.summary.scalar("discriminator_loss", model.discrim_loss)
        tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
        tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)
        tf.summary.scalar("region_mask_loss", model.region_mask_loss)
        tf.summary.scalar("generator_loss", model.gen_loss)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=10)

    logdir = os.path.join(output_dir, "log") if summary_freq > 0 and mode == "train" else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with sv.managed_session(config=tfconfig) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if resume_from != '':
            checkpoint_path = os.path.join(output_dir, "snapshot")
            checkpoint = tf.train.latest_checkpoint(checkpoint_path)
            print("loading model from checkpoint", checkpoint)
            # restorer = tf.train.Saver()
            saver.restore(sess, checkpoint)
            iter_from = int(checkpoint[len(checkpoint_path) + 10:])
        else:
            iter_from = 0
        print('iter_from', iter_from)

        if mode == "test":
            image_dir = os.path.join(output_dir, "results")
            os.makedirs(image_dir, exist_ok=True)

            for image_idx in range(nImgs):
                input_name = json_data[image_idx]['fg_name']
                target_name = json_data[image_idx]['bg_name']
                print('Processing', image_idx, '/', nImgs)

                # load inputs
                input_path = os.path.join(inputs_base_dir, input_name)
                input_data = load_image(input_path, image_size)  # [1, H, W, 3], uint8, [0-255]
                # print('input_path', input_path)

                # load targets
                target_path = os.path.join(targets_base_dir, target_name)
                target_data = load_image(target_path, image_size)  # [1, H, W, 3], uint8, [0-255]
                # print('target_path', target_path)

                # load text
                input_text = json_data[image_idx]['color_text']  # e.g. 'all things are on green grass and gray road'
                vocab_indices = preprocess_sentence(input_text, vocab_dict, T)  # list
                vocab_indices = np.array(vocab_indices, dtype=np.int32)
                vocab_indices = np.expand_dims(vocab_indices, axis=0)  # shape = [1, T]

                results, region_segment = sess.run([display_fetches, model.output_region_segment],
                                                   feed_dict={input_images: input_data,
                                                              target_images: target_data,
                                                              text_vocab_indiceses: vocab_indices,
                                                              region_labels: load_region_mask('', image_size, is_test=True)})

                # save images
                for kind in ["inputs", "outputs", "targets"]:
                    contents = results[kind][0]
                    filename = target_name[:-4] + "_" + kind + ".png"
                    out_path = os.path.join(image_dir, filename)
                    with open(out_path, "wb") as f:
                        f.write(contents)

                ## post-processing: cover the FG to generation
                segment_path = os.path.join(segment_base_dir, input_name)
                inner_mask = Image.open(segment_path).convert('RGB')
                inner_mask = np.array(inner_mask, dtype=np.uint8)[:, :, 0]  # 0 is fg

                output_path = os.path.join(image_dir, target_name[:-4] + "_outputs.png")
                post_output = Image.open(output_path).convert('RGB')
                post_output = np.array(post_output, dtype=np.uint8)
                post_output[inner_mask == 0] = input_data[0][inner_mask == 0]
                post_output = Image.fromarray(post_output, 'RGB')
                post_output.save(output_path, 'PNG')

                ## segment_map
                # region_segment = np.squeeze(region_segment)
                # segment_map = np.zeros(region_segment.shape, dtype=np.uint8)
                # segment_map[region_segment == 1] = 128  # label 'Sky' to 1
                # segment_map[region_segment == 2] = 255  # label 'Grass/Ground' to 2
                # segment_map_png = Image.fromarray(segment_map, 'L')
                # segment_map_png_path = os.path.join(image_dir, target_name[:-4] + "_segment.png")
                # segment_map_png.save(segment_map_png_path, 'PNG')

        else:
            # training
            start = time.time()

            for step in range(iter_from, max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1
                    fetches["region_mask_loss"] = model.region_mask_loss
                    fetches["gen_loss"] = model.gen_loss

                if should(summary_freq):
                    fetches["summary"] = sv.summary_op

                ## read in real data
                image_idx = random.randint(0, nImgs - 1)
                input_name = json_data[image_idx]['fg_name']
                target_name = json_data[image_idx]['bg_name']

                # load inputs
                input_path = os.path.join(inputs_base_dir, input_name)
                input_data = load_image(input_path, image_size)  # [1, H, W, 3], uint8, [0-255]
                # print('input_path', input_path)

                # load targets
                target_path = os.path.join(targets_base_dir, target_name)
                target_data = load_image(target_path, image_size)  # [1, H, W, 3], uint8, [0-255]
                # print('target_path', target_path)

                # load text
                input_text = json_data[image_idx]['color_text']

                vocab_indices = preprocess_sentence(input_text, vocab_dict, T)  # list
                vocab_indices = np.array(vocab_indices, dtype=np.int32)
                vocab_indices = np.expand_dims(vocab_indices, axis=0)  # shape = [1, T]

                # load region mask
                region_mask_path = os.path.join(segment_base_dir, input_name)
                region_mask_data = load_region_mask(region_mask_path, image_size)  # [1, H, W], int32, {0, 1, 2}

                results = sess.run(fetches, feed_dict={input_images: input_data,
                                                       target_images: target_data,
                                                       text_vocab_indiceses: vocab_indices,
                                                       region_labels: region_mask_data})

                if should(summary_freq):
                    # print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_step = results["global_step"]
                    rate = (step - iter_from + 1) * batch_size / (time.time() - start)
                    left_sec = (max_steps - step) * batch_size / rate
                    left_day = int(left_sec / 24 / 60 / 60)
                    left_hour = int((left_sec - (24 * 60 * 60) * left_day) / 60 / 60)
                    left_min = int((left_sec - (24 * 60 * 60) * left_day - (60 * 60) * left_hour) / 60)
                    print("progress step %d  image/sec %0.1f  left time:%dd %dh %dm"
                          % (train_step, rate, left_day, left_hour, left_min))

                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])
                    print("region_mask_loss", results["region_mask_loss"])
                    print("gen_loss", results["gen_loss"])

                if should(save_freq):
                    print("saving model to", save_model_base_dir)
                    snapshot_file = os.path.join(save_model_base_dir, 'snapshot')
                    saver.save(sess, snapshot_file, global_step=sv.global_step)

                if sv.should_stop():
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train', choices=["train", "test"])
    parser.add_argument("--resume_from", type=str, default='', help="where to put output files")

    parser.add_argument("--data_base_dir", type=str, default='data', help="where to put data")
    parser.add_argument("--image_size", type=int, default=768, help="image size")

    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
    parser.add_argument("--max_steps", type=int, default=100000, help="number of training steps (0 to disable)")
    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
    parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
    parser.add_argument("--seg_weight", type=float, default=100.0, help="weight on SEG term for generator gradient")
    parser.add_argument("--seg_classes", type=int, default=3, help="number of categories of seg")

    parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")

    parser.add_argument("--text_len", type=int, default=8, help="the longest length of text")
    parser.add_argument("--vocab_size", type=int, default=18, help="vocab size")
    parser.add_argument("--vocab_file", type=str, default='data/bg_vocab.txt', help="path of vocab")
    
    parser.add_argument("--summary_freq", type=int, default=200, help="update summaries every summary_freq steps")
    parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
    parser.add_argument("--save_freq", type=int, default=20000, help="save model every save_freq steps, 0 to disable")

    args = parser.parse_args()

    if args.mode == 'test':
        assert args.resume_from != ''

    run_params = {
        "mode": args.mode,
        "resume_from": args.resume_from,
        "data_base_dir": args.data_base_dir,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "l1_weight": args.l1_weight,
        "gan_weight": args.gan_weight,
        "seg_weight": args.seg_weight,
        "seg_classes": args.seg_classes,
        "ngf": args.ngf,
        "ndf": args.ndf,
        "text_len": args.text_len,
        "vocab_size": args.vocab_size,
        "vocab_file": args.vocab_file,
        "summary_freq": args.summary_freq,
        "progress_freq": args.progress_freq,
        "save_freq": args.save_freq,
    }
    
    bg_colorization(**run_params)
