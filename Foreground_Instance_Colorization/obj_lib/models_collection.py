import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as ly

from mru import embed_labels, fully_connected, conv2d, mean_pool, upsample, mru_conv, mru_deconv
from config import Config
from residual_util import nchw_conv_ex, bottleneck_residual_en, bottleneck_residual_pu, bottleneck_residual_de

SIZE = 64  # original is 64
NUM_BLOCKS = 1


def image_resize(inputs, size, method, data_format):
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
    out = tf.image.resize_images(inputs, size, method)
    if data_format == 'NCHW':
        out = tf.transpose(out, [0, 3, 1, 2])
    return out


def batchnorm(inputs, data_format=None, activation_fn=None, labels=None, n_labels=None):
    """conditional batchnorm (dumoulin et al 2016) for BCHW conv filtermaps"""
    if data_format != 'NCHW':
        raise Exception('unsupported')
    mean, var = tf.nn.moments(inputs, (0, 2, 3), keep_dims=True)
    shape = mean.get_shape().as_list()  # shape is [1,n,1,1]

    if n_labels is not None:
        offset_m = tf.get_variable('offset', initializer=np.zeros([n_labels, shape[1]], dtype='float32'))
        scale_m = tf.get_variable('scale', initializer=np.ones([n_labels, shape[1]], dtype='float32'))
        offset = tf.nn.embedding_lookup(offset_m, labels)
        scale = tf.nn.embedding_lookup(scale_m, labels)
        result = tf.nn.batch_normalization(inputs, mean, var, offset[:, :, None, None], scale[:, :, None, None], 1e-5)

    else:
        inputs_ = tf.transpose(inputs, [0, 2, 3, 1])
        input = tf.identity(inputs_)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        result = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        result = tf.transpose(result, [0, 3, 1, 2])

    return result


def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(leak * x, x)


def prelu(x, name="prelu"):
    with tf.variable_scope(name):
        leak = tf.get_variable("param", shape=None, initializer=0.2, regularizer=None,
                               trainable=True, caching_device=None)
        return tf.maximum(leak * x, x)


def miu_relu(x, miu=0.7, name="miu_relu"):
    with tf.variable_scope(name):
        return (x + tf.sqrt((1 - miu) ** 2 + x ** 2)) / 2.


def image_encoder_mru(x, num_classes, reuse=False, data_format='NCHW', labels=None, scope_name=None):
    assert data_format == 'NCHW'
    size = SIZE
    num_blocks = NUM_BLOCKS
    resize_func = tf.image.resize_bilinear
    sn = False

    if normalizer_params_e is not None and normalizer_fn_e != ly.batch_norm and normalizer_fn_e != ly.layer_norm:
        normalizer_params_e['labels'] = labels
        normalizer_params_e['n_labels'] = num_classes

    if data_format == 'NCHW':
        x_list = []
        resized_ = x
        x_list.append(resized_)

        for i in range(4):
            resized_ = mean_pool(resized_, data_format=data_format)
            x_list.append(resized_)
        x_list = x_list[::-1]
    else:
        raise NotImplementedError

    output_list = []

    h0 = conv2d(x_list[-1], 8, kernel_size=7, sn=sn, stride=2, data_format=data_format,
                activation_fn=None,
                normalizer_fn=None,
                normalizer_params=None,
                weights_initializer=weight_initializer)

    output_list.append(h0)

    # Initial memory state
    hidden_state_shape = h0.get_shape().as_list()
    hidden_state_shape[0] = 1
    hts_0 = [h0]

    hts_1 = mru_conv(x_list[-2], hts_0,
                     size * 1, sn=sn, stride=2, dilate_rate=1,
                     data_format=data_format, num_blocks=num_blocks,
                     last_unit=False,
                     activation_fn=activation_fn_e,
                     normalizer_fn=normalizer_fn_e,
                     normalizer_params=normalizer_params_e,
                     weights_initializer=weight_initializer,
                     unit_num=1)
    output_list.append(hts_1[-1])
    hts_2 = mru_conv(x_list[-3], hts_1,
                     size * 2, sn=sn, stride=2, dilate_rate=1,
                     data_format=data_format, num_blocks=num_blocks,
                     last_unit=False,
                     activation_fn=activation_fn_e,
                     normalizer_fn=normalizer_fn_e,
                     normalizer_params=normalizer_params_e,
                     weights_initializer=weight_initializer,
                     unit_num=2)
    output_list.append(hts_2[-1])
    hts_3 = mru_conv(x_list[-4], hts_2,
                     size * 4, sn=sn, stride=2, dilate_rate=1,
                     data_format=data_format, num_blocks=num_blocks,
                     last_unit=False,
                     activation_fn=activation_fn_e,
                     normalizer_fn=normalizer_fn_e,
                     normalizer_params=normalizer_params_e,
                     weights_initializer=weight_initializer,
                     unit_num=3)
    output_list.append(hts_3[-1])
    hts_4 = mru_conv(x_list[-5], hts_3,
                     size * 8, sn=sn, stride=2, dilate_rate=1,
                     data_format=data_format, num_blocks=num_blocks,
                     last_unit=True,
                     activation_fn=activation_fn_e,
                     normalizer_fn=normalizer_fn_e,
                     normalizer_params=normalizer_params_e,
                     weights_initializer=weight_initializer,
                     unit_num=4)
    output_list.append(hts_4[-1])

    return output_list


def encode_feat_with_text(visual_encoded, vocab_indices, input_e_dims, vocab_size_):
    """
    :param visual_encoded:   # [N, 512, 6, 6]
    :param vocab_indices:   # [N, 15]
    :return:
    """
    assert visual_encoded.get_shape().as_list()[0] == vocab_indices.get_shape().as_list()[0]
    print('# Use text to control color')

    # some params
    vocab_size = vocab_size_
    num_rnn_layers = 1
    batch_size = input_e_dims[0]
    w_emb_dim = input_e_dims[1]
    rnn_size = input_e_dims[1]
    mlp_dim = input_e_dims[1]
    vf_h = input_e_dims[2]
    vf_w = input_e_dims[3]
    num_steps = vocab_indices.get_shape().as_list()[1]

    lstm_output = []

    with tf.variable_scope("TextLSTM"):
        for i in range(batch_size):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            visual_encoded_s = tf.expand_dims(visual_encoded[i], axis=0)  # [1, 512, 6, 6]
            vocab_indices_s = tf.expand_dims(vocab_indices[i], axis=0)  # [1, 15]

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

            visual_feat = tf.transpose(visual_encoded_s, [0, 2, 3, 1])  # [1, 6, 6, 512]
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
            lstm_output_s = tf.nn.relu(lstm_output_s)  # [1, 6, 6, 512]
            lstm_output_s = tf.transpose(lstm_output_s, [0, 3, 1, 2])  # [1, 512, 6, 6]

            lstm_output.append(lstm_output_s)

    lstm_output = tf.concat(lstm_output, axis=0)

    return lstm_output


def generate_mru(z, text_vocab_indices, LSTM_hybrid, output_channel, num_classes, vocab_size, reuse=False,
                  data_format='NCHW',
                  labels=None, scope_name=None):
    print("MRU Generator")
    size = SIZE
    num_blocks = NUM_BLOCKS
    sn = False

    input_dims = z.get_shape().as_list()
    resize_method = tf.image.ResizeMethod.AREA

    if data_format == 'NCHW':
        height = input_dims[2]
        width = input_dims[3]
    else:
        height = input_dims[1]
        width = input_dims[2]
    resized_z = [tf.identity(z)]
    for i in range(5):
        resized_z.append(image_resize(z, [int(height / 2 ** (i + 1)), int(width / 2 ** (i + 1))],
                                      resize_method, data_format))
    resized_z = resized_z[::-1]

    if data_format == 'NCHW':
        concat_axis = 1
    else:
        concat_axis = 3

    if normalizer_params_g is not None and normalizer_fn_g != ly.batch_norm and normalizer_fn_g != ly.layer_norm:
        normalizer_params_g['labels'] = labels
        normalizer_params_g['n_labels'] = num_classes

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        z_encoded = image_encoder_mru(z, num_classes=num_classes, reuse=reuse, data_format=data_format,
                                      labels=labels, scope_name=scope_name)

        input_e_dims = z_encoded[-1].get_shape().as_list()
        batch_size = input_e_dims[0]

        # z_encoded[-1].shape = [N, 512, 6, 6], text_vocab_indices.shape = [N, 15]

        if LSTM_hybrid:

            ## Add text LSTM
            lstm_output = encode_feat_with_text(z_encoded[-1], text_vocab_indices, input_e_dims, vocab_size)
            feat_encoded_final = lstm_output  # [N, 512, 6, 6]

        else:
            feat_encoded_final = z_encoded[-1]

        channel_depth = int(input_e_dims[concat_axis] / 8.)
        if data_format == 'NCHW':
            noise_dims = [batch_size, channel_depth, int(input_e_dims[2] * 2), int(input_e_dims[3] * 2)]
        else:
            noise_dims = [batch_size, int(input_e_dims[1] * 2), int(input_e_dims[2] * 2), channel_depth]

        noise_vec = tf.random_normal(shape=(batch_size, 256), dtype=tf.float32)
        noise = fully_connected(noise_vec, int(np.prod(noise_dims[1:])), sn=sn,
                                activation_fn=activation_fn_g,
                                # normalizer_fn=normalizer_fn_g,
                                # normalizer_params=normalizer_params_g
                                )
        noise = tf.reshape(noise, shape=noise_dims)

        # Initial memory state
        hidden_state_shape = z_encoded[-1].get_shape().as_list()
        hidden_state_shape[0] = 1
        hts_0 = [feat_encoded_final]

        input_0 = tf.concat([resized_z[1], noise], axis=concat_axis)
        hts_1 = mru_deconv(input_0, hts_0,
                           size * 6, sn=sn, stride=2, data_format=data_format,
                           num_blocks=num_blocks,
                           last_unit=False,
                           activation_fn=activation_fn_g,
                           normalizer_fn=normalizer_fn_g,
                           normalizer_params=normalizer_params_g,
                           weights_initializer=weight_initializer,
                           unit_num=0)
        input_1 = tf.concat([resized_z[2], z_encoded[-3]], axis=concat_axis)
        hts_2 = mru_deconv(input_1, hts_1,
                           size * 4, sn=sn, stride=2, data_format=data_format,
                           num_blocks=num_blocks,
                           last_unit=False,
                           activation_fn=activation_fn_g,
                           normalizer_fn=normalizer_fn_g,
                           normalizer_params=normalizer_params_g,
                           weights_initializer=weight_initializer,
                           unit_num=2)
        input_2 = tf.concat([resized_z[3], z_encoded[-4]], axis=concat_axis)
        hts_3 = mru_deconv(input_2, hts_2,
                           size * 2, sn=sn, stride=2, data_format=data_format,
                           num_blocks=num_blocks,
                           last_unit=False,
                           activation_fn=activation_fn_g,
                           normalizer_fn=normalizer_fn_g,
                           normalizer_params=normalizer_params_g,
                           weights_initializer=weight_initializer,
                           unit_num=4)
        input_3 = tf.concat([resized_z[4], z_encoded[-5]], axis=concat_axis)
        hts_4 = mru_deconv(input_3, hts_3,
                           size * 2, sn=sn, stride=2, data_format=data_format,
                           num_blocks=num_blocks,
                           last_unit=False,
                           activation_fn=activation_fn_g,
                           normalizer_fn=normalizer_fn_g,
                           normalizer_params=normalizer_params_g,
                           weights_initializer=weight_initializer,
                           unit_num=6)
        hts_5 = mru_deconv(resized_z[5], hts_4,
                           size * 1, sn=sn, stride=2, data_format=data_format,
                           num_blocks=num_blocks,
                           last_unit=True,
                           activation_fn=activation_fn_g,
                           normalizer_fn=normalizer_fn_g,
                           normalizer_params=normalizer_params_g,
                           weights_initializer=weight_initializer,
                           unit_num=8)
        out = conv2d(hts_5[-1], output_channel, 7, sn=sn, stride=1, data_format=data_format,
                     normalizer_fn=None, activation_fn=tf.nn.tanh,
                     weights_initializer=weight_initializer)
        if out.get_shape().as_list()[2] != height:
            raise ValueError('Current shape', out.get_shape().as_list()[2], 'not match', height)
        return out, noise_vec


def nchw_conv(batch_input_, out_channels, stride):
    batch_input = tf.transpose(batch_input_, [0, 2, 3, 1])
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        conv = tf.transpose(conv, [0, 3, 1, 2])
        return conv


def nchw_deconv(batch_input_, out_channels):
    batch_input = tf.transpose(batch_input_, [0, 2, 3, 1])
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels],
                                      [1, 2, 2, 1], padding="SAME")
        conv = tf.transpose(conv, [0, 3, 1, 2])
        return conv
    

def image_encoder_pix2pix(x, num_classes, reuse=False, data_format='NCHW', labels=None, scope_name=None):
    """
    :param x: [batch_size, 3, H, W]
    :return: 
    """
    assert data_format == 'NCHW'
    size = SIZE

    if normalizer_params_e is not None and normalizer_fn_e != ly.batch_norm and normalizer_fn_e != ly.layer_norm:
        normalizer_params_e['labels'] = labels
        normalizer_params_e['n_labels'] = num_classes

    output_list = []

    # encoder_1: [batch, 3, 192, 192] => [batch, 64, 96, 96]
    with tf.variable_scope("encoder_1"):
        output = nchw_conv(x, size, stride=2)
        output_list.append(output)

    layer_specs = [
        size * 2,  # encoder_2: [batch, 64, 96, 96] => [batch, 128, 48, 48]
        size * 4,  # encoder_3: [batch, 128, 48, 48] => [batch, 256, 24, 24]
        size * 8,  # encoder_4: [batch, 256, 24, 24] => [batch, 512, 12, 12]
        size * 8,  # encoder_5: [batch, 512, 12, 12] => [batch, 512, 6, 6]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(output_list) + 1)):
            rectified = lrelu(output_list[-1], 0.2)
            convolved = nchw_conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved, data_format=data_format)
            output_list.append(output)

    return output_list


def generate_pix2pix(z, text_vocab_indices, LSTM_hybrid, output_channel, num_classes, vocab_size, reuse=False,
                      data_format='NCHW', labels=None, scope_name=None):
    print("Pix2pix Generator")
    size = SIZE
    sn = False

    input_dims = z.get_shape().as_list()

    if data_format == 'NCHW':
        height = input_dims[2]
        width = input_dims[3]
    else:
        height = input_dims[1]
        width = input_dims[2]

    if data_format == 'NCHW':
        concat_axis = 1
    else:
        concat_axis = 3

    if normalizer_params_g is not None and normalizer_fn_g != ly.batch_norm and normalizer_fn_g != ly.layer_norm:
        normalizer_params_g['labels'] = labels
        normalizer_params_g['n_labels'] = num_classes

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        z_encoded = image_encoder_pix2pix(z, num_classes=num_classes, reuse=reuse, data_format=data_format,
                                          labels=labels, scope_name=scope_name)  # list of hidden state

        input_e_dims = z_encoded[-1].get_shape().as_list()
        batch_size = input_e_dims[0]

        # z_encoded[-1].shape = [N, 512, 6, 6], text_vocab_indices.shape = [N, 15]

        if LSTM_hybrid:
            ## Add text LSTM
            lstm_output = encode_feat_with_text(z_encoded[-1], text_vocab_indices, input_e_dims, vocab_size)
            feat_encoded_final = lstm_output  # [N, 512, 6, 6]
        else:
            feat_encoded_final = z_encoded[-1]

        channel_depth = int(input_e_dims[concat_axis] / 8.)
        if data_format == 'NCHW':
            noise_dims = [batch_size, channel_depth, int(input_e_dims[2]), int(input_e_dims[3])]
        else:
            noise_dims = [batch_size, int(input_e_dims[1]), int(input_e_dims[2]), channel_depth]

        noise_vec = tf.random_normal(shape=(batch_size, 256), dtype=tf.float32)
        noise = fully_connected(noise_vec, int(np.prod(noise_dims[1:])), sn=sn,
                                activation_fn=activation_fn_g,
                                # normalizer_fn=normalizer_fn_g,
                                # normalizer_params=normalizer_params_g
                                )
        noise = tf.reshape(noise, shape=noise_dims)
        
        ## decoder
        layer_specs = [
            (size * 8, 0.0),  # decoder_5: [batch, 512 * 2, 6, 6] => [batch, 512, 12, 12]
            (size * 4, 0.0),  # decoder_4: [batch, 512 * 2, 12, 12] => [batch, 256, 24, 24]
            (size * 2, 0.0),  # decoder_3: [batch, 256 * 2, 24, 24] => [batch, 128, 48, 48]
            (size, 0.0),      # decoder_2: [batch, 128 * 2, 48, 48] => [batch, 64, 96, 96]
        ]

        num_encoder_layers = len(z_encoded)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    input = tf.concat([feat_encoded_final, noise], axis=concat_axis)
                else:
                    input = tf.concat([z_encoded[-1], z_encoded[skip_layer]], axis=concat_axis)

                rectified = tf.nn.relu(input)
                # [batch, in_channels, in_height, in_width] => [batch, out_channels, in_height*2, in_width*2]
                output = nchw_deconv(rectified, out_channels)
                output = batchnorm(output, data_format=data_format)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                z_encoded.append(output)

        # decoder_1: [batch, 64 * 2, 96, 96] => [batch, 3, 192, 192]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([z_encoded[-1], z_encoded[0]], axis=concat_axis)
            rectified = tf.nn.relu(input)
            output = nchw_deconv(rectified, output_channel)
            output = tf.tanh(output)
            z_encoded.append(output)
        
        if output.get_shape().as_list()[2] != height:
            raise ValueError('Current shape', output.get_shape().as_list()[2], 'not match', height)
        return output, noise_vec


def image_encoder_residual(x, num_residual_units, num_classes, reuse=False, data_format='NCHW', labels=None, scope_name=None):
    """
    :param x: [batch_size, 3, H, W]
    :return:
    """
    assert data_format == 'NCHW'
    size = SIZE

    if normalizer_params_e is not None and normalizer_fn_e != ly.batch_norm and normalizer_fn_e != ly.layer_norm:
        normalizer_params_e['labels'] = labels
        normalizer_params_e['n_labels'] = num_classes

    output_list = []

    # encoder_1: [batch, 3, 192, 192] => [batch, 64, 96, 96]
    with tf.variable_scope("encoder_1"):
        output = nchw_conv_ex(x, size, stride=2, filter_size=7)
        output = batchnorm(output, data_format=data_format)
        output = lrelu(output, 0.2)
        output_list.append(output)

    layer_specs = [
        size * 2,  # encoder_2: [batch, 64, 96, 96] => [batch, 128, 48, 48]
        size * 4,  # encoder_3: [batch, 128, 48, 48] => [batch, 256, 24, 24]
        size * 8,  # encoder_4: [batch, 256, 24, 24] => [batch, 512, 12, 12]
        size * 8,  # encoder_5: [batch, 512, 12, 12] => [batch, 512, 6, 6]
    ]
    for encoder_layer, (out_channels) in enumerate(layer_specs):
        with tf.variable_scope("encoder_%d_0" % (len(output_list) + 1)):
            output = bottleneck_residual_en(output_list[-1], out_channels, stride=2)
        for uId in range(1, num_residual_units[encoder_layer]):
            with tf.variable_scope("encoder_%d_%d" % (len(output_list) + 1, uId)):
                output = bottleneck_residual_pu(output, out_channels, True)
        output_list.append(output)

    return output_list


def generate_residual(z, text_vocab_indices, LSTM_hybrid, output_channel, num_classes, vocab_size, reuse=False,
                      data_format='NCHW', labels=None, scope_name=None):
    print("Residual Generator")
    size = SIZE
    sn = False

    input_dims = z.get_shape().as_list()

    if data_format == 'NCHW':
        height = input_dims[2]
        width = input_dims[3]
    else:
        height = input_dims[1]
        width = input_dims[2]

    if data_format == 'NCHW':
        concat_axis = 1
    else:
        concat_axis = 3

    if normalizer_params_g is not None and normalizer_fn_g != ly.batch_norm and normalizer_fn_g != ly.layer_norm:
        normalizer_params_g['labels'] = labels
        normalizer_params_g['n_labels'] = num_classes

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        num_residual_units = [3, 4, 6, 3]

        z_encoded = image_encoder_residual(z, num_residual_units, num_classes=num_classes, reuse=reuse,
                                           data_format=data_format,
                                           labels=labels, scope_name=scope_name)  # list of hidden state

        input_e_dims = z_encoded[-1].get_shape().as_list()
        batch_size = input_e_dims[0]

        # z_encoded[-1].shape = [N, 512, 6, 6], text_vocab_indices.shape = [N, 15]

        if LSTM_hybrid:
            ## Add text LSTM
            lstm_output = encode_feat_with_text(z_encoded[-1], text_vocab_indices, input_e_dims, vocab_size)
            feat_encoded_final = lstm_output  # [N, 512, 6, 6]
        else:
            feat_encoded_final = z_encoded[-1]

        channel_depth = int(input_e_dims[concat_axis] / 8.)
        if data_format == 'NCHW':
            noise_dims = [batch_size, channel_depth, int(input_e_dims[2]), int(input_e_dims[3])]
        else:
            noise_dims = [batch_size, int(input_e_dims[1]), int(input_e_dims[2]), channel_depth]

        noise_vec = tf.random_normal(shape=(batch_size, 256), dtype=tf.float32)
        noise = fully_connected(noise_vec, int(np.prod(noise_dims[1:])), sn=sn,
                                activation_fn=activation_fn_g,
                                # normalizer_fn=normalizer_fn_g,
                                # normalizer_params=normalizer_params_g
                                )
        noise = tf.reshape(noise, shape=noise_dims)

        ## decoder
        layer_specs = [
            (size * 8, 0.0),  # decoder_5: [batch, 512 * 2, 6, 6] => [batch, 512, 12, 12]
            (size * 4, 0.0),  # decoder_4: [batch, 512 * 2, 12, 12] => [batch, 256, 24, 24]
            (size * 2, 0.0),  # decoder_3: [batch, 256 * 2, 24, 24] => [batch, 128, 48, 48]
            (size, 0.0),  # decoder_2: [batch, 128 * 2, 48, 48] => [batch, 64, 96, 96]
        ]

        num_encoder_layers = len(z_encoded)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d_0" % (skip_layer + 1)):
                if decoder_layer == 0:
                    input = tf.concat([feat_encoded_final, noise], axis=concat_axis)
                else:
                    input = tf.concat([z_encoded[-1], z_encoded[skip_layer]], axis=concat_axis)
                output = bottleneck_residual_de(input, out_channels)
            for uId in range(1, num_residual_units[skip_layer - 1]):
                with tf.variable_scope("decoder_%d_%d" % (skip_layer + 1, uId)):
                    output = bottleneck_residual_pu(output, out_channels, False)

            z_encoded.append(output)

        # decoder_1: [batch, 64 * 2, 96, 96] => [batch, 3, 192, 192]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([z_encoded[-1], z_encoded[0]], axis=concat_axis)
            output = nchw_deconv(input, output_channel)
            output = batchnorm(output, data_format=data_format)
            output = tf.tanh(output)
            z_encoded.append(output)

        if output.get_shape().as_list()[2] != height:
            raise ValueError('Current shape', output.get_shape().as_list()[2], 'not match', height)
        return output, noise_vec


# MRU
def discriminate_mru(discrim_inputs, discrim_targets, num_classes, labels=None, reuse=False, 
                      data_format='NCHW', scope_name=None):
    print("MRU Discriminator")
    assert data_format == 'NCHW'
    size = SIZE
    num_blocks = NUM_BLOCKS
    resize_func = tf.image.resize_bilinear
    sn = Config.sn

    if data_format == 'NCHW':
        channel_axis = 1
    else:
        channel_axis = 3
    if type(discrim_targets) is list:
        discrim_targets = discrim_targets[-1]

    if data_format == 'NCHW':
        x_list = []
        resized_ = discrim_targets
        x_list.append(resized_)

        for i in range(5):
            resized_ = mean_pool(resized_, data_format=data_format)
            x_list.append(resized_)
        x_list = x_list[::-1]
    else:
        raise NotImplementedError

    output_dim = 1

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        h0 = conv2d(x_list[-1], 8, kernel_size=7, sn=sn, stride=1, data_format=data_format,
                    activation_fn=activation_fn_d,
                    normalizer_fn=normalizer_fn_d,
                    normalizer_params=normalizer_params_d,
                    weights_initializer=weight_initializer)

        # Initial memory state
        hidden_state_shape = h0.get_shape().as_list()
        batch_size = hidden_state_shape[0]
        hidden_state_shape[0] = 1
        hts_0 = [h0]
        for i in range(1, num_blocks):
            h0 = tf.tile(tf.get_variable("initial_hidden_state_%d" % i, shape=hidden_state_shape, dtype=tf.float32,
                                         initializer=tf.zeros_initializer()), [batch_size, 1, 1, 1])
            hts_0.append(h0)

        hts_1 = mru_conv(x_list[-1], hts_0,
                         size * 2, sn=sn, stride=2, dilate_rate=1,
                         data_format=data_format, num_blocks=num_blocks,
                         last_unit=False,
                         activation_fn=activation_fn_d,
                         normalizer_fn=normalizer_fn_d,
                         normalizer_params=normalizer_params_d,
                         weights_initializer=weight_initializer,
                         unit_num=1)
        hts_2 = mru_conv(x_list[-2], hts_1,
                         size * 4, sn=sn, stride=2, dilate_rate=1,
                         data_format=data_format, num_blocks=num_blocks,
                         last_unit=False,
                         activation_fn=activation_fn_d,
                         normalizer_fn=normalizer_fn_d,
                         normalizer_params=normalizer_params_d,
                         weights_initializer=weight_initializer,
                         unit_num=2)
        hts_3 = mru_conv(x_list[-3], hts_2,
                         size * 8, sn=sn, stride=2, dilate_rate=1,
                         data_format=data_format, num_blocks=num_blocks,
                         last_unit=False,
                         activation_fn=activation_fn_d,
                         normalizer_fn=normalizer_fn_d,
                         normalizer_params=normalizer_params_d,
                         weights_initializer=weight_initializer,
                         unit_num=3)
        hts_4 = mru_conv(x_list[-4], hts_3,
                         size * 12, sn=sn, stride=2, dilate_rate=1,
                         data_format=data_format, num_blocks=num_blocks,
                         last_unit=True,
                         activation_fn=activation_fn_d,
                         normalizer_fn=normalizer_fn_d,
                         normalizer_params=normalizer_params_d,
                         weights_initializer=weight_initializer,
                         unit_num=4)

        img = hts_4[-1]
        img_shape = img.get_shape().as_list()

        # discriminator end
        disc = conv2d(img, output_dim, kernel_size=1, sn=sn, stride=1, data_format=data_format,
                      activation_fn=None, normalizer_fn=None,
                      weights_initializer=weight_initializer)

        if Config.proj_d:
            # Projection discriminator
            assert labels is not None and (len(labels.get_shape()) == 1 or labels.get_shape().as_list()[-1] == 1)

            class_embeddings = embed_labels(labels, num_classes, img_shape[channel_axis], sn=sn)
            class_embeddings = tf.reshape(class_embeddings, (img_shape[0], img_shape[channel_axis], 1, 1))  # NCHW

            disc += tf.reduce_sum(img * class_embeddings, axis=1, keep_dims=True)

            logits = None
        else:
            # classification end
            img = tf.reduce_mean(img, axis=(2, 3) if data_format == 'NCHW' else (1, 2))
            logits = fully_connected(img, num_classes, sn=sn, activation_fn=None, normalizer_fn=None)

    return disc, logits


def discriminate_pix2pix(discrim_inputs, discrim_targets, num_classes, labels=None, reuse=False,
                          data_format='NCHW', scope_name=None):
    print("Pix2pix Discriminator")
    assert data_format == 'NCHW'
    size = SIZE
    sn = Config.sn

    if data_format == 'NCHW':
        channel_axis = 1
    else:
        channel_axis = 3
    if type(discrim_targets) is list:
        discrim_targets = discrim_targets[-1]

    output_dim = 1
    
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        n_layers = 3
        layers = []

        # 2x [batch, 3, height, width] => [batch, 3 * 2, height, width]
        input_fusion = tf.concat([discrim_inputs, discrim_targets], axis=channel_axis)

        # layer_1: [batch, 6, 192, 192] => [batch, 64, 96, 96]
        with tf.variable_scope("layer_1"):
            convolved = nchw_conv(input_fusion, size, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 64, 96, 96] => [batch, 128, 48, 48]
        # layer_3: [batch, 128, 48, 48] => [batch, 256, 24, 24]
        # layer_4: [batch, 256, 24, 24] => [batch, 512, 23, 23]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = size * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = nchw_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved, data_format=data_format)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 512, 23, 23] => [batch, 1, 22, 22], ==> discriminator end
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            disc = nchw_conv(rectified, out_channels=output_dim, stride=1)
            
        # classification end
        img = tf.reduce_mean(rectified, axis=(2, 3) if data_format == 'NCHW' else (1, 2))
        logits = fully_connected(img, num_classes, sn=sn, activation_fn=None, normalizer_fn=None)

    return disc, logits


def discriminate_residual(discrim_inputs, discrim_targets, num_classes, labels=None, reuse=False,
                          data_format='NCHW', scope_name=None):
    print("Residual Discriminator")
    assert data_format == 'NCHW'
    size = SIZE
    sn = Config.sn

    if data_format == 'NCHW':
        channel_axis = 1
    else:
        channel_axis = 3
    if type(discrim_targets) is list:
        discrim_targets = discrim_targets[-1]

    output_dim = 1

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()

        n_layers = 3
        layers = []

        # 2x [batch, 3, height, width] => [batch, 3 * 2, height, width]
        input_fusion = tf.concat([discrim_inputs, discrim_targets], axis=channel_axis)

        # layer_1: [batch, 6, 192, 192] => [batch, 64, 96, 96]
        with tf.variable_scope("layer_1"):
            rectified = bottleneck_residual_en(input_fusion, size, stride=2)
            layers.append(rectified)

        # layer_2: [batch, 64, 96, 96] => [batch, 128, 48, 48]
        # layer_3: [batch, 128, 48, 48] => [batch, 256, 24, 24]
        # layer_4: [batch, 256, 24, 24] => [batch, 512, 12, 12]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channel = size * min(2 ** (i + 1), 8)
                rectified = bottleneck_residual_en(layers[-1], out_channel, stride=2)
                layers.append(rectified)

        # layer_5: [batch, 512, 12, 12] => [batch, 1, 6, 6], ==> discriminator end
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = bottleneck_residual_en(rectified, 512, stride=2)
            disc = nchw_conv_ex(convolved, output_dim, stride=1)

        # classification end
        img = tf.reduce_mean(rectified, axis=(2, 3) if data_format == 'NCHW' else (1, 2))
        logits = fully_connected(img, num_classes, sn=sn, activation_fn=None, normalizer_fn=None)

    return disc, logits


weight_initializer = tf.random_normal_initializer(0, 0.02)


# weight_initializer = ly.xavier_initializer_conv2d()


def set_param(data_format='NCHW'):
    global model_data_format, normalizer_fn_e, normalizer_fn_g, normalizer_fn_d, \
        normalizer_params_e, normalizer_params_g, normalizer_params_d
    model_data_format = data_format
    normalizer_fn_e = batchnorm
    normalizer_params_e = {'data_format': model_data_format}
    normalizer_fn_g = batchnorm
    normalizer_params_g = {'data_format': model_data_format}
    normalizer_fn_d = None
    normalizer_params_d = None


model_data_format = None

normalizer_fn_e = None
normalizer_params_e = None
normalizer_fn_g = None
normalizer_params_g = None
normalizer_fn_d = None
normalizer_params_d = None

activation_fn_e = miu_relu
activation_fn_g = miu_relu
activation_fn_d = prelu

generator_mru = generate_mru
discriminator_mru = discriminate_mru
generator_pix2pix = generate_pix2pix
discriminator_pix2pix = discriminate_pix2pix
generator_residual = generate_residual
discriminator_residual = discriminate_residual
