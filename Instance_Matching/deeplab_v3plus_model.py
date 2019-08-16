import tensorflow as tf
import functools
from tensorflow.contrib.slim.nets import resnet_utils

slim = tf.contrib.slim

_DEFAULT_MULTI_GRID = [1, 1, 1]


@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               unit_rate=1,
               rate=1,
               outputs_collections=None,
               scope=None):
    """Bottleneck residual unit variant with BN after convolutions.

    This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
    its definition. Note that we use here the bottleneck variant which has an
    extra bottleneck layer.

    When putting together two consecutive ResNet blocks that use this unit, one
    should use stride = 2 in the last unit of the first block.

    Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    unit_rate: An integer, unit rate for atrous convolution.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.

    Returns:
    The ResNet unit's output.
    """
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(
                inputs,
                depth,
                [1, 1],
                stride=stride,
                activation_fn=None,
                scope='shortcut')

        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate * unit_rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               activation_fn=None, scope='conv3')
        output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)


def root_block_fn_for_beta_variant(net):
    """Gets root_block_fn for beta variant.

  ResNet-v1 beta variant modifies the first original 7x7 convolution to three
  3x3 convolutions.

  Args:
    net: A tensor of size [batch, height, width, channels], input to the model.

  Returns:
    A tensor after three 3x3 convolutions.
  """
    net = resnet_utils.conv2d_same(net, 64, 3, stride=2, scope='conv1_1')
    net = resnet_utils.conv2d_same(net, 64, 3, stride=1, scope='conv1_2')
    net = resnet_utils.conv2d_same(net, 128, 3, stride=1, scope='conv1_3')

    return net


def resnet_v1_beta_block(scope, base_depth, num_units, stride):
    """Helper function for creating a resnet_v1 beta variant bottleneck block.

    Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

    Returns:
    A resnet_v1 bottleneck block.
    """
    return resnet_utils.Block(scope, bottleneck, [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': 1,
        'unit_rate': 1
    }] * (num_units - 1) + [{
        'depth': base_depth * 4,
        'depth_bottleneck': base_depth,
        'stride': stride,
        'unit_rate': 1
    }])


def resnet_v1_beta(inputs,
                   blocks,
                   num_classes=None,
                   is_training=None,
                   global_pool=True,
                   output_stride=None,
                   root_block_fn=None,
                   reuse=None,
                   scope=None):
    """Generator for v1 ResNet models (beta variant).

      This function generates a family of modified ResNet v1 models. In particular,
      the first original 7x7 convolution is replaced with three 3x3 convolutions.
      See the resnet_v1_*() methods for specific model instantiations, obtained by
      selecting different block instantiations that produce ResNets of various
      depths.

      The code is modified from slim/nets/resnet_v1.py, and please refer to it for
      more details.

      Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        blocks: A list of length equal to the number of ResNet blocks. Each element
          is a resnet_utils.Block object describing the units in the block.
        num_classes: Number of predicted classes for classification tasks. If None
          we return the features before the logit layer.
        is_training: Enable/disable is_training for batch normalization.
        global_pool: If True, we perform global average pooling before computing the
          logits. Set to True for image classification, False for dense prediction.
        output_stride: If None, then the output will be computed at the nominal
          network stride. If output_stride is not None, it specifies the requested
          ratio of input to output spatial resolution.
        root_block_fn: The function consisting of convolution operations applied to
          the root input. If root_block_fn is None, use the original setting of
          RseNet-v1, which is simply one convolution with 7x7 kernel and stride=2.
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        scope: Optional variable_scope.

      Returns:
        net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
          If global_pool is False, then height_out and width_out are reduced by a
          factor of output_stride compared to the respective height_in and width_in,
          else both height_out and width_out equal one. If num_classes is None, then
          net is the output of the last ResNet block, potentially after global
          average pooling. If num_classes is not None, net contains the pre-softmax
          activations.
        end_points: A dictionary from components of the network to the corresponding
          activation.

      Raises:
        ValueError: If the target output_stride is not valid.
      """
    if root_block_fn is None:
        root_block_fn = functools.partial(resnet_utils.conv2d_same,
                                          num_outputs=64,
                                          kernel_size=7,
                                          stride=2,
                                          scope='conv1')
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            if is_training is not None:
                arg_scope = slim.arg_scope([slim.batch_norm], is_training=is_training)
            else:
                arg_scope = slim.arg_scope([])
            with arg_scope:
                net = inputs
                if output_stride is not None:
                    if output_stride % 4 != 0:
                        raise ValueError('The output_stride needs to be a multiple of 4.')
                    output_stride /= 4
                net = root_block_fn(net)  # (N, 375, 375, 128)

                net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool1')  # (N, 188, 188, 128)
                net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)  # (N, 94, 94, 2048)

                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)

                if num_classes is not None:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')

                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(net, scope='predictions')

                return net, end_points


def resnet_v1_101(inputs,
                  num_classes=None,
                  is_training=None,
                  global_pool=False,
                  output_stride=None,
                  multi_grid=None,
                  reuse=None,
                  scope='resnet_v1_101'):
    """Resnet v1 101.

    Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: Enable/disable is_training for batch normalization.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

    Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

    Raises:
    ValueError: if multi_grid is not None and does not have length = 3.
    """
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    else:
        if len(multi_grid) != 3:
            raise ValueError('Expect multi_grid to have length 3.')

    blocks = [
        resnet_v1_beta_block(
            'block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_beta_block(
            'block2', base_depth=128, num_units=4, stride=2),
        resnet_v1_beta_block(
            'block3', base_depth=256, num_units=23, stride=2),
        resnet_utils.Block('block4', bottleneck, [
            {'depth': 2048,
             'depth_bottleneck': 512,
             'stride': 1,
             'unit_rate': rate} for rate in multi_grid]),
    ]
    return resnet_v1_beta(
        inputs,
        blocks=blocks,
        num_classes=num_classes,
        is_training=is_training,
        global_pool=global_pool,
        output_stride=output_stride,
        reuse=reuse,
        scope=scope)


def resnet_v1_101_beta(inputs,
                       num_classes=None,
                       is_training=None,
                       global_pool=False,
                       output_stride=None,
                       multi_grid=None,
                       reuse=None,
                       scope='resnet_v1_101'):
    """Resnet v1 101 beta variant.

    This variant modifies the first convolution layer of ResNet-v1-101. In
    particular, it changes the original one 7x7 convolution to three 3x3
    convolutions.

    Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: Enable/disable is_training for batch normalization.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

    Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

    Raises:
    ValueError: if multi_grid is not None and does not have length = 3.
    """
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    else:
        if len(multi_grid) != 3:
            raise ValueError('Expect multi_grid to have length 3.')

    blocks = [
        resnet_v1_beta_block(
            'block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_beta_block(
            'block2', base_depth=128, num_units=4, stride=2),
        resnet_v1_beta_block(
            'block3', base_depth=256, num_units=23, stride=2),
        resnet_utils.Block('block4', bottleneck, [
            {'depth': 2048,
             'depth_bottleneck': 512,
             'stride': 1,
             'unit_rate': rate} for rate in multi_grid]),
    ]
    return resnet_v1_beta(
        inputs,
        blocks=blocks,
        num_classes=num_classes,
        is_training=is_training,
        global_pool=global_pool,
        output_stride=output_stride,
        root_block_fn=functools.partial(root_block_fn_for_beta_variant),
        reuse=reuse,
        scope=scope)


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
                           scope=None):
    """Splits a separable conv2d into depthwise and pointwise conv2d.

    This operation differs from `tf.layers.separable_conv2d` as this operation
    applies activation function between depthwise and pointwise conv2d.

    Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.

    Returns:
    Computed features after split separable conv2d.
    """
    outputs = slim.separable_conv2d(
        inputs,
        None,
        kernel_size=kernel_size,
        depth_multiplier=1,
        rate=rate,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=depthwise_weights_initializer_stddev),
        weights_regularizer=None,
        scope=scope + '_depthwise')
    return slim.conv2d(
        outputs,
        filters,
        1,
        weights_initializer=tf.truncated_normal_initializer(
            stddev=pointwise_weights_initializer_stddev),
        weights_regularizer=slim.l2_regularizer(weight_decay),
        scope=scope + '_pointwise')


# A map from network name to network function.
NETWORKS_MAP = {
    'resnet_v1_101': resnet_v1_101,
    'resnet_v1_101_beta': resnet_v1_101_beta,
}

# A map from network name to network arg scope.
ARG_SCOPES_MAP = {
    'resnet_v1_101': resnet_utils.resnet_arg_scope,
    'resnet_v1_101_beta': resnet_utils.resnet_arg_scope,
}

# A map from feature extractor name to the network name scope used in the
# ImageNet pretrained versions of these models.
NAME_SCOPE = {
    'resnet_v1_101': 'resnet_v1_101',
    'resnet_v1_101_beta': 'resnet_v1_101',
}

# A dictionary from network name to a map of end point features.
NETWORKS_TO_FEATURE_MAPS = {
    'resnet_v1_101': {
        'decoder_end_points': ['block1/unit_2/bottleneck_v1/conv3'],
    },
    'resnet_v1_101_beta': {
        'decoder_end_points': ['block1/unit_2/bottleneck_v1/conv3'],
    },
}


class DeepLab_v3plus(object):
    """DeepLab model."""

    def __init__(self, batch_size=1,
                 num_classes=47,
                 lrn_rate=0.0001,
                 lr_decay_step=70000,
                 lrn_rate_end=0.00001,
                 lrn_rate_decay_rate=0.7,
                 weight_decay_rate=0.0001,
                 optimizer='adam',  # 'sgd' or 'mom' or 'adam'
                 images=tf.placeholder(tf.float32, [None, 750, 750, 3]),
                 labels=tf.placeholder(tf.int32),
                 ignore_class_bg=True,
                 mode='test',
                 is_intermediate=False):
        """DeepLab constructor.

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
        self.lrn_rate_decay_rate = lrn_rate_decay_rate
        self.weight_decay_rate = weight_decay_rate
        self.optimizer = optimizer
        self.ignore_class_bg = ignore_class_bg
        self.mode = mode
        self.is_intermediate = is_intermediate
        self._extra_train_ops = []

        # with tf.variable_scope("DeepLab_v3plus"):
        self.build_graph()

    def build_graph(self):
        """Build a whole graph for the model."""
        self._build_model()
        if self.mode == 'train':
            self._build_train_op()

    def _build_model(self):
        self.crop_size = [750, 750]
        self.multi_grid = [1, 2, 4]
        self.atrous_rates = [6, 12, 18]
        self.enc_output_stride = 8
        self.dec_output_stride = 4
        self.encoded_size = 94
        self.decoded_size = 188
        self.model_variant = 'resnet_v1_101_beta'
        self.depth_multiplier = 1.0
        self.aspp_with_batch_norm = True
        self.aspp_with_separable_conv = True
        self.add_image_level_feature = True
        self.fine_tune_batch_norm = False
        self.decoder_use_separable_conv = True

        is_training = self.mode == 'train'

        ## encoder with ASPP
        features, end_points = self.extract_features(
            self.images,
            weight_decay=self.weight_decay_rate,
            reuse=tf.AUTO_REUSE,
            is_training=is_training,
            fine_tune_batch_norm=self.fine_tune_batch_norm)  # features: (N, H/8, W/8, 256); end_points: dict

        if self.is_intermediate:
            return

        ## decoder
        decoder_height = self.decoded_size
        decoder_width = self.decoded_size
        features = self.refine_by_decoder(
            features,
            end_points,
            decoder_height=decoder_height,
            decoder_width=decoder_width,
            decoder_use_separable_conv=self.decoder_use_separable_conv,
            model_variant=self.model_variant,
            weight_decay=self.weight_decay_rate,
            reuse=tf.AUTO_REUSE,
            is_training=is_training,
            fine_tune_batch_norm=self.fine_tune_batch_norm)  # (N, H/4, W/4, 256)

        features = self.get_branch_logits(
            features,
            self.num_classes,
            self.atrous_rates,
            aspp_with_batch_norm=self.aspp_with_batch_norm,
            weight_decay=self.weight_decay_rate,
            reuse=tf.AUTO_REUSE,
            scope_suffix='logits')  # (N, H/4, W/4, nClasses)

        logits_up = tf.image.resize_bilinear(features, self.crop_size, align_corners=True)

        # below is similar to Deeplab-v2

        self.logits_up = logits_up  # (N, H, W, num_classes)
        logits_flat = tf.reshape(self.logits_up, [-1, self.num_classes])
        pred = tf.nn.softmax(logits_flat)
        self.pred = tf.reshape(pred, tf.shape(self.logits_up))  # shape = [1, H, W, nClasses]

        pred_label = tf.argmax(self.pred, 3)  # shape = [1, H, W]
        pred_label = tf.expand_dims(pred_label, axis=3)
        self.pred_label = pred_label  # shape = [1, H, W, 1], contains [0, nClasses)

    def extract_features(self,
                         images,
                         weight_decay=0.0001,
                         reuse=None,
                         is_training=False,
                         fine_tune_batch_norm=False):
        """Extracts features by the particular model_variant.

            Args:
            images: A tensor of size [batch, height, width, channels].
            weight_decay: The weight decay for model variables.
            reuse: Reuse the model variables or not.
            is_training: Is training or not.
            fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

            Returns:
            concat_logits: A tensor of size [batch, feature_height, feature_width,
              feature_channels], where feature_height/feature_width are determined by
              the images height/width and output_stride.
            end_points: A dictionary from components of the network to the corresponding
              activation.
            """
        features, end_points = self.extract_backbone_features(
            images,
            output_stride=self.enc_output_stride,
            multi_grid=self.multi_grid,
            model_variant=self.model_variant,
            depth_multiplier=self.depth_multiplier,
            weight_decay=weight_decay,
            reuse=reuse,
            is_training=is_training,
            fine_tune_batch_norm=fine_tune_batch_norm,
            preprocess_images=False
        )  # features: (N, H/8, W/8, 2048); end_points: dict

        if self.is_intermediate:
            self.intermediate_feat = features
            return features, end_points

        ## ASPP
        if not self.aspp_with_batch_norm:
            return features, end_points
        else:
            batch_norm_params = {
                'is_training': is_training and fine_tune_batch_norm,
                'decay': 0.9997,
                'epsilon': 1e-5,
                'scale': True,
            }

            with slim.arg_scope(
                    [slim.conv2d, slim.separable_conv2d],
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=slim.batch_norm,
                    padding='SAME',
                    stride=1,
                    reuse=reuse):
                with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                    depth = 256
                    branch_logits = []

                    if self.add_image_level_feature:
                        pool_height = self.encoded_size
                        pool_width = self.encoded_size
                        image_feature = slim.avg_pool2d(
                            features, [pool_height, pool_width], [pool_height, pool_width],
                            padding='VALID')
                        image_feature = slim.conv2d(
                            image_feature, depth, 1, scope='image_pooling')
                        image_feature = tf.image.resize_bilinear(
                            image_feature, [pool_height, pool_width], align_corners=True)
                        image_feature.set_shape([None, pool_height, pool_width, depth])
                        branch_logits.append(image_feature)

                    # Employ a 1x1 convolution.
                    branch_logits.append(slim.conv2d(features, depth, 1,
                                                     scope='aspp0'))

                    if self.atrous_rates:
                        # Employ 3x3 convolutions with different atrous rates.
                        for i, rate in enumerate(self.atrous_rates, 1):
                            scope = 'aspp' + str(i)
                            if self.aspp_with_separable_conv:
                                aspp_features = split_separable_conv2d(
                                    features,
                                    filters=depth,
                                    rate=rate,
                                    weight_decay=weight_decay,
                                    scope=scope)
                            else:
                                aspp_features = slim.conv2d(
                                    features, depth, 3, rate=rate, scope=scope)
                            branch_logits.append(aspp_features)

                    # Merge branch logits.
                    concat_logits = tf.concat(branch_logits, 3)  # (N, H/8, W/8, ?)
                    concat_logits = slim.conv2d(
                        concat_logits, depth, 1, scope='concat_projection')  # (N, H/8, W/8, depth)
                    concat_logits = slim.dropout(
                        concat_logits,
                        keep_prob=0.9,
                        is_training=is_training,
                        scope='concat_projection_dropout')

                    return concat_logits, end_points

    def extract_backbone_features(self,
                                  images,
                                  output_stride=8,
                                  multi_grid=None,
                                  depth_multiplier=1.0,
                                  final_endpoint=None,
                                  model_variant=None,
                                  weight_decay=0.0001,
                                  reuse=None,
                                  is_training=False,
                                  fine_tune_batch_norm=False,
                                  regularize_depthwise=False,
                                  preprocess_images=True,
                                  global_pool=False
                                  ):
        """Extracts features by the particular model_variant.

            Args:
            images: A tensor of size [batch, height, width, channels].
            output_stride: The ratio of input to output spatial resolution.
            multi_grid: Employ a hierarchy of different atrous rates within network.
            depth_multiplier: Float multiplier for the depth (number of channels)
              for all convolution ops used in MobileNet.
            final_endpoint: The MobileNet endpoint to construct the network up to.
            model_variant: Model variant for feature extraction.
            weight_decay: The weight decay for model variables.
            reuse: Reuse the model variables or not.
            is_training: Is training or not.
            fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
            regularize_depthwise: Whether or not apply L2-norm regularization on the
              depthwise convolution weights.
            preprocess_images: Performs preprocessing on images or not. Defaults to
              True. Set to False if preprocessing will be done by other functions. We
              supprot two types of preprocessing: (1) Mean pixel substraction and (2)
              Pixel values normalization to be [-1, 1].
            num_classes: Number of classes for image classification task. Defaults
              to None for dense prediction tasks.
            global_pool: Global pooling for image classification task. Defaults to
              False, since dense prediction tasks do not use this.

            Returns:
            features: A tensor of size [batch, feature_height, feature_width,
              feature_channels], where feature_height/feature_width are determined
              by the images height/width and output_stride.
            end_points: A dictionary from components of the network to the corresponding
              activation.

            Raises:
            ValueError: Unrecognized model variant.
            """
        arg_scope = ARG_SCOPES_MAP[model_variant](
            weight_decay=weight_decay,
            batch_norm_decay=0.95,
            batch_norm_epsilon=1e-5,
            batch_norm_scale=True)
        features, end_points = self.get_network(
            model_variant, preprocess_images, arg_scope)(
            inputs=images,
            is_training=(is_training and fine_tune_batch_norm),
            global_pool=global_pool,
            output_stride=output_stride,
            multi_grid=multi_grid,
            reuse=reuse,
            scope=NAME_SCOPE[model_variant])

        return features, end_points

    def get_network(self, network_name, preprocess_images, arg_scope=None):
        """Gets the network.

      Args:
        network_name: Network name.
        preprocess_images: Preprocesses the images or not.
        arg_scope: Optional, arg_scope to build the network. If not provided the
          default arg_scope of the network would be used.

      Returns:
        A network function that is used to extract features.

      Raises:
        ValueError: network is not supported.
      """
        if network_name not in NETWORKS_MAP:
            raise ValueError('Unsupported network %s.' % network_name)
        arg_scope = arg_scope or ARG_SCOPES_MAP[network_name]()

        def _identity_function(inputs):
            return inputs

        # if preprocess_images:
        #     preprocess_function = _PREPROCESS_FN[network_name]
        # else:
        #     preprocess_function = _identity_function
        preprocess_function = _identity_function
        func = NETWORKS_MAP[network_name]

        @functools.wraps(func)
        def network_fn(inputs, *args, **kwargs):
            with slim.arg_scope(arg_scope):
                return func(preprocess_function(inputs), *args, **kwargs)

        return network_fn

    def refine_by_decoder(self,
                          features,
                          end_points,
                          decoder_height,
                          decoder_width,
                          decoder_use_separable_conv=False,
                          model_variant=None,
                          weight_decay=0.0001,
                          reuse=None,
                          is_training=False,
                          fine_tune_batch_norm=False):
        """Adds the decoder to obtain sharper segmentation results.

        Args:
        features: A tensor of size [batch, features_height, features_width,
          features_channels].
        end_points: A dictionary from components of the network to the corresponding
          activation.
        decoder_height: The height of decoder feature maps.
        decoder_width: The width of decoder feature maps.
        decoder_use_separable_conv: Employ separable convolution for decoder or not.
        model_variant: Model variant for feature extraction.
        weight_decay: The weight decay for model variables.
        reuse: Reuse the model variables or not.
        is_training: Is training or not.
        fine_tune_batch_norm: Fine-tune the batch norm parameters or not.

        Returns:
        Decoder output with size [batch, decoder_height, decoder_width,
          decoder_channels].
        """
        batch_norm_params = {
            'is_training': is_training and fine_tune_batch_norm,
            'decay': 0.9997,
            'epsilon': 1e-5,
            'scale': True,
        }

        with slim.arg_scope(
                [slim.conv2d, slim.separable_conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                activation_fn=tf.nn.relu,
                normalizer_fn=slim.batch_norm,
                padding='SAME',
                stride=1,
                reuse=reuse):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with tf.variable_scope('decoder', 'decoder', [features]):
                    feature_list = NETWORKS_TO_FEATURE_MAPS[model_variant]['decoder_end_points']
                    if feature_list is None:
                        tf.logging.info('Not found any decoder end points.')
                        return features
                    else:
                        decoder_features = features
                        for i, name in enumerate(feature_list):
                            decoder_features_list = [decoder_features]
                            feature_name = '{}/{}'.format(NAME_SCOPE[model_variant], name)
                            decoder_features_list.append(
                                slim.conv2d(end_points[feature_name], 48, 1, scope='feature_projection' + str(i)))

                            # 2x upsampling to /4
                            for j, feature in enumerate(decoder_features_list):
                                decoder_features_list[j] = tf.image.resize_bilinear(
                                    feature, [decoder_height, decoder_width], align_corners=True)
                                decoder_features_list[j].set_shape(
                                    [None, decoder_height, decoder_width, None])

                            # two 3*3 separable_conv
                            decoder_depth = 256
                            if decoder_use_separable_conv:
                                decoder_features = split_separable_conv2d(
                                    tf.concat(decoder_features_list, 3),
                                    filters=decoder_depth,
                                    rate=1,
                                    weight_decay=weight_decay,
                                    scope='decoder_conv0')
                                decoder_features = split_separable_conv2d(
                                    decoder_features,
                                    filters=decoder_depth,
                                    rate=1,
                                    weight_decay=weight_decay,
                                    scope='decoder_conv1')  # (N, H/4, W/4, 256)
                            else:
                                num_convs = 2
                                decoder_features = slim.repeat(
                                    tf.concat(decoder_features_list, 3),
                                    num_convs,
                                    slim.conv2d,
                                    decoder_depth,
                                    3,
                                    scope='decoder_conv' + str(i))

                        return decoder_features

    def get_branch_logits(self,
                          features,
                          num_classes,
                          atrous_rates=None,
                          aspp_with_batch_norm=False,
                          kernel_size=1,
                          weight_decay=0.0001,
                          reuse=None,
                          scope_suffix=''):
        """Gets the logits from each model's branch.

        The underlying model is branched out in the last layer when atrous
        spatial pyramid pooling is employed, and all branches are sum-merged
        to form the final logits.

        Args:
        features: A float tensor of shape [batch, height, width, channels].
        num_classes: Number of classes to predict.
        atrous_rates: A list of atrous convolution rates for last layer.
        aspp_with_batch_norm: Use batch normalization layers for ASPP.
        kernel_size: Kernel size for convolution.
        weight_decay: Weight decay for the model variables.
        reuse: Reuse model variables or not.
        scope_suffix: Scope suffix for the model variables.

        Returns:
        Merged logits with shape [batch, height, width, num_classes].

        Raises:
        ValueError: Upon invalid input kernel_size value.
        """
        # When using batch normalization with ASPP, ASPP has been applied before
        # in extract_features, and thus we simply apply 1x1 convolution here.
        if aspp_with_batch_norm or atrous_rates is None:
            if kernel_size != 1:
                raise ValueError('Kernel size must be 1 when atrous_rates is None or '
                                 'using aspp_with_batch_norm. Gets %d.' % kernel_size)
            atrous_rates = [1]

        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                reuse=reuse):
            with tf.variable_scope('logits', 'logits', [features]):
                branch_logits = []
                for i, rate in enumerate(atrous_rates):
                    scope = scope_suffix
                    if i:
                        scope += '_%d' % i

                    branch_logits.append(
                        slim.conv2d(
                            features,
                            num_classes,
                            kernel_size=kernel_size,
                            rate=rate,
                            activation_fn=None,
                            normalizer_fn=None,
                            scope=scope))

                return tf.add_n(branch_logits)

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
            if var.op.name.find(r'weights') > 0:
                costs.append(tf.nn.l2_loss(var))
                # tf.histogram_summary(var.op.name, var)

        return tf.multiply(self.weight_decay_rate, tf.add_n(costs))
