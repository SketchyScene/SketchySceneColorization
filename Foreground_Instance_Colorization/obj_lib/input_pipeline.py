import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import skimage.morphology as sm
from scipy import ndimage

from config import Config

num_classes = 25


def get_num_classes():
    return num_classes


def one_hot_to_dense(labels):
    # Assume on value is 1
    batch_size = int(labels.get_shape()[0])
    return tf.reshape(tf.where(tf.equal(labels, 1))[:, 1], (batch_size,))


def map_class_id_to_labels(batch_class_id):
    batch_class_id_backup = tf.identity(batch_class_id)

    for i in range(num_classes):
        comparison = tf.equal(batch_class_id_backup, tf.constant(i, dtype=tf.int64))
        batch_class_id = tf.where(comparison, tf.ones_like(batch_class_id) * i, batch_class_id)
    ret_tensor = tf.squeeze(tf.one_hot(tf.cast(batch_class_id, dtype=tf.int32), num_classes,
                                       on_value=1, off_value=0, axis=1))
    return ret_tensor


def binarize(sketch, threshold=250):
    return tf.where(sketch < threshold, x=tf.zeros_like(sketch), y=tf.ones_like(sketch) * 255.)


# SKETCH_CHANNEL = 3
SIZE = {True: (64, 64),
        False: (192, 192)}


# Distance map first, then resize
def get_paired_input(paired_filenames_, test_mode, img_dim, distance_map=True,
                     fancy_upscaling=False, data_format='NCHW'):
    if test_mode:
        num_epochs = 1
        shuffle = False
    else:
        num_epochs = None
        shuffle = True
    filename_queue = tf.train.string_input_producer(
        paired_filenames_, capacity=512, shuffle=shuffle, num_epochs=num_epochs)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'ImageName': tf.FixedLenFeature([], tf.string),
            'cartoon_data': tf.FixedLenFeature([], tf.string),
            'sketch_data': tf.FixedLenFeature([], tf.string),
            'Category': tf.FixedLenFeature([], tf.string),
            'Category_id': tf.FixedLenFeature([], tf.int64),
            'Color_text': tf.FixedLenFeature([], tf.string),
            'Text_vocab_indices': tf.FixedLenFeature([], tf.string),
        }
    )

    image = features['cartoon_data']
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [384, 384, 3])  # cannot change

    sketch = features['sketch_data']
    sketch = tf.decode_raw(sketch, tf.uint8)
    sketch = tf.cast(sketch, tf.float32)
    sketch = tf.reshape(sketch, [384, 384, 3])  # cannot change

    # Distance map
    if not Config.pre_calculated_dist_map and distance_map:
        # Binarize
        sketch = binarize(sketch)
        sketch_shape = sketch.shape

        sketch = tf.py_func(lambda x: ndimage.distance_transform_edt(x).astype(np.float32),
                            [sketch], tf.float32, stateful=False)
        sketch = tf.reshape(sketch, sketch_shape)
        # Normalize
        sketch = sketch / tf.reduce_max(sketch) * 255.

    # Resize
    if image.get_shape().as_list()[0] != img_dim[0] and image.get_shape().as_list()[1] != img_dim[1]:
        image = tf.image.resize_images(image, img_dim, method=tf.image.ResizeMethod.BILINEAR)
        sketch = tf.image.resize_images(sketch, img_dim, method=tf.image.ResizeMethod.AREA)

    # Augmentation
    # Image
    # image = tf.image.random_brightness(image, max_delta=0.3)
    # image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # image_large = tf.image.random_hue(image_large, max_delta=0.05)

    # Normalization
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image) + 1)
    image += tf.random_uniform(shape=image.shape, minval=0., maxval=1. / 256)  # dequantize
    sketch = sketch / 255.

    image = image * 2. - 1
    sketch = sketch * 2. - 1

    # Transpose for data format
    if data_format == 'NCHW':
        image = tf.transpose(image, [2, 0, 1])
        sketch = tf.transpose(sketch, [2, 0, 1])

    # Attributes
    category = features['Category']
    class_id = features['Category_id']
    imageName = features['ImageName']
    color_text = features['Color_text']
    text_vocab_indices = features['Text_vocab_indices']
    text_vocab_indices = tf.decode_raw(text_vocab_indices, tf.uint8)
    text_vocab_indices = tf.cast(text_vocab_indices, tf.int32)
    text_vocab_indices = tf.reshape(text_vocab_indices, [15])  # cannot change

    return image, sketch, class_id, category, imageName, color_text, text_vocab_indices


def build_input_queue_paired(mode, batch_size, data_format='NCHW', distance_map=False, small=False,
                             one_hot=False, capacity=8192, min_after_dequeue=512, data_base_dir='data'):
    assert mode in ['train']
    data_dir = os.path.join(data_base_dir, 'tfrecord', mode)
    # [*.tfrecord]
    paired_filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if os.path.isfile(os.path.join(data_dir, f))]
    print("build_input_queue_paired from %s: paired file num: %d" % (data_dir, len(paired_filenames)))

    image, sketch, class_id, category, image_name, color_text, text_vocab_indices = get_paired_input(
        paired_filenames, test_mode=False, distance_map=distance_map, img_dim=SIZE[small], data_format=data_format)

    images, sketches, class_ids, categories, image_names, color_texts, text_vocab_indiceses \
        = tf.train.maybe_shuffle_batch([image, sketch, class_id, category,
                                        image_name, color_text, text_vocab_indices],
                                       batch_size=batch_size, capacity=capacity,
                                       keep_input=True, min_after_dequeue=min_after_dequeue,
                                       num_threads=4)

    if one_hot:
        labels = map_class_id_to_labels(class_ids)
    else:
        labels = one_hot_to_dense(map_class_id_to_labels(class_ids))
    return images, sketches, labels, categories, image_names, color_texts, text_vocab_indiceses


def build_input_queue_paired_test(mode, batch_size, data_format='NCHW', distance_map=False, small=False,
                                  one_hot=False,
                                  capacity=8192, data_base_dir='data'):
    assert mode in ['test', 'val']
    data_dir = os.path.join(data_base_dir, 'tfrecord', mode)
    # [*.tfrecord]
    paired_filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if os.path.isfile(os.path.join(data_dir, f))]
    print("build_input_queue_paired_test from %s: paired file num: %d" % (data_dir, len(paired_filenames)))

    image, sketch, class_id, category, image_name, color_text, text_vocab_indices = get_paired_input(
        paired_filenames, test_mode=True, distance_map=distance_map, img_dim=SIZE[small], data_format=data_format)

    images, sketches, class_ids, categories, image_names, color_texts, text_vocab_indiceses \
        = tf.train.maybe_batch([image, sketch, class_id, category,
                                image_name, color_text, text_vocab_indices],
                               batch_size=batch_size, capacity=capacity,
                               keep_input=True, num_threads=2)

    if one_hot:
        labels = map_class_id_to_labels(class_ids)
    else:
        labels = one_hot_to_dense(map_class_id_to_labels(class_ids))

    return images, sketches, labels, categories, image_names, color_texts, text_vocab_indiceses


def split_inputs(input_data, batch_size, batch_portion, num_gpu):
    input_data_list = []
    dim = len(input_data.get_shape())
    start = 0
    for i in range(num_gpu):
        idx = [start]
        size = [batch_size * batch_portion[i]]
        idx.extend([0] * (dim - 1))
        size.extend([-1] * (dim - 1))
        input_data_list.append(tf.slice(input_data, idx, size))

        start += batch_size * batch_portion[i]
    return input_data_list


def resize_and_padding_mask_image(image, new_size, resample_method=Image.ANTIALIAS, margin_size=10):
    """
    :param image: in Image format
    :param new_size: an integer
    :param resample_method: Image.NEAREST/BILINEAR/BICUBIC/ANTIALIAS/HAMMING/BOX
    :param margin_size: 0 for 'road', 10 for others
    :return:
    """
    width = image.width
    height = image.height

    height += margin_size * 2
    width += margin_size * 2
    # print('ori_size', height, width)

    scale = new_size / max(height, width)
    new_h = int(round(image.height * scale))
    new_w = int(round(image.width * scale))
    # print('scale', scale)
    # print('new_size', new_h, new_w)
    assert new_h <= new_size and new_w <= new_size

    if scale != 1:
        image = image.resize((new_w, new_h), resample=resample_method)

    img_np = np.array(image, dtype=np.uint8)[:, :, 0]
    top_pad = (new_size - new_h) // 2
    bottom_pad = new_size - new_h - top_pad
    left_pad = (new_size - new_w) // 2
    right_pad = new_size - new_w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad)]
    # print('padding', padding)
    rst_img = np.pad(img_np, padding, mode='constant', constant_values=255)
    # print('rst_img.shape', rst_img.shape)
    assert rst_img.shape[0] == new_size and rst_img.shape[1] == new_size

    rst_img3 = np.zeros([rst_img.shape[0], rst_img.shape[1], 3], dtype=np.uint8)
    for i in range(3):
        rst_img3[:, :, i] = rst_img

    return rst_img3


def thicken_drawings(image):
    """
    :param image: [H, W, 3], np.float32
    :return:
    """
    img = np.array(image[:, :, 0], dtype=np.uint8)

    img = 255 - img
    dilated_img = sm.dilation(img, sm.square(2))
    dilated_img = 255 - dilated_img  # [H, W]

    rst_img3 = np.zeros([dilated_img.shape[0], dilated_img.shape[1], 3], dtype=np.uint8)
    for i in range(3):
        rst_img3[:, :, i] = dilated_img

    return rst_img3