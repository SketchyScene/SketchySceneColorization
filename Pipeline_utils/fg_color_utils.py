import os
import re
import json
import collections
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf
from PIL import Image

import Instance_Matching.data_processing.sketch_data_processing as match_sketch_data_processing
import Instance_Matching.data_processing.text_processing as match_text_processing

import Foreground_Instance_Colorization.data_processing.text_processing as fgcolor_text_processing
from Foreground_Instance_Colorization.obj_lib.graph_single import build_single_graph
from Foreground_Instance_Colorization.obj_lib.input_pipeline import resize_and_padding_mask_image, thicken_drawings

skeId_carId_map = {
    7: 0, 9: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9, 22: 10, 23: 11, 27: 12, 28: 13,
    29: 14, 30: 15, 32: 16, 34: 17, 35: 18, 36: 19, 37: 20, 39: 21, 41: 22, 43: 23, 44: 24
}

ROAD_LABEL = 36
GRASS_LABEL = 27


def judging_preposition(text, j_word):
    """
    'a man has red shirt with blue pants'
    'a man with blue pants has red shirt'
    'a man in red shirt has blue pants'
    :return:
    """
    can_split = True
    prepositions = ['with',
                    # 'in'
                    ]

    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    words = SENTENCE_SPLIT_REGEX.split(text.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]

    for prep in prepositions:
        if prep in words:
            if words.index(prep) < words.index(j_word.lower()):
                can_split = False

    return can_split


def segment_user_input_text(user_text):
    """
    :param user_text: e.g. 'the bus on the left is yellow with blue windows'
    :return: pure_text: e.g. 'the bus is yellow with blue windows'
    """
    cate, _ = match_text_processing.search_for_self_category(user_text)

    if 'has' in user_text and judging_preposition(user_text, 'has'):
        split_idx = user_text.index('has')
    elif 'have' in user_text and judging_preposition(user_text, 'have'):
        split_idx = user_text.index('have')
    elif 'is' in user_text and judging_preposition(user_text, 'is'):
        split_idx = user_text.index('is')
    elif 'are' in user_text and judging_preposition(user_text, 'are'):
        split_idx = user_text.index('are')
    else:
        return user_text
    substr = user_text[split_idx:]
    pre_substr = user_text[:split_idx]

    if match_text_processing.search_for_color(pre_substr):
        return user_text
    elif match_text_processing.search_for_color(substr):
        pure_text = 'the ' + cate + ' ' + substr
        return pure_text
    else:
        return user_text


def is_road_not_single_line(road_sketch_, parallel_width=25):
    """
    determine whether the road is a single line
    :param road_sketch_: [H, W, 3], np.uint8
    :return:
    """
    ## first binarize the sketch
    road_sketch = road_sketch_.copy()
    road_sketch[(road_sketch >= 235).all(axis=2)] = [255, 255, 255]
    road_sketch[(road_sketch != 255).all(axis=2)] = [0, 0, 0]
    road_sketch = road_sketch[:, :, 0]  # [H, W], {0, 255}
    road_sketch[road_sketch == 0] = 1
    road_sketch[road_sketch == 255] = 0  # {0, 1}
    h = road_sketch.shape[0]
    w = road_sketch.shape[1]

    # plt.imshow(road_sketch_)  # astype(np.uint8)
    # plt.show()
    # plt.imshow(road_sketch)  # astype(np.uint8)
    # plt.show()

    ## 1. check the middel vertical line crossing
    road_sketch_vert = road_sketch.copy()
    vert_valid_width = 0
    for j in range(w):
        for i in range(h - 1):
            if road_sketch_vert[i + 1][j] == 1:
                road_sketch_vert[i][j] = 0

        cross_vert = np.sum(road_sketch_vert[:, j])

        if cross_vert > 0 and cross_vert % 2 == 0:
            vert_valid_width += 1

        if vert_valid_width >= parallel_width:
            return True

    ## 2. check the middel horizonal line crossing
    road_sketch_hori = road_sketch.copy()
    hori_valid_width = 0
    for j in range(h):
        for i in range(w - 1):
            if road_sketch_hori[j][i + 1] == 1:
                road_sketch_hori[j][i] = 0

        cross_hori = np.sum(road_sketch_hori[j, :])

        if cross_hori > 0 and cross_hori % 2 == 0:
            hori_valid_width += 1

        if hori_valid_width >= parallel_width:
            return True

    print('single_line')
    return False


def reverse_resize_image(cartoon_instance, box_h, box_w, h_w_ratio=1, margin_size=10):
    """
    cut padding, scale to box size
    :param cartoon_instance: [192, 192, 3]
    :param margin_size: 0 for 'road', 10 for others
    :return:
    """
    ori_size = cartoon_instance.shape[0]
    box_h_marg = box_h + margin_size * 2
    box_w_marg = box_w + margin_size * 2
    # 1. cut padding
    if box_h_marg * h_w_ratio > box_w_marg:
        pad_size = ori_size * (box_h_marg * h_w_ratio - box_w_marg) / (box_h_marg * h_w_ratio) / 2.
        pad_size = int(round(pad_size))
        cartoon_instance_cut = cartoon_instance[:, pad_size: ori_size - pad_size]
    else:
        pad_size = ori_size * (box_w_marg - box_h_marg * h_w_ratio) / box_w_marg / 2.
        pad_size = int(round(pad_size))
        cartoon_instance_cut = cartoon_instance[pad_size: ori_size - pad_size, :]

    # 2. scale to box size
    cartoon_instance_rev = scipy.misc.imresize(cartoon_instance_cut, (box_h_marg, box_w_marg))

    # 3. cut the margin (10 pixels each side)
    cartoon_instance_rev = cartoon_instance_rev[margin_size: margin_size + box_h, margin_size: margin_size + box_w]

    return cartoon_instance_rev


def instance_result_postprocessing(generated_img, bbox, data_format, class_id46):
    """
    :param generated_img: [1, 3, H, W]
    :return:
    """
    if data_format == 'NCHW':
        assert generated_img.shape[1] == 3
        generated_img = np.transpose(generated_img, (0, 2, 3, 1))

    generated_img = ((generated_img + 1) / 2.) * 255
    generated_img = generated_img[:, :, :, :].astype(np.uint8)  # [1, INSTANCE_SIZE, INSTANCE_SIZE, 3]

    ## crop and resize color instance to box size
    color_instance = generated_img[0]
    bbox_h = bbox[2] - bbox[0]
    bbox_w = bbox[3] - bbox[1]
    margin_size = 0 if class_id46 == ROAD_LABEL else 10
    color_instance_post = reverse_resize_image(color_instance, bbox_h, bbox_w, margin_size=margin_size)

    return color_instance_post


def build_instance_colorization(data_base_dir, image_id, input_text, inst_indices, sketch_path,
                                inner_masks_mat_path, segm_data_npz_path, results_base_dir,
                                fgcolor_vocab_size, fgcolor_max_len, fgcolor_vocab_path, fgcolor_snapshot_root,
                                new_result_image_name, last_result_image_name):
    """
    instance colorization and update records
    :param image_id: int
    :param input_text: e.g. 'the bus on the left is yellow with blue windows'
    :param inst_indices: list of target inst_idx
    :return:
    """
    assert type(inst_indices) is list

    ## fixed params
    batch_size = 1
    LSTM_hybrid = True
    data_format = 'NCHW'
    INSTANCE_SIZE = 192
    IMAGE_SIZE = 768

    inst_color_vocab_dict = fgcolor_text_processing.load_vocab_dict_from_file(fgcolor_vocab_path)

    categories46 = []
    color_map_mat_path = os.path.join(data_base_dir, 'colorMapC46.mat')
    colorMap = scipy.io.loadmat(color_map_mat_path)['colorMap']
    for i in range(46):
        cat_name = colorMap[i][0][0]
        categories46.append(cat_name)

    ## 0. load common things
    sketch_image = match_sketch_data_processing.load_image2(sketch_path)  # [768, 768, 3], float32
    sketch_image = np.array(sketch_image, dtype=np.uint8)  # [768, 768, 3], uint8

    inner_mask = scipy.io.loadmat(inner_masks_mat_path)['inner_masks']  # [768, 768]

    ## 1. look for records and find the last result
    results_dir = os.path.join(results_base_dir, 'results', str(image_id))
    os.makedirs(results_dir, exist_ok=True)

    ## empty records: no images, no json
    if last_result_image_name == '':
        base_image = sketch_image.copy()
    else:
        last_result_image_path = os.path.join(results_dir, last_result_image_name)
        base_image = Image.open(last_result_image_path).convert('RGB')
        base_image = np.array(base_image, dtype=np.uint8)

    new_result_image = base_image.copy()

    ## 2. post-process input_text and instance colorization
    ## 2.1 read and process instance data
    npz = np.load(segm_data_npz_path)
    pred_class_ids = np.array(npz['pred_class_ids'], dtype=np.int32)  # [N], of the 46 ids
    pred_boxes = np.array(npz['pred_boxes'], dtype=np.int32)  # [N, 4]
    pred_masks_s = npz['pred_masks']
    pred_masks = match_sketch_data_processing.expand_small_segmentation_mask(pred_masks_s, pred_boxes)  # [N, H, W]

    ## find out grass ids
    grass_inst_idx_list = []
    for i in range(len(pred_class_ids)):
        if pred_class_ids[i] == GRASS_LABEL:
            grass_inst_idx_list.append(i)

    inst_color_text = segment_user_input_text(input_text)
    print('## segment_user_input_text: ', inst_color_text)

    input_images = tf.placeholder(tf.float32, shape=[1, 3, INSTANCE_SIZE, INSTANCE_SIZE])  # [1, 3, H, W]
    class_ids = tf.placeholder(tf.int32, shape=(1,))  # (1, )
    text_vocab_indiceses = tf.placeholder(tf.int32, shape=[1, fgcolor_max_len])  # [1, 15]

    ret_list = build_single_graph(input_images, input_images, None,
                                  class_ids, None,
                                  text_vocab_indiceses,
                                  batch_size=batch_size, training=False,
                                  LSTM_hybrid=LSTM_hybrid,
                                  vocab_size=fgcolor_vocab_size,
                                  data_format=data_format,
                                  distance_map=False)  # [image_gens, images, sketches]

    load_var = {var.op.name: var for var in tf.global_variables()
                if not var.op.name.startswith('ResNet')
                and not var.op.name.startswith('text_sketchyscene')
                }
    snapshot_loader = tf.train.Saver(load_var)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('Restore trained model:', tf.train.latest_checkpoint(fgcolor_snapshot_root))
        snapshot_loader.restore(sess, tf.train.latest_checkpoint(fgcolor_snapshot_root))

        for inst_idx in inst_indices:
            class_id46 = pred_class_ids[inst_idx]
            inst_bbox = pred_boxes[inst_idx]  # (4, )
            y1, x1, y2, x2 = inst_bbox
            inst_mask768 = pred_masks[inst_idx]  # (768, 768), {0, 1}

            if class_id46 not in skeId_carId_map:
                raise Exception('Wrong matching instance: %s' % categories46[class_id46])

            ## 2.2 Crop, resize and pad the instance sketch
            inst_mask = inst_mask768[y1: y2, x1: x2]
            inst_mask_img = np.zeros([inst_mask.shape[0], inst_mask.shape[1], 3], dtype=np.uint8)
            inst_mask_img.fill(255)
            inst_mask_img[inst_mask == 1] = [0, 0, 0]
            inst_mask_img = Image.fromarray(inst_mask_img, 'RGB')

            if inst_mask_img.width != INSTANCE_SIZE or inst_mask_img.height != INSTANCE_SIZE:
                margin_size = 0 if class_id46 == ROAD_LABEL else 10
                instance_sketch = resize_and_padding_mask_image(inst_mask_img, INSTANCE_SIZE, margin_size=margin_size)
            else:
                instance_sketch = np.array(inst_mask_img, dtype=np.uint8)  # shape = [H, W, 3]

            assert instance_sketch.shape[0] == INSTANCE_SIZE and instance_sketch.shape[1] == INSTANCE_SIZE

            if class_id46 == ROAD_LABEL:
                if not is_road_not_single_line(instance_sketch.copy()):
                    raise Exception('Road is single line')

            if class_id46 == GRASS_LABEL:
                instance_sketch = thicken_drawings(instance_sketch).astype(np.float32)

            instance_sketch = instance_sketch.astype(np.float32)
            # Normalization
            instance_sketch = instance_sketch / 255.
            instance_sketch = instance_sketch * 2. - 1

            instance_sketch = np.expand_dims(instance_sketch, axis=0)  # shape = [1, H, W, 3]
            instance_sketch = np.transpose(instance_sketch, [0, 3, 1, 2])  # shape = [1, 3, H, W]

            class_id = skeId_carId_map[class_id46]
            class_id = np.array([class_id])

            vocab_indices = fgcolor_text_processing.preprocess_sentence(inst_color_text,
                                                                        inst_color_vocab_dict, fgcolor_max_len)  # list
            vocab_indices = np.array(vocab_indices, dtype=np.int32)
            vocab_indices = np.expand_dims(vocab_indices, axis=0)  # shape = [1, 15]

            try:
                generated_img, _, _ = sess.run([ret_list[0], ret_list[1], ret_list[2]],
                                               feed_dict={input_images: instance_sketch,
                                                          class_ids: class_id,
                                                          text_vocab_indiceses: vocab_indices})
            except Exception as e:
                print('Exception:', e)
                return

            ## 2.3 crop and resize to original size as bbox: [box_h, box_w, 3]
            color_instance_post = instance_result_postprocessing(generated_img, inst_bbox, data_format, class_id46)

            ## 3. cover the instance to the last result, save and update the records
            new_result_box = new_result_image[y1: y2, x1: x2]
            inner_mask_box = inner_mask[y1: y2, x1: x2]
            new_result_box[inner_mask_box == inst_idx + 1] = color_instance_post[inner_mask_box == inst_idx + 1]
            new_result_image[y1: y2, x1: x2] = new_result_box

    ## 4.1 remove grass from inner mask (no need to cover grass drawing to color grass)
    inner_mask_no_grass = np.zeros(inner_mask.shape, dtype=np.int32)
    for i in range(len(grass_inst_idx_list)):
        grass_inst_idx = grass_inst_idx_list[i]
        inner_mask_no_grass[inner_mask == grass_inst_idx + 1] = 1

    ## 4.2 cover the sketch drawings to new result
    sketch_scene_image_moved = sketch_image.copy()
    sketch_scene_image_moved[1: IMAGE_SIZE, 1: IMAGE_SIZE] = sketch_image[0: IMAGE_SIZE - 1, 0: IMAGE_SIZE - 1]

    drawings_region = np.logical_and(sketch_scene_image_moved[:, :, 0] == 0, inner_mask_no_grass != 1)
    new_result_image[drawings_region] = sketch_scene_image_moved[drawings_region]

    ## 5. save new image result
    new_result_image = Image.fromarray(new_result_image, 'RGB')
    new_result_image_path = os.path.join(results_dir, new_result_image_name)
    new_result_image.save(new_result_image_path, 'PNG')
