import os
import re
import io
import json
import collections
import numpy as np
import scipy.io
import skimage
import skimage.color
import tensorflow as tf
from PIL import Image

import Background_Colorization.data_processing.text_processing as bg_text_processing
from Background_Colorization.data_processing.image_processing import load_region_mask
from Background_Colorization.bg_colorization_main import preprocess_examples, create_model, deprocess

GRASS_LABEL = 27
IMAGE_SIZE = 768

input_text_types = ['None', 'ground', 'sky', 'both']
ALL_COLOR = ['blue', 'green', 'cyan', 'red', 'orange', 'yellow', 'brown', 'purple', 'pink', 'black', 'gray']


def get_text_type(text):
    input_text_type_label = [0, 0]  # first for 'sky' and second for 'ground'
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    words = SENTENCE_SPLIT_REGEX.split(text.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]

    if 'sky' in words:
        input_text_type_label[0] = 1
    if 'ground' in words or 'floor' in words or 'land' in words:
        input_text_type_label[1] = 1

    type_idx = 2 * input_text_type_label[0] + input_text_type_label[1]
    text_type = input_text_types[type_idx]
    return text_type


def check_duplicated_color(text):
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    words = SENTENCE_SPLIT_REGEX.split(text.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]

    sky_color = ''
    ground_color = ''
    for word in words:
        if word in ALL_COLOR:
            if sky_color == '':
                sky_color = word
            else:
                ground_color = word
                break

    if sky_color == ground_color:
        raise Exception('It is not recommended to use the same sky and ground color.')


def combine_bg_input_text(new_text, previous_text):
    input_text_type = get_text_type(new_text)
    previous_text_type = get_text_type(previous_text)

    assert input_text_type != 'None'

    rst_text = ''

    if input_text_type == 'both':
        rst_text = new_text
    elif input_text_type == 'None':
        raise Exception('Input text contains no information.')

    elif input_text_type == 'sky':
        if previous_text_type == 'None' or previous_text_type == 'sky':
            raise Exception('No ground infomation provided and found in records.')
        elif previous_text_type == 'ground':  # 'the ground is black'
            rst_text = new_text + ' and ' + previous_text
        else:  # previous_text == 'the sky is blue and the ground is black'
            split_idx = previous_text.index('and')
            rst_text = new_text + ' ' + previous_text[split_idx:]

    else:  # input_text_type == 'ground'
        if previous_text_type == 'None' or previous_text_type == 'ground':
            raise Exception('No sky infomation provided and found in records.')
        elif previous_text_type == 'sky':  # 'the sky is black'
            rst_text = previous_text + ' and ' + new_text
        else:  # previous_text == 'the sky is blue and the ground is black'
            split_idx = previous_text.index('and')
            rst_text = previous_text[:split_idx] + 'and ' + new_text

    assert rst_text != ''
    check_duplicated_color(rst_text)

    return rst_text


def add_color_gradient(color_image, inner_mask, search_height=2, search_from=5):
    """
    :param color_image: [H, W, 3]
    :param inner_mask: [H, W]
    :param search_height: the height for searching for sky color
    :param search_from: the start row for searching for sky color
    :return:
    """
    img_h = color_image.shape[0]
    img_w = color_image.shape[1]

    img_bg = np.zeros(color_image.shape, dtype=np.uint8)
    img_bg.fill(255)
    img_bg[inner_mask == 0] = color_image[inner_mask == 0]

    ## 1. find out sky color: the most RGB in first two row (non-fg region)
    colors_container = []
    colors_count = []
    for i in range(search_height):
        for j in range(img_w):
            if inner_mask[i + search_from][j] == 0:
                rgb = img_bg[i + search_from][j].tolist()
                if rgb not in colors_container:
                    colors_container.append(rgb)
                    colors_count.append(1)
                else:
                    rgb_idx = colors_container.index(rgb)
                    colors_count[rgb_idx] = colors_count[rgb_idx] + 1

    sky_color = colors_container[int(np.argmax(colors_count))]
    # print('sky_color', sky_color)

    ## 2. search for the bottom of sky
    sky_bottom = -1
    for i in range(int(img_h / 2), -1, -1):
        row_pixels = img_bg[i, :].tolist()
        if sky_color in row_pixels:
            sky_bottom = i
            break

    assert sky_bottom != -1
    # print('sky_bottom', sky_bottom)

    ## 3. color gradient: use HSV color space
    start_height = int(sky_bottom / 4 * 3)

    sky_color = np.array(sky_color, dtype=np.float32)
    sky_color_hsv = skimage.color.rgb2hsv(np.expand_dims(np.expand_dims(sky_color / 255., 0), 0))[0][0]
    # print('sky_color_hsv', sky_color_hsv * 255.)

    img_bg_grad_hsv = skimage.color.rgb2hsv(img_bg / 255.)
    # print(img_bg_grad_hsv.shape)

    end_color_s = sky_color_hsv[1] / 3.
    end_color_v = min(1., sky_color_hsv[2] * 1.5)

    for i in range(start_height, -1, -1):
        height_s = (start_height - i) / start_height * end_color_s + i / start_height * sky_color_hsv[1]
        height_v = (start_height - i) / start_height * end_color_v + i / start_height * sky_color_hsv[2]
        # print(height_v * 255.)
        img_bg_grad_hsv[i, :, 1] = height_s
        img_bg_grad_hsv[i, :, 2] = height_v

    img_bg_grad = skimage.color.hsv2rgb(img_bg_grad_hsv)
    img_bg_grad *= 255.
    img_bg_grad = np.array(img_bg_grad, dtype=np.uint8)

    ## 4. cover original FG
    img_bg_grad[inner_mask != 0] = color_image[inner_mask != 0]

    return img_bg_grad


def build_background_colorization(image_id, input_text, sketch_path,
                                  inner_masks_mat_path, segm_data_npz_path, results_base_dir,
                                  bg_vocab_size, bg_max_len, bg_vocab_path, bg_snapshot_root,
                                  new_result_image_name, last_result_image_name, last_bg_text,
                                  color_gradient=True):
    """
        background colorization and update records
        :param image_id: int
        :param input_text: e.g. 'the sky is blue and the ground is green'
        :param color_gradient: True to add color_gradient
        :return:
        """
    ## fixed params
    ndf = 64
    ngf = 64
    gan_weight = 1.0
    l1_weight = 100.0
    seg_weight = 100.0
    lr = 0.0002
    max_steps = 100000

    vocab_dict = bg_text_processing.load_vocab_dict_from_file(bg_vocab_path)

    sketch_scene_image = Image.open(sketch_path).convert("RGB")
    sketch_scene_image = sketch_scene_image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST)
    sketch_scene_image = np.array(sketch_scene_image, dtype=np.uint8)  # shape = [H, W, 3]

    ## 1. look for records and find the last result
    results_dir = os.path.join(results_base_dir, 'results', str(image_id))
    os.makedirs(results_dir, exist_ok=True)

    ## empty records: no images, no json
    if last_result_image_name == '':
        assert last_bg_text == ""
        last_bg_text = "the sky is blue and the ground is green"
        previous_bg_image = sketch_scene_image.copy()
    else:
        last_result_image_path = os.path.join(results_dir, last_result_image_name)
        previous_bg_image = Image.open(last_result_image_path).convert('RGB')
        previous_bg_image = np.array(previous_bg_image, dtype=np.uint8)

    ## find out grass ids
    grass_inst_idx_list = []

    npz = np.load(segm_data_npz_path)
    pred_class_ids = npz['pred_class_ids']  # [N], of the 46 ids
    for i in range(len(pred_class_ids)):
        if pred_class_ids[i] == GRASS_LABEL:
            grass_inst_idx_list.append(i)

    ## 2. crop FG by inner mask
    inner_mask = scipy.io.loadmat(inner_masks_mat_path)['inner_masks']  # [768, 768]

    fg_image = np.zeros(previous_bg_image.shape, dtype=np.uint8)
    fg_image.fill(255)
    fg_image[inner_mask != 0] = previous_bg_image[inner_mask != 0]  # [H, W, 3], uint8

    fg_image_temp = fg_image.copy()

    ## 3. pre-process input text
    proc_input_text = combine_bg_input_text(input_text, last_bg_text)
    print('proc_input_text:', proc_input_text)

    ## 4. set up model and do background colorization
    input_images = tf.placeholder(tf.uint8, shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3])  # [1, H, W, 3], [0-255], uint8
    text_vocab_indiceses = tf.placeholder(tf.int32, shape=[1, bg_max_len])  # [1, T]
    region_labels = tf.placeholder(tf.int32, shape=[1, IMAGE_SIZE, IMAGE_SIZE])

    # [1, H, W, 3], [-1., 1.], float32
    input_images_pro, target_images_pro = preprocess_examples(input_images, input_images)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(input_images_pro, target_images_pro, text_vocab_indiceses, region_labels, "test",
                         ndf, ngf, gan_weight, l1_weight, seg_weight, lr, max_steps, bg_vocab_size)

    # [-1., 1.] -> [0., 1.], float32
    inputs_depro = deprocess(input_images_pro)
    outputs_depro = deprocess(model.outputs)

    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # [0., 1.] -> [0, 255], uint8
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs_depro)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs_depro)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=5)

    sv = tf.train.Supervisor(logdir=None, save_summaries_secs=0, saver=None)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with sv.managed_session(config=tfconfig) as sess:
        print("parameter_count =", sess.run(parameter_count))

        checkpoint = tf.train.latest_checkpoint(bg_snapshot_root)
        print("loading model from checkpoint", checkpoint)
        saver.restore(sess, checkpoint)

        ## 4.1 read and process instance data
        fg_data = np.expand_dims(fg_image, axis=0)  # [1, H, W, 3], uint8, [0-255]

        vocab_indices = bg_text_processing.preprocess_sentence(proc_input_text, vocab_dict, bg_max_len)  # list
        vocab_indices = np.array(vocab_indices, dtype=np.int32)
        vocab_indices = np.expand_dims(vocab_indices, axis=0)  # shape = [1, T]

        results = sess.run(display_fetches, feed_dict={input_images: fg_data,
                                                       text_vocab_indiceses: vocab_indices,
                                                       region_labels: load_region_mask('', IMAGE_SIZE, is_test=True)})

        background_image = results["outputs"][0]
        background_image = Image.open(io.BytesIO(background_image))
        background_image = np.array(background_image, dtype=np.uint8)

        ## 4.2 post-processing: cover the result with FG with the help of inner mask
        assert inner_mask.shape[0] == fg_data.shape[1] and inner_mask.shape[1] == fg_data.shape[2]
        background_image[inner_mask != 0] = np.squeeze(fg_data)[inner_mask != 0]

        ## 4.3 remove grass from inner mask
        inner_mask_no_grass = np.zeros(inner_mask.shape, dtype=np.int32)
        for i in range(len(grass_inst_idx_list)):
            grass_inst_idx = grass_inst_idx_list[i]
            inner_mask_no_grass[inner_mask == grass_inst_idx + 1] = 1

        ## 4.4 post-processing: cover the result with sketch drawings
        sketch_scene_image_moved = sketch_scene_image.copy()
        sketch_scene_image_moved[1: IMAGE_SIZE, 1: IMAGE_SIZE] = \
            sketch_scene_image[0: IMAGE_SIZE - 1, 0: IMAGE_SIZE - 1]

        drawings_region = np.logical_and(sketch_scene_image_moved[:, :, 0] == 0, inner_mask_no_grass != 1)
        background_image[drawings_region] = sketch_scene_image_moved[drawings_region]

        fg_image_temp[drawings_region] = sketch_scene_image_moved[drawings_region]
        fg_image_temp = Image.fromarray(fg_image_temp, 'RGB')
        fg_image_temp.save(os.path.join(results_dir, str(image_id) + '_fg.png'), 'PNG')

        ## 4.5 add color gradient
        if color_gradient:
            background_image = add_color_gradient(background_image, inner_mask)
            background_image[drawings_region] = sketch_scene_image_moved[drawings_region]

        background_image = Image.fromarray(background_image, 'RGB')
        background_image_save_path = os.path.join(results_dir, new_result_image_name)
        background_image.save(background_image_save_path, 'PNG')

    return proc_input_text
