import os
from time import time
import pickle

import cv2
import numpy as np
from PIL import Image
import scipy.misc
import json
import sys
import tensorflow as tf
from tensorflow.python.client import timeline

from graph_single import build_multi_tower_graph, build_single_graph
from input_pipeline import build_input_queue_paired, build_input_queue_paired_test, resize_and_padding_mask_image, \
    thicken_drawings
from text_processing import preprocess_sentence, load_vocab_dict_from_file
from config import Config

tf.logging.set_verbosity(tf.logging.INFO)
inception_v4_ckpt_path = 'model/inception-cartoon'


def log(name, arr):
    print(name, ', ', arr.shape, ', max:', np.max(arr), ', min:', np.min(arr))


def print_parameter_count(verbose=False):
    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        if verbose and len(shape) > 1:
            print(shape)
            print(variable_parametes)
        total_parameters += variable_parametes
    print('generator')
    print('total_parameters', total_parameters)

    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'):
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            # print(dim)
            variable_parametes *= dim.value
        if verbose and len(shape) > 1:
            print(shape)
            print(variable_parametes)
        total_parameters += variable_parametes
    print('discriminator')
    print('total_parameters', total_parameters)


def train(**kwargs):
    status = 0

    # Roll out the parameters
    appendix = Config.resume_from
    batch_size = Config.batch_size
    max_iter_step = Config.max_iter_step
    Diters = Config.disc_iterations
    ld = Config.ld
    optimizer = Config.optimizer
    lr_G = Config.lr_G
    lr_D = Config.lr_D
    num_gpu = Config.num_gpu
    log_dir = Config.log_dir
    ckpt_dir = Config.ckpt_dir
    data_format = Config.data_format
    distance_map = Config.distance_map
    small_img = Config.small_img
    LSTM_hybrid = Config.LSTM_hybrid
    block_type = Config.block_type
    summary_write_freq = Config.summary_write_freq
    save_model_freq = Config.save_model_freq
    count_left_time_freq = Config.count_left_time_freq
    # count_inception_score_freq = Config.count_inception_score_freq
    vocab_size = Config.vocab_size

    distance_map = distance_map != 0
    small = small_img != 0
    LSTM_hybrid = LSTM_hybrid != 0
    batch_portion = np.array([1, 1, 1, 1], dtype=np.int32)

    iter_from = kwargs['iter_from']

    # Time counter
    prev_time = float("-inf")
    curr_time = float("-inf")

    tf.reset_default_graph()
    print('Iteration starts from: %d' % iter_from)

    # assert inception_score.softmax.graph != tf.get_default_graph()
    # inception_score._init_inception()

    counter = tf.Variable(initial_value=iter_from, dtype=tf.int32, trainable=False)
    counter_addition_op = tf.assign_add(counter, 1, use_locking=True)

    # Construct data queue
    with tf.device('/cpu:0'):
        images, sketches, image_paired_class_ids, _, _, _, text_vocab_indiceses = build_input_queue_paired(
            mode='train',
            batch_size=batch_size * num_gpu,
            data_format=data_format,
            distance_map=distance_map,
            small=small, capacity=2 ** 11)  # images/sketches [2, 3, H, W], text_vocab_indiceses [2, 15]
    with tf.device('/cpu:0'):
        images_d, _, image_paired_class_ids_d, _, _, _, _ = build_input_queue_paired(
            mode='train',
            batch_size=batch_size * num_gpu,
            data_format=data_format,
            distance_map=distance_map,
            small=small, capacity=2 ** 11)  # [2, 3, H, W]

    opt_g, opt_d, loss_g, loss_d, merged_all = build_multi_tower_graph(
        images, sketches, images_d,
        image_paired_class_ids, image_paired_class_ids_d,
        text_vocab_indiceses,
        LSTM_hybrid=LSTM_hybrid,
        vocab_size=vocab_size,
        batch_size=batch_size, num_gpu=num_gpu, batch_portion=batch_portion, training=True,
        learning_rates={
            "generator": lr_G,
            "discriminator": lr_D,
        },
        counter=counter, max_iter_step=max_iter_step,
        ld=ld, data_format=data_format,
        distance_map=distance_map,
        optimizer=optimizer,
        block_type=block_type)

    saver = tf.train.Saver(max_to_keep=100)
    # try:
    #     inception_loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionV4'))
    #     perceptual_model_checkpoint_path = inception_v4_ckpt_path
    #     perceptual_model_path = tf.train.latest_checkpoint(perceptual_model_checkpoint_path)
    # except:
    #     inception_loader = None

    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # if inception_loader is not None:
        #     print('Restore:', perceptual_model_path)
        #     inception_loader.restore(sess, perceptual_model_path)

        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        if iter_from > 0:
            snapshot_loader = tf.train.Saver()
            print('Restore:', tf.train.latest_checkpoint(ckpt_dir))
            snapshot_loader.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            summary_writer.reopen()

        run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
        run_metadata = tf.RunMetadata()

        print_parameter_count(verbose=False)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        sess.run([counter.assign(iter_from)])

        for i in range(iter_from, max_iter_step):
            if status == -1:
                break

            ## count left time
            if i % count_left_time_freq == 0:
                curr_time = time()
                elapsed = curr_time - prev_time
                print("Now at iteration %d. Elapsed time: %.5fs. Average time: %.5fs/iter"
                      % (i, elapsed, elapsed / 100.))

                if elapsed != float("inf"):
                    left_iter = max_iter_step - i
                    left_sec = left_iter * (elapsed / 100.)
                    left_day = int(left_sec / 24 / 60 / 60)
                    left_hour = int((left_sec - (24 * 60 * 60) * left_day) / 60 / 60)
                    left_min = int((left_sec - (24 * 60 * 60) * left_day - (60 * 60) * left_hour) / 60)
                    print("Left time:%dd %dh %dm" % (left_day, left_hour, left_min))

                prev_time = curr_time

            diters = Diters

            # Train Discriminator
            for j in range(diters):
                ## summary
                if i % summary_write_freq == 0 and j == 0:
                    _, merged, loss_d_out = sess.run([opt_d, merged_all, loss_d],
                                                     options=run_options,
                                                     run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.flush()
                else:
                    _, loss_d_out = sess.run([opt_d, loss_d])
                # print('loss_d', loss_d_out)
                if np.isnan(np.sum(loss_d_out)):
                    status = -1
                    print("NaN occurred during training D")
                    return status

            # Train Generator
            if i % summary_write_freq == 0:
                _, merged, loss_g_out, counter_out, _ = sess.run(
                    [opt_g, merged_all, loss_g, counter, counter_addition_op],
                    options=run_options,
                    run_metadata=run_metadata)
                summary_writer.add_summary(merged, i)
                summary_writer.flush()
            else:
                _, loss_g_out, counter_out, _ = sess.run([opt_g, loss_g, counter, counter_addition_op])
            # print('loss_g', loss_g_out)
            if np.isnan(np.sum(loss_g_out)):
                status = -1
                print("NaN occurred during training G")
                return status

            ## save model
            if i % save_model_freq == save_model_freq - 1:
                saver.save(sess, os.path.join(ckpt_dir, 'model_{}.ckpt'.format(i)), global_step=i)
                print('Save model_{}.ckpt'.format(i))

        coord.request_stop()
        coord.join(threads)

    return status


def validation(**kwargs):
    # Roll out the parameters
    dataset_type = Config.dataset_type
    batch_size = Config.batch_size
    ckpt_dir = Config.ckpt_dir
    results_dir = Config.results_dir
    data_format = Config.data_format
    distance_map = Config.distance_map
    small_img = Config.small_img
    LSTM_hybrid = Config.LSTM_hybrid
    block_type = Config.block_type
    vocab_size = Config.vocab_size

    channel = 3
    distance_map = distance_map != 0
    small = small_img != 0
    LSTM_hybrid = LSTM_hybrid != 0

    if LSTM_hybrid:
        output_folder = os.path.join(results_dir, 'with_text')
    else:
        output_folder = os.path.join(results_dir, 'without_text')
    print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Time counter
    prev_time = float("-inf")
    curr_time = float("-inf")
    # Construct data queue
    with tf.device('/cpu:0'):
        images, sketches, class_ids, categories, image_names, color_texts, text_vocab_indiceses \
            = build_input_queue_paired_test(
            mode=dataset_type,
            batch_size=batch_size, data_format=data_format,
            distance_map=distance_map, small=small, capacity=512)  # [2, 3, H, W]

    ret_list = build_single_graph(images, sketches, None,
                                  class_ids, None,
                                  text_vocab_indiceses,
                                  batch_size=batch_size, training=False,
                                  LSTM_hybrid=LSTM_hybrid,
                                  vocab_size=vocab_size,
                                  data_format=data_format,
                                  distance_map=distance_map,
                                  block_type=block_type)  # [image_gens, images, sketches]

    snapshot_loader = tf.train.Saver()

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('Restore trained model:', tf.train.latest_checkpoint(ckpt_dir))
        snapshot_loader.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
        counter = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        while True:
            try:
                generated_img, gt_image, input_sketch, category, image_name = sess.run(
                    [ret_list[0], ret_list[1], ret_list[2], categories, image_names])
            except Exception as e:
                print(e.args)
                break

            if counter % 100 == 0:
                curr_time = time()
                elapsed = curr_time - prev_time
                print(
                    "Now at iteration %d. Elapsed time: %.5fs." % (counter, elapsed))
                prev_time = curr_time

            if data_format == 'NCHW':
                generated_img = np.transpose(generated_img, (0, 2, 3, 1))
                gt_image = np.transpose(gt_image, (0, 2, 3, 1))
                input_sketch = np.transpose(input_sketch, (0, 2, 3, 1))

            # log('before, generated_img', generated_img)
            # log('before, gt_image', gt_image)
            # log('before, input_sketch', input_sketch)
            generated_img = ((generated_img + 1) / 2.) * 255
            gt_image = ((gt_image + 1) / 2.) * 255
            input_sketch = ((input_sketch + 1) / 2.) * 255
            generated_img = generated_img[:, :, :, ::-1].astype(np.uint8)
            gt_image = gt_image[:, :, :, ::-1].astype(np.uint8)
            input_sketch = input_sketch.astype(np.uint8)
            # log('after, generated_img', generated_img)
            # log('after, gt_image', gt_image)  # (2, H, W, 3)
            # log('after, input_sketch', input_sketch)

            for i in range(batch_size):
                this_prefix = '%s' % (category[i].decode('ascii'))
                img_out_filename = this_prefix + '_' + image_name[i].decode()[:-4] + '_output.png'
                img_gt_filename = this_prefix + '_' + image_name[i].decode()[:-4] + '_target.png'
                sketch_in_filename = this_prefix + '_' + image_name[i].decode()[:-4] + '_input.png'

                # Save file
                # file_path = os.path.join(output_folder, 'output_%d.jpg' % int(counter / batch_size))
                cv2.imwrite(os.path.join(output_folder, img_out_filename), generated_img[i])
                cv2.imwrite(os.path.join(output_folder, img_gt_filename), gt_image[i])
                cv2.imwrite(os.path.join(output_folder, sketch_in_filename), input_sketch[i])
                # output_img = np.zeros((img_dim * 2, img_dim * batch_size, channel))

                print('Saved file %s' % this_prefix)

            counter += 1

        coord.request_stop()
        coord.join(threads)


def test():
    SIZE = {True: (64, 64),
            False: (192, 192)}
    T = 15  # the longest length of text
    vocab_file = 'data/vocab.txt'
    test_data_base_dir = 'data'

    captions_base_dir = os.path.join(test_data_base_dir, 'captions')
    images_base_dir = os.path.join(test_data_base_dir, 'images')
    categories = os.listdir(captions_base_dir)
    categories.sort()
    print(categories)

    # Roll out the parameters
    batch_size = 1
    ckpt_dir = Config.ckpt_dir
    results_dir = Config.results_dir
    data_format = Config.data_format
    distance_map = Config.distance_map
    small_img = Config.small_img
    LSTM_hybrid = Config.LSTM_hybrid
    block_type = Config.block_type
    vocab_size = Config.vocab_size

    distance_map = distance_map != 0
    small = small_img != 0
    LSTM_hybrid = LSTM_hybrid != 0

    img_dim = SIZE[small]

    output_folder = results_dir
    os.makedirs(output_folder, exist_ok=True)

    vocab_dict = load_vocab_dict_from_file(vocab_file)

    input_images = tf.placeholder(tf.float32, shape=[1, 3, img_dim[0], img_dim[1]])  # [1, 3, H, W]
    class_ids = tf.placeholder(tf.int32, shape=(1,))  # (1, )
    text_vocab_indiceses = tf.placeholder(tf.int32, shape=[1, 15])  # [1, 15]

    ret_list = build_single_graph(input_images, input_images, None,
                                  class_ids, None,
                                  text_vocab_indiceses,
                                  batch_size=batch_size, training=False,
                                  LSTM_hybrid=LSTM_hybrid,
                                  vocab_size=vocab_size,
                                  data_format=data_format,
                                  distance_map=distance_map,
                                  block_type=block_type)  # [image_gens, images, sketches]

    snapshot_loader = tf.train.Saver()

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('Restore trained model:', tf.train.latest_checkpoint(ckpt_dir))
        snapshot_loader.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

        for cate in categories:
            testing_json = os.path.join(captions_base_dir, cate, 'test.json')
            fp = open(testing_json, "r")
            json_data = fp.read()
            json_data = json.loads(json_data)
            print(len(json_data), 'inference datas')

            for i in range(len(json_data)):
                input_name = json_data[i]['key']  # e.g. '228_1.png'
                input_text = json_data[i]['color_text']  # e.g. 'A yellow bus with blue window'

                sketch_path = os.path.join(images_base_dir, cate, 'sketch', input_name)
                sketch_image = Image.open(sketch_path)
                sketch_image = sketch_image.convert("RGB")
                # Resize
                if sketch_image.width != img_dim[0] or sketch_image.height != img_dim[1]:
                    margin_size = 0 if cate in ['road'] else 10
                    sketch_image = resize_and_padding_mask_image(sketch_image, img_dim[0],
                                                                 margin_size=margin_size).astype(np.float32)
                else:
                    sketch_image = np.array(sketch_image, dtype=np.float32)  # shape = [H, W, 3]

                if cate in ['house', 'road']:
                    sketch_image = thicken_drawings(sketch_image).astype(np.float32)  # shape = [H, W, 3]

                # Normalization
                sketch_image = sketch_image / 255.
                sketch_image = sketch_image * 2. - 1

                sketch_image = np.expand_dims(sketch_image, axis=0)  # shape = [1, H, W, 3]
                sketch_image = np.transpose(sketch_image, [0, 3, 1, 2])  # shape = [1, 3, H, W]

                class_id = categories.index(cate)
                class_id = np.array([class_id])

                vocab_indices = preprocess_sentence(input_text, vocab_dict, T)  # list
                vocab_indices = np.array(vocab_indices, dtype=np.int32)
                vocab_indices = np.expand_dims(vocab_indices, axis=0)  # shape = [1, 15]

                try:
                    # print('class_id', class_id)
                    # print('vocab_indices', vocab_indices)
                    generated_img, _, input_sketch = sess.run(
                        [ret_list[0], ret_list[1], ret_list[2]],
                        feed_dict={input_images: sketch_image,
                                   class_ids: class_id,
                                   text_vocab_indiceses: vocab_indices})
                except Exception as e:
                    print(e.args)
                    break

                if data_format == 'NCHW':
                    generated_img = np.transpose(generated_img, (0, 2, 3, 1))
                    input_sketch = np.transpose(input_sketch, (0, 2, 3, 1))

                # log('before, generated_img', generated_img)
                # log('before, input_sketch', input_sketch)
                generated_img = ((generated_img + 1) / 2.) * 255
                input_sketch = ((input_sketch + 1) / 2.) * 255
                generated_img = generated_img[:, :, :, ::-1].astype(np.uint8)
                input_sketch = input_sketch.astype(np.uint8)
                # log('after, generated_img', generated_img)
                # log('after, input_sketch', input_sketch)

                img_out_filename = cate + '_' + input_name[:-4] + '_output.png'
                sketch_in_filename = cate + '_' + input_name[:-4] + '_input.png'

                # Save file
                cv2.imwrite(os.path.join(output_folder, img_out_filename), generated_img[0])
                cv2.imwrite(os.path.join(output_folder, sketch_in_filename), input_sketch[0])

                print('Saved file %s' % img_out_filename)


def inference(img_name, instruction):
    wild_data_base_dir = 'examples'
    wild_text = instruction

    wild_cate = img_name[:img_name.find('.png')]

    SIZE = {True: (64, 64),
            False: (192, 192)}
    T = 15  # the longest length of text
    vocab_file = 'data/vocab.txt'

    captions_base_dir = os.path.join('data', 'captions')
    categories = os.listdir(captions_base_dir)
    categories.sort()

    if wild_cate not in categories:
        wild_cate = categories[2]

    # Roll out the parameters
    batch_size = 1
    ckpt_dir = Config.ckpt_dir
    results_dir = Config.results_dir
    data_format = Config.data_format
    distance_map = Config.distance_map
    small_img = Config.small_img
    LSTM_hybrid = Config.LSTM_hybrid
    block_type = Config.block_type
    vocab_size = Config.vocab_size

    distance_map = distance_map != 0
    small = small_img != 0
    LSTM_hybrid = LSTM_hybrid != 0

    img_dim = SIZE[small]

    output_folder = results_dir
    print('output_folder:', output_folder)
    os.makedirs(output_folder, exist_ok=True)

    vocab_dict = load_vocab_dict_from_file(vocab_file)

    input_images = tf.placeholder(tf.float32, shape=[1, 3, img_dim[0], img_dim[1]])  # [1, 3, H, W]
    class_ids = tf.placeholder(tf.int32, shape=(1,))  # (1, )
    text_vocab_indiceses = tf.placeholder(tf.int32, shape=[1, 15])  # [1, 15]

    ret_list = build_single_graph(input_images, input_images, None,
                                  class_ids, None,
                                  text_vocab_indiceses,
                                  batch_size=batch_size, training=False,
                                  LSTM_hybrid=LSTM_hybrid,
                                  vocab_size=vocab_size,
                                  data_format=data_format,
                                  distance_map=distance_map,
                                  block_type=block_type)  # [image_gens, images, sketches]

    snapshot_loader = tf.train.Saver()

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('Restore trained model:', tf.train.latest_checkpoint(ckpt_dir))
        snapshot_loader.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

        input_name = img_name
        input_category = wild_cate  # e.g. 'bus'
        input_text = wild_text  # e.g. 'A yellow bus with blue window'

        sketch_path = os.path.join(wild_data_base_dir, input_name)
        sketch_image = Image.open(sketch_path)
        sketch_image = sketch_image.convert("RGB")
        # Resize
        if sketch_image.width != img_dim[0] or sketch_image.height != img_dim[1]:
            margin_size = 0 if input_category in ['road'] else 10
            sketch_image = resize_and_padding_mask_image(sketch_image, img_dim[0],
                                                         margin_size=margin_size).astype(np.float32)
        else:
            sketch_image = np.array(sketch_image, dtype=np.float32)  # shape = [H, W, 3]

        # Normalization
        sketch_image = sketch_image / 255.
        sketch_image = sketch_image * 2. - 1

        sketch_image = np.expand_dims(sketch_image, axis=0)  # shape = [1, H, W, 3]
        sketch_image = np.transpose(sketch_image, [0, 3, 1, 2])  # shape = [1, 3, H, W]

        class_id = categories.index(input_category)
        class_id = np.array([class_id])

        vocab_indices = preprocess_sentence(input_text, vocab_dict, T)  # list
        vocab_indices = np.array(vocab_indices, dtype=np.int32)
        vocab_indices = np.expand_dims(vocab_indices, axis=0)  # shape = [1, 15]

        try:
            # print('class_id', class_id)
            # print('vocab_indices', vocab_indices)
            generated_img, _, input_sketch = sess.run(
                [ret_list[0], ret_list[1], ret_list[2]],
                feed_dict={input_images: sketch_image,
                           class_ids: class_id,
                           text_vocab_indiceses: vocab_indices})
        except Exception as e:
            print(e.args)

        if data_format == 'NCHW':
            generated_img = np.transpose(generated_img, (0, 2, 3, 1))
            input_sketch = np.transpose(input_sketch, (0, 2, 3, 1))

        # log('before, generated_img', generated_img)
        # log('before, input_sketch', input_sketch)
        generated_img = ((generated_img + 1) / 2.) * 255
        input_sketch = ((input_sketch + 1) / 2.) * 255
        generated_img = generated_img[:, :, :, ::-1].astype(np.uint8)
        input_sketch = input_sketch.astype(np.uint8)
        # log('after, generated_img', generated_img)
        # log('after, input_sketch', input_sketch)

        img_out_filename = input_name[:-4] + '_output.png'
        sketch_in_filename = input_name[:-4] + '_input.png'

        # Save file
        cv2.imwrite(os.path.join(output_folder, img_out_filename), generated_img[0])
        cv2.imwrite(os.path.join(output_folder, sketch_in_filename), input_sketch[0])

        print('Saved file %s' % img_out_filename)
