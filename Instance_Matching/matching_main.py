import os
import tensorflow as tf
import json
import random
import numpy as np
import argparse
import scipy.io
import matplotlib.pyplot as plt
import time
from datetime import timedelta

from RMI_model import RMI_model
from utils import eval_tools, visualization_util
from data_processing import text_processing, sketch_data_processing, im_processing

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

FLAGS = tf.app.flags.FLAGS

###############################
# Temp flags, no need to fix

tf.app.flags.DEFINE_string('mode', 'train', '')
tf.app.flags.DEFINE_string('model', 'deeplab', '')
tf.app.flags.DEFINE_string('data_base_dir', '../data', '')
tf.app.flags.DEFINE_string('captions_base_dir', 'data', '')
tf.app.flags.DEFINE_string('dataset', 'val', '')
tf.app.flags.DEFINE_string('seg_data_dir', 'outputs/inst_segm_output_data', '')
tf.app.flags.DEFINE_string('instruction', '', '')
tf.app.flags.DEFINE_integer('visualized', 0, '')
tf.app.flags.DEFINE_integer('mask_ap', 1, '')
tf.app.flags.DEFINE_integer('image_id', -1, '')

###############################


tf.app.flags.DEFINE_string(
    'model_name', 'RMI',
    'RMI or LSTM.')
tf.app.flags.DEFINE_string(
    'log_root', 'outputs/log',
    'Directory to store tensorboard.')
tf.app.flags.DEFINE_string(
    'snapshot_root', 'outputs/snapshots',
    'Directory to store models.')
tf.app.flags.DEFINE_string(
    'eval_result_root', 'outputs/eval_results',
    'Directory to store eval results.')
tf.app.flags.DEFINE_string(
    'match_result_root', 'outputs/matching_results',
    'Directory to store matching results.')
tf.app.flags.DEFINE_string(
    'visualize_pred_base_dir', 'outputs/visualization',
    'The path of vocabulary.')
tf.app.flags.DEFINE_string(
    'vocab_path', 'data/vocab.txt',
    'The path of vocabulary.')
tf.app.flags.DEFINE_integer(
    'vocab_size', 76,
    'The number of words in vocabulary.')
tf.app.flags.DEFINE_integer(
    'MAX_LEN', 15,
    'The max length of a sentence.')

tf.app.flags.DEFINE_integer(
    'max_iteration', 100000,
    'Max iteration.')
tf.app.flags.DEFINE_integer(
    'count_left_time_freq', 50,
    'Max iteration.')
tf.app.flags.DEFINE_integer(
    'summary_write_freq', 200,
    'Max iteration.')
tf.app.flags.DEFINE_integer(
    'save_model_freq', 10000,
    'Max iteration.')

mu = np.array((104.00698793, 116.66876762, 122.67891434))


def train(weights_name, data_base_dir, captions_base_dir):
    dataset_base_dir = os.path.join(data_base_dir, 'train')
    caption_json_path = os.path.join(captions_base_dir, 'sentence_instance_train.json')
    vocab_dict = text_processing.load_vocab_dict_from_file(FLAGS.vocab_path)

    snapshot_file = os.path.join(FLAGS.snapshot_root, weights_name + '_' + FLAGS.model_name + '_iter_%d.tfmodel')
    os.makedirs(FLAGS.snapshot_root, exist_ok=True)
    os.makedirs(FLAGS.log_root, exist_ok=True)

    cls_loss_avg = 0
    decay = 0.99

    start_iter = 0

    model = RMI_model(mode='train', vocab_size=FLAGS.vocab_size,
                      weights=weights_name)

    print('-' * 100)

    # Calculate trainable params.
    print('Network params:')
    t_vars = tf.trainable_variables()
    count_t_vars = 0
    for var in t_vars:
        num_param = np.prod(var.get_shape().as_list())
        count_t_vars += num_param
        print('%s | shape: %s | num_param: %i' % (var.name, str(var.get_shape()), num_param))
    print('Total network variables %i.' % count_t_vars)
    print('-' * 100)

    print('Optimizing params:')
    for var in model.optim_params:
        print('%s | shape: %s' % (var.name, str(var.get_shape())))
    print('-' * 100)

    snapshot_saver = tf.train.Saver(max_to_keep=10)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(FLAGS.snapshot_root)
    if not ckpt:
        if weights_name == 'deeplab':
            ckpt = tf.train.get_checkpoint_state('./models/SketchyScene_DeepLabv2')
            load_var = {var.op.name: var for var in tf.global_variables() if var.op.name.startswith('ResNet/group')}
        elif weights_name == 'fcn_8s':
            ckpt = tf.train.get_checkpoint_state('./models/SketchyScene_FCN8s')
            load_var = {var.op.name: var for var in tf.global_variables() if var.op.name.startswith('FCN_8s')}
        elif weights_name == 'segnet':
            ckpt = tf.train.get_checkpoint_state('./models/SketchyScene_SegNet')
            load_var = {var.op.name: var for var in tf.global_variables() if var.op.name.startswith('SegNet')}
        elif weights_name == 'deeplab_v3plus':
            ckpt = tf.train.get_checkpoint_state('./models/SketchyScene_DeepLabv3plus')
            load_var = {var.op.name: var for var in tf.global_variables() if var.op.name.startswith('resnet_v1_101')}
        else:
            raise ValueError('Unknown weights_name %s' % weights_name)

        snapshot_loader = tf.train.Saver(load_var)
        print('firstly train, loaded', ckpt.model_checkpoint_path)
        snapshot_loader.restore(sess, ckpt.model_checkpoint_path)  # pretrained_model
    else:
        snapshot_path = ckpt.model_checkpoint_path
        print('loaded', snapshot_path)
        snapshot_saver.restore(sess, snapshot_path)
        start_iter = int(snapshot_path[snapshot_path.rfind('_') + 1:snapshot_path.rfind('.')])

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.log_root, graph=sess.graph)

    duration_time_n_step = 0

    fp = open(caption_json_path, "r")
    json_data = fp.read()
    json_data = json.loads(json_data)
    print('data_len', len(json_data))

    train_info_list = []
    for i in range(len(json_data)):
        img_idx = json_data[i]['key']

        sen_instIdx_map = json_data[i]['sen_instIdx_map']
        sen_instIdx_map_keys = list(sen_instIdx_map.keys())

        for inst_data_idx in range(len(sen_instIdx_map_keys)):
            caption = sen_instIdx_map_keys[inst_data_idx]
            inst_indices = sen_instIdx_map[caption]

            tuple_map = {'img_idx': img_idx, 'inst_indices': inst_indices, 'caption': caption}
            train_info_list.append(tuple_map)

    train_info_indices = np.arange(len(train_info_list))
    temp_data_idx = -1
    print(len(train_info_list), 'tuples of data.')

    print('start_iter', start_iter)

    for n_iter in range(start_iter, FLAGS.max_iteration):
        start_time = time.time()

        temp_data_idx = (temp_data_idx + 1) % len(train_info_list)
        if temp_data_idx == 0:
            random.shuffle(train_info_indices)

        data_idx = train_info_indices[temp_data_idx]
        img_idx = train_info_list[data_idx]['img_idx']
        inst_indices = train_info_list[data_idx]['inst_indices']
        caption_thin = train_info_list[data_idx]['caption']
        # print('img_idx', img_idx, '; inst_indices', inst_indices, ':', caption_thin)

        # Load image, and target mask
        sketch_image, target_mask = sketch_data_processing.load_data_gt(dataset_base_dir, img_idx,
                                                                        fast_version=True,
                                                                        inst_indices=inst_indices)
        sketch_image -= mu

        # load text and augment the caption with random attributes
        caption = text_processing.augment_the_caption_with_attr(caption_thin)
        vocab_indices, seq_len = text_processing.preprocess_sentence(caption, vocab_dict, FLAGS.MAX_LEN)
        # print(caption, vocab_indices, seq_len)

        feed_dict = {
            model.words: np.expand_dims(vocab_indices, axis=0),  # [N, MAX_LEN]
            model.sequence_lengths: [seq_len],  # [N]
            model.im: np.expand_dims(sketch_image, axis=0),  # [N, H, W, 3]
            model.target_mask: np.expand_dims(np.expand_dims(target_mask.astype(np.float32), axis=0), axis=3),
        }

        _, cls_loss_, lr_, scores_ = sess.run([model.train_step,
                                               model.cls_loss,
                                               model.learning_rate,
                                               model.pred],
                                              feed_dict=feed_dict)

        duration_time = time.time() - start_time
        duration_time_n_step += duration_time

        if n_iter % FLAGS.count_left_time_freq == 0:
            if n_iter != 0:
                cls_loss_avg = decay * cls_loss_avg + (1 - decay) * cls_loss_
                print('iter = %d, loss (cur) = %f, loss (avg) = %f, lr = %f' % (n_iter, cls_loss_, cls_loss_avg, lr_))

                left_step = FLAGS.max_iteration - n_iter
                left_sec = left_step / FLAGS.count_left_time_freq * duration_time_n_step
                print("### duration_time_%d_step:%.3f(sec), left time:%s\n"
                      % (FLAGS.count_left_time_freq, duration_time_n_step, str(timedelta(seconds=left_sec))))
                duration_time_n_step = 0

        if n_iter % FLAGS.summary_write_freq == 0:
            if n_iter != 0:
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, n_iter)
                summary_writer.flush()

        # Save model
        if (n_iter + 1) % FLAGS.save_model_freq == 0 or (n_iter + 1) >= FLAGS.max_iteration:
            snapshot_saver.save(sess, snapshot_file % (n_iter + 1))
            print('model saved to ' + snapshot_file % (n_iter + 1))

    print('Optimization done.')


def test(weights_name, data_base_dir, dataset_split, captions_base_dir, seg_data_base_dir, cal_mask_AP, visualize):
    dataset_base_dir = os.path.join(data_base_dir, dataset_split)
    caption_json_path = os.path.join(captions_base_dir, 'sentence_instance_' + dataset_split + '.json')
    vocab_dict = text_processing.load_vocab_dict_from_file(FLAGS.vocab_path)

    os.makedirs(FLAGS.eval_result_root, exist_ok=True)

    score_thresh = 1e-9
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    cum_I, cum_U = 0, 0
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)

    APs = []
    iou_threshold = None  # None for mAP@[0.5:0.95]

    seg_total = 0.

    model = RMI_model(mode='eval', vocab_size=FLAGS.vocab_size,
                      weights=weights_name)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Load pretrained model
    snapshot_restorer = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.snapshot_root)
    print('Restore:', ckpt.model_checkpoint_path)
    snapshot_restorer.restore(sess, ckpt.model_checkpoint_path)

    fp = open(caption_json_path, "r")
    json_data = fp.read()
    json_data = json.loads(json_data)
    print('data_len', len(json_data))

    for data_idx in range(len(json_data)):
        img_idx = json_data[data_idx]['key']
        print('Processing', data_idx + 1, '/', len(json_data), ', img_idx:', img_idx)

        # Load image, and target mask
        sketch_image, gt_class_ids, gt_bboxes, gt_masks = sketch_data_processing.load_data_gt(dataset_base_dir, img_idx)
        sketch_image_vis = np.array(np.squeeze(sketch_image), dtype=np.uint8)
        sketch_image -= mu

        bin_drawing = sketch_image_vis.copy()[:, :, 0]
        bin_drawing[bin_drawing == 0] = 1
        bin_drawing[bin_drawing == 255] = 0

        # load text and target_mask
        sen_instIdx_map = json_data[data_idx]['sen_instIdx_map']
        sen_instIdx_map_keys = list(sen_instIdx_map.keys())

        segm_data_npz_path = os.path.join(seg_data_base_dir, dataset_split, 'seg_data',
                                          str(img_idx) + '_datas.npz')

        for inst_data_idx in range(len(sen_instIdx_map_keys)):
            caption = sen_instIdx_map_keys[inst_data_idx]
            inst_indices = sen_instIdx_map[caption]
            target_mask = np.zeros((gt_masks.shape[0], gt_masks.shape[1]), dtype=np.int32)
            caption_gt_masks = np.zeros((gt_masks.shape[0], gt_masks.shape[1], len(inst_indices)), dtype=np.int32)

            for t_i, inst_idx in enumerate(inst_indices):
                target_mask = np.logical_or(target_mask, gt_masks[:, :, inst_idx])
                caption_gt_masks[:, :, t_i] = gt_masks[:, :, inst_idx]

            # augment the caption with random attributes
            caption = text_processing.augment_the_caption_with_attr(caption)

            vocab_indices, seq_len = text_processing.preprocess_sentence(caption, vocab_dict, FLAGS.MAX_LEN)

            scores_val, up_val, sigm_val = sess.run(
                [model.pred, model.up, model.sigm],
                feed_dict={
                    model.words: np.expand_dims(vocab_indices, axis=0),  # [N, MAX_LEN]
                    model.sequence_lengths: [seq_len],  # [N]
                    model.im: np.expand_dims(sketch_image, axis=0),  # [N, H, W, 3]
                })

            up_val = np.squeeze(up_val)  # shape = [768, 768]
            pred_raw = (up_val >= score_thresh).astype(np.float32)  # 0.0/1.0
            predicts = im_processing.resize_and_crop(pred_raw, target_mask.shape[0], target_mask.shape[1])
            predicts = predicts * bin_drawing

            if visualize:
                save_dir = os.path.join(FLAGS.visualize_pred_base_dir, dataset_split,
                                        str(img_idx))
                os.makedirs(save_dir, exist_ok=True)

                save_path_gt = os.path.join(save_dir, str(inst_data_idx) + '_gt.png')
                visualization_util.visualize_sem_seg(sketch_image_vis.copy(), target_mask, 'GT: ' + caption,
                                                     save_path_gt)

                save_path = os.path.join(save_dir, str(inst_data_idx) + '_out.png')
                visualization_util.visualize_sem_seg(sketch_image_vis.copy(), predicts, 'Pred: ' + caption, save_path)

            I, U = eval_tools.compute_mask_IU(predicts.copy(), target_mask)
            cum_I += I
            cum_U += U
            msg = 'cumulative IoU = %f' % (cum_I / cum_U)
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (I / U >= eval_seg_iou)

            # Compute mask AP
            if cal_mask_AP:
                # get pred_instance_mask by segm_data and predicts
                pred_masks, pred_scores, _, _ = sketch_data_processing.get_pred_instance_mask(segm_data_npz_path,
                                                                                              predicts.copy())
                # print('caption_gt_masks', caption_gt_masks.shape)
                # print('pred_masks', pred_masks.shape)
                # print('pred_scores', pred_scores.shape, pred_scores)
                # visualization_util.visualize_inst_seg(sketch_image_vis.copy(), pred_masks,
                #                                       'Instance pred: ' + caption)

                if iou_threshold is None:
                    iou_thresholds = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
                    AP_list = np.zeros([len(iou_thresholds)], dtype=np.float32)
                    if pred_scores.shape[0] != 0:
                        for j in range(len(iou_thresholds)):
                            iouThr = iou_thresholds[j]
                            AP_single_iouThr, precisions, recalls, overlaps = \
                                eval_tools.compute_ap(caption_gt_masks, pred_scores, pred_masks,
                                                      iou_threshold=iouThr)
                            AP_list[j] = AP_single_iouThr

                    AP = AP_list
                    # print('iou_thresholds', str(iou_thresholds), ', AP', AP)
                else:
                    if pred_scores.shape[0] != 0:
                        AP, precisions, recalls, overlaps = \
                            eval_tools.compute_ap(caption_gt_masks, pred_scores, pred_masks,
                                                  iou_threshold=iou_threshold)
                    else:
                        AP = 0
                    # print('iou_threshold', str(iou_threshold), ', AP', AP)

                APs.append(AP)

            # print(msg)
            seg_total += 1

    # Print results
    result_str = '\n' + ckpt.model_checkpoint_path + '\n'
    result_str += 'Segmentation evaluation (without DenseCRF):\n'
    for n_eval_iou in range(len(eval_seg_iou_list)):
        result_str += 'precision@%s = %f\n' % \
                      (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
    result_str += 'overall IoU = %f\n' % (cum_I / cum_U)
    print(ckpt.model_checkpoint_path)
    print(dataset_split, 'overall IoU = %f\n' % (cum_I / cum_U))

    if cal_mask_AP:
        mAP = np.mean(APs)
        mAP_list = np.mean(APs, axis=0)
        if iou_threshold is None:
            iou_str = '@[0.5:0.95]'
        else:
            iou_str = '@[' + str(iou_threshold) + ']'

        print("iou_threshold: ", iou_str, ", mAP = ", mAP)
        result_str += 'iou_threshold %s,  mAP = %s\n' % (iou_str, str(mAP))

        if iou_threshold is None:
            print("mAP_list: ", mAP_list)
            result_str += 'mAP_list = %s\n' % (str(mAP_list))

    # write validation result to txt
    write_path = os.path.join(FLAGS.eval_result_root,
                              weights_name + '_' + FLAGS.model_name +
                              '_iter_' + str(FLAGS.max_iteration) + '_' + dataset_split + '_result.txt')
    fp = open(write_path, 'a')
    fp.write(result_str)
    fp.close()


def inference(weights_name, data_base_dir, dataset_split, seg_data_base_dir, image_id, input_text):
    sketch_image_base_dir = os.path.join(data_base_dir, dataset_split, 'DRAWING_GT')
    vocab_dict = text_processing.load_vocab_dict_from_file(FLAGS.vocab_path)

    dataset_class_names = ['bg']
    color_map_mat_path = os.path.join(data_base_dir, 'colorMapC46.mat')
    colorMap = scipy.io.loadmat(color_map_mat_path)['colorMap']
    for i in range(46):
        cat_name = colorMap[i][0][0]
        dataset_class_names.append(cat_name)

    score_thresh = 1e-9

    model = RMI_model(mode='eval', vocab_size=FLAGS.vocab_size,
                      weights=weights_name)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Load pretrained model
    snapshot_restorer = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.snapshot_root)
    print('Restore:', ckpt.model_checkpoint_path)
    snapshot_restorer.restore(sess, ckpt.model_checkpoint_path)

    # Load image
    sketch_image = sketch_data_processing.load_image(sketch_image_base_dir, image_id)  # [768, 768, 3], float32
    sketch_image_vis = np.array(np.squeeze(sketch_image), dtype=np.uint8)
    sketch_image -= mu

    bin_drawing = sketch_image_vis.copy()[:, :, 0]
    bin_drawing[bin_drawing == 0] = 1
    bin_drawing[bin_drawing == 255] = 0

    caption = input_text
    vocab_indices, seq_len = text_processing.preprocess_sentence(caption, vocab_dict, FLAGS.MAX_LEN)

    up_val, sigm_val = sess.run([model.up, model.sigm],
                                feed_dict={
                                    model.words: np.expand_dims(vocab_indices, axis=0),  # [N, T]
                                    model.sequence_lengths: [seq_len],  # [N]
                                    model.im: np.expand_dims(sketch_image, axis=0),  # [N, H, W, 3]
                                })

    up_val = np.squeeze(up_val)  # shape = [768, 768]
    predicts = (up_val >= score_thresh).astype(np.float32)  # 0.0/1.0
    predicts = predicts * bin_drawing  # [768, 768] {0, 1}

    # get pred_instance_mask by segm_data and predicts
    segm_data_npz_path = os.path.join(seg_data_base_dir, dataset_split, 'seg_data',
                                      str(image_id) + '_datas.npz')
    pred_masks, pred_scores, pred_boxes, pred_class_ids = sketch_data_processing.get_pred_instance_mask(
        segm_data_npz_path, predicts.copy())
    print('pred_masks', pred_masks.shape)
    print('pred_scores', pred_scores.shape, pred_scores)
    print('pred_boxes', pred_boxes.shape)
    print('pred_class_ids', pred_class_ids.shape, pred_class_ids)

    # visualization_util.visualize_sem_seg(sketch_image_vis.copy(), predicts, 'Binary pred: ' + caption,
    #                                      save_path=os.path.join(FLAGS.match_result_root, 'seg_vis_bin.png'))
    # visualization_util.visualize_inst_seg(sketch_image_vis.copy(), pred_masks, 'Instance pred: ' + caption)
    # visualize_instance.display_instances(sketch_image_vis.copy(), pred_boxes, pred_masks, pred_class_ids,
    #                                      dataset_class_names, scores=None, title=caption,
    #                                      save_path=save_path, fix_color=False)

    visualization_util.visualize_sem_inst_mask(sketch_image_vis.copy(), predicts,
                                               pred_boxes, pred_masks, pred_class_ids, dataset_class_names, caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-md', type=str, choices=['train', 'eval', 'inference'],
                        default='train', help="choose a mode")
    parser.add_argument('--model', '-mdl', type=str, choices=['deeplab', 'deeplab_v3plus', 'fcn_8s', 'segnet'],
                        default='deeplab', help="choose a model")
    parser.add_argument('--data_base_dir', '-db', type=str, default='../data',
                        help="set the data base dir of SketchyScene")
    parser.add_argument('--captions_base_dir', '-cb', type=str, default='data',
                        help="set the data base dir of captions")

    # Used only in testing mode
    parser.add_argument('--dataset', '-ds', type=str, choices=['val', 'test'],
                        default='val', help="choose a dataset")
    parser.add_argument('--visualized', '-vs', type=int, choices=[0, 1],
                        default=0, help="whether to visualize results when evaluating")
    parser.add_argument('--mask_ap', '-cap', type=int, choices=[0, 1],
                        default=1, help="whether to calculate mask AP")
    parser.add_argument('--seg_data_dir', '-sd', type=str, default='outputs/inst_segm_output_data',
                        help="the dir of instance segmentation output data")

    # Used only in inference mode
    parser.add_argument('--image_id', '-id', type=int, default=-1, help="choose an image.")
    parser.add_argument('--instruction', '-it', type=str, default='',
                        help="the input instruction")

    args = parser.parse_args()

    if args.mode == 'train':
        print('Training mode.')
        train(weights_name=args.model,
              data_base_dir=args.data_base_dir,
              captions_base_dir=args.captions_base_dir)
    elif args.mode == 'eval':
        print('Eval mode.')
        test(weights_name=args.model,
             data_base_dir=args.data_base_dir,
             dataset_split=args.dataset,
             captions_base_dir=args.captions_base_dir,
             seg_data_base_dir=args.seg_data_dir,
             visualize=args.visualized,
             cal_mask_AP=args.mask_ap)
    elif args.mode == 'inference':
        print('Inference mode.')
        assert args.image_id != -1 and args.instruction != ''
        inference(weights_name=args.model,
                  data_base_dir=args.data_base_dir,
                  dataset_split=args.dataset,
                  seg_data_base_dir=args.seg_data_dir,
                  image_id=args.image_id,
                  input_text=args.instruction)
