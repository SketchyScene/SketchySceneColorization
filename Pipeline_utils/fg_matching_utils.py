import os
import numpy as np
import scipy.io
import tensorflow as tf

import Instance_Matching.data_processing.text_processing as match_text_processing
import Instance_Matching.data_processing.sketch_data_processing as match_sketch_data_processing
import Instance_Matching.utils.visualization_util as match_visualization_util
from Instance_Matching.RMI_model import RMI_model

mu = np.array((104.00698793, 116.66876762, 122.67891434))


def build_instance_matching(data_base_dir, sketch_path, input_text, segm_data_npz_path,
                            match_vocab_path, match_vocab_size, match_snapshot_root, match_max_len):
    vocab_dict = match_text_processing.load_vocab_dict_from_file(match_vocab_path)

    dataset_class_names = ['bg']
    color_map_mat_path = os.path.join(data_base_dir, 'colorMapC46.mat')
    colorMap = scipy.io.loadmat(color_map_mat_path)['colorMap']
    for i in range(46):
        cat_name = colorMap[i][0][0]
        dataset_class_names.append(cat_name)

    score_thresh = 1e-9

    model = RMI_model(mode='eval', vocab_size=match_vocab_size, weights='deeplab')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Load pretrained model
    snapshot_restorer = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(match_snapshot_root)
    print('Restore:', ckpt.model_checkpoint_path)
    snapshot_restorer.restore(sess, ckpt.model_checkpoint_path)

    # Load image
    sketch_image = match_sketch_data_processing.load_image2(sketch_path)  # [768, 768, 3], float32
    sketch_image_vis = np.array(np.squeeze(sketch_image), dtype=np.uint8)
    sketch_image -= mu

    bin_drawing = sketch_image_vis.copy()[:, :, 0]
    bin_drawing[bin_drawing == 0] = 1
    bin_drawing[bin_drawing == 255] = 0

    caption = input_text
    vocab_indices, seq_len = match_text_processing.preprocess_sentence(caption, vocab_dict, match_max_len)

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
    pred_masks, pred_scores, pred_boxes, pred_class_ids, matched_inst_indices \
        = match_sketch_data_processing.get_pred_instance_mask(segm_data_npz_path, predicts.copy())
    print('pred_masks', pred_masks.shape)
    print('pred_scores', pred_scores.shape, pred_scores)
    print('pred_boxes', pred_boxes.shape)
    print('pred_class_ids', pred_class_ids.shape, pred_class_ids)

    # match_visualization_util.visualize_sem_inst_mask(sketch_image_vis.copy(), predicts,
    #                                                  pred_boxes, pred_masks, pred_class_ids, dataset_class_names,
    #                                                  caption)

    sess.close()

    return matched_inst_indices
