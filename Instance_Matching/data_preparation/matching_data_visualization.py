import os
import json
import sys
import numpy as np
import random
import argparse
import scipy.io

sys.path.append('../data_processing')
sys.path.append('../utils')
import text_processing, sketch_data_processing
from visualization_util import visualize_sem_inst_mask


def matching_data_visualization(**kwargs):
    dataset_type = kwargs['dataset']
    img_id = kwargs['image_id']
    caption_base_dir = kwargs['caption_base_dir']
    sketchyscene_data_base_dir = kwargs['data_base_dir']

    dataset_class_names = ['bg']
    color_map_mat_path = os.path.join(sketchyscene_data_base_dir, 'colorMapC46.mat')
    colorMap = scipy.io.loadmat(color_map_mat_path)['colorMap']
    for i in range(46):
        cat_name = colorMap[i][0][0]
        dataset_class_names.append(cat_name)

    caption_json_path = os.path.join(caption_base_dir, 'sentence_instance_' + dataset_type + '.json')
    dataset_base_dir = os.path.join(sketchyscene_data_base_dir, dataset_type)

    fp = open(caption_json_path, "r")
    json_data = fp.read()
    json_data = json.loads(json_data)
    print('data_len', len(json_data))

    if img_id == -1:
        data_rand_idx = random.randint(0, len(json_data) - 1)
        img_id = json_data[data_rand_idx]['key']

    for data_idx in range(len(json_data)):
        img_idx = json_data[data_idx]['key']

        if img_idx != img_id:
            if data_idx == len(json_data) - 1:
                print('Error: Data of this image not exists!')
            continue

        sketch_image, gt_class_ids, gt_bboxes, gt_masks = sketch_data_processing.load_data_gt(dataset_base_dir,
                                                                                              str(img_idx))
        sketch_image_vis = np.array(np.squeeze(sketch_image), dtype=np.uint8)

        # load text and target_mask
        sen_instIdx_map = json_data[data_idx]['sen_instIdx_map']
        sen_instIdx_map_keys = list(sen_instIdx_map.keys())

        for inst_data_idx in range(len(sen_instIdx_map_keys)):
            caption = sen_instIdx_map_keys[inst_data_idx]
            inst_indices = sen_instIdx_map[caption]

            print(caption, inst_indices)
            target_mask = np.zeros((gt_masks.shape[0], gt_masks.shape[1]), dtype=np.int32)

            pred_masks_list = []
            pred_class_ids_list = []
            pred_boxes_list = []

            for inst_idx in inst_indices:
                target_mask = np.logical_or(target_mask, gt_masks[:, :, inst_idx])
                # target_bbox = gt_bboxes[inst_idx]  # (y1, x1, y2, x2)
                pred_masks_list.append(gt_masks[:, :, inst_idx])
                pred_class_ids_list.append(gt_class_ids[inst_idx])
                pred_boxes_list.append(gt_bboxes[inst_idx])

            # augment the caption with random attributes
            caption = text_processing.augment_the_caption_with_attr(caption)

            visualize_sem_inst_mask(sketch_image_vis.copy(), target_mask,
                                    np.stack(pred_boxes_list), np.stack(pred_masks_list, axis=2),
                                    np.stack(pred_class_ids_list), dataset_class_names, caption)

        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds', type=str, choices=['train', 'val', 'test'],
                        default='val', help="choose a dataset")
    parser.add_argument('--image_id', '-id', type=int, default=-1, help="choose an image. -1 for random.")
    parser.add_argument('--data_base_dir', '-db', type=str, default='../data',
                        help="set the data base dir of SketchyScene")
    parser.add_argument('--caption_base_dir', '-cb', type=str, default='../data',
                        help="set the data base dir of captions")

    args = parser.parse_args()

    run_params = {
        "dataset": args.dataset,
        "image_id": args.image_id,
        "data_base_dir": args.data_base_dir,
        "caption_base_dir": args.caption_base_dir
    }

    matching_data_visualization(**run_params)
