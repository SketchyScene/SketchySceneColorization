import os
import scipy.io
import json
import argparse
import collections
import sys
sys.path.append('../')

import png_to_text
from data_processing import sketch_data_processing

img_num_maps = {'train': 5617, 'val': 535, 'test': 1113}


CATEGORIES_UNMOVABLE = ["house", "bus", "truck", "car", "bench", "chair"]
CATEGORIES_TREE = ["tree"]
CATEGORIES_MOVABLE = ["people", "horse", "cow", "sheep", "pig", "cat", "dog", "chicken",
                      "duck", "rabbit", "bird", "butterfly"]
# 16 valid categories
INSTANCE = CATEGORIES_UNMOVABLE + CATEGORIES_TREE + CATEGORIES_MOVABLE + \
           ["cloud", "sun", "moon", "star"] + \
           ["road", "grass"] + ["others"]


def matching_data_generation(**kwargs):
    dataset_type = kwargs['dataset']
    dataset_base_dir = kwargs['data_base_dir']
    save = kwargs['save']

    color_map_mat_path = os.path.join(dataset_base_dir, 'colorMapC46.mat')
    colorMap = scipy.io.loadmat(color_map_mat_path)['colorMap']  # cat_name = colorMap[i][0][0]

    if dataset_type == 'all':
        modes = ['train', 'val', 'test']
    else:
        modes = [dataset_type]

    for mode in modes:
        split_dataset_base_dir = os.path.join(dataset_base_dir, mode)

        nImages = img_num_maps[mode]
        print(nImages, 'images')

        not_include_count = 0
        summary_data = []

        min_inst_count = 1000
        max_inst_count = 0
        avg_inst_count = 0

        for i in range(nImages):
            image_id = i + 1
            print('\nProcessing', image_id, '/', nImages)

            original_image, gt_class_id, gt_bbox, gt_mask = sketch_data_processing.load_data_gt(split_dataset_base_dir,
                                                                                                str(image_id))
            # original_image = np.array(np.squeeze(original_image), dtype=np.uint8)

            # filter out the excluding categories
            excluded = False
            for id in gt_class_id:
                if colorMap[id - 1][0][0] not in INSTANCE:
                    not_include_count += 1
                    print('not_include_count', not_include_count)
                    excluded = True
                    break
            if excluded:
                continue

            # categoroes = [colorMap[j - 1][0][0] for j in gt_class_id]
            # print(categoroes)
            caption, sorted_indices, sen_boxes_map_list \
                = png_to_text.png2text(gt_bbox.copy().tolist(), gt_class_id.copy().tolist(), dataset_base_dir)
            print('>>> caption')
            print(caption)
            print('>>> sen_boxes_map_list')
            print(sen_boxes_map_list)
            caption_set = caption.split('.')[:-1]
            caption_set = [item.strip() for item in caption_set]
            assert len(caption_set) == len(sen_boxes_map_list)

            sen_inst_idx_map = {}
            for sen_idx, sen_boxes_map in enumerate(sen_boxes_map_list):
                assert -1 not in sen_boxes_map
                sen_inst_idx_map[caption_set[sen_idx]] = sen_boxes_map

            if len(sen_inst_idx_map) > 0:
                min_inst_count = min(min_inst_count, len(sen_inst_idx_map))
                max_inst_count = max(max_inst_count, len(sen_inst_idx_map))
                avg_inst_count += len(sen_inst_idx_map)

                order_dict = collections.OrderedDict()
                order_dict["key"] = image_id
                order_dict["sen_instIdx_map"] = sen_inst_idx_map
                summary_data.append(order_dict)

        print('not_include_count', not_include_count)
        print('min_inst_count', min_inst_count)
        print('max_inst_count', max_inst_count)
        print('total_inst_count', avg_inst_count)

        avg_inst_count = avg_inst_count / (nImages - not_include_count)
        print('avg_inst_count', avg_inst_count)

        if save:
            instIdx_sen_json_path = '../data/sentence_instance_' + mode + '.json'
            sen_inst_summary = open(instIdx_sen_json_path, "w")
            summary_data = json.dumps(summary_data, indent=4)
            sen_inst_summary.write(summary_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds', type=str, choices=['train', 'val', 'test', 'all'],
                        default='val', help="choose a dataset")
    parser.add_argument('--data_base_dir', '-db', type=str, default='../data',
                        help="set the data base dir of SketchyScene")
    parser.add_argument('--save', '-sa', type=int, choices=[0, 1],
                        default=1, help="whether save the data")

    args = parser.parse_args()

    run_params = {
        "dataset": args.dataset,
        "data_base_dir": args.data_base_dir,
        "save": args.save
    }

    matching_data_generation(**run_params)
