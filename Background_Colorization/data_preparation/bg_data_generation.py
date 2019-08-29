import os
import json
import random
import argparse
import collections
import numpy as np
from PIL import Image


SKY_COLOR = ['blue', 'green', 'cyan', 'red', 'orange', 'yellow', 'brown', 'purple', 'pink', 'black', 'gray']
GROUND_COLOR = ['yellow', 'green', 'black', 'gray', 'brown']
color_map = {'blue': [153, 217, 234], 'green': [181, 230, 29], 'cyan': [128, 255, 215],
             'red': [237, 28, 36], 'orange': [255, 127, 39], 'yellow': [255, 242, 0],
             'brown': [185, 122, 87], 'purple': [163, 73, 164], 'pink': [255, 174, 201],
             'black': [30, 30, 30], 'gray': [127, 127, 127]}


def is_bg_color_blue_and_green(bg_img, mask, times=0):
    """
    weak validation function, maybe slow
    :param times: cannot search for more than 3 times
    :param bg_img: [H, W, 3]
    :param mask: [H, W], 0 for fg, 255 for bg
    :return:
    """
    if times >= 3:
        return False

    sky_rand = random.randint(0, bg_img.shape[1] - 1)
    ground_rand = random.randint(0, bg_img.shape[1] - 1)

    if mask[0][sky_rand] == 0 or mask[bg_img.shape[0] - 1][ground_rand] == 0:
        return is_bg_color_blue_and_green(bg_img, mask)
    else:
        if not bg_img[0][sky_rand].tolist() == color_map['blue'] \
                and bg_img[bg_img.shape[0] - 1][ground_rand].tolist() == color_map['green']:
            return is_bg_color_blue_and_green(bg_img, mask, times + 1)
        else:
            return True


def gen_random_color_pair(former_pair):
    assert type(former_pair) is list

    sky_color = SKY_COLOR[random.randint(0, len(SKY_COLOR) - 1)]
    ground_color = GROUND_COLOR[random.randint(0, len(GROUND_COLOR) - 1)]

    if sky_color == ground_color:
        return gen_random_color_pair(former_pair)

    if (sky_color, ground_color) in former_pair:
        return gen_random_color_pair(former_pair)

    return sky_color, ground_color


def gen_bg_caption(up_color, down_color):
    caption = 'the sky is ' + up_color + ' and the ground is ' + down_color
    return caption


def bg_data_generation(**kwargs):
    data_base_dir = kwargs['data_base_dir']
    aug_num = kwargs['aug_num']
    data_splits = ['train', 'test']

    user_paint_base = os.path.join(data_base_dir, 'user_paint')
    fg_base = os.path.join(data_base_dir, 'foreground')
    inner_mask_base = os.path.join(data_base_dir, 'inner_mask')

    save_bg_base_dir = os.path.join(data_base_dir, 'background')
    save_caption_base_dir = os.path.join(data_base_dir, 'captions')
    save_segment_base_dir = os.path.join(data_base_dir, 'segment')

    for data_split in data_splits:
        split_user_paint_base = os.path.join(user_paint_base, data_split)
        split_fg_base = os.path.join(fg_base, data_split)
        split_inner_mask_base = os.path.join(inner_mask_base, data_split)

        split_save_bg_base_dir = os.path.join(save_bg_base_dir, data_split)
        split_save_caption_base_dir = os.path.join(save_caption_base_dir, data_split)
        split_save_segment_base_dir = os.path.join(save_segment_base_dir, data_split)
        os.makedirs(split_save_bg_base_dir, exist_ok=True)
        os.makedirs(split_save_caption_base_dir, exist_ok=True)
        os.makedirs(split_save_segment_base_dir, exist_ok=True)

        split_json_path = os.path.join(save_caption_base_dir, data_split + '.json')
        summary_data = []

        all_files = os.listdir(split_user_paint_base)
        all_files.sort()
        print(data_split, len(all_files))

        for file_idx, file_name in enumerate(all_files):
            print('Processing', file_idx, '/', len(all_files), file_name)
            
            bg_img_user_paint_path = os.path.join(split_user_paint_base, file_name)
            fg_ori_path = os.path.join(split_fg_base, file_name)
            mask_path = os.path.join(split_inner_mask_base, file_name)

            bg_img_after_user = Image.open(bg_img_user_paint_path).convert('RGB')
            bg_img_after_user = np.array(bg_img_after_user, dtype=np.uint8)
            bg_img_proc = bg_img_after_user.copy()

            fg_img_ori = Image.open(fg_ori_path).convert('RGB')
            fg_img_ori = np.array(fg_img_ori, dtype=np.uint8)

            mask_img = Image.open(mask_path).convert('RGB')
            mask_img = np.array(mask_img, dtype=np.uint8)[:, :, 0]  # 0 for fg, 255 for bg

            ## first remove the seperating line across the fg
            bg_img_proc[mask_img == 0] = fg_img_ori[mask_img == 0]

            bg_img_proc_png = Image.fromarray(bg_img_proc, 'RGB')
            bg_img_proc_path = os.path.join(split_save_bg_base_dir, file_name)
            bg_img_proc_png.save(bg_img_proc_path, 'PNG')

            assert is_bg_color_blue_and_green(bg_img_proc.copy(), mask_img.copy())

            ## then generate the segment map
            segment_map = np.zeros(mask_img.shape, dtype=np.uint8)
            # label: {'FG': 0, 'Sky': 128, 'Ground': 255}
            segment_map[mask_img == 0] = 0  # label 'FG' to 0
            segment_map[np.logical_and(mask_img == 255, (bg_img_proc == color_map['blue']).all(axis=2))] = 128
            segment_map[np.logical_and(mask_img == 255, (bg_img_proc == color_map['green']).all(axis=2))] = 255
            segment_map_png = Image.fromarray(segment_map, 'L')
            segment_map_png_path = os.path.join(split_save_segment_base_dir, file_name)
            segment_map_png.save(segment_map_png_path, 'PNG')

            ## then caption the default bg: blue sky and green ground
            former_color_pair_set = [('blue', 'green')]

            order_dict = collections.OrderedDict()
            order_dict["fg_name"] = file_name
            order_dict["bg_name"] = file_name
            order_dict["color_text"] = gen_bg_caption(former_color_pair_set[0][0], former_color_pair_set[0][1])
            summary_data.append(order_dict)

            ## aug the bg color to non-fg area (mask_img == 255)
            for aug_i in range(aug_num):
                sky_color, ground_color = gen_random_color_pair(former_color_pair_set)
                former_color_pair_set.append((sky_color, ground_color))

                aug_idx = aug_i + 1
                aug_img = bg_img_proc.copy()
                aug_img[np.logical_and(mask_img == 255, (bg_img_proc == color_map['blue']).all(axis=2))] = color_map[sky_color]
                aug_img[np.logical_and(mask_img == 255, (bg_img_proc == color_map['green']).all(axis=2))] = color_map[ground_color]

                aug_img = Image.fromarray(aug_img, 'RGB')
                aug_image_name = file_name[:-4] + '_' + str(aug_idx) + '.png'
                aug_img_proc_path = os.path.join(split_save_bg_base_dir, aug_image_name)
                aug_img.save(aug_img_proc_path, 'PNG')

                caption = gen_bg_caption(sky_color, ground_color)

                order_dict = collections.OrderedDict()
                order_dict["fg_name"] = file_name
                order_dict["bg_name"] = aug_image_name
                order_dict["color_text"] = caption
                summary_data.append(order_dict)

        color_summary = open(split_json_path, "w")
        summary_data = json.dumps(summary_data, indent=4)
        color_summary.write(summary_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_base_dir', '-db', type=str, default='../data',
                        help="set the data base dir")
    parser.add_argument('--aug_num', '-an', type=int,
                        default=3, help="the number of augmentation")

    args = parser.parse_args()

    run_params = {
        "data_base_dir": args.data_base_dir,
        "aug_num": args.aug_num
    }
    
    bg_data_generation(**run_params)
