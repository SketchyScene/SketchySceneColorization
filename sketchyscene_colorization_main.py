import os
import argparse

from Pipeline_utils.customization_util import judge_colorize_type, fetch_records, update_records, withdraw_records
from Pipeline_utils.fg_matching_utils import build_instance_matching
from Pipeline_utils.fg_color_utils import build_instance_colorization
from Pipeline_utils.bg_utils import build_background_colorization

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def withdraw_last_record(image_id, results_base_dir):
    withdraw_records(image_id, results_base_dir)


def colorization_main(image_id, input_text, data_base_dir, results_base_dir,
                      match_vocab_path, match_vocab_size, match_snapshot_root, match_max_len,
                      fgcolor_vocab_path, fgcolor_vocab_size, fgcolor_snapshot_root, fgcolor_max_len,
                      bg_vocab_path, bg_vocab_size, bg_snapshot_root, bg_max_len):
    ## 1. sentence parsing
    colorization_type = judge_colorize_type(input_text)
    print('colorization_type:', colorization_type)

    sketch_path = os.path.join(data_base_dir, 'sketches', str(image_id) + '.png')
    segm_data_npz_path = os.path.join(data_base_dir, 'seg_data', str(image_id) + '_datas.npz')
    inner_masks_mat_path = os.path.join(data_base_dir, 'inner_masks', str(image_id) + '.mat')

    new_result_image_name, last_result_image_name, last_bg_text, summary_data = \
        fetch_records(image_id, results_base_dir)

    if colorization_type == 'FG':
        assert input_text != '' and input_text is not None

        ## 1.1 get customized mask and find the most related instance (indices)
        matched_inst_indices = build_instance_matching(data_base_dir, sketch_path, input_text, segm_data_npz_path,
                                                       match_vocab_path, match_vocab_size, match_snapshot_root,
                                                       match_max_len)
        assert type(matched_inst_indices) is list
        print('matched_inst_indices', matched_inst_indices)

        ## 1.2 instance colorization and update records
        build_instance_colorization(data_base_dir, image_id, input_text, matched_inst_indices, sketch_path,
                                    inner_masks_mat_path, segm_data_npz_path, results_base_dir,
                                    fgcolor_vocab_size, fgcolor_max_len, fgcolor_vocab_path, fgcolor_snapshot_root,
                                    new_result_image_name, last_result_image_name)
        proc_bg_text = last_bg_text

    else:
        proc_bg_text = build_background_colorization(image_id, input_text, sketch_path,
                                                     inner_masks_mat_path, segm_data_npz_path, results_base_dir,
                                                     bg_vocab_size, bg_max_len, bg_vocab_path, bg_snapshot_root,
                                                     new_result_image_name, last_result_image_name, last_bg_text)

    update_records(image_id, input_text, results_base_dir,
                   colorization_type, new_result_image_name, proc_bg_text, summary_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--command', '-c', type=str, choices=['color', 'withdraw'],
                        default='color', help="choose a command from 'color' or 'withdraw'")
    parser.add_argument('--image_id', '-id', type=int, default=-1, help="choose an image.")
    parser.add_argument('--instruction', '-it', type=str, default='',
                        help="the input instruction")

    # fixed params
    parser.add_argument('--data_base_dir', '-dbd', type=str, default='examples',
                        help="the base dir of examples")
    parser.add_argument('--results_base_dir', '-rbd', type=str, default='outputs',
                        help="the dir of results")

    parser.add_argument('--match_snapshot_root', '-msr', type=str, default='Instance_Matching/outputs/snapshots',
                        help="the dir of instance matching models")
    parser.add_argument('--match_vocab_path', '-mvp', type=str, default='Instance_Matching/data/vocab.txt',
                        help="the dir of instance matching vocab")
    parser.add_argument('--match_vocab_size', '-mvs', type=int, default=76, help="vocab size of matching")
    parser.add_argument('--match_max_len', '-ml', type=int, default=15, help="max sentence length of matching")

    parser.add_argument('--fgcolor_snapshot_root', '-fgsr', type=str,
                        default='Foreground_Instance_Colorization/outputs/2019-00-00-00-00-00/snapshot',
                        help="the dir of foreground colorization models")
    parser.add_argument('--fgcolor_vocab_path', '-fgvp', type=str,
                        default='Foreground_Instance_Colorization/data/vocab.txt',
                        help="the dir of foreground colorization vocab")
    parser.add_argument('--fgcolor_vocab_size', '-fgvs', type=int, default=58, help="vocab size of fg colorization")
    parser.add_argument('--fgcolor_max_len', '-fgl', type=int, default=15,
                        help="max sentence length of fg colorization")

    parser.add_argument('--bg_snapshot_root', '-bgsr', type=str,
                        default='Background_Colorization/outputs/2019-00-00-00-00-00/snapshot',
                        help="the dir of background colorization models")
    parser.add_argument('--bg_vocab_path', '-bgvp', type=str,
                        default='Background_Colorization/data/bg_vocab.txt',
                        help="the dir of background colorization vocab")
    parser.add_argument('--bg_vocab_size', '-bgvs', type=int, default=18, help="vocab size of bg colorization")
    parser.add_argument('--bg_max_len', '-bgl', type=int, default=8, help="max sentence length of bg colorization")

    args = parser.parse_args()

    assert args.image_id != -1

    if args.command == 'color':
        assert args.instruction != ''
        colorization_main(args.image_id, args.instruction, args.data_base_dir, args.results_base_dir,
                          args.match_vocab_path, args.match_vocab_size, args.match_snapshot_root, args.match_max_len,
                          args.fgcolor_vocab_path, args.fgcolor_vocab_size, args.fgcolor_snapshot_root, args.fgcolor_max_len,
                          args.bg_vocab_path, args.bg_vocab_size, args.bg_snapshot_root, args.bg_max_len)
    elif args.command == 'withdraw':
        withdraw_last_record(args.image_id, args.results_base_dir)
    else:
        raise Exception('Unknown user command:', args.command)
