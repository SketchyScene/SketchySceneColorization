import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.append('../data_processing')
from text_processing import preprocess_sentence, load_vocab_dict_from_file


def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _to_tfexample_raw(image_name, cartoon_data, sketch_data, category,
                      category_id, color_text, vocab_indices):
    """ write a raw input"""
    return tf.train.Example(features=tf.train.Features(feature={
        'ImageName': _bytes_feature(image_name),
        'cartoon_data': _bytes_feature(cartoon_data),
        'sketch_data': _bytes_feature(sketch_data),
        'Category': _bytes_feature(category),
        'Category_id': _int64_feature(category_id),
        'Color_text': _bytes_feature(color_text),
        'Text_vocab_indices': _bytes_feature(vocab_indices),
    }))


def data_preparation(**kwargs):
    dataset = kwargs['dataset']
    data_base_dir = kwargs['data_base_dir']
    text_len = kwargs['text_len']

    if dataset == 'both':
        dataset_types = ['train', 'val']
    else:
        dataset_types = [dataset]

    caption_data_base_dir = os.path.join(data_base_dir, 'captions')
    image_data_base_dir = os.path.join(data_base_dir, 'images')

    categories = os.listdir(caption_data_base_dir)
    categories.sort()

    vocab_file = os.path.join(data_base_dir, 'vocab.txt')
    vocab_dict = load_vocab_dict_from_file(vocab_file)

    for dataset_type in dataset_types:
        data_save_split_base = os.path.join(data_base_dir, 'tfrecord', dataset_type)
        os.makedirs(data_save_split_base, exist_ok=True)

        for category_id, category_name in enumerate(categories):
            record_filename = os.path.join(data_save_split_base, category_name + '.tfrecord')

            with tf.python_io.TFRecordWriter(record_filename) as tfrecord_writer:
                json_file_path = os.path.join(caption_data_base_dir, category_name, dataset_type + '.json')
                fp = open(json_file_path, "r")
                json_data = fp.read()
                json_data = json.loads(json_data)
                nImgs = len(json_data)
                print(dataset_type, category_name, nImgs)

                for j in range(nImgs):
                    image_name = json_data[j]['key']

                    cartoon_path = os.path.join(image_data_base_dir, category_name, 'cartoon', image_name)
                    sketch_path = os.path.join(image_data_base_dir, category_name, 'edgemap', image_name)

                    cartoon_image = Image.open(cartoon_path)
                    cartoon_image = cartoon_image.convert("RGB")
                    cartoon_image = np.array(cartoon_image, dtype=np.uint8)  # shape = [H, W, 3]
                    cartoon_image_raw = cartoon_image.tobytes()

                    sketch_image = Image.open(sketch_path)
                    sketch_image = sketch_image.convert("RGB")
                    sketch_image = np.array(sketch_image, dtype=np.uint8)  # shape = [H, W, 3]
                    sketch_image_raw = sketch_image.tobytes()

                    color_text = json_data[j]['color_text']
                    vocab_indices = preprocess_sentence(color_text, vocab_dict, text_len)  # list
                    vocab_indices_raw = np.array(vocab_indices, dtype=np.uint8).tobytes()  # [15]
                    # print(color_text)
                    # vocab_indices_display = [item + 1 for item in vocab_indices]
                    # print(vocab_indices_display)

                    example = _to_tfexample_raw(image_name.encode(), cartoon_image_raw,
                                                sketch_image_raw,
                                                category_name.encode(), category_id, color_text.encode(),
                                                vocab_indices_raw)
                    tfrecord_writer.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-ds', type=str, choices=['train', 'val', 'both'],
                        default='both', help="choose a dataset")
    parser.add_argument('--data_base_dir', '-db', type=str, default='../data',
                        help="set the data base dir")
    parser.add_argument('--text_len', '-tl', type=int,
                        default=15, help="the longest length of text")

    args = parser.parse_args()

    run_params = {
        "dataset": args.dataset,
        "data_base_dir": args.data_base_dir,
        "text_len": args.text_len
    }

    data_preparation(**run_params)
