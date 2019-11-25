import re
import json
import numpy as np
import random

UNK_IDENTIFIER = '<unk>'  # <unk> is the word used to identify unknown words
PAD_IDENTIFIER = '<pad>'
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def sentence2vocab_indices(sentence, vocab_dict):
    words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0 and w != '-']
    # remove .
    if words[-1] == '.':
        words = words[:-1]
    vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
                     for w in words]
    return vocab_indices


color_list = ['dark brown', 'light brown', 'light gray', 'dark gray',
              'black', 'red', 'dark green', 'light green', 'dark blue', 'light blue', 'yellow',
              'orange', 'pink', 'purple']

simple_color_list = ['brown', 'gray', 'black', 'red', 'green', 'blue', 'yellow', 'orange',
                     'pink', 'purple', 'cyan', 'white']

category_list = ['bench', 'bird', 'bus', 'butterfly',
                 'car', 'cat', 'chair', 'chicken', 'cloud', 'cow',
                 'dog', 'duck', 'horse', 'house', 'grass',
                 'moon', 'person', 'pig', 'rabbit', 'road',
                 'sheep', 'star', 'sun', 'tree', 'truck']

category_es_list = ['benches', 'birds', 'buses', 'butterflies',
                    'cars', 'cats', 'chairs', 'chickens', 'clouds', 'cows',
                    'dogs', 'ducks', 'horses', 'houses', 'grasses',
                    'moons', 'people', 'pigs', 'rabbits', 'roads',
                    'sheep', 'stars', 'suns', 'trees', 'trucks']

es_attr = ['both', 'all', 'two', 'three', 'four', 'five', 'six']


def search_for_self_category(caption):
    words = SENTENCE_SPLIT_REGEX.split(caption.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0 and w != '-']

    is_es = False
    self_category = None

    for w in words:
        if w in es_attr:
            is_es = True

        if w in category_list:
            self_category = w
            break

        if w in category_es_list:
            self_category = category_list[category_es_list.index(w)]
            is_es = True
            break

    return self_category, is_es


def search_for_color(caption):
    words = SENTENCE_SPLIT_REGEX.split(caption.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0 and w != '-']

    have_color = False

    for w in words:
        if w in simple_color_list:
            have_color = True
            break

    return have_color


## Public
################################################

def load_vocab_dict_from_file(dict_file):
    with open(dict_file) as f:
        words = [w.strip() for w in f.readlines()]
    vocab_dict = {words[n]: n for n in range(len(words))}
    return vocab_dict


def preprocess_sentence(sentence, vocab_dict, T):
    vocab_indices = sentence2vocab_indices(sentence, vocab_dict)
    # Truncate long sentences
    if len(vocab_indices) > T:
        # raise Exception('The length of text is oversize than ', T)
        vocab_indices = vocab_indices[:T]

    ori_len = len(vocab_indices)
    # Pad short sentences at the beginning with the special symbol '<pad>'
    if ori_len < T:
        vocab_indices = vocab_indices + [vocab_dict[PAD_IDENTIFIER]] * (T - ori_len)
    return vocab_indices, ori_len


COLOR_MAPS = {
    'bench': ['light brown', 'dark brown', 'yellow', 'orange', 'dark blue', 'light blue', 'red', 'pink', 'purple'],
    'cat': ['yellow', 'orange', 'dark gray', 'pink', 'light gray'],
    'chair': ['light brown', 'dark brown'],
    'cloud': ['dark gray', 'light blue', 'dark blue'],
    'dog': ['light brown', 'dark brown', 'orange'],
    'duck': ['yellow', 'orange'],
    'grass': ['dark green', 'light green'],
    'horse': ['light brown', 'dark brown', 'orange', 'dark gray', 'light gray', 'dark blue', 'purple'],
    'moon': ['yellow', 'orange'],
    'pig': ['pink', 'red'],
    'rabbit': ['pink', 'dark gray'],
    'road': ['yellow', 'orange', 'dark gray', 'black', 'light brown', 'dark brown'],
    'sheep': ['red', 'yellow', 'dark blue', 'light blue', 'orange', 'pink', 'light green', 'dark green', 'purple',
              'cyan', 'dark brown', 'dark gray', 'light brown', 'light gray', 'black'],
    'star': ['yellow', 'orange', 'red'],
    'sun': ['yellow'],
    'tree': ['light green', 'dark green'],
    'truck': ['red', 'yellow', 'orange', 'light green', 'dark blue', 'light blue'],
    'chicken': ['yellow', 'orange', 'light brown', 'dark brown'],
    'cow': ['light brown', 'dark brown', 'yellow', 'dark gray', 'light gray'],
}


def augment_the_caption_with_attr(ori_caption):
    """
    augment the text like 'the dog on the left' to 'the dog on the left is brown'
    to make the model more robust
    """
    self_category, is_es = search_for_self_category(ori_caption)
    assert self_category is not None

    rst_caption = ori_caption
    rand_color_0 = color_list[random.randint(0, len(color_list) - 1)]
    rand_color_1 = color_list[random.randint(0, len(color_list) - 1)]

    verb = ' are' if is_es else ' is'

    if self_category == 'person':
        ## '.. is in blue' / '.. is in blue shirt and green pants' / '.. is in blue shirt and green skirt'
        kind = random.randint(0, 2)
        if kind == 0:
            rst_caption += verb + ' in ' + rand_color_0
        elif kind == 1:
            rst_caption += verb + ' in ' + rand_color_0 + ' shirt and ' + rand_color_1 + ' pants'
        elif kind == 2:
            rst_caption += verb + ' in ' + rand_color_0 + ' shirt and ' + rand_color_1 + ' skirt'

    elif self_category in ['bus', 'car', 'house']:
        ## '.. is red' / '.. is red with blue windows'
        kind = random.randint(0, 1)
        if kind == 0:
            rst_caption += verb + ' ' + rand_color_0
        elif kind == 1:
            sub_part = ' roof' if self_category == 'house' else ' windows'
            rst_caption += verb + ' ' + rand_color_0 + ' with ' + rand_color_1 + sub_part

    elif self_category in ['bird']:
        ## '.. is red' / '.. is red with blue wing' / '.. is red with blue head and freen wing'
        kind = random.randint(0, 1)
        if kind == 0:
            rst_caption += verb + ' ' + rand_color_0
        elif kind == 1:
            rst_caption += verb + ' ' + rand_color_0 + ' with ' + rand_color_1 + ' wings'

    elif self_category in ['butterfly']:
        ## '.. has red body and blue wings'
        verb = ' have' if is_es else ' has'
        rst_caption += verb + ' ' + rand_color_0 + ' body and ' + rand_color_1 + ' wings'

    else:
        self_color_list = COLOR_MAPS[self_category]
        rand_color_0 = self_color_list[random.randint(0, len(self_color_list) - 1)]
        rst_caption += verb + ' ' + rand_color_0

    assert rst_caption != ori_caption
    return rst_caption


################################################


json_file_path = '../datas/captions/sentence_instance_train.json'
TT = 15  # the longest length of text
vocab_file = '../datas/vocab/vocab_new.txt'


def text_processing():
    vocab_dict = load_vocab_dict_from_file(vocab_file)
    print('vocab_dict.len', len(vocab_dict))
    print('vocab_dict', vocab_dict)

    fp = open(json_file_path, "r")
    json_data = fp.read()
    json_data = json.loads(json_data)
    print('data_len', len(json_data))

    for k in range(len(json_data)):
        instIdx_sen_map = json_data[k]['instIdx_sen_map']
        instIdx_sen_map_keys = list(instIdx_sen_map.keys())

        for j in range(len(instIdx_sen_map_keys)):
            caption = instIdx_sen_map[instIdx_sen_map_keys[j]]

            # augment the caption with random attributes
            caption = augment_the_caption_with_attr(caption)

            print(caption)

            vocab_indices, seq_len = preprocess_sentence(caption, vocab_dict, TT)  # list, [15]

            vocab_indices = [item + 1 for item in vocab_indices]

            print(np.array(vocab_indices, dtype=np.uint8).shape)
            print(np.expand_dims(vocab_indices, axis=0).shape)
            print(vocab_indices)
            print('\n')


if __name__ == "__main__":
    text_processing()
