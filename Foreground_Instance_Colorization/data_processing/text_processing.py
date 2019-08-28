import re
import json
import numpy as np

UNK_IDENTIFIER = '<unk>'  # <unk> is the word used to identify unknown words
PAD_IDENTIFIER = '<pad>'
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def sentence2vocab_indices(sentence, vocab_dict):
    words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [w.lower() for w in words if len(w.strip()) > 0]
    # remove .
    if words[-1] == '.':
        words = words[:-1]
    if words[0] in ['a']:
        words = words[1:]
    while 'the' in words:  # 'the' is no need
        words.remove('the')

    for i in range(len(words)):
        word = words[i]
        if word == ',':  # ',' = 'and'
            words[i] = 'and'
        elif word == ', ':  # ', ' = 'and'
            words[i] = 'and'

    vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
                     for w in words]
    return vocab_indices


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

    # Pad short sentences at the beginning with the special symbol '<pad>'
    if len(vocab_indices) < T:
        vocab_indices = [vocab_dict[PAD_IDENTIFIER]] * (T - len(vocab_indices)) + vocab_indices
    return vocab_indices

################################################
