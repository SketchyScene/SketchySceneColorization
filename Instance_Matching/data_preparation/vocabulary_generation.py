import re
import os
import json

dataset_type = ['train', 'val', 'test']
captions_base_dir = '../data'
vocab_save_base_dir = '../data'

IGNORE_WORDS = []
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


def vocabulary_generation():
    if not os.path.exists(vocab_save_base_dir):
        os.makedirs(vocab_save_base_dir)

    vocab_list = []
    max_len = 0
    min_len = 1000
    avg_len = 0
    sentence_count = 0

    for kind in dataset_type:
        json_file_path = os.path.join(captions_base_dir, 'sentence_instance_' + kind + '.json')
        fp = open(json_file_path, "r")
        json_data = fp.read()
        json_data = json.loads(json_data)
        print('data_len', len(json_data))

        for k in range(len(json_data)):
            sen_instIdx_map = json_data[k]['sen_instIdx_map']
            sen_instIdx_map_keys = list(sen_instIdx_map.keys())

            for inst_data_idx in range(len(sen_instIdx_map_keys)):
                caption = sen_instIdx_map_keys[inst_data_idx]
                print(caption)

                words = SENTENCE_SPLIT_REGEX.split(caption.strip())
                words = [w.lower() for w in words if len(w.strip()) > 0 and w != '-']
                # print(words)

                max_len = max(max_len, len(words))
                min_len = min(min_len, len(words))
                avg_len += len(words)
                sentence_count += 1

                for l in range(len(words)):
                    if words[l] not in vocab_list and words[l] not in IGNORE_WORDS:
                        vocab_list.append(words[l])

    outstr = '<pad>' + '\n'
    outstr += '<unk>' + '\n'
    for i in range(len(vocab_list)):
        outstr += vocab_list[i] + '\n'

    # write validation result to txt
    write_path = os.path.join(vocab_save_base_dir, 'vocab_new.txt')
    fp = open(write_path, 'w')
    fp.write(outstr)
    fp.close()

    print('max_len', max_len)
    print('min_len', min_len)
    avg_len = avg_len / sentence_count
    print('avg_len', avg_len)


if __name__ == "__main__":
    vocabulary_generation()