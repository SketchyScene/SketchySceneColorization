import os
import json
import collections

from Instance_Matching.data_processing.text_processing import search_for_self_category


def judge_colorize_type(text):
    """
    judge if it is FG or BG colorization
    :param text:
    :return: 0 for iFG, 1 for BG
    """
    # Special case: 'the moon in the sky is yellow'

    category, _ = search_for_self_category(text)  # category name or None
    return 'BG' if category is None else 'FG'


def fetch_records(image_id, results_base_dir):
    records_dir = os.path.join(results_base_dir, 'update_records')
    os.makedirs(records_dir, exist_ok=True)

    records_file_path = os.path.join(records_dir, str(image_id) + '_records.json')

    summary_data = []
    last_bg_text = ""

    ## empty records: no images, no json
    if not os.path.isfile(records_file_path):
        # new record
        new_result_image_name = str(image_id) + '_1.png'
        last_result_image_name = ''
    else:
        fp = open(records_file_path, "r")
        record_json_data = fp.read()
        record_json_data = json.loads(record_json_data)
        print(len(record_json_data), 'editing records')

        # store old records
        for i in range(len(record_json_data)):
            last_bg_text = record_json_data[i]["proc_bg_text"]

            order_dict = collections.OrderedDict()
            order_dict["colorization_type"] = record_json_data[i]["colorization_type"]
            order_dict["result_name"] = record_json_data[i]["result_name"]
            order_dict["input_text"] = record_json_data[i]["input_text"]
            order_dict["proc_bg_text"] = last_bg_text
            summary_data.append(order_dict)
        # new record
        new_result_image_name = str(image_id) + '_' + str(len(record_json_data) + 1) + '.png'
        last_result_image_name = record_json_data[len(record_json_data) - 1]['result_name']

    return new_result_image_name, last_result_image_name, last_bg_text, summary_data


def update_records(image_id, input_text, results_base_dir,
                   colorization_type, new_result_image_name, proc_bg_text, summary_data):
    records_dir = os.path.join(results_base_dir, 'update_records')
    os.makedirs(records_dir, exist_ok=True)
    records_file_path = os.path.join(records_dir, str(image_id) + '_records.json')

    order_dict = collections.OrderedDict()
    order_dict["colorization_type"] = colorization_type
    order_dict["result_name"] = new_result_image_name
    order_dict["input_text"] = input_text
    order_dict["proc_bg_text"] = proc_bg_text
    summary_data.append(order_dict)

    info_summary = open(records_file_path, "w")
    summary_data = json.dumps(summary_data, indent=4)
    info_summary.write(summary_data)


def withdraw_records(image_id, results_base_dir):
    results_dir = os.path.join(results_base_dir, 'results', str(image_id))
    records_dir = os.path.join(results_base_dir, 'update_records')
    records_file_path = os.path.join(records_dir, str(image_id) + '_records.json')

    if not os.path.isfile(records_file_path):
        raise Exception('No record to withdraw.')
    else:
        fp = open(records_file_path, "r")
        record_json_data = fp.read()
        record_json_data = json.loads(record_json_data)
        print('Original: ', len(record_json_data), 'editing records')

        last_result_img_path = os.path.join(results_dir, str(image_id) + '_' + str(len(record_json_data)) + '.png')
        os.remove(last_result_img_path)

        if len(record_json_data) == 1:
            fp.close()
            os.remove(records_file_path)
        else:
            summary_data = []
            for i in range(len(record_json_data) - 1):
                order_dict = collections.OrderedDict()
                order_dict["colorization_type"] = record_json_data[i]["colorization_type"]
                order_dict["result_name"] = record_json_data[i]["result_name"]
                order_dict["input_text"] = record_json_data[i]["input_text"]
                order_dict["proc_bg_text"] = record_json_data[i]["proc_bg_text"]
                summary_data.append(order_dict)

            info_summary = open(records_file_path, "w")
            summary_data = json.dumps(summary_data, indent=4)
            info_summary.write(summary_data)
