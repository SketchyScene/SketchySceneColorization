import os
import numpy as np
import scipy.io
import scipy.ndimage
from PIL import Image
import matplotlib.pyplot as plt

image_subfolder = 'DRAWING_GT'
semantic_subfolder = 'CLASS_GT'
instance_subfolder = 'INSTANCE_GT'

IMAGE_SIZE = 768


def load_image(image_dir, image_id):
    image_name = os.path.join(image_dir, 'L0_sample' + str(image_id) + '.png')
    image = Image.open(image_name).convert("RGB")
    if image.width != IMAGE_SIZE or image.height != IMAGE_SIZE:
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST)
    image = np.array(image, dtype=np.float32)  # shape = [H, W, 3]
    return image


def load_testing_image(image_dir, image_id):
    image_name = os.path.join(image_dir, str(image_id) + '.png')
    image = Image.open(image_name).convert("RGB")
    if image.width != IMAGE_SIZE or image.height != IMAGE_SIZE:
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.NEAREST)
    image = np.array(image, dtype=np.float32)  # shape = [H, W, 3]
    return image


def load_mask(instance_dir, semantic_dir, image_id):
    mask_class_name = 'sample_' + str(image_id) + '_class.mat'
    mask_instance_name = 'sample_' + str(image_id) + '_instance.mat'

    mask_class_path = os.path.join(semantic_dir, mask_class_name)
    mask_instance_path = os.path.join(instance_dir, mask_instance_name)

    INSTANCE_GT = scipy.io.loadmat(mask_instance_path)['INSTANCE_GT']
    INSTANCE_GT = np.array(INSTANCE_GT, dtype=np.uint8)  # shape=(750, 750)
    CLASS_GT = scipy.io.loadmat(mask_class_path)['CLASS_GT']  # (750, 750)

    # print(np.max(INSTANCE_GT))  # e.g. 101
    instance_count = np.bincount(INSTANCE_GT.flatten())
    # print(instance_count.shape)  # e.g. shape=(102,)

    instance_count = instance_count[1:]  # e.g. shape=(101,)
    nonzero_count = np.count_nonzero(instance_count)  # e.g. 16
    # print("nonzero_count", nonzero_count)  # e.g. shape=(102,)

    mask_set = np.zeros([nonzero_count, INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.uint8)
    class_id_set = np.zeros([nonzero_count], dtype=np.uint8)

    real_instanceIdx = 0
    for i in range(instance_count.shape[0]):
        if instance_count[i] == 0:
            continue

        instanceIdx = i + 1

        ## mask
        mask = np.zeros([INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.uint8)
        mask[INSTANCE_GT == instanceIdx] = 1
        mask_set[real_instanceIdx] = mask

        # plt.imshow(mask)  # astype(np.uint8)
        # plt.show()

        ## new version: fast
        class_gt_filtered = CLASS_GT * mask
        class_gt_filtered = np.bincount(class_gt_filtered.flatten())
        class_gt_filtered = class_gt_filtered[1:]
        class_id = np.argmax(class_gt_filtered) + 1

        class_id_set[real_instanceIdx] = class_id

        real_instanceIdx += 1

    mask_set = np.transpose(mask_set, (1, 2, 0))  # [H, W, nInst]

    ## scaling
    if mask_set.shape[0] != IMAGE_SIZE:
        scale = IMAGE_SIZE / mask_set.shape[0]
        mask_set = scipy.ndimage.zoom(mask_set, zoom=[scale, scale, 1], order=0)
        mask_set = np.array(mask_set, dtype=np.uint8)

    return mask_set, class_id_set


def load_mask_simp(instance_dir, image_id, selected_instance_ids):
    assert type(selected_instance_ids) is list
    selected_instance_ids_ = [item for item in selected_instance_ids]
    """A fast version of loading mask (without class id) given the instance index"""
    mask_instance_name = 'sample_' + str(image_id) + '_instance.mat'
    mask_instance_path = os.path.join(instance_dir, mask_instance_name)

    INSTANCE_GT = scipy.io.loadmat(mask_instance_path)['INSTANCE_GT']
    INSTANCE_GT = np.array(INSTANCE_GT, dtype=np.int32)  # shape=(750, 750)
    # print(np.max(INSTANCE_GT))  # e.g. 101
    instance_count = np.bincount(INSTANCE_GT.flatten())
    # print(instance_count.shape)  # e.g. shape=(102,)

    instance_count = instance_count[1:]  # e.g. shape=(101,)
    # nonzero_count = np.count_nonzero(instance_count)  # e.g. 16
    # print("nonzero_count", nonzero_count)  # e.g. shape=(102,)

    selected_mask = np.zeros([INSTANCE_GT.shape[0], INSTANCE_GT.shape[1]], dtype=np.int32)

    real_instanceIdx = 0
    for i in range(instance_count.shape[0]):
        if instance_count[i] == 0:
            continue

        instanceIdx = i + 1

        if real_instanceIdx in selected_instance_ids:
            selected_mask[INSTANCE_GT == instanceIdx] = 1
            selected_instance_ids_.remove(real_instanceIdx)
            if len(selected_instance_ids_) == 0:
                # print('done')
                break

        real_instanceIdx += 1

    assert np.sum(selected_mask) != 0

    ## scaling
    if selected_mask.shape[0] != IMAGE_SIZE:
        scale = IMAGE_SIZE / selected_mask.shape[0]
        selected_mask = scipy.ndimage.zoom(selected_mask, zoom=[scale, scale], order=0)
        selected_mask = np.array(selected_mask, dtype=np.int32)

    return selected_mask


def extract_bboxes(mask):
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])

    return boxes.astype(np.int32)


def load_data_gt(dataset_base_dir, image_id, fast_version=False, inst_indices=None):
    image_dir = os.path.join(dataset_base_dir, image_subfolder)
    semantic_dir = os.path.join(dataset_base_dir, semantic_subfolder)
    instance_dir = os.path.join(dataset_base_dir, instance_subfolder)

    sketch_image = load_image(image_dir, image_id)  # [768, 768, 3], {0-255}, np.float32

    if fast_version:
        assert inst_indices is not None
        mask = load_mask_simp(instance_dir, image_id, inst_indices)  # [768, 768]
        return sketch_image, mask
    else:
        masks, class_ids = load_mask(instance_dir, semantic_dir, image_id)  # [768, 768, nInst], [nInst]
        bboxes = extract_bboxes(masks)  # [nInst, (y1, x1, y2, x2)]

        return sketch_image, class_ids, bboxes, masks


def compute_mask_iou(maskA, maskB):
    """
    Calculates mask IoU of the given two masks
    :param maskA: [H, W]
    :param maskB: [H, W]
    :return:
    """
    maskA_sum = np.sum(maskA)
    maskB_sum = np.sum(maskB)

    # Calculate intersection areas
    intersection = np.sum(np.logical_and(maskA, maskB))
    union = maskA_sum + maskB_sum - intersection
    iou = intersection / union
    return iou


def post_processing_mask_with_segmentation(segm_data_path, pred_mask, iou_threshold=0.9):
    npz = np.load(segm_data_path)
    pred_masks = np.array(npz['pred_masks'], dtype=np.uint8)  # [N, H, W]

    mask_IoU_list = []

    for i in range(pred_masks.shape[0]):
        candidate_mask = pred_masks[i]
        candidate_IoU = compute_mask_iou(pred_mask.copy(), candidate_mask)
        mask_IoU_list.append(candidate_IoU)

    assert len(mask_IoU_list) == pred_masks.shape[0]

    if np.max(mask_IoU_list) < iou_threshold:
        print('# Max iou is less than iou_threshold', iou_threshold)
        return pred_mask
    else:
        print('# Refined the mask')
        maxIoU_idx = np.argmax(mask_IoU_list)
        return pred_masks[maxIoU_idx]


def compute_mask_occupied_percentage(mask_overall, mask_instance):
    """
    Calculates mask IoU of the given two masks
    :param maskA: [H, W]
    :param maskB: [H, W]
    :return:
    """
    intersection = np.sum(np.logical_and(mask_overall, mask_instance))
    union = np.sum(mask_instance)
    percentage = intersection / union
    return percentage


def get_pred_instance_mask(segm_data_path, pred_overall_mask, mask_occupied_threshold=0.5):
    npz = np.load(segm_data_path)
    pred_masks = np.array(npz['pred_masks'], dtype=np.uint8)  # [N, H, W]
    pred_class_ids = np.array(npz['pred_class_ids'], dtype=np.int32)  # [N], of the 46 ids
    pred_boxes = np.array(npz['pred_boxes'], dtype=np.int32)  # [N, 4]

    pred_masks_list = []
    pred_scores_list = []
    pred_class_ids_list = []
    pred_boxes_list = []

    for i in range(pred_masks.shape[0]):
        candidate_mask = pred_masks[i]
        mask_occupied_percentage = compute_mask_occupied_percentage(pred_overall_mask.copy(), candidate_mask)
        if mask_occupied_percentage > mask_occupied_threshold:
            pred_masks_list.append(candidate_mask.copy())
            pred_scores_list.append(mask_occupied_percentage)
            pred_class_ids_list.append(pred_class_ids[i])
            pred_boxes_list.append(pred_boxes[i])

    if len(pred_masks_list) != 0:
        return np.stack(pred_masks_list, axis=2), np.stack(pred_scores_list), \
               np.stack(pred_boxes_list), np.stack(pred_class_ids_list)
    else:
        return np.array(()), np.array(()), np.array(()), np.array(())


def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]
