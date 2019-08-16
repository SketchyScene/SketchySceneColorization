import numpy as np

# all boxes are [xmin, ymin, xmax, ymax] format, 0-indexed, including xmax and ymax
def compute_bbox_iou(bboxes, target):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    bboxes = bboxes.reshape((-1, 4))

    if isinstance(target, list):
        target = np.array(target)
    target = target.reshape((-1, 4))

    A_bboxes = (bboxes[..., 2]-bboxes[..., 0]+1) * (bboxes[..., 3]-bboxes[..., 1]+1)
    A_target = (target[..., 2]-target[..., 0]+1) * (target[..., 3]-target[..., 1]+1)
    assert(np.all(A_bboxes >= 0))
    assert(np.all(A_target >= 0))
    I_x1 = np.maximum(bboxes[..., 0], target[..., 0])
    I_y1 = np.maximum(bboxes[..., 1], target[..., 1])
    I_x2 = np.minimum(bboxes[..., 2], target[..., 2])
    I_y2 = np.minimum(bboxes[..., 3], target[..., 3])
    A_I = np.maximum(I_x2 - I_x1 + 1, 0) * np.maximum(I_y2 - I_y1 + 1, 0)
    IoUs = A_I / (A_bboxes + A_target - A_I)
    assert(np.all(0 <= IoUs) and np.all(IoUs <= 1))
    return IoUs

# # all boxes are [num, height, width] binary array
def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U


def compute_overlaps_masks(masks1, masks2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    '''
    # flatten masks
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def compute_ap(gt_masks, pred_scores, pred_masks,
               iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    gt_masks: (768, 768, nGT)
    pred_scores: (nRoIs), the mask occupied percentage
    pred_masks: (768, 768, nRoIs)

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding and sort predictions by score from high to low
    # TODO: cleaner to do zero unpadding upstream
    indices = np.argsort(pred_scores)[::-1]
    pred_masks = pred_masks[..., indices]

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)

    # Loop through ground truth boxes and find matching predictions
    match_count = 0
    pred_match = np.zeros([pred_masks.shape[2]])
    gt_match = np.zeros([gt_masks.shape[2]])
    for i in range(pred_masks.shape[2]):
        # Find best matching ground truth box
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] == 1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            else:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                break

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps
