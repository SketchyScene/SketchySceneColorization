import os
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import random


def visualize_sem_seg(im, predicts, sent, save_path=''):
    im_seg = im.copy()
    im_seg[:, :, 0] += predicts.astype('uint8') * 250
    plt.imshow(im_seg.astype('uint8'))
    plt.title(sent)

    if save_path != '':
        plt.savefig(save_path)
        # im_seg_png = Image.fromarray(im_seg, 'RGB')
        # im_seg_png.save(save_path)
    else:
        plt.show()


def visualize_inst_seg(im, predict_inst_seg, sent):
    """

    :param im:
    :param predict_inst_seg: [H, W, N]
    :param sent:
    :return:
    """
    predicts = np.zeros((im.shape[0], im.shape[1]), dtype=np.int32)
    if predict_inst_seg.shape[0] != 0:
        for inst_idx in range(predict_inst_seg.shape[2]):
            predicts = np.logical_or(predicts, predict_inst_seg[:, :, inst_idx])

    im_seg = im.copy()
    im_seg[:, :, 0] += predicts.astype('uint8') * 250
    plt.imshow(im_seg.astype('uint8'))
    plt.title(sent)
    plt.show()


def visualize_sem_inst_mask(im, sem_mask, boxes, inst_masks, class_ids, class_names, sent, scores=None):
    rows = 1
    cols = 2
    plt.figure(figsize=(8 * cols, 8 * rows))

    ## semantic mask
    im_mask = im.copy()
    im_mask[:, :, 0] += sem_mask.astype('uint8') * 250

    plt.subplot(rows, cols, 1)
    plt.title('Pred semantic: ' + sent, fontsize=14)
    plt.axis('on')
    plt.imshow(im_mask.astype('uint8'))

    ## instance mask
    def apply_mask(image, mask, color, alpha=1.):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def generate_colors(N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def draw_dash_line(x1, y1, x2, y2, dash_gap):
        assert x1 - x2 == 0 or y1 - y2 == 0
        len = abs(x1 - x2) + abs(y1 - y2)
        segm = len // dash_gap + 1
        for seg_idx in range(segm):
            if x1 - x2 == 0:
                draw.line((x1, y1 + seg_idx * dash_gap, x2, min((y1 + seg_idx * dash_gap + 20), y2)),
                          fill=color_str, width=3)
            else:
                draw.line((x1 + seg_idx * dash_gap, y1, min((x1 + seg_idx * dash_gap + 20), x2), y2),
                          fill=color_str, width=3)

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == inst_masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = generate_colors(N)

    masked_image = im.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]
        mask = inst_masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

    # ax.imshow(masked_image.astype(np.uint8))
    masked_image_out = Image.fromarray(np.array(masked_image, dtype=np.uint8))
    draw = ImageDraw.Draw(masked_image_out)
    font_path = 'data/TakaoPGothic.ttf'

    if not os.path.exists(font_path):
        font_path = '../data/TakaoPGothic.ttf'

    font = ImageFont.truetype(font_path, 26)
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1_, x1_, y2_, x2_ = boxes[i]

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = "{} {:.3f}".format(label, score) if score else label

        draw.text((x1_ + 2, y1_ + 2), caption, font=font, fill='#000000')
        color_str = '#'

        def get_color_str(color_val):
            nor_color = int(color_val * 255)
            if nor_color < 16:
                return '0' + str(hex(nor_color))[2:]
            else:
                return str(hex(nor_color))[2:]

        color_str += get_color_str(color[0])
        color_str += get_color_str(color[1])
        color_str += get_color_str(color[2])

        dash_gap_ = 30

        draw_dash_line(x1_, y1_, x1_, y2_, dash_gap_)
        draw_dash_line(x2_, y1_, x2_, y2_, dash_gap_)
        draw_dash_line(x1_, y1_, x2_, y1_, dash_gap_)
        draw_dash_line(x1_, y2_, x2_, y2_, dash_gap_)

    plt.subplot(rows, cols, 2)
    plt.title('Pred instance: ' + sent, fontsize=14)
    plt.axis('on')
    plt.imshow(np.array(masked_image_out, dtype=np.uint8))
    plt.show()