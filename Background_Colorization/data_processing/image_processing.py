import numpy as np
from PIL import Image


def load_image(imname, image_size):
    im = Image.open(imname).convert("RGB")
    if im.width != image_size or im.height != image_size:
        im = im.resize((image_size, image_size), resample=Image.BILINEAR)
    im = np.array(im, dtype=np.uint8)
    im = np.expand_dims(im, axis=0)  # shape = [1, H, W, 3]
    return im


def load_region_mask(seg_path, image_size, is_test=False):
    if is_test:
        return np.zeros([1, image_size, image_size], dtype=np.int32)

    seg_img = Image.open(seg_path).convert('RGB')
    seg_img = np.array(seg_img, dtype=np.uint8)[:, :, 0]
    region_label = np.zeros(seg_img.shape, dtype=np.int32)
    region_label[seg_img == 128] = 1
    region_label[seg_img == 255] = 2
    region_label = np.expand_dims(region_label, axis=0)  # shape = [1, H, W]
    return region_label
