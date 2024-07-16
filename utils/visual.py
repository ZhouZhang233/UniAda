import os
import numpy as np
from PIL import Image
from skimage import measure
from skimage.measure import label, regionprops
from scipy import ndimage
import scipy

def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))

def save_per_img_prostate(patch_image, data_save_path, img_name, prob_map, gt=None, mask_path=None, ext="bmp"):
    # path1 = os.path.join(data_save_path, 'overlay', img_name.split('.')[0]+'.png')
    path1 = os.path.join(data_save_path, 'overlay', img_name + '.png')
    path0 = os.path.join(data_save_path, 'original_image', img_name + '.png')
    if not os.path.exists(os.path.dirname(path0)):
        os.makedirs(os.path.dirname(path0))
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))

    patch_image = (patch_image - np.min(patch_image)) / (np.max(patch_image) - np.min(patch_image)) * 255
    # patch_image = patch_image[..., 1:2]

    patch_image_orign = patch_image.astype(np.uint8)
    patch_image_orign = Image.fromarray(patch_image_orign)
    patch_image_orign.save(path0)

    mask = get_largest_fillhole(gt).astype(np.uint8)  # return 0,1

    contours = measure.find_contours(mask, 0.5)
    red = [255, 0, 0]
    for n, contour in enumerate(contours):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red

    # map = prob_map[0]
    map = prob_map
    size = map.shape
    map[:, 0] = np.zeros(size[0])
    map[:, size[1] - 1] = np.zeros(size[0])
    map[0, :] = np.zeros(size[1])
    map[size[0] - 1, :] = np.zeros(size[1])

    contours = measure.find_contours(map, 0.5)

    green = [0, 255, 0]
    for n, contour in enumerate(contours):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = green
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = green
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = green
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = green
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = green
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = green
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = green

    patch_image = patch_image.astype(np.uint8)
    patch_image = Image.fromarray(patch_image)

    patch_image.save(path1)