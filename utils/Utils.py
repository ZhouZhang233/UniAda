import os.path as osp
import numpy as np
import os
import cv2
from skimage import morphology
import scipy
from PIL import Image, ImageDraw, ImageFont
from matplotlib.pyplot import imsave
# from keras.preprocessing import image
from skimage.measure import label, regionprops
from skimage.transform import rotate, resize
from skimage import measure, draw
import torch
from skimage.morphology import disk, erosion, dilation, opening, closing, white_tophat
from scipy import ndimage
# import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import ttest_1samp, wilcoxon
import scipy.stats as stats
import pandas as pd

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def statistics_analysis(dice_values):
    # 将列表转换为numpy数组
    dice_values = np.array(dice_values)

    # 创建DataFrame
    df = pd.DataFrame(dice_values, columns=['Dice'])
    # 描述性统计
    desc_stats = df.describe()
    print(desc_stats)

    # 正态性检验
    stat, p_value_normality = shapiro(dice_values)
    print(f"正态性检验 p-value: {p_value_normality}")

    if p_value_normality > 0.05:
        print("数据服从正态分布")
    else:
        print("数据不服从正态分布")

    # 基准值
    benchmark = 0.8

    # 检验
    if p_value_normality > 0.05:
        # 单样本t检验
        t_stat, p_value = ttest_1samp(dice_values, benchmark)
        test_used = '单样本t检验'
    else:
        # 单样本Wilcoxon检验
        t_stat, p_value = wilcoxon(dice_values - benchmark)
        test_used = '单样本Wilcoxon检验'

    print(f"使用的统计检验方法: {test_used}")
    print(f"统计量: {t_stat}")
    print(f"p-value: {p_value}")

    # 置信区间计算
    confidence_level = 0.95
    degrees_freedom = len(dice_values) - 1
    sample_mean = np.mean(dice_values)
    sample_standard_error = stats.sem(dice_values)
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)

    print(f"95% 置信区间: {confidence_interval}")

    return t_stat, p_value

def _connectivity_region_analysis(mask):
    s = [[0,1,0],
         [1,1,1],
         [0,1,0]]
    label_im, nb_labels = ndimage.label(mask)#, structure=s)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)        
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im

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

def postprocessing(prediction, threshold=0.75, dataset='G'):
    if dataset[0] == 'D':
        # prediction = prediction.numpy()
        prediction_copy = np.copy(prediction)
        disc_mask = prediction[1]
        cup_mask = prediction[0]
        disc_mask = (disc_mask > 0.5)  # return binary mask
        cup_mask = (cup_mask > 0.1)  # return binary mask
        disc_mask = disc_mask.astype(np.uint8)
        cup_mask = cup_mask.astype(np.uint8)
        # for i in range(5):
        #     disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        #     cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        # cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        return prediction_copy
    else:
        # prediction = torch.sigmoid(prediction).data.cpu().numpy()

        # disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        # cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
        # disc_mask = morphology.erosion(disc_mask, morphology.diamond(3))  # return 0,1
        # cup_mask = morphology.erosion(cup_mask, morphology.diamond(3))  # return 0,1

        prediction_copy = np.copy(prediction)
        prediction_copy = (prediction_copy > threshold)  # return binary mask
        prediction_copy = prediction_copy.astype(np.uint8)
        disc_mask = prediction_copy[1]
        cup_mask = prediction_copy[0]
        disc_mask = get_largest_fillhole(disc_mask).astype(np.bool)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.bool)
        prediction_copy[0] = cup_mask
        prediction_copy[1] = disc_mask
        # selem = disk(6)
        # disc_mask = morphology.closing(disc_mask, selem)
        # cup_mask = morphology.closing(cup_mask, selem)
        # print(sum(disc_mask))


        return prediction_copy

def postprocessing_prostate(prediction, threshold=0.75, dataset='prostate'):
    prediction = np.asarray((prediction > threshold), dtype=np.bool)  # return binary mask
    # prediction_copy = prediction_copy
    prediction_copy = np.asarray(get_largest_fillhole(prediction), dtype=np.bool)  # return 0,1
    return prediction_copy

def joint_val_image(image, prediction, mask):
    ratio = 0.5
    _pred_cup = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _pred_disc = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    _mask = np.zeros([mask.shape[-2], mask.shape[-1], 3])
    image = np.transpose(image, (1, 2, 0))

    _pred_cup[:, :, 0] = prediction[0]
    _pred_cup[:, :, 1] = prediction[0]
    _pred_cup[:, :, 2] = prediction[0]
    _pred_disc[:, :, 0] = prediction[1]
    _pred_disc[:, :, 1] = prediction[1]
    _pred_disc[:, :, 2] = prediction[1]
    _mask[:,:,0] = mask[0]
    _mask[:,:,1] = mask[1]

    pred_cup = np.add(ratio * image, (1 - ratio) * _pred_cup)
    pred_disc = np.add(ratio * image, (1 - ratio) * _pred_disc)
    mask_img = np.add(ratio * image, (1 - ratio) * _mask)

    joint_img = np.concatenate([image, mask_img, pred_cup, pred_disc], axis=1)
    return joint_img


def save_val_img(path, epoch, img):
    name = osp.join(path, "visualization", "epoch_%d.png" % epoch)
    out = osp.join(path, "visualization")
    if not osp.exists(out):
        os.makedirs(out)
    img_shape = img[0].shape
    stack_image = np.zeros([len(img) * img_shape[0], img_shape[1], img_shape[2]])
    for i in range(len(img)):
        stack_image[i * img_shape[0] : (i + 1) * img_shape[0], :, : ] = img[i]
    imsave(name, stack_image)


def save_per_img_prostate(patch_image, data_save_path, img_name, prob_map, gt=None, mask_path=None, ext="bmp"):
    # path1 = os.path.join(data_save_path, 'overlay', img_name.split('.')[0]+'.png')

    path0 = os.path.join(data_save_path, 'original_image', img_name +'.png')
    if not os.path.exists(os.path.dirname(path0)):
        os.makedirs(os.path.dirname(path0))


    patch_image  = (patch_image - np.min(patch_image))/(np.max(patch_image)-np.min(patch_image)) *255
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

    from medpy.metric import binary
    dice = binary.dc(prob_map, gt)
    dice = str(np.around(dice, decimals=4))
    path1 = os.path.join(data_save_path, 'overlay_out2', img_name + "_dice" + dice + '.png')

    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))

    # draw = ImageDraw.Draw(patch_image)
    # dice = str(np.around(dice, decimals=4))
    # disc_dice = str(np.around(disc_dice, decimals=4))
    # font = ImageFont.truetype("arial.ttf", 40)
    # position_cup = (10, 10)
    # position_disc = (10, 90)
    # draw.text(position_cup, "dice: " + dice, font=font)

    patch_image.save(path1)


def save_per_img_npc(patch_image, data_save_path, img_name, prob_map, gt=None, mask_path=None, ext="bmp"):
    # path1 = os.path.join(data_save_path, 'overlay', img_name.split('.')[0]+'.png')

    path0 = os.path.join(data_save_path, 'original_image', img_name + '.png')
    if not os.path.exists(os.path.dirname(path0)):
        os.makedirs(os.path.dirname(path0))


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
        patch_image[(contour[:, 0] ).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] ).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = red
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

    from medpy.metric import binary
    dice = binary.dc(prob_map, gt)

    # draw = ImageDraw.Draw(patch_image)
    dice = str(np.around(dice, decimals=4))
    # font = ImageFont.truetype("arial.ttf", 40)
    # position_cup = (10, 10)
    # position_disc = (10, 90)
    # draw.text(position_cup, "dice: " + dice, font=font)
    path1 = os.path.join(data_save_path, '2d', img_name + "_dice" + dice +'.png')
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))

    patch_image.save(path1)

def save_per_img(patch_image, data_save_path, img_name, prob_map, gt, cup_dice, disc_dice, mask_path=None, ext="bmp"):
    path1 = os.path.join(data_save_path, 'overlay', img_name.split('.')[0]+'.png')
    path0 = os.path.join(data_save_path, 'original_image', img_name.split('.')[0]+'.png')
    if not os.path.exists(os.path.dirname(path0)):
        os.makedirs(os.path.dirname(path0))
    if not os.path.exists(os.path.dirname(path1)):
        os.makedirs(os.path.dirname(path1))

    disc_map = prob_map[0]
    cup_map = prob_map[1]
    size = disc_map.shape
    disc_map[:, 0] = np.zeros(size[0])
    disc_map[:, size[1] - 1] = np.zeros(size[0])
    disc_map[0, :] = np.zeros(size[1])
    disc_map[size[0] - 1, :] = np.zeros(size[1])
    size = cup_map.shape
    cup_map[:, 0] = np.zeros(size[0])
    cup_map[:, size[1] - 1] = np.zeros(size[0])
    cup_map[0, :] = np.zeros(size[1])
    cup_map[size[0] - 1, :] = np.zeros(size[1])

    # disc_mask = (disc_map > 0.75) # return binary mask
    # cup_mask = (cup_map > 0.75)
    # disc_mask = disc_mask.astype(np.uint8)
    # cup_mask = cup_mask.astype(np.uint8)


    contours_disc = measure.find_contours(disc_map, 0.5)
    contours_cup = measure.find_contours(cup_map, 0.5)

    # green
    for n, contour in enumerate(contours_cup):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 255, 0]
    # blue
    for n, contour in enumerate(contours_disc):
        patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = [0, 0, 255]

    disc_mask = get_largest_fillhole(gt[0]).astype(np.uint8)  # return 0,1
    cup_mask = get_largest_fillhole(gt[1]).astype(np.uint8)

    contours_disc = measure.find_contours(disc_mask, 0.5)
    contours_cup = measure.find_contours(cup_mask, 0.5)
    red = [255, 0, 0]
    for n, contour in enumerate(contours_cup):
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red

    for n, contour in enumerate(contours_disc):
        patch_image[contour[:, 0].astype(int), contour[:, 1].astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] + 1.0).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] + 1.0).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1]).astype(int), :] = red
        patch_image[(contour[:, 0] - 1.0).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red
        patch_image[(contour[:, 0]).astype(int), (contour[:, 1] - 1.0).astype(int), :] = red


    patch_image = patch_image.astype(np.uint8)
    patch_image = Image.fromarray(patch_image)

    draw = ImageDraw.Draw(patch_image)
    cup_dice = str(np.around(cup_dice, decimals=4))
    disc_dice = str(np.around(disc_dice, decimals=4))
    font = ImageFont.truetype("arial.ttf", 80)
    position_cup = (10, 10)
    position_disc = (10, 90)
    draw.text(position_cup, "cup:  "+cup_dice, font=font)
    draw.text(position_disc, "disc:  " + disc_dice, font=font)

    patch_image.save(path1)