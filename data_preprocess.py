import SimpleITK as sitk
import numpy as np
import os
import os.path as osp

root_im = "../data/prostate/Domain1/image_volume"
root_label = "../data/prostate/Domain1/mask_volume"

save_im = "../data/prostate/Domain1/image_npy"
save_label = "../data/prostate/Domain1/mask_npy"

if not osp.exists(save_im):
    os.makedirs(save_im)
if not osp.exists(save_label):
    os.makedirs(save_label)

im_list = sorted(os.listdir(root_im))
label_list = sorted(os.listdir(root_label))

for i, im_name in enumerate(im_list):

    label_name = label_list[i]
    print(im_name, label_name)
    itk_image = sitk.ReadImage(osp.join(root_im, im_name))
    image = sitk.GetArrayFromImage(itk_image).transpose(1,2,0)
    itk_gt = sitk.ReadImage(osp.join(root_label, label_name))
    gt = sitk.GetArrayFromImage(itk_gt).transpose(1,2,0)

    im_min = np.min(image)
    im_max = np.max(image)
    image = (image - im_min) / (im_max - im_min)

    gt = np.asarray(gt > 0, dtype=np.uint8)

    for s in range(1, image.shape[2]-1):
        im_slice = image[:, :, s-1:s+2]
        label_name = gt[:, :, s]

        if np.sum(label_name) > 0:
            np.save(osp.join(save_im, im_name.split(".")[0]+"_"+str(s)), im_slice)
            np.save(osp.join(save_label, im_name.split(".")[0] + "_" + str(s)), label_name)





