
import cv2
from skimage.measure import compare_psnr, compare_ssim
import torch
import os
import sys
import numpy as np
from PIL import  Image
import data
import skimage

'''
def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)


def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)
'''

def calc_psnr(im1, im2):

    x = compare_psnr(im1[:,:,0], im2[:,:,0])
    y = compare_psnr(im1[:,:,1], im2[:,:,1])
    z = compare_psnr(im1[:,:,2], im2[:,:,2])
    return (x+y+z)/3



def calc_ssim(im1, im2):
    x = compare_ssim(im1[:,:,0], im2[:,:,0])
    y = compare_ssim(im1[:,:,1], im2[:,:,1])
    z = compare_ssim(im1[:,:,2], im2[:,:,2])
    return (x+y+z)/3



def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img


def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
    return norm_ip(t, t.min(), t.max())


def get_residual_tensor(im1, im2):
    '''
    :param im1: （B,C,W,H）tensor
    :param im2: （B,C,W,H）tensor
    :return: residual of im1 and im2
    '''
    return im1-im2


def save_image_from_tensor(tensor_images, path):
    '''
    :param tensor_images: (B, C ,W, H)
    :return: images saved in disk
    '''

    tensor_images = norm_range(tensor_images, None)
    numpy_images = tensor_images.mul(255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()

    for i in range(numpy_images.shape[0]):        #batch
        image = Image.fromarray(numpy_images[i])  # W H C
        image.save(path+'/test_{}.jpg'.format(i))

    print('convert tensor_images to numpy_image, and save in disk...done!')




###### TEST ######

def test_get_residual():
    rain_path='E:\All_My_Files\All_My_DataSets\derain_datasets\Derain_ML_Proj\\final_testset\\rainy_image'
    gt_path = 'E:\All_My_Files\All_My_DataSets\derain_datasets\Derain_ML_Proj\\final_testset\ground_truth'
    batch=3
    dataloader = iter( data.get_dataloader(gt_path=gt_path, rain_path=rain_path, batchSize=batch ) )
    gt_image, rain_image= next(dataloader)
    print(gt_image.size())
    print(rain_image.size())

    residual = get_residual_tensor(gt_image, rain_image)
    print(residual.size())
    save_image_from_tensor(residual, './test')

###### TEST ######
