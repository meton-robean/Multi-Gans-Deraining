#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
#Models lib
from models import *
#Metrics lib
from metrics import calc_psnr, calc_ssim
from data import make_dataset
from PIL import Image
from torchvision import  transforms
from metrics import  norm_range
from data import read_data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_level', default=3, type=int, help='levels of GAN')
    parser.add_argument("--mode", default='test1v1', type=str)
    parser.add_argument("--input_dir", default='/home/cmt/Baiduyun FIles/All_My_Files/All_My_DataSets/derain_datasets/rainy_image_dataset/testing/rainy_image/',  type=str)
    parser.add_argument("--output_dir",default='/home/cmt/Baiduyun FIles/All_My_Files/All_My_Coding_Tests/ML_class_derain_proj/Derain_ML_proj_v2.8/test/output/', type=str)
    parser.add_argument("--gt_dir",    default='/home/cmt/Baiduyun FIles/All_My_Files/All_My_DataSets/derain_datasets/rainy_image_dataset/testing/ground_truth/', type=str)
    parser.add_argument("--weight_dir",
                        default='/home/cmt/Baiduyun FIles/All_My_Files/All_My_Coding_Tests/ML_class_derain_proj/Derain_ML_proj_v2.8/weights',
                        type=str)
    args = parser.parse_args()
    return args


transform = transforms.Compose(
    [
        #transforms.Resize(128),
        #transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


#cmt 裁剪成4的倍数
def align_to_four(img):
    #print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    #align to four
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]
    #print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img


def predict(model, image):
    image = np.array(image, dtype='float32')/255.
    image = image.transpose((2, 0, 1))  # C W H
    image = image[np.newaxis, :, :, :]  #(1, C, W, H)
    image = torch.from_numpy(image)
    image = Variable(image).cuda()
    #image = Variable(image)   #cmt cpu
    out = model(image)[-1]

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))  #b c w h --> b w h c
    out = out[0, :, :, :]*255. #w h c
    
    return out


def predict_v2(lapgan_model, image):
    #image = np.array(image, dtype='float32') / 255.
    #image = image.transpose((2, 0, 1))  # C W H
    #image = image[np.newaxis, :, :, :]  # (1, C, W, H)
    #image = torch.from_numpy(image)
    #image = Variable(image).cuda()
    image = transform(image)
    image = image.unsqueeze(0)
    image = Variable(image).cuda()

    z=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    #z = [0.05, 0.05, 0.05, 0.05]
    for i in range(args.n_level):
        lapgan_model.Generator[i].eval()
        image=lapgan_model.Generator[i](image, z[i])
        #image = torch.abs(image - res)
    out=image

    #out = out.cpu().data
    #out = out.numpy()
    #out = out.transpose((0, 2, 3, 1))  # b c w h --> b w h c
    #out = out[0, :, :, :] * 255.  # w h c
    out = torch.squeeze(out.data.cpu())
    out = norm_range(out, None)
    out = out.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()  # W H C

    return out

def predict_v3(model, image):

    image = np.array(image, dtype='float32') / 255.
    image = image.transpose((2, 0, 1))  # C W H
    image = image[np.newaxis, :, :, :]  # (1, C, W, H)
    image = torch.from_numpy(image)
    image = Variable(image).cuda()
    #image = transform(image)
    #image = image.unsqueeze(0)
    #image = Variable(image).cuda()

    z=[0.1, 0.2, 0.4, 0.8]

    out = model(image)[-1]
    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))  # b c w h --> b w h c
    out = out[0, :, :, :] * 255.  # w h c
    #out = torch.squeeze(out.data.cpu())
    #out = norm_range(out, None)
    #out = out.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()  # W H C

    return out


def predict_v4(lapgan_model, image):
    image = np.array(image, dtype='float32') / 255.
    image = image.transpose((2, 0, 1))  # C W H
    image = image[np.newaxis, :, :, :]  # (1, C, W, H)
    image = torch.from_numpy(image)
    image = Variable(image).cuda()
    #image = transform(image)
    #image = image.unsqueeze(0)
    #image = Variable(image).cuda()

    z=[0.1, 0.2, 0.4, 0.8]

    for i in range(args.n_level):
        image=lapgan_model.Generator[i](image, z[i])
        #image = torch.abs(image - res)
    out=image

    out = out.cpu().data
    out = out.numpy()
    out = out.transpose((0, 2, 3, 1))  # b c w h --> b w h c
    out = out[0, :, :, :] * 255.  # w h c
    #out = torch.squeeze(out.data.cpu())
    #out = norm_range(out, None)
    #out = out.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()  # W H C

    return out



####### TEST ########

def test_norm(image):
    image = transform(image)
    out = torch.squeeze(image)
    out = norm_range(out, None)
    out = out.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()  # W H C
    return out


def test_batch_image():
    data = read_data(gt_path='E:\All_My_Files\All_My_DataSets\derain_datasets\Derain_ML_Proj\\final_testset\ground_truth\\',
              rain_path='E:\All_My_Files\All_My_DataSets\derain_datasets\Derain_ML_Proj\\final_testset\\rainy_image\\',
              num_channel=3,
              size_input=128,
              batch_size=1
              )
    rain_image, gt_image = data
    out1 = torch.squeeze(rain_image)
    out1 = norm_range(out1, None)
    out1 = out1.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()  # W H C
    imag1 = Image.fromarray(out1)  # W H C

    out2 = torch.squeeze(gt_image)
    out2 = norm_range(out2, None)
    out2 = out2.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()  # W H C
    imag2 = Image.fromarray(out2)  # W H C
    imag1.save('C:\\Users\Administrator\Desktop\Derain_ML_proj_v2.5\\test\output\\rain.jpg')
    imag2.save('C:\\Users\Administrator\Desktop\Derain_ML_proj_v2.5\\test\output\\gt.jpg')
    #return  out

############# TEST #################################################################

if __name__ == '__main__':
    args = get_args()

    lpagan = LPAGAN(args.n_level)   #cuda lapgan model

    # load weight
    if args.weight_dir:
        print('load weight dir from %s ....'%args.weight_dir)

        for idx in range(lpagan.n_level):
            gen_checkpoint = torch.load(args.weight_dir+'/LPAGAN_Generator_{}.pkl'.format(idx))
            print('load '+args.weight_dir+'/LPAGAN_Generator_{}.pkl'.format(idx))
            lpagan.Generator[idx].load_state_dict(gen_checkpoint)
        print('load weight file done!')

    if args.mode == 'demo':
        input_list = sorted(os.listdir(args.input_dir))
        num = len(input_list)
        for i in range(num):
            print ('Processing image: %s'%(input_list[i]))
            img = cv2.imread(args.input_dir + input_list[i])
            img = align_to_four(img)
            result = predict(model, img)
            img_name = input_list[i].split('.')[0]
            cv2.imwrite(args.output_dir + img_name + '.jpg', result)

    elif args.mode == 'test1v1':   #1 rain_img, 1 ground truth
        input_list = sorted(os.listdir(args.input_dir))
        gt_list = sorted(os.listdir(args.gt_dir))
        num = len(input_list)
        cumulative_psnr = 0
        cumulative_ssim = 0
        for i in range(num):
            print ('Processing image: %s'%(input_list[i]))

            print('test images %d path: '%(i), args.input_dir + input_list[i])
            img = cv2.imread(args.input_dir + input_list[i])
            print('gt images %d path: '%(i), args.gt_dir + gt_list[i])
            gt = cv2.imread(args.gt_dir + gt_list[i])

            img = align_to_four(img)
            gt = align_to_four(gt)

            result = predict_v2(lpagan, img)
            result = np.array(result, dtype = 'uint8')

            cur_psnr = calc_psnr(result, gt)
            cur_ssim = calc_ssim(result, gt)
            #print('save {} image in '.format(i) + args.output_dir+"%d_derain.jpg" % (i) )
            #cv2.imwrite(args.output_dir+"%d_derain.jpg" % (i), result)  # save the result 发现路径有中文保存不了
            cv2.imwrite(args.output_dir+'{}.jpg'.format(i), result)  # save the result
            print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
            cumulative_psnr += cur_psnr
            cumulative_ssim += cur_ssim
        print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr/num, cumulative_ssim/num))

    else:
        print ('Mode Invalid!')

