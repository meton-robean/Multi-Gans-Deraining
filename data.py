
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.utils.data as data
import random
import matplotlib.image as imgplot
import cv2



IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]


transform = transforms.Compose(
    [
        #transforms.Resize(128),
        #transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def sample_data(path, batch_size):
    '''
      生成器 ，需要iter, next ..来使用
    '''
    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=False, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def make_dataset(dir):
    images = []
    if not os.path.isdir(dir):
        raise Exception('Check dataroot')
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(dir, fname)
                item = path
                images.append(item)

    return images


#自定义的data读取函数，在预处理方面更加灵活
def read_data(gt_path, rain_path,  num_channel, size_input, batch_size=32, transform=transform ):
    '''
      load a batch data into memory ....
    '''
    gt_imgs_list = make_dataset(gt_path)
    rain_imgs_list = make_dataset(rain_path)
    Data = torch.zeros(batch_size, num_channel, size_input, size_input)
    Label = torch.zeros(batch_size, num_channel, size_input, size_input)

    for i in range(batch_size):
        r_idx = np.random.randint(0, len(rain_imgs_list)-1)
        #rain_img = Image.open(rain_imgs_list[r_idx]).convert('RGB')
        rain_img = cv2.imread(rain_imgs_list[r_idx])
        #print('rain_img:', rain_imgs_list[r_idx])
        #gt_img = Image.open( gt_imgs_list[r_idx]).convert('RGB')
        gt_img = cv2.imread(gt_imgs_list[r_idx])
        #print( 'gt:', gt_imgs_list[r_idx] )

        x = random.randint(0,rain_img.shape[0] - size_input)
        y = random.randint(0,rain_img.shape[1] - size_input)
        subim_input = rain_img[x : x+size_input, y : y+size_input, :]
        subim_label = gt_img[x : x+size_input, y : y+size_input, :]

        subim_input = transform(subim_input)  #auto change (B W H C)->(B C W H)
        subim_label = transform(subim_label)
        Data[i,:,:,:] =  subim_input
        Label[i,:,:,:] = subim_label

    return Data, Label    #rain_img bacth , gt_img batch


def read_data_test2(input_path, gt_path, size_input, num_channel, batch_size):
    Data = np.zeros((batch_size, size_input, size_input, num_channel))
    Label = np.zeros((batch_size, size_input, size_input, num_channel))
    input_files=make_dataset(input_path)
    gt_files=make_dataset(gt_path)
    for i in range(batch_size):

        r_idx = random.randint(0, len(input_files) - 1)

        rainy = imgplot.imread(input_files[r_idx])
        # print(input_path + input_files[r_idx])
        if np.max(rainy) > 1:
            rainy = rainy / 255.0

        label = imgplot.imread(gt_files[r_idx])
        # print(gt_path + gt_files[r_idx])
        if np.max(label) > 1:
            label = label / 255.0

        x = random.randint(0, rainy.shape[0] - size_input)
        y = random.randint(0, rainy.shape[1] - size_input)
        subim_input = rainy[x: x + size_input, y: y + size_input, :]
        subim_label = label[x: x + size_input, y: y + size_input, :]
        Data[i, :, :, :] = subim_input
        Label[i, :, :, :] = subim_label

    return Data, Label



class RainDataset(data.Dataset):
    '''
        custom rain Dataset CLASS
    '''
    def __init__(self, gt_path, rain_path, transform=None ):
        gt_imgs_list = make_dataset(gt_path)
        rain_imgs_list = make_dataset(rain_path)

        if len(gt_imgs_list) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + gt_path + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        if len(rain_imgs_list) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + rain_path + "\n"
                    "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.gt_imgs_list = gt_imgs_list
        self.rain_imgs_list = rain_imgs_list
        self.transform = transform

        #print('gt_imgs_list:', gt_imgs_list)
        #print('rain_imgs_list:', rain_imgs_list)

        gt_rainimg14_dic={}
        rainimg14=[]

        for rain_img_idx, rain_img_path in enumerate(self.rain_imgs_list):
            rainimg14.append(rain_img_path)
            if ((rain_img_idx + 1) % 14) == 0:
                gt_img_path=self.gt_imgs_list[((rain_img_idx+1)//14)-1]
                gt_rainimg14_dic[gt_img_path]=rainimg14
                rainimg14=[]
        self.gt_rain_img_dic = gt_rainimg14_dic
        #print('gt_rain_img_dic: ', self.gt_rain_img_dic, len(self.gt_rain_img_dic))



    def __getitem__(self, index):
        random_rain_img_idx = np.random.randint(0, 14)  # 1 gt image -->14 rain images
        random_gt_img_idx = np.random.randint(0, 99)    # 100 gt images

        gt_img = Image.open( self.gt_imgs_list[random_gt_img_idx] ).convert('RGB')
        rain_img = Image.open( self.gt_rain_img_dic[ self.gt_imgs_list[random_gt_img_idx] ][random_rain_img_idx] ).convert('RGB')
        #print( 'get gt_img: ', self.gt_imgs_list[random_gt_img_idx] )
        #print( 'get rain_img: ', self.gt_rain_img_dic[self.gt_imgs_list[random_gt_img_idx]][random_rain_img_idx] )
        if self.transform is not None:
            gt_img = self.transform(gt_img)
            rain_img = self.transform(rain_img)

        return  rain_img, gt_img


    def __len__(self):
        return len(self.rain_imgs_list)


#使用dataloader.....
def get_dataloader(gt_path, rain_path, batchSize=32, shuffle=True, workers=4, transform=transform):
    '''
    :param gt_path:    ground truth path
    :param rain_path:  rain images path
    :param transform:  transform operation
    :return: rain_img gt_img pair
    '''
    dataset = RainDataset(gt_path,  rain_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batchSize,
                                           shuffle=shuffle,
                                           num_workers=int(workers))

    return dataloader





##### TEST #####

def test_dataloader():
    datalaoder = get_dataloader(gt_path='E:\All_My_Files\All_My_DataSets\derain_datasets\Derain_ML_Proj\\final_testset\ground_truth',
                   rain_path='E:\All_My_Files\All_My_DataSets\derain_datasets\Derain_ML_Proj\\final_testset\\rainy_image',
                   batchSize=1,
                   shuffle=True,
                   transform=transform
                   )

    for i, data in enumerate(datalaoder):
        rain_img, gt_img =data
        print(i+1, rain_img.size(), gt_img.size())


def test_read_data():
    read_data(gt_path='E:\All_My_Files\All_My_DataSets\derain_datasets\\rainy_image_dataset\\training\ground_truth\\',
              rain_path='E:\All_My_Files\All_My_DataSets\derain_datasets\\rainy_image_dataset\\training\\rainy_image\\',
              num_channel=3,
              size_input=128,
              batch_size=32
              )


##### TEST #####
