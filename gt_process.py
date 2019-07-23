import os
from PIL import Image
'''
  因为DDN数据集是 1 gt 对应 14张雨图，这里将 1 gt 复制14份，并按照格式命好名字对应14张雨图
'''
IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_gt_dataset(dir):
    images = []
    if not os.path.isdir(dir):
        raise Exception('Check dataroot')
    image_names=os.listdir(dir)
    print(image_names)
    for fname in image_names:
        print(fname)
        if is_image_file(fname):
            path = os.path.join(dir, fname)
            print(path[:-4])
            img = Image.open(path).convert('RGB')
            for j in range(1,15):
                path2 = path[:-4]+'_{}.jpg'.format(j)
                img.save(path2)
            os.remove(path)
    return images

#make_gt_dataset('E:\All_My_Files\All_My_DataSets\derain_datasets\\rainy_image_dataset\\training\ground_truth\\')
make_gt_dataset('E:\All_My_Files\All_My_DataSets\derain_datasets\\rainy_image_dataset\\testing\ground_truth\\')