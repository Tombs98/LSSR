import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class TrainSetLoader(Dataset):
    def __init__(self, trainset_dir, scale_factor):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = trainset_dir + '/patches_x' + str(scale_factor)
        self.file_list = os.listdir(self.dataset_dir)
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr1.png')
        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr0.png')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr1.png')
        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        img_hr_left, img_hr_right, img_lr_left, img_lr_right = augmentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)
        img_gt = np.concatenate([img_hr_left, img_hr_right], axis=-1)
        img_lq = np.concatenate([img_lr_left, img_lr_right], axis=-1)
        return toTensor(img_gt), toTensor(img_lq)
        # return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)

    def __len__(self):
        return len(self.file_list)
    
class TrainSetLoader2(Dataset):
    def __init__(self, trainset_dir, scale_factor):
        super(TrainSetLoader2, self).__init__()
        self.dataset_dir = trainset_dir
        self.file_list = os.listdir(self.dataset_dir)
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hl_'+ self.file_list[index]+'.png')
        img_hr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/hr_'+ self.file_list[index]+'.png')

        img_lr_left  = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/ll_'+ self.file_list[index]+'.png')
        img_lr_right = Image.open(self.dataset_dir + '/' + self.file_list[index] + '/lr_'+ self.file_list[index]+'.png')
        # 指定裁剪的大小
        crop_size = (256, 256)

        # 随机裁剪图像
        img_hr_left = random_crop(img_hr_left, crop_size)
        img_hr_right = random_crop(img_hr_right, crop_size)
        img_lr_left = random_crop(img_lr_left, crop_size)
        img_lr_right = random_crop(img_lr_right, crop_size)


        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        img_hr_left, img_hr_right, img_lr_left, img_lr_right = augmentation(img_hr_left, img_hr_right, img_lr_left, img_lr_right)
        img_gt = np.concatenate([img_hr_left, img_hr_right], axis=-1)
        img_lq = np.concatenate([img_lr_left, img_lr_right], axis=-1)
        return toTensor(img_gt), toTensor(img_lq)
        # return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)

    def __len__(self):
        return len(self.file_list)

class ValSetLoader(Dataset):
    def __init__(self, trainset_dir, scale_factor):
        super(ValSetLoader, self).__init__()
        self.gt_dir = trainset_dir + '/hr' 
        self.low_dir = trainset_dir + '/lr_x' + str(scale_factor) 
        self.file_list = os.listdir(self.gt_dir)
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.gt_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.gt_dir + '/' + self.file_list[index] + '/hr1.png')

        img_lr_left  = Image.open(self.low_dir + '/' + self.file_list[index] + '/lr0.png')
        img_lr_right = Image.open(self.low_dir + '/' + self.file_list[index] + '/lr1.png')

        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        img_gt = np.concatenate([img_hr_left, img_hr_right], axis=-1)
        img_lq = np.concatenate([img_lr_left, img_lr_right], axis=-1)
        filrname = self.file_list[index] 
        return toTensor(img_gt), toTensor(img_lq), filrname
        # return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)

    def __len__(self):
        return len(self.file_list)
    
class ValSetLoader2(Dataset):
    def __init__(self, trainset_dir, scale_factor):
        super(ValSetLoader2, self).__init__()
        self.gt_dir = trainset_dir + '/hr' 
        self.file_list = os.listdir(self.gt_dir)
    def __getitem__(self, index):
        img_hr_left  = Image.open(self.gt_dir + '/' + self.file_list[index] + '/hr0.png')
        img_hr_right = Image.open(self.gt_dir + '/' + self.file_list[index] + '/hr1.png')

        img_lr_left  = Image.open(self.gt_dir + '/' + self.file_list[index] + '/hr0_noisy20.png')
        img_lr_right = Image.open(self.gt_dir + '/' + self.file_list[index] + '/hr1_noisy20.png')

          # 指定裁剪的大小
        crop_size = (256, 256)

        # 随机裁剪图像
        img_hr_left = random_crop(img_hr_left, crop_size)
        img_hr_right = random_crop(img_hr_right, crop_size)
        img_lr_left = random_crop(img_lr_left, crop_size)
        img_lr_right = random_crop(img_lr_right, crop_size)



        img_hr_left  = np.array(img_hr_left,  dtype=np.float32)
        img_hr_right = np.array(img_hr_right, dtype=np.float32)
        img_lr_left  = np.array(img_lr_left,  dtype=np.float32)
        img_lr_right = np.array(img_lr_right, dtype=np.float32)
        img_gt = np.concatenate([img_hr_left, img_hr_right], axis=-1)
        img_lq = np.concatenate([img_lr_left, img_lr_right], axis=-1)

        return toTensor(img_gt), toTensor(img_lq)
        # return toTensor(img_hr_left), toTensor(img_hr_right), toTensor(img_lr_left), toTensor(img_lr_right)

    def __len__(self):
        return len(self.file_list)

def random_crop(image, crop_size):
    width, height = image.size
    crop_width, crop_height = crop_size
    
    if crop_width > width or crop_height > height:
        raise ValueError("Crop size exceeds image dimensions.")
    
    x = random.randint(0, width - crop_width)
    y = random.randint(0, height - crop_height)
    
    cropped_image = image.crop((x, y, x + crop_width, y + crop_height))
    
    return cropped_image

def augmentation(hr_image_left, hr_image_right, lr_image_left, lr_image_right):
        if random.random()<0.5:     # flip horizonly
            lr_image_left_ = lr_image_right[:, ::-1, :]
            lr_image_right_ = lr_image_left[:, ::-1, :]
            hr_image_left_ = hr_image_right[:, ::-1, :]
            hr_image_right_ = hr_image_left[:, ::-1, :]
            lr_image_left, lr_image_right = lr_image_left_, lr_image_right_
            hr_image_left, hr_image_right = hr_image_left_, hr_image_right_

        if random.random()<0.5:     #flip vertically
            lr_image_left = lr_image_left[::-1, :, :]
            lr_image_right = lr_image_right[::-1, :, :]
            hr_image_left = hr_image_left[::-1, :, :]
            hr_image_right = hr_image_right[::-1, :, :]
        idx = [
                    [0, 1, 2],
                    [0, 2, 1],
                    [1, 0, 2],
                    [1, 2, 0],
                    [2, 0, 1],
                    [2, 1, 0],
                ][int(np.random.rand() * 6)]
        lr_image_left = lr_image_left[:, :, idx]
        lr_image_right = lr_image_right[:, :, idx]
        hr_image_left = hr_image_left[:, :, idx]
        hr_image_right = hr_image_right[:, :, idx]

        return np.ascontiguousarray(hr_image_left), np.ascontiguousarray(hr_image_right), \
                np.ascontiguousarray(lr_image_left), np.ascontiguousarray(lr_image_right)

def toTensor(img):
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)

