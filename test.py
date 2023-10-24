# from torch.autograd import Variable
# from PIL import Image
# from torchvision.transforms import ToTensor
# import argparse
# import os
# from new import NAFSSR
# import utils
# import torch
# from torchvision import transforms

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--testset_dir', type=str, default='../datasets/test/')
#     parser.add_argument('--scale_factor', type=int, default=4)
#     parser.add_argument('--device', type=str, default='cuda:2')
#     parser.add_argument('--model_name', type=str, default='iPASSR_4xSR_epoch80')
#     return parser.parse_args()


# def test(cfg):
#     net = NAFSSR().to(cfg.device)
#     model_restoration = NAFSSR()
#     utils.load_checkpoint(model_restoration,'./new_res4x_48_8/model_best.pth')
#     file_list = os.listdir(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor))
#     for idx in range(len(file_list)):
#         LR_left = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr0.png')
#         LR_right = Image.open(cfg.testset_dir + cfg.dataset + '/lr_x' + str(cfg.scale_factor) + '/' + file_list[idx] + '/lr1.png')

#         LR_left, LR_right = ToTensor()(LR_left), ToTensor()(LR_right)
#         LR_left, LR_right = Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)
#         scene_name = file_list[idx]
#         print('Running Scene ' + scene_name + ' of ' + cfg.dataset + ' Dataset......')
#         input = torch.cat([LR_left,LR_right],axis=1)
#         print(input.shape)
#         with torch.no_grad():
#              restored = model_restoration = net(input)
#         restored = restored.chunk(2, dim=1)
#         SR_left, SR_right = restored[0], restored[1]
#         SR_left, SR_right = torch.clamp(SR_left, 0, 1), torch.clamp(SR_right, 0, 1)
#         save_path = './results2/' +  str(cfg.scale_factor)+ '/' + cfg.dataset + '/' + scene_name
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
#         SR_left_img.save(save_path +  '/hr0.png')
#         SR_right_img = transforms.ToPILImage()(torch.squeeze(SR_right.data.cpu(), 0))
#         SR_right_img.save(save_path +  '/hr1.png')


# if __name__ == '__main__':
#     cfg = parse_args()
#     # dataset_list = ['Flickr1024', 'KITTI2012', 'KITTI2015', 'Middlebury']
#     dataset_list = [ 'KITTI2012']
#     for i in range(len(dataset_list)):
#         cfg.dataset = dataset_list[i]
#         test(cfg)
#     print('Finished!')


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

from data_RGB import get_validation_data

from new import NAFSSR

from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deblurring using MPRNet')

parser.add_argument('--input_dir', default='../datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./newfo_v/4x1', type=str, help='Directory for results')
parser.add_argument('--weights', default='./new_res4x_48_81/checkpoint_epoch134.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='Flickr1024', type=str, help='Test Dataset') # ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--gpus', default='2', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = NAFSSR()
print("===>Testing using weights: ",args.weights)
utils.load_checkpoint(model_restoration,args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

dataset = args.dataset
rgb_dir_test = os.path.join(args.input_dir, dataset)
test_dataset = get_validation_data(rgb_dir_test, 4)
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

result_dir  = os.path.join(args.result_dir, dataset)
utils.mkdir(result_dir)

with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        input_    = data_test[1].cuda()
        filenames = data_test[2]
   


        restored = model_restoration(input_)
        restored = restored.chunk(2, dim=1)
        left = restored[0]
        right= restored[1]

        left = torch.clamp(left,0,1)
        
        right = torch.clamp(right,0,1)

        left = left.permute(0, 2, 3, 1).cpu().detach().numpy()
        right = right.permute(0, 2, 3, 1).cpu().detach().numpy()

        for batch in range(len(right)):
            right = img_as_ubyte(right[batch])
            save_path = os.path.join(result_dir,filenames[batch])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            utils.save_img(save_path+'/hr1.png', right)

            left = img_as_ubyte(left[batch])
            save_path = os.path.join(result_dir,filenames[batch])
            utils.save_img(save_path+'/hr0.png', left)