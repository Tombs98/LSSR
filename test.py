

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
