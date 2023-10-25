import os
from config import Config

opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

import random
import time
import numpy as np
from pathlib import Path

import utils
from data_RGB import get_training_data, get_validation_data
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx
from torchstat import  stat
#from naf import NAFNet
from new import NAFNetSR, NAFSSR


dir_checkpoint = Path('./new_res4x_48_81/')

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models',  session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir   = opt.TRAINING.VAL_DIR
scale = opt.TRAINING.SCALE
is_training = opt.TRAINING.IS_TRAINING

######### Model ###########

model_restoration = NAFSSR()
print("Total number of param  is ", sum(x.numel() for x in model_restoration.parameters()))
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")


new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.9),eps=1e-8)


######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS-warmup_epochs, eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest    = './new_res4x_48_81/model_best.pth'
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    # start_epoch = 45
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
      scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)
    print("duoka")

######### Loss ###########
criterion_L1 = nn.L1Loss()


######### DataLoaders ###########
train_dataset = get_training_data(train_dir, scale)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False, pin_memory=True)

val_dataset = get_validation_data(val_dir, scale)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)

val_dataset2 = get_validation_data('../datasets/test/KITTI2015', scale)
val_loader2 = DataLoader(dataset=val_dataset2, batch_size=1, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)

val_dataset3 = get_validation_data('../datasets/test/Flickr1024', scale)
val_loader3 = DataLoader(dataset=val_dataset3, batch_size=1, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)

val_dataset4 = get_validation_data('../datasets/test/Middlebury', scale)
val_loader4 = DataLoader(dataset=val_dataset4, batch_size=1, shuffle=False, num_workers=16, drop_last=False, pin_memory=True)



print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0
global_step = 0

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    psnr_train_rgb = []
    psnr_train_rgb2 = []
    psnr_train_rgb3 = []
    psnr_train_rgb4 = []
    psnr_tr = 0
    psnr_tr1 = 0
    ssim_tr1 = 0
    
    model_restoration.train()
    accumulation_steps = 4
    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None
        target = data[0].cuda()
        input_ = data[1].cuda()
        # print(target.shape)
        # print(input_.shape)
        
        restored = model_restoration(input_)
        restored = restored.chunk(2, dim=1)
        target = target.chunk(2, dim=1)

        ''' SR Loss '''
        loss = criterion_L1(target[0], restored[0]) + criterion_L1(target[1], restored[1])


        loss.backward()


   

        optimizer.step()
        epoch_loss += loss.item()
        global_step = global_step+1


    psnr_te = 0
    psnr_te_1 = 0
    ssim_te_1 = 0
    #### Evaluation ####
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        psnr_val_rgb2 = []
        psnr_val_rgb3 = []
        psnr_val_rgb4 = []
        for ii, data in enumerate((val_loader), 0):
            target = data[0].cuda()
            input_ = data[1].cuda()


            with torch.no_grad():
                 restored = model_restoration(input_)
            
            restored = restored.chunk(2, dim=1)
            target = target.chunk(2, dim=1)
            p1 = utils.torchPSNR(restored[0], target[0])
            p2 = utils.torchPSNR(restored[1], target[1])
            p = (p1+p2)/2
            psnr_val_rgb.append(p)
     
        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()


        for ii, data in enumerate((val_loader2), 0):
            target2 = data[0].cuda()
            input_2 = data[1].cuda()


            with torch.no_grad():
                 restored2 = model_restoration(input_2)
            
            restored2 = restored2.chunk(2, dim=1)
            target2 = target2.chunk(2, dim=1)
            p12 = utils.torchPSNR(restored2[0], target2[0])
            p22 = utils.torchPSNR(restored2[1], target2[1])
            p02 = (p12+p22)/2
            psnr_val_rgb2.append(p02)
     
        psnr_val_rgb2 = torch.stack(psnr_val_rgb2).mean().item()


        for ii, data in enumerate((val_loader3), 0):
            target = data[0].cuda()
            input_ = data[1].cuda()


            with torch.no_grad():
                 restored = model_restoration(input_)
            
            restored = restored.chunk(2, dim=1)
            target = target.chunk(2, dim=1)
            p1 = utils.torchPSNR(restored[0], target[0])
            p2 = utils.torchPSNR(restored[1], target[1])
            p = (p1+p2)/2
            psnr_val_rgb3.append(p)
     
        psnr_val_rgb3 = torch.stack(psnr_val_rgb3).mean().item()


        for ii, data in enumerate((val_loader4), 0):
            target = data[0].cuda()
            input_ = data[1].cuda()


            with torch.no_grad():
                 restored = model_restoration(input_)
            
            restored = restored.chunk(2, dim=1)
            target = target.chunk(2, dim=1)
            p1 = utils.torchPSNR(restored[0], target[0])
            p2 = utils.torchPSNR(restored[1], target[1])
            p = (p1+p2)/2
            psnr_val_rgb4.append(p)
     
        psnr_val_rgb4 = torch.stack(psnr_val_rgb4).mean().item()
        # psnr_val_rgb2 = 0
        # psnr_val_rgb3 = 0
        # psnr_val_rgb4 = 0
        pcr = (psnr_val_rgb + psnr_val_rgb2+ psnr_val_rgb3 + psnr_val_rgb4)/4
        if pcr > best_psnr:
            best_psnr = pcr
            best_epoch = epoch
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, str(dir_checkpoint / "model_best.pth"))


        print("[epoch %d PSNR: %.4f PSNR: %.4f  PSNR: %.4f  PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, psnr_val_rgb2, psnr_val_rgb3, psnr_val_rgb4, best_epoch, best_psnr))
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    # Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    # torch.save({'epoch': epoch,
    #             'state_dict': model_restoration.state_dict(),
    #             'optimizer': optimizer.state_dict()
    #             }, str(dir_checkpoint /  "model_latest.pth"))

