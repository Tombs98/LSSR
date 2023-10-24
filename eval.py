import os
import numpy as np
from glob import glob
import cv2
from natsort import natsorted

from skimage.metrics import structural_similarity,peak_signal_noise_ratio
from cal import calculate_psnr,calculate_ssim,calculate_psnr_left,calculate_skimage_ssim,calculate_skimage_ssim_left

def read_img(path):
    return cv2.imread(path)




def main():
    file_path = os.path.join('newfo_v/4x1', 'KITTI2015')
    gt_path = os.path.join('../datasets/test', 'KITTI2015/hr')
    file_list = os.listdir(gt_path)
    # print(file_list)

    # path_fake = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
    # path_real = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))
    # print(len(path_fake))
    list_psnr = []
    list_psnr_l = []
    list_ssim_l = []
    list_ssim = []
    for i in range(len(file_list)):
        h0 = read_img(gt_path+'/'+file_list[i]+'/hr0.png')
        h1 = read_img(gt_path+'/'+file_list[i]+'/hr1.png')

        l0 = read_img(file_path+'/'+file_list[i]+'/hr0.png')
        l1 = read_img(file_path+'/'+file_list[i]+'/hr1.png')
        #result1 = np.zeros(t1.shape,dtype=np.float32)
        #result2 = np.zeros(t2.shape,dtype=np.float32)
        #cv2.normalize(t1,result1,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        #cv2.normalize(t2,result2,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
       

        
        

        psnr_num0 = calculate_psnr(h0, l0,0)
        psnr_num1 = calculate_psnr(h1, l1,0)
        psnr_num2 = calculate_psnr_left(h0, l0,0)

        ssim_num0 = calculate_skimage_ssim(h0, l0)
        ssim_num1 = calculate_skimage_ssim(h1, l1)
        ssim_num2 = calculate_skimage_ssim_left(h0, l0)

        psnr_num = (psnr_num0+psnr_num1)/2
        ssim_num = (ssim_num0+ssim_num1)/2

        list_ssim_l.append(ssim_num2)
        list_ssim.append(ssim_num)

        list_psnr.append(psnr_num)
        list_psnr_l.append(psnr_num2)
  


    print("AverSSIM:", np.mean(list_ssim))  # ,list_ssim)
    print("AverSSIMl:", np.mean(list_ssim_l))  # ,list_ssim)
    print("AverPSNR:", np.mean(list_psnr))  # ,list_ssim)
    print("AverPSNRl:", np.mean(list_psnr_l))  # ,list_ssim)
   

if __name__ == '__main__':
    main()