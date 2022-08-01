import numpy as np
import torch
import impaintingLib as imp

from torchvision import transforms
from PIL import Image
from os import listdir
from os.path import isfile, join
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Propage le masque généré sur tous les channels
def propagate(imgs,masks):

    n, c, h, w = imgs.shape     # c+1
    imgs_masked = torch.empty((n, c, h, w), dtype=imgs.dtype, device=imgs.device)
    for i, (img, mask) in enumerate(zip(imgs, masks)):
        propag_img = img.clone()
        mask_bit = (mask == 0) * 1.
        for j,channel in enumerate(img[:3]) :
            propag_img[j] = channel * mask_bit

        #imgs_masked[i] = torch.cat((propag_img,mask),0)
        imgs_masked[i] = propag_img

    return imgs_masked

class Alter :

    def __init__(self, min_cut=15, max_cut=45, seed=0, resize="low", test=False):
        self.min_cut = min_cut
        self.max_cut = max_cut
        self.seed    = seed
        self.test    = test
        
        # self.maskLoader = imp.data.getMasks(resize=resize,seed=seed,test=test)
        # self.maskIter   = iter(self.maskLoader)
    
    # Generate square mask
    def squareMask(self,imgs):
        
        if self.seed != 0:
            np.random.seed(self.seed)
        
        n, c, h, w = imgs.shape
        w1 = np.random.randint(self.min_cut, self.max_cut, n)
        h1 = np.random.randint(self.min_cut, self.max_cut, n)
        
        w2 = np.random.randint(self.min_cut, self.max_cut, n)
        h2 = np.random.randint(self.min_cut, self.max_cut, n)
        
        masks = torch.empty((n, 1, h, w), dtype=imgs.dtype, device=imgs.device)
        for i, (img, w11, h11, w22, h22) in enumerate(zip(imgs, w1, h1, w2, h2)):
            cut_img = torch.full((1,h,w),0, dtype=img.dtype, device=img.device)
            cut_img[:, h11:h11 + h11, w11:w11 + w11] = 1
            cut_img[:, h22:h22 + h11, w22:w22 + w22] = 1
            masks[i] = cut_img
            
        imgs_masked = propagate(imgs,masks)
        return imgs_masked
    
    def downScale(self,imgs, scale_factor=2, upscale=True):
        imgs_low = torch.nn.MaxPool2d(kernel_size=scale_factor)(imgs)
        if upscale:
            imgs_low = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)(imgs_low)
        return imgs_low
    
    # deprecated
    def irregularMaskOLD(self,imgs):
    
        maskPath = "./data/masks/"
        files = [f for f in listdir(maskPath) if isfile(join(maskPath, f))]
        n,c,w,h = imgs.shape

        masks = torch.empty((n, 1, h, w), dtype=imgs.dtype, device=imgs.device)
        for i,img in enumerate(imgs) : 
            path = maskPath + random.choice(files)
            with Image.open(path) as mask:
                mask = transforms.ToTensor()(mask)
                mask = transforms.Resize((w,h))(mask)
                masks[i] = mask

        imgs_masked = propagate(imgs,masks)
        return imgs_masked
    
    def irregularMask(self,imgs):
        
        try:
            masks,_ = next(self.maskIter)
        except StopIteration:
            self.maskIter = iter(self.maskLoader)
            masks,_ = next(self.maskIter)
            
        masks = masks[:,:1].to(device)
        imgs_masked = propagate(imgs,masks)
        return imgs_masked


# Generate random mask