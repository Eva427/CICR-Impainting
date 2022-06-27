import numpy as np
import torch

class Alter :

    def __init__(self, min_cut=15, max_cut=45, seed=0):
        self.min_cut = min_cut
        self.max_cut = max_cut
        self.seed    = seed
    
    # Propage le masque généré sur tous les channels
    def propagate(self,imgs,masks):
        
        n, c, h, w = imgs.shape     # 4
        imgs_masked = torch.empty((n, 3, h, w), dtype=imgs.dtype, device=imgs.device)
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            propag_img = img.clone()
            mask_bit = (mask != 0) * 1.
            for j,channel in enumerate(img) :
                propag_img[j] = channel * mask_bit
                
            #imgs_masked[i] = torch.cat((propag_img,mask),0)
            imgs_masked[i] = propag_img
            
        return imgs_masked
    
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
            cut_img = torch.full((1,h,w),255, dtype=img.dtype, device=img.device)
            cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
            cut_img[:, h22:h22 + h11, w22:w22 + w22] = 0
            masks[i] = cut_img
            
        imgs_masked = self.propagate(imgs,masks)
        return imgs_masked
    
    def downScale(self,imgs, scale_factor=2, upscale=True):
        imgs_low = torch.nn.MaxPool2d(kernel_size=scale_factor)(imgs)
        if upscale:
            imgs_low = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)(imgs_low)
        return imgs_low
    
    def none(self):
        pass


# Generate random mask