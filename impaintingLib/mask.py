import numpy as np
import torch

class Alter :

    def __init__(self, min_cut=15, max_cut=45, seed=0):
        self.min_cut = min_cut
        self.max_cut = max_cut
        self.seed    = seed

    # Generate square mask
    def squareMaskDirectly(self,imgs):
        
        np.random.seed(self.seed)
        
        n, c, h, w = imgs.shape
        w1 = np.random.randint(self.min_cut, self.max_cut, n)
        h1 = np.random.randint(self.min_cut, self.max_cut, n)
        
        w2 = np.random.randint(self.min_cut, self.max_cut, n)
        h2 = np.random.randint(self.min_cut, self.max_cut, n)
        
        cutouts = torch.empty((n, c, h, w), dtype=imgs.dtype, device=imgs.device)
        for i, (img, w11, h11, w22, h22) in enumerate(zip(imgs, w1, h1, w2, h2)):
            cut_img = img.clone()
            cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
            cut_img[:, h22:h22 + h11, w22:w22 + w22] = 0
            cutouts[i] = cut_img
        return cutouts
    
    # Generate square mask
    def squareMask(self,imgs):
        
        np.random.seed(self.seed)
        
        n, c, h, w = imgs.shape
        w1 = np.random.randint(self.min_cut, self.max_cut, n)
        h1 = np.random.randint(self.min_cut, self.max_cut, n)
        
        w2 = np.random.randint(self.min_cut, self.max_cut, n)
        h2 = np.random.randint(self.min_cut, self.max_cut, n)
        
        cutouts = torch.empty((n, 4, h, w), dtype=imgs.dtype, device=imgs.device)
        for i, (img, w11, h11, w22, h22) in enumerate(zip(imgs, w1, h1, w2, h2)):
            cut_img = torch.full((1,h,w),255, dtype=img.dtype, device=img.device)
            cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
            cut_img[:, h22:h22 + h11, w22:w22 + w22] = 0
            cutouts[i] = torch.cat((img,cut_img),0)
        return cutouts
    
    def none(self):
        pass


# Generate random mask