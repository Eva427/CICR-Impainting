import numpy as np
import torch

# Generate square mask
def squareMask(imgs, min_cut=15,max_cut=45):
    n, c, h, w = imgs.shape
    w1 = np.random.randint(min_cut, max_cut, n)
    h1 = np.random.randint(min_cut, max_cut, n)
    
    w2 = np.random.randint(min_cut, max_cut, n)
    h2 = np.random.randint(min_cut, max_cut, n)
    
    cutouts = torch.empty((n, c, h, w), dtype=imgs.dtype, device=imgs.device)
    for i, (img, w11, h11, w22, h22) in enumerate(zip(imgs, w1, h1, w2, h2)):
        cut_img = img.clone()
        cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
        cut_img[:, h22:h22 + h11, w22:w22 + w22] = 0
        cutouts[i] = cut_img
    return cutouts


# Generate random mask