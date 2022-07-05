from torchvision.datasets.folder import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

sizeTrain = 12000
sizeTest  = 1233

resize = (120, 120)
crop   = (64 , 64 )

resize = (240, 240)
crop   = (128, 128)

resize = (360, 360)
crop   = (192, 192)

# def downloadFaces():
#     !wget http://vis-www.cs.umass.edu/lfw/lfw.tgz > /dev/null 2>&1
#     !tar zxvf lfw.tgz > /dev/null 2>&1
#     !mkdir data > /dev/null 2>&1
#     !mv lfw data > /dev/null 2>&1

def addChannel(imgs):
    n, c, h, w = imgs.shape
    all_img = torch.empty((n, 4, h, w), dtype=imgs.dtype, device=imgs.device)
    for i,img in enumerate(imgs):
        blank_layer = torch.full((1,h,w),255, dtype=img.dtype, device=img.device)
        all_img[i] = torch.cat((img,blank_layer),0)
    return all_img

def getData(path,**kwargs):
    transformations = [
         transforms.Resize(resize), 
         transforms.CenterCrop(crop),
         transforms.ToTensor()
    ]
    
    if not kwargs["shuffle"] :
        torch.manual_seed(0)
    
    process = transforms.Compose(transformations)
    dataset = ImageFolder(path, process)
    lengths = [sizeTrain, sizeTest]
    train_set, val_set = torch.utils.data.random_split(dataset, lengths)
    return DataLoader(train_set, **kwargs), DataLoader(val_set, **kwargs)

def getFaces(batch_size=32,shuffle=True,doNormalize=True):
    return getData(path='data/lfw', 
                    batch_size=batch_size, 
                    shuffle=shuffle, 
                    num_workers=0) # 2 normalement, test à 1 ou 0

def getMasks(batch_size=32,shuffle=True,num_workers=2):
    path = "data/masks"

    transformations = [
         transforms.Resize(crop), 
         transforms.ToTensor()
        ]

    process = transforms.Compose(transformations)
    dataset = ImageFolder(path, process)
    masks   = DataLoader(dataset, batch_size=batch_size, 
                                  shuffle=shuffle, 
                                  num_workers=num_workers)
    return masks
    

def normalize(x):
    transfo = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std =[0.229, 0.224, 0.225])
    return transfo(x)

def inv_normalize(x):
    transfo = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                   std=[1/0.229, 1/0.224, 1/0.225])
    x = x[:,:3]
    return transfo(x)

# ------------- AUGMENTATION

from PIL import Image, ImageEnhance
import numpy as np
import math
import random

def zoom(img,factor=0):
    size = (width, height) = (img.size)

    # Si on ne lui donne pas d'arg alors c'est aléatoire
    if factor < 1 :
        (mu,sigma) = (1,3)
        factor = abs(factor)
        factor = np.random.normal(mu, sigma)

    (left, upper, right, lower) = (factor, factor, height-factor, width-factor)
    img = img.crop((left, upper, right, lower))
    img = img.resize(size)
    return img

def rotation(img):
    (mu,sigma) = (1,1)
    factor = np.random.normal(mu, sigma)
    factor = abs(factor)
    img = img.rotate(factor)
    #img = zoom(img,20)
    return img

def mirror(img):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def enhance(img,enhancer):
    (mu,sigma) = (1,0.3)
    factor = np.random.normal(mu, sigma)
    #print(factor)
    img = enhancer.enhance(factor)
    return img

def lumi(img):
    func = ImageEnhance.Brightness(img)
    return enhance(img,func)

def contrast(img):
    func = ImageEnhance.Contrast(img)
    return enhance(img,func)

def color(img):
    func = ImageEnhance.Color(img)
    return enhance(img,func)

def sharpness(img):
    func = ImageEnhance.Sharpness(img)
    return enhance(img,func)

def randomTransfo(imgs):
    
    #(mu,sigma) = (1,0.15)
    (mu,sigma) = (1.3,0.4)
    
    for k,img in enumerate(imgs) : 
        img = transforms.ToPILImage()(img)
        nbTransfo = np.random.normal(mu, sigma)
        nbTransfo = abs(nbTransfo)
        nbTransfo = int(nbTransfo)

        #print(nbTransfo)
        #transfos = [zoom, rotation, mirror, lumi, contrast, color, sharpness]
        transfos = [mirror, lumi, contrast, color, sharpness]
        for i in range(nbTransfo):
            func = random.choice(transfos)
            #print(func. __name__)
            transfos.remove(func)
            img = func(img)
            
        imgs[k] = transforms.ToTensor()(img.copy())
    return imgs

#inv_tensor = inv_normalize(tensor)

# getCeleba
# getFacesPlusCeleba

# méthodes de dataAugmentation ...
# contraste / rotation / couleur / zoom