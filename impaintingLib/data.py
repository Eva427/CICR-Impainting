from torchvision.datasets.folder import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

sizeTrain = 12000
sizeTest  = 1233

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

def getSize(factor=1):
    numWorker = 2 
    wr,hr = resize = (120, 120)
    wc,hc = crop = (64 , 64 )
    batchSize = 32
        
    if factor > 1 :
        numWorker = 0
        resize = (wr*factor,hr*factor)
        crop   = (wc*factor,hc*factor)
        
    if factor > 3 :
        batchSize = 16
        
    return resize,crop,numWorker,batchSize

def getData(path,**kwargs):
   
    if not kwargs["shuffle"] :
        torch.manual_seed(0)
        
    resize,crop,numWorker,batchSize = getSize(kwargs["resize"])
    transformations = [
         transforms.Resize(resize), 
         transforms.CenterCrop(crop),
         transforms.ToTensor()
    ]
    
    kwargs["num_workers"] = numWorker
    kwargs["batch_size"]  = batchSize
    kwargs.pop("resize")
    
    process = transforms.Compose(transformations)
    dataset = ImageFolder(path, process)
    
    sizeTest = int(len(dataset) / 10)
    sizeTrain = len(dataset) - sizeTest
    
    lengths = [sizeTrain, sizeTest]
    gen = torch.Generator()
    gen.manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dataset, lengths, generator=gen)
    
    return DataLoader(train_set, **kwargs), DataLoader(val_set, **kwargs)

def getFaces(shuffle=True,doNormalize=True,resize=1):
    return getData(path='data/lfw', 
                    shuffle=shuffle, 
                    resize=resize) 

def getMasks(seed=0,resize=1,test=False):
    
    if test:
        path = "data/test_masks"
    else : 
        path = "data/masks"

    _,crop,numWorker,batchSize = getSize(resize)
    transformations = [
         transforms.Resize(crop), 
         transforms.ToTensor()
        ]

    g = None
    if seed != 0 : 
        g = torch.Generator()
        g.manual_seed(seed)
    
    process = transforms.Compose(transformations)
    dataset = ImageFolder(path, process)
    masks   = DataLoader(dataset, batch_size = batchSize,
                                  shuffle=True, 
                                  generator=g,
                                  num_workers=numWorker)
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

def crop(img):
    (mu,sigma) = (40,10)
    _,_,size,_ = img.shape
    factor = np.random.normal(mu, sigma)
    
    transfo = transforms.Compose([transforms.RandomCrop(size*factor/100),
                        transforms.Resize((size,size))])
    
    return transfo(img)

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