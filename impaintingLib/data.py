from torchvision.datasets.folder import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

sizeTrain = 12000
sizeTest  = 1233

resize = (120, 120)
crop   = (64 , 64 )

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
    
    if kwargs["doNormalize"] :
        transformations.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]))
    if not kwargs["shuffle"] :
        torch.manual_seed(0)
    
    kwargs.pop("doNormalize")
    process = transforms.Compose(transformations)
    dataset = ImageFolder(path, process)
    lengths = [sizeTrain, sizeTest]
    train_set, val_set = torch.utils.data.random_split(dataset, lengths)
    return DataLoader(train_set, **kwargs), DataLoader(val_set, **kwargs)

def getFaces(batch_size=32,shuffle=True,doNormalize=True):
    return getData(path='data/lfw', 
                    batch_size=batch_size, 
                    doNormalize=doNormalize,
                    shuffle=shuffle, 
                    num_workers=2)

def inv_normalize(x):
    transfo = transforms.Normalize(
                       mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                       std=[1/0.229, 1/0.224, 1/0.225])
    x = x[:,:3]
    x = transfo(x)
    return x

#inv_tensor = inv_normalize(tensor)

# getCeleba
# getFacesPlusCeleba

# méthodes de dataAugmentation ...
# contraste / rotation / couleur / zoom