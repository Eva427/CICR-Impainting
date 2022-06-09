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

def getData(path,**kwargs):
    process = transforms.Compose(
        [transforms.Resize(resize), transforms.CenterCrop(crop),transforms.ToTensor()])
    dataset = ImageFolder(path, process)
    lengths = [sizeTrain, sizeTest]
    train_set, val_set = torch.utils.data.random_split(dataset, lengths)
    return DataLoader(train_set, **kwargs), DataLoader(val_set, **kwargs)

def getFaces():
    return getData(path='data/lfw', 
                    batch_size=128, 
                    shuffle=True, 
                    num_workers=2)

# getCeleba
# getFacesPlusCeleba

# méthodes de dataAugmentation ...
# contraste / rotation / couleur / zoom