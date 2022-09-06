import impaintingLib as imp
import torch
import torchvision
import numpy as np
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#### Segmentation

classif = imp.loss.getTrainedModel()

def simplifyChannels(x):
    x = np.where(x == 3, 0, x) 
    x = np.where(x == 4, 3, x) 
    x = np.where(x == 5, 3, x) 
    x = np.where(x == 6, 4, x) 
    x = np.where(x == 7, 4, x) 
    x = np.where(x == 8, 5, x) 
    x = np.where(x == 9, 5, x) 
    x = np.where(x == 10 , 6, x) 
    x = np.where(x == 11, 7, x) 
    x = np.where(x == 12, 7, x)  
    x = np.where(x > 12, 0, x)
    return x

def npToTensor(x):
    c,w,h = x.shape
    x = torch.from_numpy(x).to(device)
    x = torch.reshape(x, (c,1,w,h))
    return x.float()

def get_segmentation(x, segmenter=classif, scale_factor=4, simplify=True):
    n,c,w,h = x.shape
    with torch.no_grad():
        if scale_factor > 0 :
            x = torch.nn.functional.interpolate(x, scale_factor=scale_factor)
        x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)
        y = classif(x)
        if scale_factor > 0 :
            y = torch.nn.functional.avg_pool2d(y, scale_factor)
            
    y = imp.loss.generate_label_plain(y,w)
    if simplify: 
        y = simplifyChannels(y)
    y =  (y / (np.amax(y)+2)) + 0.1
    y = npToTensor(y)
    return y


#### Keypoints

keypointModel = imp.model.XceptionNet().to(device) 
keypointModel.load_state_dict(torch.load("./modelSave/keypoint.pt",map_location=device))
keypointModel.eval()
    
def addKeypoints(images_list, landmarks_list):
    n,c,w,h = images_list.shape
    image_dim = w
    layers = torch.zeros((n,1, w, h), dtype=images_list.dtype, device=images_list.device)
    for i,(image, landmarks) in enumerate(zip(images_list, landmarks_list)):
        image = (image - image.min())/(image.max() - image.min())
        landmarks = landmarks.view(-1, 2)
        landmarks = (landmarks + 0.5) * image_dim
        landmarks = landmarks.cpu().detach().numpy().tolist()
        landmarks = np.array([(x, y) for (x, y) in landmarks if 0 <= x <= image_dim and 0 <= y <= image_dim])
        landmarks = torch.from_numpy(landmarks).to(device)
        
        layer = torch.empty((1, w, h), dtype=images_list.dtype, device=images_list.device).fill_(0.1)
        for x,y in landmarks:
            x = int(x.item()) - 1
            y = int(y.item()) - 1
            layer[0][y][x] = 1
        layers[i] = layer
    return layers

def getKeypoints(x, model=keypointModel):
    x = torchvision.transforms.Grayscale()(x)
    with torch.no_grad():
        keypoints = model(x)
        layers = addKeypoints(x, keypoints)
    return layers

##### ESRGAN

suprRes_path = 'modelSave/RRDB_ESRGAN_x4.pth'
superResModel = imp.model.RRDBNet(3, 3, 64, 23, gc=32)
superResModel.load_state_dict(torch.load(suprRes_path,map_location=device), strict=True)
superResModel.eval()
superResModel = superResModel.to(device)

def superRes(x):
    with torch.no_grad():
        return superResModel(x)