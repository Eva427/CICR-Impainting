from quart import Blueprint, request, redirect, render_template
from io import BytesIO
import base64
import torch
import impaintingLib as imp
import numpy as np
from torchvision import transforms
from PIL import Image, ImageChops
from datetime import datetime

# ---------------

impBP = Blueprint('imp', __name__, template_folder='templates', static_folder='static')

# ---------------

factorResize = 2
scale_factor = 2

# Impainter
impainter = imp.model.UNet(5, netType="partial")
impainter_weight_path = './modelSave/09_01/bigRun2115' #02_08/partial4channels_mid'
impainter.load_state_dict(torch.load(impainter_weight_path, map_location=torch.device('cpu')))
impainter.eval()

# Classifier
classifier_weight_path = "./modelSave/classifierUNet.pth"
classif = imp.model.ClassifierUNet()
classif.load_state_dict(torch.load(classifier_weight_path,map_location=torch.device('cpu')))
classif.eval()

# Enhancer
enhancer_weight_path = './modelSave/RRDB_ESRGAN_x4.pth'
enhancer = imp.model.RRDBNet(3, 3, 64, 23, gc=32)
enhancer.load_state_dict(torch.load(enhancer_weight_path,map_location=torch.device('cpu')))
enhancer.eval()

# Keypoints
keypoint_weight_path = "./modelSave/keypoint.pt"
keypointModel = imp.model.XceptionNet()
keypointModel.load_state_dict(torch.load(keypoint_weight_path,map_location=torch.device('cpu')))
keypointModel.eval()

# ---------------


def convertImage(image,doCrop=True):
    w,h = image.size
    resize = (120*factorResize, 120*factorResize)
    crop   = (64*factorResize, 64*factorResize)

    if doCrop : 
        process = transforms.Compose([
            transforms.Resize(resize), 
            transforms.CenterCrop(crop),
            transforms.ToTensor()
    ])
    else : 
        process = transforms.Compose([
            transforms.Resize(crop), 
            transforms.ToTensor()
    ])

    image = process(image)
    c,w,h = image.shape
    image = image.view(1,c,w,h)
    return image

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
    x = torch.from_numpy(x)
    x = x.float()
    return x

def segment(image):
    image = convertImage(image)
    with torch.no_grad():
        if scale_factor > 0 :
            image = torch.nn.functional.interpolate(image, scale_factor=scale_factor)
        normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        classifiedImage = classif(normalized)
        # classifiedImage = torch.nn.functional.avg_pool2d(classifiedImage, scale_factor)
        _,_,w,_ = classifiedImage.shape
        classifPlain = imp.loss.generate_label_plain(classifiedImage,w)
        classifPlain = simplifyChannels(classifPlain)
        # classifPlain = npToTensor(classifPlain)
        classifPlain = (classifPlain + 1) * 25
        classifPlain = classifPlain.astype(np.uint8)
        pil_image = Image.fromarray(classifPlain[0])
    return pil_image

def addKeypoints(images_list, landmarks_list):
    n,_,w,h = images_list.shape
    image_dim = w
    layers = torch.zeros((n,1, w, h), dtype=images_list.dtype, device=images_list.device)
    for i,(image, landmarks) in enumerate(zip(images_list, landmarks_list)):
        image = (image - image.min())/(image.max() - image.min())
        landmarks = landmarks.view(-1, 2)
        landmarks = (landmarks + 0.5) * image_dim
        landmarks = landmarks.cpu().detach().numpy().tolist()
        landmarks = np.array([(x, y) for (x, y) in landmarks if 0 <= x <= image_dim and 0 <= y <= image_dim])
        landmarks = torch.from_numpy(landmarks)
        
        layer = torch.empty((1, w, h), dtype=images_list.dtype, device=images_list.device).fill_(0.1)
        for x,y in landmarks:
            x = int(x.item()) - 1
            y = int(y.item()) - 1
            layer[0][y][x] = 1
        layers[i] = layer
    return layers

def getKeypoints(x, model=keypointModel):
    x = transforms.Grayscale()(x)
    keypoints = model(x)
    layers = addKeypoints(x, keypoints)
    return layers

def predict(image):
    image = impainter(image)
    image = torch.clip(image,0,1)
    return image[:,:3]

def impaint(mask,segment,enhance):
    image = Image.open("./impaintingWeb/static/image/original.jpg")
    image   = convertImage(image)
    mask    = convertImage(mask)
    segment = convertImage(segment,doCrop=False)

    segment = segment * 255
    segment = (segment / 25) - 1
    segment = torch.round(segment)
    segment = (segment / 9) + 0.1

    n, c, h, w = image.shape
    x_prime = torch.empty((n, c, h, w), dtype=image.dtype, device=image.device)
    for i, (img, mask) in enumerate(zip(image, mask)):
        propag_img = img.clone()
        mask_bit = (mask > 0.5) * 1.
        for j,channel in enumerate(img[:3]) :
            propag_img[j] = channel * mask_bit
        x_prime[i] = propag_img

    x_prime2 = torch.cat((x_prime,segment),dim=1)
    keypointLayer = getKeypoints(image)
    x_prime3 = torch.cat((x_prime2, keypointLayer),dim=1)

    with torch.no_grad():
        image_hat = predict(x_prime3)
        if enhance : 
            image_hat = enhancer(image_hat)

    image_hat = transforms.ToPILImage()(image_hat[0])
    return image_hat

# ---------------

@impBP.route("/")
async def home():
    return await render_template('home.html')

@impBP.route("/segment", methods=['POST'])
async def segmentPOST():
    ask        = await request.form
    dataB64    = ask["imgB64"]
    img = Image.open(BytesIO(base64.b64decode(dataB64))).convert('RGB')
    img.save("./impaintingWeb/static/image/original.jpg")
    
    original_crop = convertImage(img)
    original_crop = transforms.ToPILImage()(original_crop[0])
    newsize = (128,128)
    original_crop = original_crop.resize(newsize)
    original_crop.save("./impaintingWeb/static/image/original_crop.jpg")

    img = segment(img)
    img.save("./impaintingWeb/static/image/mask.jpg")

    return {"ok" : True}

@impBP.route("/predict", methods=['POST'])
async def predictPOST():
    ask        = await request.form
    maskB64    = ask["maskB64"]
    segmentB64 = ask["segmentB64"]
    doEnhance  = ask["doEnhance"]
    doEnhance = doEnhance == "true" 

    mask = Image.open(BytesIO(base64.b64decode(maskB64)))
    mask.load() # required for png.split()
    maskRGB = Image.new("RGB", mask.size, (255, 255, 255))
    maskRGB.paste(mask, mask=mask.split()[3]) # 3 is the alpha channel
    mask = maskRGB.convert("L")
    segment = Image.open(BytesIO(base64.b64decode(segmentB64))).convert("L")

    predict = impaint(mask,segment,doEnhance)
    predict.save("./impaintingWeb/static/image/predict.jpg")
    return {"ok" : True}

@impBP.route("/download", methods=['POST'])
async def download():
    ask        = await request.form
    maskB64    = ask["maskB64"]
    segmentB64 = ask["segmentB64"]

    mask = Image.open(BytesIO(base64.b64decode(maskB64)))
    mask.load() # required for png.split()
    maskRGB = Image.new("RGB", mask.size, (255, 255, 255))
    maskRGB.paste(mask, mask=mask.split()[3]) # 3 is the alpha channel
    mask = maskRGB.convert("L")
    segment = Image.open(BytesIO(base64.b64decode(segmentB64))).convert("L")

    # mask = convertImage(mask)
    # mask = (mask < 0.5) * 1.
    # mask = transforms.ToPILImage()(mask[0])

    mask.save("./impaintingWeb/static/image/downloaded/mask.jpg")
    segment.save("./impaintingWeb/static/image/downloaded/seg.jpg")
    return {"ok" : True}