from quart import Blueprint, request, redirect, render_template
from io import BytesIO
import base64
import torch
import impaintingLib as imp
import numpy as np
from torchvision import transforms
from PIL import Image, ImageChops
from datetime import datetime
import json
import cv2

import impaintingLib as imp
from impaintingLib.model import keypoint

# ---------------

impBP = Blueprint('imp', __name__, template_folder='templates', static_folder='static')

# ---------------

factorResize = 2
scale_factor = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Impainter
impainter = imp.model.UNet(5, netType="partial")
impainter_weight_path = './modelSave/impainter.pth'
impainter.load_state_dict(torch.load(impainter_weight_path, map_location=device))
impainter.eval()

# Classifier
classifier_weight_path = "./modelSave/classifierUNet.pth"
classif = imp.model.ClassifierUNet()
classif.load_state_dict(torch.load(classifier_weight_path,map_location=device))
classif.eval()

# Background remover
torch.hub.set_dir("./modelSave/bg")
bgRemove = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
bgRemove.eval()

# ---------------

resize = (120*factorResize, 120*factorResize) # Default one 

def convertImage(image,resize=None,transparentFill=False):
    w,h = image.size
    maxs = max(w,h)

    if w != h : 
        filling = (255,255,255)
        if transparentFill :
            filling = 255
        result = Image.new(image.mode, (maxs, maxs), filling)
        result.paste(image, ((maxs-w)//2, (maxs-h)//2))
        image = result

    crop   = (64*factorResize, 64*factorResize)
    if resize : 
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

def segment(image):
    with torch.no_grad():
        if scale_factor > 0 :
            image = torch.nn.functional.interpolate(image, scale_factor=scale_factor)
        normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        classifiedImage = classif(normalized)
        # classifiedImage = torch.nn.functional.avg_pool2d(classifiedImage, scale_factor)
        _,_,w,_ = classifiedImage.shape
        classifPlain = imp.loss.generate_label_plain(classifiedImage,w)
        classifPlain = imp.components.simplifyChannels(classifPlain)
        classifPlain = (classifPlain + 1) * 25
        classifPlain = classifPlain.astype(np.uint8)
        pil_image = Image.fromarray(classifPlain[0])
    return pil_image

def make_transparent_foreground(pic, mask):
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
    a = np.ones(mask.shape, dtype='uint8') * 255
    alpha_im = cv2.merge([b, g, r, a], 4)
    bg = np.zeros(alpha_im.shape)
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)
    return foreground


def remove_background(input_image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        bgRemove.to('cuda')

    with torch.no_grad():
        output = bgRemove(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a binary (black and white) mask of the profile foreground
    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)

    foreground = make_transparent_foreground(input_image, bin_mask)
    return foreground, bin_mask

def impaint(mask,segmented,enhance,removeBg,keypoints,modelOpt,resize):
    image = Image.open("./impaintingWeb/static/image/original.png")
    image   = convertImage(image,resize)
    mask    = convertImage(mask,resize,transparentFill=True)
    segmented = convertImage(segmented)

    segmented = segmented * 255
    segmented = (segmented / 25) - 1
    segmented = torch.round(segmented)
    segmented = (segmented / 9) + 0.1

    mask = (mask < 0.5) * 1.
    x_prime = imp.mask.propagate(image,mask)

    x_prime2 = torch.cat((x_prime,segmented),dim=1)
    keypointLayer = imp.components.addKeypoints(image, keypoints)
    x_prime3 = torch.cat((x_prime2, keypointLayer),dim=1)

    with torch.no_grad():

        if modelOpt == "default" : 
            image_hat = impainter(x_prime3)
        else : 
            print("Error no model found")
            image_hat = x_prime3
            
        image_hat = torch.clip(image_hat,0,1)[:,:3]
        if enhance : 
            image_hat = imp.components.superRes(image_hat)
            
        if removeBg :
            image_hat = transforms.ToPILImage()(image_hat[0])
            image_hat, _ = remove_background(image_hat)
            image_hat = Image.fromarray(image_hat)
        else : 
            image_hat = transforms.ToPILImage()(image_hat[0])

    return image_hat

# ---------------

@impBP.route("/")
async def home():
    return await render_template('homeImp.html')

# Recevoir l'image encodé en Base 64
# Générer la segmentation et croper/resizer au bon format
# Stocker ces informations sous forme d'image
# Générer les keypoints et les envoyer en réponse sous forme de liste  
@impBP.route("/segment", methods=['POST'])
async def segmentPOST():
    ask        = await request.form
    dataB64    = ask["imgB64"]
    size       = int(ask["size"])
    resize = [size*factorResize, size*factorResize]

    img = Image.open(BytesIO(base64.b64decode(dataB64))).convert('RGB')
    img.save("./impaintingWeb/static/image/original.png")

    original_crop = convertImage(img,resize)
    segmented = segment(original_crop)
    segmented.save("./impaintingWeb/static/image/mask.png")

    keypoints = imp.components.getLandmarks(original_crop)[0]
    keypoints = keypoints.tolist()

    original_crop = transforms.ToPILImage()(original_crop[0])
    original_crop.save("./impaintingWeb/static/image/original_crop.png")

    return {"keypoints" : keypoints}

# Recevoir masques, segementation, keypoints et choix sur la super résolution
# Convertir les informations reçues au bon format
# Faire la prédiction et la sauvegarder
@impBP.route("/predict", methods=['POST'])
async def predictPOST():
    ask        = await request.form
    maskB64    = ask["maskB64"]
    segmentB64 = ask["segmentB64"]
    doEnhance  = ask["doEnhance"]
    doRemoveBg = ask["doRemoveBg"]
    size       = int(ask["size"])
    modelOpt   = ask["modelOpt"]
    keypoints  = ask["keypoints"]
    doEnhance = doEnhance == "true" 
    doRemoveBg = doRemoveBg == "true" 

    resize = [size*factorResize, size*factorResize]

    mask = Image.open(BytesIO(base64.b64decode(maskB64)))
    mask.load() # required for png.split()
    maskRGB = Image.new("RGB", mask.size, (255, 255, 255))

    maskRGB.paste(mask, mask=mask.split()[3]) # 3 is the alpha channel
    mask = maskRGB.convert("L")
    segment = Image.open(BytesIO(base64.b64decode(segmentB64))).convert("L")

    keypoints = json.loads(keypoints)
    keypoints = torch.Tensor(keypoints)
    keypoints = torch.unsqueeze(keypoints, dim=0)

    predict = impaint(mask,segment,doEnhance,doRemoveBg,keypoints,modelOpt,resize)
    predict.save("./impaintingWeb/static/image/predict.png")
    return {"ok" : True}

# Recevoir le masque et la segmentation
# Les télécharger
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

    mask.save("./impaintingWeb/static/image/downloaded/mask.png")
    segment.save("./impaintingWeb/static/image/downloaded/seg.png")
    return {"ok" : True}