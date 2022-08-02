import gradio as gr
import torch
import impaintingLib as imp
import numpy as np
from torchvision import transforms
from PIL import Image, ImageChops

factorResize = 2
scale_factor = 2
enhance = True

class Obj : 
    "temporaire"
o = Obj()

# Impainter
impainter = imp.model.UNet(4, netType="partial")
impainter_weight_path = './modelSave/02_08/partial4channels_mid'
# impainter_weight_path = "./modelSave/david/partial4channel_low.pth"
impainter.load_state_dict(torch.load(impainter_weight_path, map_location=torch.device('cpu')))
impainter.eval()

# Classifier
classifier_weight_path = "./modelSave/classifierUNet.pth"
classif = imp.model.ClassifierUNet()
classif.load_state_dict(torch.load(classifier_weight_path,map_location=torch.device('cpu')))
classif.eval()

# Enhancer
enhancer_weight_path = 'modelSave/RRDB_ESRGAN_x4.pth'
enhancer = imp.model.RRDBNet(3, 3, 64, 23, gc=32)
enhancer.load_state_dict(torch.load(enhancer_weight_path,map_location=torch.device('cpu')))
enhancer.eval()

def convertImage(image):
    w,h = image.size
    min_dim = min(w,h)
    resize = (120*factorResize, 120*factorResize)
    crop   = (64*factorResize, 64*factorResize)

    process = transforms.Compose([
        # transforms.CenterCrop((min_dim,min_dim)),
        # transforms.Resize(crop),
        transforms.Resize(resize), 
        transforms.CenterCrop(crop),
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

def segment(input):
    image = input["image"]
    image = convertImage(image)
    with torch.no_grad():
        image = torch.nn.functional.interpolate(image, scale_factor=scale_factor)
        normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        classifiedImage = classif(normalized)
        classifiedImage = torch.nn.functional.avg_pool2d(classifiedImage, scale_factor)
        _,_,w,_ = classifiedImage.shape
        classifPlain = imp.loss.generate_label_plain(classifiedImage,w)
        classifPlain = simplifyChannels(classifPlain)
        # classifPlain = npToTensor(classifPlain)
        classifPlain = (classifPlain + 1) * 25
        classifPlain = classifPlain.astype(np.uint8)
        # pil_image=Image.fromarray(classifPlain[0])
        # pil_image.save("mask.jpg")
        # o.t1 = classifPlain[0]
    return classifPlain[0]

def predict(image):
    image = impainter(image)
    image = torch.clip(image,0,1)
    return image[:,:3]

def impaint(original,segment):
    image, mask = original["image"], original["mask"]
    image   = convertImage(image)
    mask    = convertImage(mask)
    # segment = convertImage(segment)
    mask  = mask[:,:1]
    
    # with torch.no_grad():
    #     segment = torch.nn.functional.interpolate(image, scale_factor=scale_factor)
    #     normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(segment)
    #     classifiedImage = classif(normalized)
    #     classifiedImage = torch.nn.functional.avg_pool2d(classifiedImage, scale_factor)
    #     _,_,w,_ = classifiedImage.shape
    #     classifPlain = imp.loss.generate_label_plain(classifiedImage,w)
    #     classifPlain = simplifyChannels(classifPlain)
    #     segment = npToTensor(classifPlain)
    #     segment = segment.view(1,1,64*factorResize,64*factorResize)

    # equality = o.t1 == segment
    # differences = np.where(equality == False)
    # print(differences[0],differences[0][0])

    segment = (segment / 25) - 1
    segment = npToTensor(segment)
    segment = torch.round(segment)
    segment = (segment/9) + 0.1

    # o.t2 = segment[0][0]
    # equality = torch.eq(o.t1,o.t2)
    # equality = ~equality
    # equality = (equality==True).nonzero().squeeze()
    # print(equality)
    # print("nombre de différents : ",len(equality))
    # print(o.t1[0][6],o.t2[0][6])
    # print(o.t1[63][50],o.t2[63][50])

    w,h = segment.shape
    segment = segment.view(1,1,w,h)
    x_prime = imp.mask.propagate(image,mask)
    x_prime2 = torch.cat((x_prime,segment),dim=1)
    with torch.no_grad():
        image_hat = predict(x_prime2)
        if enhance : 
            image_hat = enhancer(image_hat)

    image_hat = transforms.ToPILImage()(image_hat[0])
    return image_hat

def update():
    return

css = ".output_image {height: 40rem !important; width: 100% !important;}"
css = ""
interface = gr.Blocks(css=css)

with interface:
    with gr.Row():
        with gr.Column():
            img = gr.Image(
                tool="sketch", source="upload", label="Input", type="pil"
            )
            with gr.Row():
                btnSegment = gr.Button("Segment")
        with gr.Column():
            imgSegment = gr.Image(label="Classifer",interactive=False, type="numpy", image_mode="L")
            html = gr.HTML(visible=False, value="<img src='file/current.png' width='200px'>")
            with gr.Row():
                btnUpdate = gr.Button("Update")
            with gr.Row():
                btnRun    = gr.Button("Run")
        with gr.Column():
            imgOutput = gr.Image(label="Output",interactive=False)

    #btn.click(fn=impaint, inputs=img, outputs=img2)
    btnSegment.click(fn=segment, inputs=img, outputs=imgSegment)
    btnUpdate.click(fn=update, inputs=None, outputs=imgSegment)
    btnRun.click(fn=impaint, inputs=[img,imgSegment], outputs=imgOutput)

interface.launch()