import gradio as gr
import torch
import impaintingLib as imp
import numpy as np
from torchvision import transforms
from PIL import Image

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#models.append(imp.model.SubPixelNetwork(3))
#models[0].load_state_dict(torch.load('./modelSave/gated/impainter.pth', map_location=torch.device('cpu')))
#models[1].load_state_dict(torch.load('./modelSave/gated/augmenter.pth', map_location=torch.device('cpu')))

factorResize = 2
resize = (64*factorResize, 64*factorResize)

# Impainter
impainter = imp.model.UNet(4, netType="partial") # , convType="gated"
impainter_weight_path = './modelSave/david/partial4channel_low.pth'
impainter.load_state_dict(torch.load(impainter_weight_path, map_location=torch.device('cpu')))

# Classifier
classifier_weight_path = "./modelSave/classifierUNet.pth"
classif = imp.model.ClassifierUNet()
classif.load_state_dict(torch.load(classifier_weight_path,map_location=torch.device('cpu')))

# Enhancer
enhancer_weight_path = 'modelSave/RRDB_ESRGAN_x4.pth'
enhancer = imp.model.RRDBNet(3, 3, 64, 23, gc=32)
enhancer.load_state_dict(torch.load(enhancer_weight_path,map_location=torch.device('cpu')))

def convertImage(image):
    image = transforms.ToTensor()(image)
    c,w,h = image.shape
    min_dim = min(w,h)
    image = transforms.CenterCrop((min_dim,min_dim))(image)
    image = transforms.Resize(resize)(image)
    image = image.view(1,c,resize[0],resize[0])
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
    return x / 10

def npToTensor(x):
    c,w,h = x.shape
    x = torch.from_numpy(x)
    x = torch.reshape(x, (c,1,w,h))
    return x.float()

def segment(input):
    image = input["image"]
    image = convertImage(image)
    image = torch.nn.functional.interpolate(image, scale_factor=4)
    normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
    classifiedImage = classif(normalized)
    classifiedImage = torch.nn.functional.avg_pool2d(classifiedImage, 4)
    _,_,w,_ = classifiedImage.shape
    classifPlain = imp.loss.generate_label_plain(classifiedImage,w)
    classifPlain = simplifyChannels(classifPlain)
    classifPlain = npToTensor(classifPlain)
    return transforms.ToPILImage()(classifPlain[0])

def predict(image):
    image = impainter(image)
    image = torch.clip(image,0,1)
    return image[:,:3]

def impaint(original,segment):
    image, mask = original["image"], original["mask"]
    image   = convertImage(image)
    mask    = convertImage(mask)
    segment = convertImage(segment)
    mask  = mask[:,:1]

    segment = segment * 10
    segment = torch.round(segment)
    # le = int(len(segment[0][0])/2)  
    # for i in segment[0][0][le-2:le]:
    #     print(i)
    segment = (segment / 9) + 0.11
    x_prime = imp.mask.propagate(image,mask)

    x_prime2 = torch.cat((x_prime,segment),dim=1)
    image_hat = predict(x_prime2)
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
            imgSegment = gr.Image(label="Classifer",interactive=False, type="pil", image_mode="L")
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