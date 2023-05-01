import os
import orjson
import numpy as np
import torch
import yaml
import torchvision.transforms as transforms
import torchvision
import pytesseract
import boto3
import botocore
import io

from PIL import Image, ImageOps
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 
from flask import Response

from pdf2image import convert_from_bytes

#Load types of boxes

with open("modules/reader/data.yaml", "r") as stream:
    try:
        classes_dict={k:v for k,v in enumerate(yaml.safe_load(stream)['names'])}
        inv_classes_dict={v: k for k, v in classes_dict.items()}
    except yaml.YAMLError as exc:
        print(exc)


# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
num_classes = 5 # background, item,description,title and price
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# We will set parent items and childs 
device = torch.device("cpu")
model.to(device)
state_dict=torch.load('nn_fasterrcnn_resnet50_fpn_fine_tuned_721_images.pth',map_location=device)
model.load_state_dict(state_dict)
model.eval()



def contains(parent, child):
        return parent[0] < child[0] < child[2] < parent[2] and parent[1] < child[1]< child[3] < parent[3]
def isOverlapping1D(parent, child):
    return parent[1] >= child[0] and child[1] >= parent[0]
def overlaps(parent, child):
    return isOverlapping1D([parent[0],parent[2]], [child[0],child[2]]) and \
            isOverlapping1D([parent[1],parent[3]], [child[1],child[3]])

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result
def process_image(image):
    return add_margin(image, int(image.size[1]*0.1), int(image.size[0]*0.1), int(image.size[1]*0.1), int(image.size[0]*0.1), (255,255,255))

def read_menu(img_path,language="spa", BUCKET='contler-test'):
    s3_resource = boto3.resource('s3')
    try:
        content_object=s3_resource.Object(BUCKET, img_path.split(".")[-2]+'.json')
        file_content = content_object.get()['Body'].read().decode('utf-8')
        return Response(file_content, mimetype='application/json')
    except botocore.exceptions.ClientError as e:
        img_object=s3_resource.Object(BUCKET, img_path)
        file_stream = io.BytesIO()
        img_object.download_fileobj(file_stream)
        if(img_path.split(".")[-1] in ['PDF', 'pdf']):
            pdf_image_lst = convert_from_bytes(file_stream.getvalue())
            img = pdf_image_lst[0]
        else:
            img=Image.open(file_stream)
        img = img.convert("RGB")
        img.format = 'jpeg'
        if is_background_dark(img):
            img = ImageOps.invert(img)
        transforms_img = transforms.Compose([transforms.ToTensor()])
        img_T=transforms_img(img)
        result=model([img_T])

        indexes=[item.item() for item in result[0]['scores']>0.5]
        parents=[]
        for obj in range(len(result[0]['boxes'])):
            if ((result[0]['labels'][obj].item()-1)==inv_classes_dict['item']) and indexes[obj]:
                parents.append({'box':result[0]['boxes'][obj].detach().numpy(),'class':result[0]['labels'][obj].item()})
            
        for parent in parents:
            for obj in range(len(result[0]['boxes'])):
                if overlaps(parent['box'], result[0]['boxes'][obj].detach().numpy()) and\
                (result[0]['labels'][obj].item()-1)!=1 and indexes[obj]:
                    parent[classes_dict[result[0]['labels'][obj].item()-1]]=result[0]['boxes'][obj].detach().numpy()
        for parent in parents:
            #print('Item \n')
            if 'tittle' in parent.keys():
                parent['title_text']=pytesseract.image_to_string(process_image(img.crop(box=parent['tittle'])),config=r'--psm 11',lang=language).replace("\n", " ")
                #print('Title:',parent['title_text'])
            if 'price'in parent.keys():
                parent['price_text']=pytesseract.image_to_string(process_image(img.crop(box=parent['price'])),config=r'--psm 11').replace("\n", " ")
                #print('Price:',parent['price_text'])
            if 'description' in parent.keys():
                parent['description_text']=pytesseract.image_to_string(process_image(img.crop(box=parent['description'])),config=r'--psm 11',lang=language).replace("\n", " ")
                #print('Description:',parent['description_text'])
        #print(parents)
        to_list=lambda x: {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in x.items()}
        res=list(map(to_list,parents))
        s3object =s3_resource.Object(BUCKET, img_path.split(".")[-2]+'.json')
        s3object.put(Body=orjson.dumps(res))
        return Response(orjson.dumps(res).decode("utf-8"), mimetype='application/json')

def is_background_dark(img):
    # Get the average color of the image
    array = np.asarray(img) # a is readonly
    avg_color = np.mean(array, axis=(0,1))
    #print(avg_color)
    # Convert the average color to HSL color space
    r, g, b = avg_color
    r /= 255
    g /= 255
    b /= 255
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    l = (c_max + c_min) / 2

    # Check if the lightness of the average color is less than 0.5
    return l < 0.5