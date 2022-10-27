import numpy as np
import cv2
from PIL import Image, ImageDraw
import albumentations as A
import math


def poly_to_mask(poly):
    filee = open(poly, 'r')
    mask = np.zeros((512, 512))
    lines = filee.readlines()
    for line in lines:
        line = line.replace('\n', '')
        line = line.split(',')
        line = [int(i) for i in line]

        polygon = line
        width = 512
        height = 512

        img = Image.fromarray(np.zeros((512, 512), dtype='uint8'))
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask += np.array(img)
    mask = np.expand_dims((mask > 0).astype('uint8'), axis=2)

    return np.concatenate((mask, mask, mask), axis=2)*255


def bbox_to_mask(img, txt):
    img = cv2.imread(img)
    mm = open(txt, 'r')
    mask = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)

    for i in mm.readlines():
        i = i.split(',')
        i = [int(k) for k in i]
        mask[i[1]:i[3], i[0]:i[2]] = 255
    return mask


def transforms(x1, x2, x3):

    if random.uniform(0,1) > 0.4:
        t2 = A.HorizontalFlip(p=1)
        x1 = t2(image=x1)
        x1 = x1['image']
        x2 = t2(image=x2)
        x2 = x2['image']
        x3 = t2(image=x3, mask=x3)
        x3 = x3['mask']
 

    elif random.uniform(0,1) > 0.4:
        t2 = A.RandomBrightnessContrast(p=1)
        x1 = t2(image=x1)
        x1 = x1['image']
        x2 = t2(image=x2)
        x2 = x2['image']
        x3 = t2(image=x3, mask=x3)
        x3 = x3['mask']
 

    elif random.uniform(0,1) > 0.5:
        t2 = A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.5, alpha_affine=120 * 0.3)
        x1 = t2(image=x1)
        x1 = x1['image']
        x2 = t2(image=x2)
        x2 = x2['image']
        x3 = t2(image=x3, mask=x3)
        x3 = x3['mask']
 

    
    elif random.uniform(0,1) > 0.5:
        t2 = A.GridDistortion(p=1)
        x1 = t2(image=x1)
        x1 = x1['image']
        x2 = t2(image=x2)
        x2 = x2['image']
        x3 = t2(image=x3, mask=x3)
        x3 = x3['mask']
 

    elif random.uniform(0,1) > 0.:
        t2 = A.OpticalDistortion(distort_limit=8, shift_limit=0.7, p=1)
        x1 = t2(image=x1)
        x1 = x1['image']
        x2 = t2(image=x2)
        x2 = x2['image']
        x3 = t2(image=x3, mask=x3)
        x3 = x3['mask']
 
    return x1, x2, x3


def psnr(pred, gt):
    pred=pred.clamp(0,1).detach().cpu().numpy()
    gt=gt.clamp(0,1).detach().cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10( 1.0 / rmse)


