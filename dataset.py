import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from cfg import Config as C
from utils import *

class DeTextDataset(Dataset):
  def __init__(self, img_paths=None, mask_paths=None, poly_paths=None, a=True, test=False):
    self.a = a
    self.img_paths = img_paths
    self.mask_paths = mask_paths
    self.poly_paths = poly_paths
    assert len(self.img_paths) == len(self.mask_paths)
    self.images = len(self.img_paths) #list all the files present in that folder...
    self.test = test
  
  def __len__(self):
    return len(self.img_paths) #length of dataset

  def Lowpass(self, img):
    temp = img.copy()
    dst = cv2.GaussianBlur(temp,(5,5),cv2.BORDER_DEFAULT) 
    return dst
  
  def Highpass(self, img):
    temp = img.copy()
    dst = cv2.GaussianBlur(temp, (3,3) ,0) 
    source_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    dest = cv2.Laplacian(source_gray, cv2.CV_16S, ksize=3)
    abs_dest = cv2.convertScaleAbs(dest)
    return abs_dest
  
  def __getitem__(self, index):
    img_path = self.img_paths[index]
    mask_path = self.mask_paths[index]
    poly_path = self.poly_paths[index]

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))

    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (512, 512))

    poly = poly_to_mask(poly_path)
    if self.a:
      image, mask, poly = transforms(image, mask, poly)

    highpass = self.Highpass(mask)
    
    image = image.astype(np.float32)
    image = image/255.0
    image = torch.from_numpy(image)
    image = image.permute(2,0,1)

    poly = poly.astype('float32')
    poly = torch.from_numpy(poly)
    poly = (poly.permute(2, 0, 1) / 255.0)


    highpass = highpass.astype(np.float32)
    highpass = highpass[:,:,np.newaxis]
    highpass = highpass/255.0
    highpass = torch.from_numpy(highpass)
    highpass = highpass.permute(2,0,1)

    mask = mask.astype(np.float32)
    mask = mask/255.0
    mask = torch.from_numpy(mask)
    mask = mask.permute(2,0,1)

    return image, poly[0:1, :, :], highpass, mask


class TestDeTextDataset(Dataset):
  def __init__(self, img_paths=None, mask_paths=None, size=1024):

    self.img_paths = img_paths
    self.mask_paths = mask_paths
    self.size = size
    assert len(self.img_paths) == len(self.mask_paths)

  
  def __len__(self):
    return len(self.img_paths) 
  
  def __getitem__(self, index):
    img_path = self.img_paths[index]
    mask_path = self.mask_paths[index]

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (self.size, self.size))

    image = image.astype(np.float32)
    image = image/255.0
    image = torch.from_numpy(image)
    image = image.permute(2,0,1)
    
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (self.size, self.size))

    mask = mask.astype(np.float32)
    mask = mask/255.0
    mask = torch.from_numpy(mask)
    mask = mask.permute(2,0,1)

    return image, mask


def get_loaders():

    x_train=glob.glob(C.train_x)
    y_train=glob.glob(C.train_y)
    mask_trian=glob.glob(C.train_mask)

    x_test=glob.glob(C.test_x)
    y_test=glob.glob(C.test_y)
    mask_test=glob.glob(C.mask_test)

    train_ds = DeTextDataset(x_train, y_train, 
    mask_train, 
    a=True, test=False)

    test_ds = DeTextDataset(x_test, y_test, 
    mask_test, 
    a=False, test=True)


    train_loader = DataLoader(train_ds, batch_size=C.batch_size, num_workers=C.num_worker, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=C.batch_size, num_workers=C.num_worker, shuffle=False)

    return train_loader, test_loader


