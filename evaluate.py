from utils import *
from ssim import ssim
from model import TFPNet
from dataset import *
import glob
from cfg import Config as C 
from torch.utils.data import DataLoader
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_test = glob.glob(C.test_x)
y_test = glob.glob(C.test_y)

test_ds = TestDeTextDataset(x_test, y_test)
test_loader = DataLoader(test_ds, 1, shuffle=False)

checkpoint = torch.load(C.saved_model_path+'weight_best.pth')
model = TFPNet()
model.load_state_dict(checkpoint['weights'])
model.to(device)


psnr1 = []
ssim1 = []

with torch.no_grad():
    net.eval()
    for i, data in tqdm(enumerate(test_loader)):

        x = data[0].to(device)
        y = data[1].to(device)

        _, _, pred = net(x)

        ps = psnr(pred, y)
        sm = ssim(pred, y)

        psnr1.append(ps)
        ssim1.append(sm)

print('PSNR: ',sum(psnr1)/len(psnr1))
print('SSIM: ',sum(ssim1)/len(ssim1))