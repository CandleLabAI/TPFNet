import torch
import cv2
from model import TFPNet
from cfg import Config as C

model = TFPNet()
checkpoint = torch.load(C.saved_model_path+'weight_best.pth')
model.load_state_dict(checkpoint['weights'])



def get_image_predections(img_path, size):

    with torch.no_grad():
        f, axrr = plt.subplots(1,3, figsize=(25, 25))

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))

        axrr[0].imshow(img)
        image = img

        img = img/255.
        img = torch.from_numpy(img.astype('float32')).permute(2, 0, 1).unsqueeze(0).to(device)

        pred1, pred2, pred3 = net(img)
        
        pred3 = pred3.detach().cpu().clamp(0,1).squeeze(0).permute(1, 2, 0).numpy()
        pred1 = pred1.detach().cpu().clamp(0,1).squeeze(0).permute(1, 2, 0).squeeze(-1).numpy()

        
        axrr[1].imshow(pred3)
        axrr[2].imshow(pred1>0.4)
        plt.show()

if __name__ == '__main__':
    get_image_predections

