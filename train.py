from tqdm import tqdm
from cfg import Config as C
from utils import *
from ssim import ssim

def train(model, epochs, train_loader, test_loader, optimizer, critean):
    for epoch in range(epochs, 500):
        print('Epoch: {}'.format(epoch))
        l_train = []
        ps_train = []
        l_test = []
        ps_test = []
        mss = []
        model.train()
        current_psnr = 15

        for i, data in tqdm(enumerate(train_loader)):
            x, y1, y2, y3 = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            optimizer.zero_grad()
            pred1, pred2, pred3 = model(x)
            loss = critean(pred1, pred2, pred3, y1, y2, y3)
            # ssim1 = ssim(pred3, y3)
            ssim2 = ssim(pred2, y2)
            # ssim3 = ssim(pred1, y1)
            loss_ssim =  (1 - ssim2)*2
            loss += loss_ssim
        
            psnr1=psnr(pred3.detach(),y3)
            loss.backward()
            optimizer.step()
            l_train.append(loss.item())
            # mss.append(ssim1)
            ps_train.append(psnr1)
        print("Epoch loss: ", sum(l_train)/len(l_train))
        print('Epoch {} PSNR: '.format(epoch), sum(ps_train)/len(ps_train))
        

        with torch.no_grad():
          model.eval()
          mss_val = []
          for i, data in tqdm(enumerate(test_loader)):
            x, y1, y1, y3 = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            pred1, pred2, pred3 = model(x)

            psnr1=psnr(pred3,y3)
            val_mss = ssim(pred3, y3)

            mss_val.append(val_mss)
            ps_test.append(psnr1)


        print('Val Epoch {} PSNR: '.format(epoch), sum(ps_test)/len(ps_test))
        print('VAL SSIM: ', sum(mss_val)/len(mss_val))

        if current_psnr < sum(ps_test)/len(ps_test):
          checkpoint = {
          'weights': model.state_dict(),
          'optimizer':optimizer.state_dict()
          } 
          print("saving best one.....")
          torch.save(checkpoint, C.saved_model_path + 'weight_best.pth')
          current_psnr = sum(ps_test)/len(ps_test)

          
        if (epoch+1) % 10 == 0:
          checkpoint = {
          'weights': model.state_dict(),
          'optimizer':optimizer.state_dict()
          } 
          torch.save(checkpoint, C.saved_model_path+"checkpoint_SCUT{}.pth".format(epoch+1))