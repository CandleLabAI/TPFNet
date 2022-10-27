import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-7):
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class Loss(nn.Module):
  def __init__(self):
    super(Loss, self).__init__()

    self.l1 = nn.L1Loss()
    self.l2 = nn.L1Loss()
    self.l3 = nn.L1Loss()
    self.bce = nn.BCELoss()
    self.dice = DiceLoss()

  def forward(self, pred1, pred2, pred3, y1, y2, y3):
    
    l1 = self.l1(pred1, y1)
    l2 = self.l2(pred2, y2)
    l3 = self.l3(pred3, y3)

    bi = self.bce(pred1, y1)
    dice = self.dice(pred1, y1)
    loss = (l1 + l2 + l3)/3.0
    return loss + 0.5*(bi + dice)