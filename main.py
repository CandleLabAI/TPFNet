from utils import *
from cfg import Config as C 
from model import TFPNet
from losses import *
import torch
from dataset import *


def main()
    epochs=C.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TFPNet()
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=C.lr)
    loss = Loss().to(device)
    train_loader, test_loader = get_loaders()
    train(net, epochs, train_loader, test_loader, optimizer, loss)

if __name__ == "__main__":
    main()