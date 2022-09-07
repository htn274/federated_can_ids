from networks.resnet import SupResNet
import torch

def init_dataloader():
    pass

def init_model():
    pass

def train():
    pass

def validate():
    pass

if __name__ == '__main__':
    model = SupResNet(num_classes=3)
    X = torch.rand(20, 1, 29, 29)
    print(model(X).shape)