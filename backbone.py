import torch
import torch.nn as nn
import torchvision


def MobileNetV2(embedding_size):
    model = torchvision.models.mobilenet_v2()
        
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(in_features=1280, out_features=embedding_size, bias=True)
    )
    
    return model


def ResNet50(embedding_size):
    model = torchvision.models.resnet50()
        
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=embedding_size, bias=True)
    )
    
    return model


if __name__ == '__main__':
    model = ResNet50(512)
    print(model)