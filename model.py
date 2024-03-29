import pandas as pd
import torch
import torchvision
from torch import nn, optim

#图片32x32，3通道

class cnn(nn.Module):
    def __init__(self, in_dim, n_class):
        super(cnn, self).__init__()
        self.conv = nn.Sequential(
            #图像尺寸32*32，3通道
            nn.Conv2d(in_dim, 6, 3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.fc = nn.Sequential(
            nn.Linear(400,120),
            nn.Linear(120,84),
            nn.Linear(84, n_class)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out