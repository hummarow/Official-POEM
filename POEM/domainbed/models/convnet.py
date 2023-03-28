import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .mixstyle import MixStyle

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, r):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, int(round(out_channels/r)), kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(int(round(out_channels/r)), int(round(out_channels/r)), kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2, padding='same')

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, r):
        super(DenseBlock, self).__init__()
        self.fc1 = nn.Linear(int(round(in_features/r*2)), int(round(out_features/r*2)))
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(round(out_features/r*2)), int(round(out_features/r*2)))
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(int(round(out_features/r*2)), 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, r):
        super(ConvNet, self).__init__()
        self.conv1 = ConvBlock(1, 64, r)
        self.conv2 = ConvBlock(int(round(64/r)), 128, r)
        self.conv3 = ConvBlock(int(round(128/r)), 256, r)
        self.conv4 = ConvBlock(int(round(256/r)), 512, r)
        self.flatten = nn.Flatten()
        self.dense1 = DenseBlock(512*2*10, 2048, r)
        self.dense2 = DenseBlock(2048*r*2, 2048, r)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

