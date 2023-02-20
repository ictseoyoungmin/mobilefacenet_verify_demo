import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

class OOnetFCN(nn.Module):
    def __init__(self,n_classes=3):
        super(OOnetFCN,self).__init__()
        self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 32, 3, 1)),
                ('prelu1', nn.PReLU(32)),
                ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

                ('conv2', nn.Conv2d(32, 64, 3, 1)),
                ('prelu2', nn.PReLU(64)),
                ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

                ('conv3', nn.Conv2d(64, 64, 3, 1)),
                ('prelu3', nn.PReLU(64)),
                ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

                ('conv4', nn.Conv2d(64, 128, 2, 1)),
                ('prelu4', nn.PReLU(128)),
                ]))
        self.clf1 = nn.Conv2d(128,64,3,2,1)
        self.clf2 = nn.Conv2d(64,n_classes,3,2,1)

    def forward(self,x):
        x = self.features(x)
        x = self.clf1(x)
        x = self.clf2(x)
        return x
class OOnet(nn.Module):
    def __init__(self,n_classes=3):
        super(OOnet,self).__init__()
        self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(3, 32, 3, 1)),
                ('prelu1', nn.PReLU(32)),
                ('bnorm1',nn.BatchNorm2d(32)),
                ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

                ('conv2', nn.Conv2d(32, 64, 3, 1)),
                ('prelu2', nn.PReLU(64)),
                ('bnorm2',nn.BatchNorm2d(64)),
                ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

                ('conv3', nn.Conv2d(64, 64, 3, 1)),
                ('prelu3', nn.PReLU(64)),
                ('bnorm3',nn.BatchNorm2d(64)),
                ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

                ('conv4', nn.Conv2d(64, 128, 2, 1)),
                ('prelu4', nn.PReLU(128)),
                ('bnorm4',nn.BatchNorm2d(128)),
                ]))
        self.clf1 = nn.Conv2d(128,64,3,2,1)
        self.clf2 = nn.Conv2d(64,32,3,2,1)
        ##
        self.out1 = nn.Linear(32*6*6,128,bias=False)
        self.out2 = nn.Linear(128, n_classes, bias=False)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self,x):
        x = self.features(x)
        x = self.clf1(x)
        x = self.relu(self.clf2(x))
        x = x.view(-1, 32*6*6)
        x = self.drop(self.relu(self.out1(x)))
        out = self.out2(x)

        return out

##############################

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

    
class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.res_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128))
        
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256))
            
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512))

        # self.res_conv4 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3,stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(128))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layer1(x) + x
        x = self.layer2(x) + self.res_conv1(x)
        x = self.layer3(x) + self.res_conv2(x)
        x = self.layer4(x) + self.res_conv3(x)
        x = self.avgpool(x)
        x = x.view(-1,512)
        out = self.fc(x)

        return out
