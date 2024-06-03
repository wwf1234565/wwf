# Machine learning models
import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy
from torchvision import datasets, transforms
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class LogisticRegression(nn.Module):
    def __init__(self, num_feature=10, output_size=1):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_feature, output_size)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmod(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, output_dim))

    def forward(self, x):
        return self.model(x)

class MnistCNN(nn.Module):
    def __init__(self ):
        super(MnistCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CIAFARCNN(nn.Module):  #CNN  CIAFAR10 3*32*32
    def __init__(self):  
        super(CIAFARCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(18, 36, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(36 * 8 * 8, 10)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x 

class KF_grad:
    def __init__(self,Q,R,P,Net_Piror):
        self.Q = Q
        self.R = R
        self.P = P
        self.Net_Piror = {}         #记录梯度
        for name, param in Net_Piror.named_parameters():
            self.lastlayer_name = name          #记录最后一层的名字
            self.Net_Piror[name] = torch.zeros(param.size())
        self.first = True      #是否是第一个net
    def KF_Fliter(self,grad, name):
        if self.first:          #若是第一个，则直接返回
            self.Net_Piror[name] = grad
            if name == self.lastlayer_name:
                self.first = False
            return grad

        K = self.P / (self.P + self.R)
        grad = self.Net_Piror[name] + K * (grad - self.Net_Piror[name])
        
        if name == self.lastlayer_name:
            self.P = (1-K)*self.P + self.Q
        self.Net_Piror[name] = copy.deepcopy(grad)
        return   grad
class KF_param:
    def __init__(self,Q,R,P,alfa,Net_Piror=None):
        self.Q = Q
        self.R = R
        self.P = P
        self.Net_Piror = None
        self.alfa = alfa
        self.iter = 0
 
    def KF_Fliter(self,net,epoch=False):
        self.iter += 1
        if self.Net_Piror == None:
            self.Net_Piror = copy.deepcopy(net)
            return net
        if epoch:
            self.Q = self.Q * self.alfa
        K = self.P / (self.P + self.R)
        for name in net:
            net[name] = self.Net_Piror[name] + K * (net[name] - self.Net_Piror[name])
        self.P = (1-K)*self.P + self.Q
        self.Net_Piror = copy.deepcopy(net)
        return net