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
import pickle as pk 

class My_Dataset(Dataset):
    def __init__(self, data):
        self.data = data  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本

    def __getitem__(self, idx):  # 函数功能是根据index索引去返回图片img以及标签label
        feat, label = self.data[idx]
        return feat, label

    def __len__(self):   # 函数功能是用来查看数据的长度，也就是样本的数量
        return len(self.data)

        
def load_eplision_torch(batch_size, nums,data_path="data/eplision_cen.csv"):
    data = np.array(pd.read_csv(data_path))
    train_data = data[:,1:-1] 
    train_label = data[:,-1:]
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    train_data, test_data, train_label, test_label = train_test_split(train_data, train_label, test_size=0.1, random_state=0, shuffle=False)
    train_data = torch.tensor(train_data)
    train_label = torch.tensor(train_label)
    test_data = torch.tensor(test_data)
    test_label = torch.tensor(test_label)
    train_data ,train_label, test_data,test_label= train_data.float() ,train_label.float(), test_data.float(),test_label.float()

    train_list, test_list = [], []
    for i in range(train_data.size()[0]):
        train_list.append([train_data[i],train_label[i]])
    for i in range(test_data.size()[0]):
        test_list.append([test_data[i], test_label[i]])

    Train_dloader = []
    length, step = len(train_list), len(train_list)//nums
    for i in range(nums):
        ds = My_Dataset(train_list[i*step:(i+1)*step])
        Train_dloader.append( DataLoader(dataset=ds,batch_size=batch_size,shuffle=True) )

    test_ds = My_Dataset(test_list)
    test_dloader = DataLoader(dataset=test_ds ,batch_size=test_data.size()[0] ,shuffle=True)
    return [Train_dloader, test_dloader]


def load_mnist(batch_size, nums, model = "MLP"):
    train = datasets.MNIST(root="~/data/", train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(root="~/data/", train=False, download=True, transform=transforms.ToTensor())
    print(train.data.size())
    if model == "MLP" :
        train_data = train.data.float().flatten(1)
        test_data = test.data.float().flatten(1)
    elif model == "CNN":
        train_data = train.data.unsqueeze(1).float()
        test_data = test.data.unsqueeze(1).float()
    else:
        print("the data process exist error!")
    
    train_label, test_label = train.targets, test.targets 
    mean = train_data.mean()
    std = train_data.std()

    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    
    train_list, test_list = [], []
    for i in range(train_data.size()[0]):
        train_list.append([train_data[i],train_label[i]])
    for i in range(test_data.size()[0]):
        test_list.append([test_data[i], test_label[i]])


    Train_dloader = []
    length, step = len(train_list), len(train_list)//nums
    for i in range(nums):
        ds = My_Dataset(train_list[i*step:(i+1)*step])
        Train_dloader.append( DataLoader(dataset=ds,batch_size=batch_size,shuffle=True) )

    test_ds = My_Dataset(test_list)
    test_dloader = DataLoader(dataset=test_ds ,batch_size=test_data.size()[0] ,shuffle=True)
    return [Train_dloader, test_dloader]


def load_CIFAR10(batch_size, nums):
    train = datasets.CIFAR10(root="~/data/", train=True, download=True, transform=transforms.ToTensor())
    test = datasets.CIFAR10(root="~/data/", train=False, download=True, transform=transforms.ToTensor())

    train_data, train_label = torch.tensor(train.data).float(), torch.tensor(train.targets).long()
    test_data, test_label = torch.tensor(test.data).float(), torch.tensor(test.targets).long()
    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    train_data, test_data = train_data.transpose(1,3), test_data.transpose(1,3)

    train_list, test_list = [], [] 
    for i in range(train_data.size()[0]):
        train_list.append([train_data[i],train_label[i]])
    for i in range(test_data.size()[0]):
        test_list.append([test_data[i], test_label[i]])
        
    Train_dloader = []
    length, step = len(train_list), len(train_list)//nums
    for i in range(nums):
        ds = My_Dataset(train_list[i*step:(i+1)*step])
        Train_dloader.append( DataLoader(dataset=ds,batch_size=batch_size,shuffle=True) )

    test_ds = My_Dataset(test_list)
    test_dloader = DataLoader(dataset=test_ds ,batch_size=test_data.size()[0] ,shuffle=True)
    return [Train_dloader, test_dloader]