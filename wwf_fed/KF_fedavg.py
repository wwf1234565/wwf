import torch
from torch import nn
from torchvision import datasets, transforms
import torch
import numpy as np
import torch.utils.data as Data
import copy
from torch import nn
import os
import matplotlib.pyplot as plt
from model import LogisticRegression,MLP,CIAFARCNN,MnistCNN,KF_grad,KF_param
from data_process import load_eplision_torch,load_mnist,load_CIFAR10
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 2022
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



class KFDPSGD():
    def __init__(self ,param, data, test_data, model, KF_Grad, KF_Param):
        self.param = param    #{"lr":0.01,"clip":20,"sigma":2,"epochs":epochs,"Ada":3,"Momentum":0.9,"PID_I":1,"PID_D":0}
        self.data, self.test_data = data, test_data

        self.model = model
        self.model_KF = copy.deepcopy(self.model)
        self.model_PID = copy.deepcopy(self.model)
        self.KF_Grad = KF_Grad
        self.KF_Param = KF_Param
        self.Momentum = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}  #记录Momentum动量
        self.Momentum_KF = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}  #记录Momentum动量

    #测试DPSGD-KFGrad
    def DPSGD_KFGrad_update(self):
        if type(self.model) == LogisticRegression:
            LR = True
        else:
            LR= False

        train_dl = self.data 
        if LR==False:
            criterion, criterion_KF = nn.CrossEntropyLoss(),nn.CrossEntropyLoss()
        else:
            criterion, criterion_KF = nn.BCELoss(),nn.BCELoss()

        optimizer, optimizer_KF = torch.optim.SGD(self.model.parameters(), lr=self.param["lr"]),torch.optim.SGD(self.model_KF.parameters(), lr=self.param["lr"])

        for epoch in range(self.param["epochs"]):
            for (batch_x,batch_y) in train_dl:
                y_pred, y_pred_KF = self.model(batch_x), self.model_KF(batch_x)
                loss, loss_KF = criterion(y_pred, batch_y), criterion_KF(y_pred_KF,batch_y)
                optimizer.zero_grad(),optimizer_KF.zero_grad()
                loss.backward(),loss_KF.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.param["clip"])
                torch.nn.utils.clip_grad_norm_(self.model_KF.parameters(), max_norm=self.param["clip"])
                noise = {name: torch.normal(0, self.param["sigma"]*self.param["clip"], param.shape)/self.param["Lots"] for name, param in model.named_parameters()}

                for name,param in self.model.named_parameters(): 
                    param.grad += noise[name]
    
                for name,param in self.model_KF.named_parameters(): 
                    param.grad += noise[name]
                    param.grad = self.KF_Grad.KF_Fliter(copy.deepcopy(param.grad), name)        # 对DP+KF的方法进行过滤
                    param.grad *= self.param["Ada"]
                optimizer.step(),optimizer_KF.step()
        return  

    #KFParam的作用，Momentum VS KF_Param
    def DPSGD_Momentum_KFParam(self,flag):   
        if type(self.model) == LogisticRegression: 
            LR = True
            criterion, criterion_KF = nn.BCELoss(),nn.BCELoss()
        else: 
            LR = False
            criterion, criterion_KF = nn.CrossEntropyLoss(),nn.CrossEntropyLoss()
        optimizer =  torch.optim.SGD(self.model.parameters(), lr=self.param["lr"])
        optimizer_KF = torch.optim.SGD(self.model_KF.parameters(), lr=self.param["lr"])
        for epoch in range(self.param["epochs"]):
            epoch_flag = True
            for (batch_x,batch_y) in self.data:
                y_pred, y_pred_KF= self.model(batch_x), self.model_KF(batch_x)
                loss, loss_KF = criterion(y_pred, batch_y), criterion_KF(y_pred_KF,batch_y)
                optimizer.zero_grad(), optimizer_KF.zero_grad()
                loss.backward(), loss_KF.backward()
                #梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.param["clip"]),torch.nn.utils.clip_grad_norm_(self.model_KF.parameters(), max_norm=self.param["clip"])
                noise = {name: torch.normal(0, self.param["sigma"]*self.param["clip"], param.shape)/self.param["Lots"] for name, param in self.model.named_parameters()}                
                #加噪
                for name,param in self.model.named_parameters():  
                    param.grad += noise[name]
                    self.Momentum[name] = self.Momentum[name] * (self.param["Momentum"]-0.1) + param.grad
                    param.grad = param.grad + self.Momentum[name]

                for name,param in self.model_KF.named_parameters():  
                    param.grad += noise[name]
                    self.Momentum_KF[name] = self.Momentum_KF[name] * self.param["Momentum"] + param.grad
                    param.grad = param.grad + self.Momentum_KF[name]
                optimizer.step(),optimizer_KF.step()
                if flag:
                    self.model_KF.load_state_dict( self.KF_Param.KF_Fliter( copy.deepcopy(self.model_KF.state_dict()), epoch_flag) )  #使用KF_Param对模型参数进行过滤
                    epoch_flag = False
        return

    #KFParam+KFParam的作用，与Momentum的对比
    def DPSGD_Momentum_KFGrad_KFParam(self, flag):   #两种方法
        if type(self.model) == LogisticRegression:
            LR = True
            criterion, criterion_KF = nn.BCELoss(),nn.BCELoss()
        else:
            LR = False 
            criterion, criterion_KF = nn.CrossEntropyLoss(),nn.CrossEntropyLoss()

        optimizer =  torch.optim.SGD(self.model.parameters(), lr=self.param["lr"] )
        optimizer_KF = torch.optim.SGD(self.model_KF.parameters(), lr=self.param["lr"])
        for epoch in range(self.param["epochs"]):
            epoch_flag = True
            for (batch_x,batch_y) in self.data:
                y_pred, y_pred_KF = self.model(batch_x), self.model_KF(batch_x) 
                loss, loss_KF = criterion(y_pred, batch_y), criterion_KF(y_pred_KF,batch_y) 
                optimizer.zero_grad(),optimizer_KF.zero_grad() 
                loss.backward(),loss_KF.backward() 
                #梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.param["clip"]),torch.nn.utils.clip_grad_norm_(self.model_KF.parameters(), max_norm=self.param["clip"]) 
                noise = {name: torch.normal(0, self.param["sigma"]*self.param["clip"], param.shape)/self.param["Lots"] for name, param in self.model.named_parameters()}
                #加噪
                for name,param in self.model.named_parameters():  param.grad += noise[name]
                for name,param in self.model_KF.named_parameters():
                    param.grad += noise[name]
                    param.grad = self.KF_Grad.KF_Fliter(copy.deepcopy(param.grad), name)        # 对DP+KF的方法进行过滤
                    param.grad *= self.param["Ada"]
                    self.Momentum_KF[name] = self.Momentum_KF[name] * self.param["Momentum"] + param.grad
                    param.grad = param.grad + self.Momentum_KF[name]

                optimizer.step(),optimizer_KF.step() 
                if flag:
                    self.model_KF.load_state_dict(self.KF_Param.KF_Fliter(copy.deepcopy(self.model_KF.state_dict()),epoch_flag))  #使用KF_Param对模型参数进行过滤
                    epoch_flag = False
        return

    def get_test_acc(self, model):
        for test_data,test_label in self.test_data:
            y_pred_test = model(test_data)
        if type(model) != LogisticRegression:
            _,mask = torch.max(y_pred_test,1)
        else:
            mask = y_pred_test.ge(0.5).float()
        correct = (mask == test_label).sum() 
        acc = correct.item() / test_label.size(0)
        return acc


def train_kfgrad(Clients, epoch, sample_rate):
    clients = random.sample(Clients, int(len(Clients)*sample_rate))  #从中抽取一定比例的客户端参与训练
    ACC, ACC_KF = [], []
    for e in range(epoch): 
        for client in clients:
            client.DPSGD_KFGrad_update()
        base1, base2 = clients[0].model.state_dict(), clients[0].model_KF.state_dict()
        # 聚合
        for i in range(1, len(clients)):   #求和  
            for name in base1:
                base1[name] += clients[i].model.state_dict()[name] 
                base2[name] += clients[i].model_KF.state_dict()[name] 
        for name in base1:  #平均
            base1[name] /= len(clients)
            base2[name] /= len(clients)
        # 分发
        for cl in Clients:
            cl.model.load_state_dict(copy.deepcopy(base1))
            cl.model_KF.load_state_dict(copy.deepcopy(base2))

        acc, acc_KF = Clients[0].get_test_acc(Clients[0].model), Clients[0].get_test_acc(Clients[0].model_KF)
        ACC.append(acc), ACC_KF.append(acc_KF)
        print("第",e,"次迭代,DPSGD的准确率:", acc,"KF-DPSGD的准确率:",acc_KF )
    #画图
    if type(Clients[0].model) == LogisticRegression:
        m = "LR_Epl"
    elif type(Clients[0].model) == MLP:
        m = "MLP_MINIST"
    elif type(Clients[0].model) == MnistCNN:
        m = "CNN_MINIST"
    elif type(Clients[0].model) == CIAFARCNN:
        m = "CNN_CIAFAR10"
    else:
        print("model name exist error!")
    
    X = [i for i in range(0, epochs)]
    plt.xlabel("epoch"), plt.ylabel("Accuracy")

    plt.plot(X, ACC, label="Fed-DPSGD", linestyle='--', markersize=6, alpha=0.8, linewidth=3)
    plt.plot(X, ACC_KF, label="Fed-KFGrad", linestyle='-', markersize=6, alpha=0.8, linewidth=3)
    plt.grid(True)  # 打开网格
    # 添加标题和标签
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(loc='lower right')

    path = "./picture_fed/KF_Grad结果/" + m + ".png"
    plt.savefig(path)
    plt.close()

    data = {"DPSGD":ACC,"DPSGD-KFGrad":ACC_KF}
    df = pd.DataFrame(data)
    df.to_csv("./picture_fed/KF_Grad结果/" +m+".csv")

def train_kfparam(Clients, epoch, sample_rate):
    clients = random.sample(Clients, int(len(Clients)*sample_rate))  #从中抽取一定比例的客户端参与训练
    ACC, ACC_KF, ACC_PID = [], [], []
    for e in range(epoch): 
        for client in clients:
            if e > client.param["threshold"]:
                client.DPSGD_Momentum_KFParam(True)
            else:
                client.DPSGD_Momentum_KFParam(False)
        base1, base2 = clients[0].model.state_dict(), clients[0].model_KF.state_dict()
        Momentum1, Momentum2 = clients[0].Momentum, clients[0].Momentum_KF
        # 聚合
        for i in range(1, len(clients)):   #求和  
            for name in base1:
                base1[name] += clients[i].model.state_dict()[name]
                base2[name] += clients[i].model_KF.state_dict()[name]
                Momentum1[name] += clients[i].Momentum[name]
                Momentum2[name] += clients[i].Momentum_KF[name]
        for name in base1:  #平均
            base1[name] /= len(clients)
            base2[name] /= len(clients)
            Momentum1[name] /= len(clients)
            Momentum2[name] /= len(clients)
        # 分发
        for cl in Clients:
            cl.model.load_state_dict(copy.deepcopy(base1))
            cl.model_KF.load_state_dict(copy.deepcopy(base2))
            cl.Momentum = copy.deepcopy(Momentum1)
            cl.Momentum_KF = copy.deepcopy(Momentum2)

        acc, acc_KF  = Clients[0].get_test_acc(Clients[0].model), Clients[0].get_test_acc(Clients[0].model_KF)
        ACC.append(acc), ACC_KF.append(acc_KF)
        print("第",e,"次迭代,DPSGD的准确率:", acc,"KF-DPSGD的准确率:",acc_KF)
    #画图
    if type(Clients[0].model) == LogisticRegression:
        m = "LR_Epl"
    elif type(Clients[0].model) == MLP:
        m = "MLP_MINIST"
    elif type(Clients[0].model) == MnistCNN:
        m = "CNN_MINIST"
    elif type(Clients[0].model) == CIAFARCNN:
        m = "CNN_CIAFAR10"
    else:
        print("model name exist error!")
    
    X = [i for i in range(0, epochs)]
    plt.xlabel("epoch"), plt.ylabel("Accuracy")


    plt.plot(X, ACC, label="Fed-DPSGD-M", linestyle='--', markersize=6, alpha=1, linewidth=2)
    plt.plot(X, ACC_KF, label="Fed-KFParam", linestyle='-', markersize=6, alpha=1, linewidth=2)
    plt.grid(True)  # 打开网格
    # 添加标题和标签
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(loc='lower right')

    path = "./picture_fed/KF_Param结果/" + m + ".png"
    plt.savefig(path)
    print(path)
    plt.close()

    data = {"DPSGD":ACC,"DPSGD-KFGrad":ACC_KF}
    df = pd.DataFrame(data)
    print("./picture_fed/KF_Param结果/" +m+".csv")
    df.to_csv("./picture_fed/KF_Param结果/" +m+".csv")

def train_kf2(Clients, epoch, sample_rate):
    clients = random.sample(Clients, int(len(Clients)*sample_rate))  #从中抽取一定比例的客户端参与训练
    ACC, ACC_KF, ACC_PID = [], [], []
    for e in range(epoch): 
        for client in clients:
            if e > client.param["threshold"]:
                client.DPSGD_Momentum_KFGrad_KFParam(True)
            else:
                client.DPSGD_Momentum_KFGrad_KFParam(False)
        base1, base2 = clients[0].model.state_dict(), clients[0].model_KF.state_dict()
        Momentum2  = clients[0].Momentum_KF 
        # 聚合
        for i in range(1, len(clients)):   #求和  
            for name in base1:
                base1[name] += clients[i].model.state_dict()[name]
                base2[name] += clients[i].model_KF.state_dict()[name]
                Momentum2[name] += clients[i].Momentum_KF[name]
        for name in base1:  #平均
            base1[name] /= len(clients)
            base2[name] /= len(clients)
            Momentum2[name] /= len(clients)
        # 分发
        for cl in Clients:
            cl.model.load_state_dict(copy.deepcopy(base1))
            cl.model_KF.load_state_dict(copy.deepcopy(base2))
            cl.Momentum_KF = copy.deepcopy(Momentum2)

        acc, acc_KF  = Clients[0].get_test_acc(Clients[0].model), Clients[0].get_test_acc(Clients[0].model_KF)
        ACC.append(acc), ACC_KF.append(acc_KF)
        print("第",e,"次迭代,DPSGD的准确率:", acc,"KF-DPSGD的准确率:",acc_KF)
    #画图
    if type(Clients[0].model) == LogisticRegression:
        m = "LR_Epl"
    elif type(Clients[0].model) == MLP:
        m = "MLP_MINIST"
    elif type(Clients[0].model) == MnistCNN:
        m = "CNN_MINIST"
    elif type(Clients[0].model) == CIAFARCNN:
        m = "CNN_CIAFAR10"
    else:
        print("model name exist error!")
    
    X = [i for i in range(0, epochs)]
    plt.xlabel("epoch"), plt.ylabel("Accuracy")


    plt.plot(X, ACC, label="Fed-DPSGD", linestyle='--', markersize=6, alpha=1, linewidth=2)
    plt.plot(X, ACC_KF, label="Fed-KF2", linestyle='-', markersize=6, alpha=1, linewidth=2)
    plt.grid(True)  # 打开网格
    # 添加标题和标签
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(loc='lower right')

    path = "./picture_fed/KF2结果/" + m + ".png"
    plt.savefig(path)
    plt.close()

    data = {"DPSGD":ACC,"DPSGD-KF2":ACC_KF}
    df = pd.DataFrame(data)
    df.to_csv("./picture_fed/KF2结果/" +m+".csv")

# 客户端的参数配置
param = {"lr":0.002,"clip":1,"sigma": 5,"epochs":10,"Lots":50,
        "Ada":1.5,        #  KFGrad方法的参数
        "Momentum":0.5, "threshold":40,"Q_decent": 0.95, #KFParam方法的参数
        "PID_D":0.004}    #PID方法的参数

# 联邦学习参数配置
nums = 20  # 联邦学习参与方数量
epochs = 50
sample_rate = 0.2


# #####################################测试Fed-KFGrad#####################################

# Data_Model = [
#              #  [load_eplision_torch(batch_size=param["Lots"], nums=nums),LogisticRegression()],
#               [load_mnist(batch_size=param["Lots"], nums=nums,model = "MLP"),  MLP()],
#               [load_mnist(batch_size=param["Lots"], nums=nums,model = "CNN"),  MnistCNN()],
#               [load_CIFAR10(batch_size=param["Lots"], nums=nums) ,CIAFARCNN()]
#               ]

# for dm in Data_Model:
#     data, model = dm[0], dm[1]
#     KF_Grad = KF_grad(Q=10,R=10,P=10, Net_Piror=copy.deepcopy(model))
#     KF_Param = KF_param(Q=10 ,R=1 ,P=10, alfa=param["Q_decent"], Net_Piror =copy.deepcopy(model))
    
#     Clients = []
#     for i in range(nums):    #初始化各个参与方的模型参数
#         Clients.append( KFDPSGD(param = param, data = data[0][i], test_data = copy.deepcopy(data[1]), model = copy.deepcopy(model),  KF_Grad=copy.deepcopy(KF_Grad), KF_Param=copy.deepcopy(KF_Param) ) )
    
#     train_kfgrad(Clients, epochs, sample_rate)

# #####################################测试Fed-KFParam#####################################

Data_Model = [
              [load_eplision_torch(batch_size=param["Lots"], nums=nums),LogisticRegression()],
              [load_mnist(batch_size=param["Lots"], nums=nums,model = "MLP"),  MLP()],
              [load_mnist(batch_size=param["Lots"], nums=nums,model = "CNN"),  MnistCNN()],
              # [load_CIFAR10(batch_size=param["Lots"], nums=nums) ,CIAFARCNN()]
              ]

for dm in Data_Model:
    data, model = dm[0], dm[1]
    KF_Grad = KF_grad(Q=10,R=10,P=10, Net_Piror=copy.deepcopy(model))
    KF_Param = KF_param(Q=10 ,R=1 ,P=10, alfa=param["Q_decent"], Net_Piror =copy.deepcopy(model.state_dict()))
    
    Clients = []
    for i in range(nums):    #初始化各个参与方的模型参数
        Clients.append( KFDPSGD(param = param, data = data[0][i], test_data = copy.deepcopy(data[1]), model = copy.deepcopy(model),  KF_Grad=copy.deepcopy(KF_Grad), KF_Param=copy.deepcopy(KF_Param) ) )
    train_kfparam(Clients, epochs, sample_rate)


# # #####################################测试Fed-KF2#####################################

# Data_Model = [
#              [load_eplision_torch(batch_size=param["Lots"], nums=nums),LogisticRegression()],
#               [load_mnist(batch_size=param["Lots"], nums=nums,model = "MLP"),  MLP()],
#               [load_mnist(batch_size=param["Lots"], nums=nums,model = "CNN"),  MnistCNN()],
#               [load_CIFAR10(batch_size=param["Lots"], nums=nums) ,CIAFARCNN()]
#               ]

# for dm in Data_Model:
#     data, model = dm[0], dm[1]
#     KF_Grad = KF_grad(Q=10,R=10,P=10, Net_Piror=copy.deepcopy(model))
#     KF_Param = KF_param(Q=10 ,R=1 ,P=10, alfa=param["Q_decent"], Net_Piror =copy.deepcopy(model))
#     Clients = []
#     for i in range(nums):    #初始化各个参与方的模型参数
#         Clients.append( KFDPSGD(param = param, data = data[0][i], test_data = copy.deepcopy(data[1]), model = copy.deepcopy(model),  KF_Grad=copy.deepcopy(KF_Grad), KF_Param=copy.deepcopy(KF_Param) ) )
    
#     train_kf2(Clients, epochs, sample_rate)