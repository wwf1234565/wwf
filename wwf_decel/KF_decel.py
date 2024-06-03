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

class Client():
    def __init__(self, param, Graph, model, kcif_grad, kcif_param, train_data, test_data):
        self.param = param
        self.nums = len(Graph)  #客户端数量
        self.model = model
        self.model_KCIF =  [copy.deepcopy(model) for i in range(self.nums)]
        if type(model) == LogisticRegression: self.LR = True
        else: self.LR = False

        self.Momentums = [{name: torch.zeros_like(param) for name, param in self.model.named_parameters()} for i in range(self.nums)]

        self.Graph = Graph
        self.kcif_grad = [copy.deepcopy(kcif_grad) for i in range(self.nums)]
        self.kcif_param = [copy.deepcopy(kcif_param) for i in range(self.nums)]

        # 将数据集转换成list
        for i in range(len(train_data)):
            train_data[i] = list(train_data[i])
        self.train_data = train_data 
        self.batch_nums = len(train_data[0])
        self.test_data = test_data

    def train_KCIF_Grad(self,epochs):
        if self.LR==False: criterion_KCIF = [nn.CrossEntropyLoss() for i in range(self.nums)] 
        else: criterion_KCIF = [nn.BCELoss() for i in range(self.nums)]    # 计算梯度信息并传送
        optimizer_KCIF = [torch.optim.SGD(self.model_KCIF[i].parameters(), lr=self.param["lr"]) for i in range(self.nums)]

        ACC_KCIF = [[] for i in range(self.nums)]
        for epoch in range(epochs):  #迭代次数
            for batch_step in range(self.batch_nums):  #针对每个batch
                grads = []  #保留每个客户端的梯度
                for i in range(self.nums):  #针对每个客户端
                    (batch_x,batch_y) = self.train_data[i][batch_step]
                    y_pred_KCIF = self.model_KCIF[i](batch_x)
                    loss_KCIF = criterion_KCIF[i](y_pred_KCIF,batch_y)
                    optimizer_KCIF[i].zero_grad()
                    loss_KCIF.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_KCIF[i].parameters(), max_norm=self.param["clip"])
                    noise = {name: torch.normal(0, self.param["sigma"]*self.param["clip"], param.shape)/self.param["Lots"] for name, param in self.model_KCIF[i].named_parameters()}
                    grad_kcif = {}
                    for name,param in self.model_KCIF[i].named_parameters(): 
                        param.grad += noise[name]
                        param.grad = self.kcif_grad[i].KF_Fliter(copy.deepcopy(param.grad), name)        # 对DP+KF的方法进行过滤
                        param.grad *= (self.param["Ada"]+0.1)
                        grad_kcif[name] = param.grad

                    grads.append(copy.deepcopy(grad_kcif))

                # 计算聚合梯度
                for i in range(self.nums):  #对每个客户端进行聚合并梯度下降
                    grad_batch = copy.deepcopy(grads[i])  #聚合并平均该节点的梯度
                    for name in grad_batch:  #自身扩大
                        grad_batch[name] *= 1

                    cnt = np.sum(self.Graph[:,i])+1    #该节点的度
                    for j in range(self.nums):  # 列,累计梯度
                        if self.Graph[i][j] == 0 or i==j:
                            continue
                        for name in grads[j]:
                            grad_batch[name] += grads[j][name]

                    for name,param in self.model_KCIF[i].named_parameters():  #平均梯度
                        param.grad = grad_batch[name]/cnt
                    optimizer_KCIF[i].step()  #梯度下降

            # 聚合并分发参数
            Params_epoch = []
            for i in range(self.nums): 
                param_epoch = copy.deepcopy(self.model_KCIF[i].state_dict())
                for j in range(self.nums):
                    if self.Graph[i][j] == 0 or i==j:
                        continue
                    for name in self.model_KCIF[j].state_dict():
                        param_epoch[name] += self.model_KCIF[j].state_dict()[name]
                cnt = np.sum(self.Graph[:,i])+1 
                for name in param_epoch:
                    param_epoch[name] /= cnt
                Params_epoch.append(param_epoch)
            for i in range(self.nums):
                self.model_KCIF[i].load_state_dict(Params_epoch[i])

            for i in range(self.nums):
                acc = self.get_test_acc(self.model_KCIF[i])  #计算准确率
                ACC_KCIF[i].append(acc)
            print("KCIF-第{}个epoch，准确率为:{}".format(epoch,ACC_KCIF[0][epoch]))
        return ACC_KCIF

    def train_KCIF_Param(self,epochs):
        if self.LR==False: criterion_KCIF = [nn.CrossEntropyLoss() for i in range(self.nums)] 
        else: criterion_KCIF = [nn.BCELoss() for i in range(self.nums)]    # 计算梯度信息并传送
        optimizer_KCIF = [torch.optim.SGD(self.model_KCIF[i].parameters(), lr=self.param["lr"]) for i in range(self.nums)]
        ACC_KCIF = [[] for i in range(self.nums)]
        for epoch in range(epochs):  #迭代次数
            epoch_flag = True
            for batch_step in range(self.batch_nums):  #针对每个batch
                grads = []  #保留每个客户端的梯度
                for i in range(self.nums):  #针对每个客户端
                    (batch_x,batch_y) = self.train_data[i][batch_step]
                    y_pred_KCIF = self.model_KCIF[i](batch_x)
                    loss_KCIF = criterion_KCIF[i](y_pred_KCIF,batch_y)
                    optimizer_KCIF[i].zero_grad()
                    loss_KCIF.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_KCIF[i].parameters(), max_norm=self.param["clip"])
                    noise = {name: torch.normal(0, self.param["sigma"]*self.param["clip"], param.shape)/self.param["Lots"] for name, param in self.model_KCIF[i].named_parameters()}
                    for name,param in self.model_KCIF[i].named_parameters(): 
                        param.grad += noise[name]
                        self.Momentums[i][name] = self.Momentums[i][name] * (self.param["Momentum"]) + param.grad
                        param.grad = param.grad + self.Momentums[i][name]
                    optimizer_KCIF[i].step()  #梯度下降

                # 计算聚合参数
                for i in range(self.nums):  #对每个客户端进行聚合并梯度下降
                    param_batch = copy.deepcopy(self.model_KCIF[i].state_dict())  #聚合并平均该节点的梯度
                    cnt = np.sum(self.Graph[:,i])+1    #该节点的度
                    for j in range(self.nums):  # 列,累计梯度
                        if self.Graph[i][j] == 0 or i==j:
                            continue
                        for name in param_batch:
                            param_batch[name] += self.model_KCIF[j].state_dict()[name]

                    for name in param_batch:  #平均梯度
                        param_batch[name] /= cnt
                    self.model_KCIF[i].load_state_dict(copy.deepcopy(param_batch))

                    if epoch > self.param["threshold"]:
                        self.model_KCIF[i].load_state_dict( self.kcif_param[i].KF_Fliter( copy.deepcopy(self.model_KCIF[i].state_dict()), epoch_flag) )  #使用KF_Param对模型参数进行过滤
                        epoch_flag = False

            # 聚合并分发Momentum
            Momentum_epoch = []
            for i in range(self.nums): 
                momentum_epoch = copy.deepcopy(self.Momentums[i])
                for j in range(self.nums):
                    if self.Graph[i][j] == 0 or i==j:
                        continue
                    for name in self.model_KCIF[j].state_dict():
                        momentum_epoch[name] += self.Momentums[j][name]
                cnt = np.sum(self.Graph[:,i])+1 
                for name in momentum_epoch:
                    momentum_epoch[name] /= cnt

                Momentum_epoch.append(momentum_epoch)
            for i in range(self.nums):
                self.Momentums[i] = copy.deepcopy(Momentum_epoch[i])

            for i in range(self.nums):
                acc = self.get_test_acc(self.model_KCIF[i])  #计算准确率
                ACC_KCIF[i].append(acc)
            print("KCIF-第{}个epoch，准确率为:{}".format(epoch,ACC_KCIF[0][epoch]))
        return ACC_KCIF

    def train_KCIF2(self,epochs):
        if self.LR==False: criterion_KCIF = [nn.CrossEntropyLoss() for i in range(self.nums)] 
        else: criterion_KCIF = [nn.BCELoss() for i in range(self.nums)]    # 计算梯度信息并传送
        optimizer_KCIF = [torch.optim.SGD(self.model_KCIF[i].parameters(), lr=self.param["lr"]) for i in range(self.nums)]
        ACC_KCIF = [[] for i in range(self.nums)]
        for epoch in range(epochs):  #迭代次数
            epoch_flag = True
            for batch_step in range(self.batch_nums):  #针对每个batch
                grads = []  #保留每个客户端的梯度
                for i in range(self.nums):  #针对每个客户端
                    (batch_x,batch_y) = self.train_data[i][batch_step]
                    y_pred_KCIF = self.model_KCIF[i](batch_x)
                    loss_KCIF = criterion_KCIF[i](y_pred_KCIF,batch_y)
                    optimizer_KCIF[i].zero_grad()
                    loss_KCIF.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_KCIF[i].parameters(), max_norm=self.param["clip"])
                    noise = {name: torch.normal(0, self.param["sigma"]*self.param["clip"], param.shape)/self.param["Lots"] for name, param in self.model_KCIF[i].named_parameters()}
                    grad_kcif = {}
                    for name,param in self.model_KCIF[i].named_parameters(): 
                        param.grad += noise[name]
                        param.grad = self.kcif_grad[i].KF_Fliter(copy.deepcopy(param.grad), name)        # 对DP+KF的方法进行过滤
                        param.grad *= (self.param["Ada"]+0.1)

                        self.Momentums[i][name] = self.Momentums[i][name] * (self.param["Momentum"]) + param.grad
                        param.grad = param.grad + self.Momentums[i][name]
                        grad_kcif[name] = param.grad

                    grads.append(copy.deepcopy(grad_kcif))
                
                # 计算聚合梯度
                for i in range(self.nums):  #对每个客户端进行聚合并梯度下降
                    grad_batch = copy.deepcopy(grads[i])  #聚合并平均该节点的梯度
                    for name in grad_batch:  #自身扩大
                        grad_batch[name] *= 1

                    cnt = np.sum(self.Graph[:,i])+1    #该节点的度
                    for j in range(self.nums):  # 列,累计梯度
                        if self.Graph[i][j] == 0 or i==j:
                            continue
                        for name in grads[j]:
                            grad_batch[name] += grads[j][name]

                    for name,param in self.model_KCIF[i].named_parameters():  #平均梯度
                        param.grad = grad_batch[name]/cnt
                    optimizer_KCIF[i].step()  #梯度下降


                # 计算聚合参数+梯度
                for i in range(self.nums):  #对每个客户端进行聚合并梯度下降
                    param_batch = copy.deepcopy(self.model_KCIF[i].state_dict())  #聚合并平均该节点的梯度
                    cnt = np.sum(self.Graph[:,i])+1    #该节点的度
                    for j in range(self.nums):  # 列,累计梯度
                        if self.Graph[i][j] == 0 or i==j:
                            continue
                        for name in param_batch:
                            param_batch[name] += self.model_KCIF[j].state_dict()[name]

                    for name in param_batch:  #平均梯度
                        param_batch[name] /= cnt
                    self.model_KCIF[i].load_state_dict(copy.deepcopy(param_batch))

                    if epoch > self.param["threshold"]:
                        self.model_KCIF[i].load_state_dict( self.kcif_param[i].KF_Fliter( copy.deepcopy(self.model_KCIF[i].state_dict()), epoch_flag) )  #使用KF_Param对模型参数进行过滤
                        epoch_flag = False

            for i in range(self.nums):
                acc = self.get_test_acc(self.model_KCIF[i])  #计算准确率
                ACC_KCIF[i].append(acc)
            print("KCIF-第{}个epoch，准确率为:{}".format(epoch,ACC_KCIF[0][epoch]))
        return ACC_KCIF


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

    #KFParam的作用，与PID的对比，Momentum VS KF_Param
    def DPSGD_Momentum_KFParam(self,flag):   #三种方法
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


def train_kfgrad(Clients, Graph, epoch):
    nums = len(Graph)
    ACC, ACC_KF = [[] for i in range(nums)], [[] for i in range(nums)]
    for e in range(epoch): 
        for client in Clients:  #每个模型进行单独更新
            client.DPSGD_KFGrad_update()
        # 计算聚合
        for i in range(nums):  #对每个客户端进行聚合
            base1, base2 = Clients[i].model.state_dict(), Clients[i].model_KF.state_dict()
            for j in range(nums):  # 聚合参数
                if Graph[i][j] == 0 or i==j:
                    continue
                for name in base1:
                    base1[name] += Clients[j].model.state_dict()[name] 
                    base2[name] += Clients[j].model_KF.state_dict()[name] 
                
            cnt = np.sum(Graph[:,i])+1    #该节点的度
            for name in base1:  #平均
                base1[name] /= cnt
                base2[name] /= cnt

            Clients[i].model.load_state_dict(copy.deepcopy(base1))   #分发
            Clients[i].model_KF.load_state_dict(copy.deepcopy(base2))  #分发
            acc, acc_KF = Clients[i].get_test_acc(Clients[i].model), Clients[i].get_test_acc(Clients[i].model_KF)
            ACC[i].append(acc), ACC_KF[i].append(acc_KF)
            
        print("Bseline-第",e,"个epoch，DPSGD的准确率:", acc,"KF-DPSGD的准确率:",acc_KF )
    return ACC, ACC_KF


def train_kfparam(Clients, Graph, epoch):
    nums = len(Graph)
    ACC, ACC_KF = [[] for i in range(nums)], [[] for i in range(nums)]

    for e in range(epoch): 
        if e > Clients[0].param["threshold"]:
            flag = True
        else:
            flag = False 
        for client in Clients:  #每个模型进行单独更新
            client.DPSGD_Momentum_KFParam(flag)
        # 计算聚合
        for i in range(nums):  #对每个客户端进行聚合
            base1, base2 = Clients[i].model.state_dict(), Clients[i].model_KF.state_dict()
            Momentum1, Momentum2 = Clients[i].Momentum, Clients[i].Momentum_KF
            for j in range(nums):  # 聚合参数
                if Graph[i][j] == 0 or i==j:
                    continue
                for name in base1:
                    base1[name] += Clients[j].model.state_dict()[name] 
                    base2[name] += Clients[j].model_KF.state_dict()[name]
                    Momentum1[name] += Clients[j].Momentum[name]
                    Momentum2[name] += Clients[j].Momentum_KF[name] 
                
            cnt = np.sum(Graph[:,i])+1    #该节点的度
            for name in base1:  #平均
                base1[name] /= cnt
                base2[name] /= cnt
                Momentum1[name] /= cnt
                Momentum2[name] /= cnt

            Clients[i].model.load_state_dict(copy.deepcopy(base1))   #分发
            Clients[i].model_KF.load_state_dict(copy.deepcopy(base2))  #分发
            Clients[i].Momentum = copy.deepcopy(Momentum1)
            Clients[i].Momentum_KF = copy.deepcopy(Momentum2)

            acc, acc_KF = Clients[i].get_test_acc(Clients[i].model), Clients[i].get_test_acc(Clients[i].model_KF)
            ACC[i].append(acc), ACC_KF[i].append(acc_KF)

        print("Bseline-第",e,"个epoch，DPSGD的准确率:", acc,"KF-DPSGD的准确率:",acc_KF )
    return ACC, ACC_KF


def train_kf2(Clients, Graph, epoch):
    nums = len(Graph)
    ACC, ACC_KF = [[] for i in range(nums)], [[] for i in range(nums)]

    for e in range(epoch): 
        if e > Clients[0].param["threshold"]:
            flag = True
        else:
            flag = False 
        for client in Clients:  #每个模型进行单独更新
            client.DPSGD_Momentum_KFGrad_KFParam(flag)
        # 计算聚合
        for i in range(nums):  #对每个客户端进行聚合
            base1, base2 = Clients[i].model.state_dict(), Clients[i].model_KF.state_dict()
            Momentum1, Momentum2 = Clients[i].Momentum, Clients[i].Momentum_KF
            for j in range(nums):  # 聚合参数
                if Graph[i][j] == 0 or i==j:
                    continue
                for name in base1:
                    base1[name] += Clients[j].model.state_dict()[name] 
                    base2[name] += Clients[j].model_KF.state_dict()[name]
                    Momentum1[name] += Clients[j].Momentum[name]
                    Momentum2[name] += Clients[j].Momentum_KF[name] 
                
            cnt = np.sum(Graph[:,i])+1    #该节点的度
            for name in base1:  #平均
                base1[name] /= cnt
                base2[name] /= cnt
                Momentum1[name] /= cnt
                Momentum2[name] /= cnt

            Clients[i].model.load_state_dict(copy.deepcopy(base1))   #分发
            Clients[i].model_KF.load_state_dict(copy.deepcopy(base2))  #分发
            Clients[i].Momentum = copy.deepcopy(Momentum1)
            Clients[i].Momentum_KF = copy.deepcopy(Momentum2)

            acc, acc_KF = Clients[i].get_test_acc(Clients[i].model), Clients[i].get_test_acc(Clients[i].model_KF)
            ACC[i].append(acc), ACC_KF[i].append(acc_KF)

        print("Bseline-第",e,"个epoch，DPSGD的准确率:", acc,"KF-DPSGD的准确率:",acc_KF )
    return ACC, ACC_KF


def draw(acc, acc_kf, acc_kcif, model,sparity, y1="Decel", y2="KF-Grad", y3="KCIF-Grad", res= "KF_Grad"):
    ACC, ACC_KF, ACC_KCIF = np.mean(np.array(acc),axis=0), np.mean(np.array(acc_kf),axis=0),np.mean(np.array(acc_kcif),axis=0)
    #画图
    if type(model) == LogisticRegression:
        m = "LR_Epl"
    elif type(model) == MLP:
        m = "MLP_MINIST"
    elif type(model) == MnistCNN:
        m = "CNN_MINIST"
    elif type(model) == CIAFARCNN:
        m = "CNN_CIAFAR10"
    else:
        print("model name exist error!")
    
    X = [i for i in range(0, len(ACC))]
    plt.xlabel("epoch"), plt.ylabel("Accuracy")

    plt.plot(X, ACC, label=y1, linestyle='--', markersize=6, alpha=1, linewidth=2)
    plt.plot(X, ACC_KF, label=y2, linestyle=':', markersize=6, alpha=1, linewidth=2)
    plt.plot(X, ACC_KCIF, label=y3, linestyle='-', markersize=6, alpha=1, linewidth=2)
    plt.grid(True)  # 打开网格
    # 添加标题和标签
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(loc='lower right')
    base_path = "./picture_decel/"+res+"结果/" 
    path = base_path + m +str(sparity)+ ".png"
    plt.savefig(path)
    plt.close()

    data = {"Decel":ACC,"KFGrad":ACC_KF, "KCIFGrad":ACC_KCIF}
    df = pd.DataFrame(data)
    df.to_csv(base_path+m+str(sparity)+".csv")
    return 

def Gen_Graph(numOfNodes=100, sparity=0.2, Empty = False):  # 生成连通图, sparity为稀疏度，是在最基本的连通图的基础上对0进行取1的概率
    num = 2 * numOfNodes
    Graph = np.zeros((numOfNodes, numOfNodes))  # 生成全0矩阵
    if Empty:
        return Graph,0
    Graph[0][numOfNodes - 1] = 1
    Graph[numOfNodes - 1][0] = 1
    for i in range(numOfNodes - 1):
        Graph[i][i + 1] = 1
        Graph[i + 1][i] = 1
    for i in range(numOfNodes):
        for j in range(i+1,numOfNodes):
            if Graph[i][j] == 1:
                continue
            else:
                if (np.random.rand(1) > 1 - sparity):
                    Graph[i][j] = 1
                    Graph[j][i] = 1
                    num += 2
    return Graph ,num/numOfNodes

# 客户端的参数配置，本地迭代次数为1
param = {"lr":0.001,"clip":1,"sigma": 5,"epochs":1,"Lots":50,
        "Ada":1.5,        #  KFGrad方法的参数
        "Momentum":0.4, "threshold":50,"Q_decent": 0.95, #KFParam方法的参数
        "PID_D":0.004}    #PID方法的参数

# 联邦学习参数配置
numOfNodes = 15  # 联邦学习参与方数量
epochs = 70
 
Data_Model = [
              [load_eplision_torch(batch_size=param["Lots"], nums=numOfNodes),LogisticRegression()],
               [load_mnist(batch_size=param["Lots"], nums=numOfNodes,model = "MLP"),  MLP()],
               [load_mnist(batch_size=param["Lots"], nums=numOfNodes,model = "CNN"),  MnistCNN()],
               [load_CIFAR10(batch_size=param["Lots"], nums=numOfNodes) ,CIAFARCNN()]
            ]

#############################  测试KCIF_Grad方法  ############################################
# for sparity in [ 0.3 ]:
#     Graph, degree = Gen_Graph(numOfNodes,sparity)
#     for dm in Data_Model:
#         data, model = dm[0], dm[1]
#         #初始化卡尔曼滤波器
#         KF_Grad = KF_grad(Q=10,R=10,P=10, Net_Piror=copy.deepcopy(model))
#         KCIF_Grad = KF_grad(Q=10,R=10,P=10, Net_Piror=copy.deepcopy(model))
#         KF_Param = KF_param(Q=10 ,R=1 ,P=10, alfa=param["Q_decent"] , Net_Piror =copy.deepcopy(model))
#         KCIF_Param = KF_param(Q=10 ,R=1 ,P=10, alfa=param["Q_decent"] , Net_Piror =copy.deepcopy(model))

#         # KCIF方法  param, Graph, model, kcif_grad, kcif_param, train_data, test_data
#         client_kcif = Client(param=param, Graph=Graph, model=copy.deepcopy(model), kcif_grad=copy.deepcopy(KCIF_Grad),kcif_param= copy.deepcopy(KCIF_Param), train_data= copy.deepcopy(data[0]), test_data= copy.deepcopy(data[1]))
#         ACC_KCIF = client_kcif.train_KCIF_Grad(epochs)
#         print("KCIF方法结束")

#         #Baseline和KF方法         
#         Clients = []
#         for i in range(numOfNodes):    #初始化各个参与方的模型参数
#             Clients.append( KFDPSGD(param = param, data = data[0][i], test_data = copy.deepcopy(data[1]), model = copy.deepcopy(model),  KF_Grad=copy.deepcopy(KF_Grad), KF_Param=copy.deepcopy(KF_Param) ) )
#         ACC,ACC_KF=train_kfgrad(Clients, Graph, epochs)
#         print("Baseline方法结束")

#         #acc, acc_kf, acc_kcif, model, y1="Decel", y2="KF-Grad", y3="KCIF-Grad", res= "KF_Grad"
#         draw(ACC,ACC_KF,ACC_KCIF,model, sparity,y1="Decel", y2="KF-Grad", y3="KCIF-Grad", res= "KF_Grad")


#############################  测试KCIF_Param方法  ############################################
for sparity in [ 0.3 ]:
    Graph, degree = Gen_Graph(numOfNodes,sparity)
    for dm in Data_Model:
        data, model = dm[0], dm[1]
        #初始化卡尔曼滤波器
        KF_Grad = KF_grad(Q=10,R=10,P=10, Net_Piror=copy.deepcopy(model))
        KCIF_Grad = KF_grad(Q=10,R=10,P=10, Net_Piror=copy.deepcopy(model))
        KF_Param = KF_param(Q=10 ,R=1 ,P=10, alfa=param["Q_decent"] , Net_Piror =copy.deepcopy(model))
        KCIF_Param = KF_param(Q=10 ,R=1 ,P=10, alfa=param["Q_decent"] , Net_Piror =copy.deepcopy(model))

        # KCIF方法  param, Graph, model, kcif_grad, kcif_param, train_data, test_data
        client_kcif = Client(param=param, Graph=Graph, model=copy.deepcopy(model), kcif_grad=copy.deepcopy(KCIF_Grad),kcif_param= copy.deepcopy(KCIF_Param), train_data= copy.deepcopy(data[0]), test_data= copy.deepcopy(data[1]))
        ACC_KCIF = client_kcif.train_KCIF_Param(epochs)
        print("KCIF方法结束")

        #Baseline和KF方法         
        Clients = []
        for i in range(numOfNodes):    #初始化各个参与方的模型参数
            Clients.append( KFDPSGD(param = param, data = data[0][i], test_data = copy.deepcopy(data[1]), model = copy.deepcopy(model),  KF_Grad=copy.deepcopy(KF_Grad), KF_Param=copy.deepcopy(KF_Param) ) )
        ACC,ACC_KF=train_kfparam(Clients, Graph, epochs)
        print("Baseline方法结束")

        #acc, acc_kf, acc_kcif, model, y1="Decel", y2="KF-Grad", y3="KCIF-Grad", res= "KF_Grad"
        draw(ACC,ACC_KF,ACC_KCIF,model, sparity,y1="Decel", y2="KF-Param", y3="KCIF-Param", res= "KF_Param")


#############################  测试KCIF2方法  ############################################
# for sparity in [0.1, 0.3, 0.5, 0.7 ]:
#     Graph, degree = Gen_Graph(numOfNodes,sparity)
#     for dm in Data_Model:
#         data, model = dm[0], dm[1]
#         #初始化卡尔曼滤波器
#         KF_Grad = KF_grad(Q=10,R=10,P=10, Net_Piror=copy.deepcopy(model))
#         KCIF_Grad = KF_grad(Q=10,R=10,P=10, Net_Piror=copy.deepcopy(model))
#         KF_Param = KF_param(Q=10 ,R=1 ,P=10, alfa=param["Q_decent"] , Net_Piror =copy.deepcopy(model))
#         KCIF_Param = KF_param(Q=10 ,R=1 ,P=10, alfa=param["Q_decent"] , Net_Piror =copy.deepcopy(model))

#         # KCIF方法  param, Graph, model, kcif_grad, kcif_param, train_data, test_data
#         client_kcif = Client(param=param, Graph=Graph, model=copy.deepcopy(model), kcif_grad=copy.deepcopy(KCIF_Grad),kcif_param= copy.deepcopy(KCIF_Param), train_data= copy.deepcopy(data[0]), test_data= copy.deepcopy(data[1]))
#         ACC_KCIF = client_kcif.train_KCIF2(epochs)
#         print("KCIF方法结束")

#         #Baseline和KF方法         
#         Clients = []
#         for i in range(numOfNodes):    #初始化各个参与方的模型参数
#             Clients.append( KFDPSGD(param = param, data = data[0][i], test_data = copy.deepcopy(data[1]), model = copy.deepcopy(model),  KF_Grad=copy.deepcopy(KF_Grad), KF_Param=copy.deepcopy(KF_Param) ) )
#         ACC,ACC_KF=train_kf2(Clients, Graph, epochs)
#         print("Baseline方法结束")

#         #acc, acc_kf, acc_kcif, model, y1="Decel", y2="KF-Grad", y3="KCIF-Grad", res= "KF_Grad"
#         draw(ACC,ACC_KF,ACC_KCIF,model, sparity,y1="Decel", y2="KF2", y3="KCIF2", res= "KF2")