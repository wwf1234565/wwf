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
    def __init__(self ,param):
        self.param = param    #{"lr":0.01,"clip":20,"sigma":2,"epochs":epochs,"Ada":3,"Momentum":0.9,"PID_I":1,"PID_D":0}

    #测试加KF前后梯度变化
    def KF_Grad_compare(self, model, data, KF_Grad ):
        LR = False
        if type(model) == LogisticRegression:
            LR = True
        model_KF, model_GD = copy.deepcopy(model), copy.deepcopy(model)
        train_data ,train_label ,test_data ,test_label = data[0], data[1], data[2], data[3] 
        if LR == False:
            criterion, criterion_KF, criterion_GD = nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss()
            train_label = train_label.long()
        else:
            criterion, criterion_KF, criterion_GD = nn.BCELoss(),nn.BCELoss(),nn.BCELoss()
        optimizer, optimizer_KF, optimizer_GD = torch.optim.SGD(model.parameters(), lr=self.param["lr"]),torch.optim.SGD(model_KF.parameters(), lr=self.param["lr"]),torch.optim.SGD(model_GD.parameters(), lr=self.param["lr"]) 
        bound = {} 
        ACC, ACC_KF, ACC_GD = [], [], []
        Grad, Grad_KF, Grad_GD = {}, {}, {}
        for name,param in model.named_parameters():
            Grad[name] = []
            Grad_KF[name] = []
            Grad_GD[name] = []

        for epoch in range(self.param["epochs"]):
            idx = np.where(np.random.rand(train_data.size(0)) < 0.5)[0]
            batch_x, batch_y = train_data[idx], train_label[idx] 

            y_pred_GD = model_GD(train_data)
            loss_GD = criterion_GD(y_pred_GD, train_label)

            y_pred, y_pred_KF = model(batch_x), model_KF(batch_x)
            loss, loss_KF = criterion(y_pred, batch_y), criterion_KF(y_pred_KF,batch_y)

            optimizer.zero_grad(),optimizer_KF.zero_grad(),optimizer_GD.zero_grad()
            loss.backward(), loss_KF.backward(), loss_GD.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.param["clip"])
            #torch.nn.utils.clip_grad_norm_(model_KF.parameters(), max_norm=self.param["clip"])
            
            if epoch == 0:   #记录梯度最大值的索引
                for name,param in model_GD.named_parameters(): 
                    if name[-4:] != "bias":
                        index = (param.grad == torch.max(param.grad)).nonzero(as_tuple=True)
                        print("maxgrad:",torch.max(param.grad))
                        if len(index) == 2:
                            bound[name] = [index[0].item(),index[1].item()]
                        elif len(index) == 4:
                            bound[name] = [index[0].item(),index[1].item(),index[2].item(),index[3].item()]
                        else:
                            print("Index exist error")
                print("index:", bound)
            
            noise = {name: torch.normal(0, self.param["sigma"], param.shape) for name, param in model.named_parameters()}

            for name,param in model.named_parameters():     #DPSGD梯度
                param.grad += noise[name]
                if name[-4:] != "bias":
                    if len(bound[name]) == 2:
                        Grad[name].append(param.grad[bound[name][0]][bound[name][1]].item())
                    else:
                        Grad[name].append(param.grad[bound[name][0]][bound[name][1]][bound[name][2]][bound[name][3]].item())
            
            for name,param in model_KF.named_parameters():  #KFDPSGD梯度
                param.grad += noise[name]
                k, param.grad = KF_Grad.KF_Fliter(copy.deepcopy(param.grad), name)        # 对DP+KF的方法进行过滤
                if name[-4:] != "bias":
                    if len(bound[name]) == 2:
                        Grad_KF[name].append(param.grad[bound[name][0]][bound[name][1]].item())
                    else:
                        Grad_KF[name].append(param.grad[bound[name][0]][bound[name][1]][bound[name][2]][bound[name][3]].item())
            print("!!!!!!!epoch {}:".format(epoch),k)
            for name,param in model_GD.named_parameters(): #GD梯度
                if name[-4:] != "bias":
                    if len(bound[name]) == 2:
                        Grad_GD[name].append(param.grad[bound[name][0]][bound[name][1]].item())
                    else:
                        Grad_GD[name].append(param.grad[bound[name][0]][bound[name][1]][bound[name][2]][bound[name][3]].item())

            optimizer.step(),optimizer_KF.step(), optimizer_GD.step()
            #测试
            y_pred_test, y_pred_KF_test, y_pred_GD_test = model(test_data), model_KF(test_data), model_GD(test_data)
            if LR== False:
                _,mask = torch.max(y_pred_test,1)
                _,mask_KF = torch.max(y_pred_KF_test,1)
                _,mask_GD = torch.max(y_pred_GD_test, 1)
            else:
                mask,mask_KF,mask_GD = y_pred_test.ge(0.5).float(), y_pred_KF_test.ge(0.5).float(), y_pred_GD_test.ge(0.5).float()

            correct, correct_KF, correct_GD = (mask == test_label).sum(), (mask_KF == test_label).sum(), (mask_GD == test_label).sum()

            acc = correct.item() / test_label.size(0)
            acc_KF = correct_KF.item() / test_label.size(0)  
            acc_GD = correct_GD.item() / test_label.size(0)
            ACC.append(acc), ACC_KF.append(acc_KF), ACC_GD.append(acc_GD)
            print(epoch,"次迭代,DPSGD的Loss:",loss.item(),"KF-DPSGD的Loss:",loss_KF.item(),"GD的Loss:",loss_GD.item())

        #画图
        X = [i for i in range(0,self.param["epochs"])]
        plt.xlabel("epoch"), plt.ylabel("Accuracy")
        plt.plot(X,ACC,label="DPSGD"), plt.plot(X,ACC_KF,label="DPSGD_KFGrad"), plt.plot(X,ACC_GD,label="GD")
        plt.legend()
        plt.savefig("./picture/GD_DPSGD_KFGrad.png")
        plt.close()
        plt.xlabel("epoch"), plt.ylabel("Grad")

        df = pd.DataFrame()
        for name,param in model.named_parameters():
            if name[-4:] != "bias":
                df[name+"_BGD"], df[name+"_DPSGD"], df[name+"_DPSGD_KF"] = Grad_GD[name], Grad[name], Grad_KF[name]
                plt.plot(X, Grad[name],label="DPSGD"), plt.plot(X, Grad_KF[name],label="DPSGD-KF"), plt.plot(X, Grad_GD[name],label="BGD")
                plt.title(name+"'s Grad")
                plt.legend()
                plt.savefig("./picture/"+name +".png")
                plt.close()

        df.to_csv("LR_Grad.csv")
        return

    #测试DPSGD-KFGrad
    def DPSGD_KFGrad_update(self, model, data, KF_Grad):
        if type(model) == LogisticRegression:
            LR = True
        else:
            LR= False
         
        train_dl, test_dl = data[0], data[1]
        model_KF = copy.deepcopy(model)
        if LR==False:
            criterion, criterion_KF = nn.CrossEntropyLoss(),nn.CrossEntropyLoss()
        else:
            criterion, criterion_KF = nn.BCELoss(),nn.BCELoss()
        
        optimizer, optimizer_KF = torch.optim.SGD(model.parameters(), lr=self.param["lr"]),torch.optim.SGD(model_KF.parameters(), lr=self.param["lr"])
        ACC, ACC_KF = [], []
        for epoch in range(self.param["epochs"]):
            for (batch_x,batch_y) in train_dl:
                y_pred, y_pred_KF = model(batch_x), model_KF(batch_x)
                loss, loss_KF = criterion(y_pred, batch_y), criterion_KF(y_pred_KF,batch_y)
                optimizer.zero_grad(),optimizer_KF.zero_grad()
                loss.backward(),loss_KF.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.param["clip"])
                torch.nn.utils.clip_grad_norm_(model_KF.parameters(), max_norm=self.param["clip"])
                noise = {name: torch.normal(0, self.param["sigma"]*self.param["clip"], param.shape)/self.param["Lots"] for name, param in model.named_parameters()}

                for name,param in model.named_parameters(): 
                    param.grad += noise[name]
    
                for name,param in model_KF.named_parameters(): 
                    param.grad += noise[name]
                    param.grad = KF_Grad.KF_Fliter(copy.deepcopy(param.grad), name)        # 对DP+KF的方法进行过滤
                    param.grad *= self.param["Ada"]
                optimizer.step(),optimizer_KF.step()
            #测试
            for batch_x,test_label in test_dl:
                y_pred_test, y_pred_KF_test = model(batch_x), model_KF(batch_x)
            if LR== False:
                _,mask = torch.max(y_pred_test,1)
                _,mask_KF = torch.max(y_pred_KF_test,1)
            else:
                mask,mask_KF = y_pred_test.ge(0.5).float(), y_pred_KF_test.ge(0.5).float() 
            correct,correct_KF = (mask == test_label).sum(), (mask_KF == test_label).sum()
            acc = correct.item() / test_label.size(0)
            acc_KF = correct_KF.item() / test_label.size(0)  
            ACC.append(acc), ACC_KF.append(acc_KF) 
            print(epoch,"次迭代,DPSGD的Loss:",loss.item(),"KF-DPSGD的Loss:",loss_KF.item())
        #画图
        if type(model) == LogisticRegression:
            m = "LR_Epl_"
        elif type(model) == MLP:
            m = "MLP_MINIST_"
        elif type(model) == MnistCNN:
            m = "CNN_MINIST_"
        elif type(model) == CIAFARCNN:
            m = "CNN_CIAFAR10_"
        else:
            print("model name exist error!")
        
        X = [i for i in range(0,self.param["epochs"])]
        plt.xlabel("epoch"), plt.ylabel("Accuracy")
        plt.plot(X,ACC,label="DPSGD"), plt.plot(X,ACC_KF,label="KF_Grad")
        plt.legend()
        path = "./picture/KF_Grad结果/" + m + "KF_Grad.png"
        print(path)
        plt.savefig(path)
        plt.close()

        file = open("./picture/KF_Grad结果/record",'a')
        file.write(path)
        file.write("ACC:"+str(ACC)+"\n" )
        file.write("ACC_KF:"+str(ACC_KF))
        file.write('\n')

        data = {"ACC":ACC,"ACC_KF":ACC_KF}
        df = pd.DataFrame(data)
        df.to_csv("Res.csv")
        return
   
    #KFParam的作用，与PID的对比，Momentum VS KF_Param VS PID
    def DPSGD_Momentum_KFParam(self, model, data, KF_Param):   #三种方法
        if type(model) == LogisticRegression:
            LR = True
        else:
            LR = False 
        model_KF = copy.deepcopy(model)
        model_PID = copy.deepcopy(model)

        PID_Pre_Grad = {name: torch.zeros_like(param) for name, param in model.named_parameters()} # 记录上一次的梯度
        Momentum = {name: torch.zeros_like(param) for name, param in model.named_parameters()}  #记录Momentum动量
        D_Grad = {name: torch.zeros_like(param) for name, param in model.named_parameters()} # 记录PID的D项

        train_dl, test_dl = data[0], data[1]

        if LR==False:
            criterion, criterion_KF, criterion_PID = nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss()
        else :
            criterion, criterion_KF, criterion_PID = nn.BCELoss(),nn.BCELoss(),nn.BCELoss()

        optimizer =  torch.optim.SGD(model.parameters(), lr=self.param["lr"],momentum = self.param['Momentum'])
        optimizer_KF = torch.optim.SGD(model_KF.parameters(), lr=self.param["lr"],momentum = self.param['Momentum'])
        optimizer_PID =  torch.optim.SGD(model_PID.parameters(), lr=self.param["lr"])

        ACC, ACC_KF, ACC_PID = [], [], []
        for epoch in range(self.param["epochs"]):
            flag = True
            for (batch_x,batch_y) in train_dl:
                y_pred, y_pred_KF, y_pred_PID = model(batch_x), model_KF(batch_x), model_PID(batch_x)
                loss, loss_KF, loss_PID = criterion(y_pred, batch_y), criterion_KF(y_pred_KF,batch_y), criterion_PID(y_pred_PID,batch_y)
                optimizer.zero_grad(),optimizer_KF.zero_grad(), optimizer_PID.zero_grad()
                loss.backward(),loss_KF.backward(),loss_PID.backward()

                #梯度裁剪
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.param["clip"]),torch.nn.utils.clip_grad_norm_(model_KF.parameters(), max_norm=self.param["clip"]),torch.nn.utils.clip_grad_norm_(model_PID.parameters(), max_norm=self.param["clip"])

                noise = {name: torch.normal(0, self.param["sigma"]*self.param["clip"], param.shape)/self.param["Lots"] for name, param in model.named_parameters()}                
                deta_PID = {} #用于记录PID方法的D项

                #加噪
                for name,param in model.named_parameters():  param.grad += noise[name]
                for name,param in model_KF.named_parameters():  param.grad += noise[name]
                for name,param in model_PID.named_parameters():  
                    param.grad += noise[name]
                    Momentum[name] = Momentum[name] * self.param["Momentum"] + param.grad  # Momentum
                    D_Grad[name] = self.param['Momentum']*D_Grad[name] +  (1- self.param['Momentum'])*(param.grad - PID_Pre_Grad[name])
                    PID_Pre_Grad[name] = param.grad*1
                    param.grad = param.grad + Momentum[name] + self.param["PID_D"]*D_Grad[name] 

                optimizer.step(),optimizer_KF.step(),optimizer_PID.step()
                if epoch > self.param["threshold"]:
                    model_KF.load_state_dict(KF_Param.KF_Fliter(copy.deepcopy(model_KF.state_dict()), flag))  #使用KF_Param对模型参数进行过滤
                    flag=False

            #测试
            for test_data,test_label in test_dl:
                y_pred_test, y_pred_KF_test,y_pred_PID_test = model(test_data),model_KF(test_data),model_PID(test_data)
            if LR== False:
                _,mask = torch.max(y_pred_test,1)
                _,mask_KF = torch.max(y_pred_KF_test,1)
                _,mask_PID = torch.max(y_pred_PID_test,1)
            else:
                mask,mask_KF,mask_PID = y_pred_test.ge(0.5).float(), y_pred_KF_test.ge(0.5).float() , y_pred_PID_test.ge(0.5).float() 
            
            correct,correct_KF,correct_PID = (mask == test_label).sum(), (mask_KF == test_label).sum(), (mask_PID == test_label).sum()
            acc,acc_KF,acc_PID = correct.item() / test_label.size(0), correct_KF.item() / test_label.size(0) , correct_PID.item() / test_label.size(0) 
            ACC.append(acc), ACC_KF.append(acc_KF) , ACC_PID.append(acc_PID) 
            print(epoch,"次迭代,DPSGD的Loss:",loss.item(),"KF-DPSGD的Loss:",loss_KF.item(),"PID-DPSGD的Loss:",loss_PID.item())

        if type(model) == LogisticRegression:
            m = "LR_Epl_"
        elif type(model) == MLP:
            m = "MLP_MINIST_"
        elif type(model) == MnistCNN:
            m = "CNN_MINIST_"
        elif type(model) == CIAFARCNN:
            m = "CNN_CIAFAR10_"
        else:
            print("model name exist error!")

        #画图
        X = [i for i in range(0,self.param["epochs"])]
        
        plt.xlabel("epoch"), plt.ylabel("Accuracy")
        plt.plot(X,ACC,label="DPSGD-M"), plt.plot(X,ACC_KF,label="DPSGD-M-KFParam"), plt.plot(X,ACC_PID,label="DPSGD-M-PID")
        plt.legend()
        path = "./picture/KF_Param结果/" + m + "KFParam.png"
        plt.savefig(path)
        plt.close()
        data = {"ACC":ACC,"DPSGD-M-KFParam":ACC_KF,"DPSGD-M-PID": ACC_PID}
        df = pd.DataFrame(data)
        df.to_csv(m+"KFM_Res.csv")
        return

    #KF2算法,即KFGrad+KFParam的作用，与Momentum的对比
    def DPSGD_Momentum_KFGrad_KFParam(self, model, data, KF_Param,KF_Grad):   #两种方法
        if type(model) == LogisticRegression:
            LR = True
        else:
            LR = False 
        model_KF = copy.deepcopy(model)

        Momentum = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        train_dl, test_dl = data[0], data[1]
        if LR==False:
            criterion, criterion_KF, criterion_PID = nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss()
        else:
            criterion, criterion_KF, criterion_PID = nn.BCELoss(),nn.BCELoss(),nn.BCELoss()

        optimizer =  torch.optim.SGD(model.parameters(), lr=self.param["lr"] )
        optimizer_KF = torch.optim.SGD(model_KF.parameters(), lr=self.param["lr"],momentum=self.param['Momentum'])
        ACC, ACC_KF = [], [] 
        for epoch in range(self.param["epochs"]):
            flag = True
            for (batch_x,batch_y) in train_dl:
                y_pred, y_pred_KF = model(batch_x), model_KF(batch_x) 
                loss, loss_KF = criterion(y_pred, batch_y), criterion_KF(y_pred_KF,batch_y) 
                optimizer.zero_grad(),optimizer_KF.zero_grad() 
                loss.backward(),loss_KF.backward() 

                #梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.param["clip"]),torch.nn.utils.clip_grad_norm_(model_KF.parameters(), max_norm=self.param["clip"]) 
                noise = {name: torch.normal(0, self.param["sigma"]*self.param["clip"], param.shape)/self.param["Lots"] for name, param in model.named_parameters()}
                #加噪
                for name,param in model.named_parameters():  param.grad += noise[name]
                for name,param in model_KF.named_parameters():
                    param.grad += noise[name]
                    param.grad = KF_Grad.KF_Fliter(copy.deepcopy(param.grad), name)        # 对DP+KF的方法进行过滤
                    param.grad *= self.param["Ada"]
                optimizer.step(),optimizer_KF.step() 
                if epoch > self.param["threshold"]:
                    model_KF.load_state_dict(KF_Param.KF_Fliter(copy.deepcopy(model_KF.state_dict()), flag))  #使用KF_Param对模型参数进行过滤
                    flag=False   
            #测试
            for test_data,test_label in test_dl:
                y_pred_test, y_pred_KF_test  = model(test_data),model_KF(test_data) 
            if LR== False:
                _,mask = torch.max(y_pred_test,1)
                _,mask_KF = torch.max(y_pred_KF_test,1)
            else:
                mask,mask_KF = y_pred_test.ge(0.5).float(), y_pred_KF_test.ge(0.5).float() 
            
            correct,correct_KF = (mask == test_label).sum(), (mask_KF == test_label).sum() 
            acc,acc_KF = correct.item() / test_label.size(0), correct_KF.item() / test_label.size(0) 
            ACC.append(acc), ACC_KF.append(acc_KF)  
            print(epoch,"次迭代,DPSGD的Loss:",loss.item(),"KF-DPSGD的Loss:",loss_KF.item())
        
        #画图
        if type(model) == LogisticRegression:
            m = "LR_Epl_"
        elif type(model) == MLP:
            m = "MLP_MINIST_"
        elif type(model) == MnistCNN:
            m = "CNN_MINIST_"
        elif type(model) == CIAFARCNN:
            m = "CNN_CIAFAR10_"
        else:
            print("model name exist error!")
        
        X = [i for i in range(0,self.param["epochs"])]
        plt.xlabel("epoch"), plt.ylabel("Accuracy")
        plt.plot(X,ACC,label="DPSGD"), plt.plot(X,ACC_KF,label="DPSGD-KFGrad-M-KFParam") 
        plt.legend()
        path = "./picture/KF2结果/" + m + "KF1_KF2.png"
        plt.savefig(path)
        plt.close()

        data = {"DPSGD-M":ACC,"DPSGD-KFGrad-M-KFParam":ACC_KF}
        df = pd.DataFrame(data)
        df.to_csv(m+"KF2_Res.csv")
        return


    # #KFParam的作用，与PID的对比，Momentum VS KF_Param VS PID
    # def test1(self, model, data, KF_Param):   #三种方法
    #     if type(model) == LogisticRegression:
    #         LR = True
    #     else:
    #         LR = False 
    #     model_KF = copy.deepcopy(model)
    #     model_PID = copy.deepcopy(model)

    #     train_dl, test_dl = data[0], data[1]

    #     if LR==False:
    #         criterion, criterion_KF, criterion_PID = nn.CrossEntropyLoss(),nn.CrossEntropyLoss(),nn.CrossEntropyLoss()
    #     else :
    #         criterion, criterion_KF, criterion_PID = nn.BCELoss(),nn.BCELoss(),nn.BCELoss()

    #     optimizer =  torch.optim.SGD(model.parameters(), lr=self.param["lr"] )  #SGD
    #     optimizer_KF = torch.optim.SGD(model_KF.parameters(), lr=self.param["lr"] )  #DPSGD
    #     optimizer_PID =  torch.optim.SGD(model_PID.parameters(), lr=self.param["lr"],momentum = self.param['Momentum'])  #DPSGD-M

    #     ACC, ACC_KF, ACC_PID = [], [], []
    #     for epoch in range(self.param["epochs"]):
    #         flag = True
    #         for (batch_x,batch_y) in train_dl:
    #             y_pred, y_pred_KF, y_pred_PID = model(batch_x), model_KF(batch_x), model_PID(batch_x)
    #             loss, loss_KF, loss_PID = criterion(y_pred, batch_y), criterion_KF(y_pred_KF,batch_y), criterion_PID(y_pred_PID,batch_y)
    #             optimizer.zero_grad(),optimizer_KF.zero_grad(), optimizer_PID.zero_grad()
    #             loss.backward(),loss_KF.backward(),loss_PID.backward()

    #             #梯度裁剪
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.param["clip"]),torch.nn.utils.clip_grad_norm_(model_KF.parameters(), max_norm=self.param["clip"]),torch.nn.utils.clip_grad_norm_(model_PID.parameters(), max_norm=self.param["clip"])

    #             noise = {name: torch.normal(0, self.param["sigma"]*self.param["clip"], param.shape)/self.param["Lots"] for name, param in model.named_parameters()}                

    #             #加噪
    #             for name,param in model_KF.named_parameters():  param.grad += noise[name]  #DPSGD
    #             for name,param in model_PID.named_parameters(): param.grad += noise[name]  #DPSGD-M

    #             optimizer.step(),optimizer_KF.step(),optimizer_PID.step()

    #         #测试
    #         for test_data,test_label in test_dl:
    #             y_pred_test, y_pred_KF_test,y_pred_PID_test = model(test_data),model_KF(test_data),model_PID(test_data)
    #         if LR== False:
    #             _,mask = torch.max(y_pred_test,1)
    #             _,mask_KF = torch.max(y_pred_KF_test,1)
    #             _,mask_PID = torch.max(y_pred_PID_test,1)
    #         else:
    #             mask,mask_KF,mask_PID = y_pred_test.ge(0.5).float(), y_pred_KF_test.ge(0.5).float() , y_pred_PID_test.ge(0.5).float() 

    #         correct,correct_KF,correct_PID = (mask == test_label).sum(), (mask_KF == test_label).sum(), (mask_PID == test_label).sum()
    #         acc,acc_KF,acc_PID = correct.item() / test_label.size(0), correct_KF.item() / test_label.size(0) , correct_PID.item() / test_label.size(0) 
    #         ACC.append(acc), ACC_KF.append(acc_KF) , ACC_PID.append(acc_PID) 
    #         print(epoch,"次迭代,DPSGD的Loss:",loss.item(),"KF-DPSGD的Loss:",loss_KF.item(),"PID-DPSGD的Loss:",loss_PID.item())

    #     if type(model) == LogisticRegression:
    #         m = "LR_Epl_"
    #     elif type(model) == MLP:
    #         m = "MLP_MINIST_"
    #     elif type(model) == MnistCNN:
    #         m = "CNN_MINIST_"
    #     elif type(model) == CIAFARCNN:
    #         m = "CNN_CIAFAR10_"
    #     else:
    #         print("model name exist error!")

    #     X = [i for i in range(0,self.param["epochs"])]
    #     plt.xlabel("epoch"), plt.ylabel("Accuracy")

    #     plt.plot(X, ACC, label="SGD", linestyle='--', markersize=6, alpha=1, linewidth=2)
    #     plt.plot(X, ACC_KF, label="DPSGD", linestyle='--', markersize=6, alpha=1, linewidth=2)
    #     plt.plot(X, ACC_PID, label="DPSGD-M", linestyle='-', markersize=6, alpha=1, linewidth=2)
    #     plt.grid(True)  # 打开网格
    #     # 添加标题和标签
    #     plt.xlabel("Epoch", fontsize=14)
    #     plt.ylabel("Accuracy", fontsize=14)
    #     plt.legend(loc='lower right')

    #     path = "./picture/test/" + m + ".png"
    #     plt.savefig(path)
    #     print(path)
    #     plt.close()

    #     data = {"DPSGD":ACC,"DPSGD-KFGrad":ACC_KF}
    #     df = pd.DataFrame(data)
    #     print("./picture/test/" +m+".csv")
    #     df.to_csv("./picture/test/" +m+".csv")
    #     return




param = {"lr":0.0005,"clip":1,"sigma": 5,"epochs":100,"Lots":25,
        "Ada":1,        #  KFGrad方法的参数
        "Momentum":0.5, "threshold":10,"Q_decent": 0.95, #KFParam方法的参数
        "PID_D":0.004}    #PID方法的参数


# #####################################测试KFGrad#####################################
# Data_Model = [
#               [load_eplision_torch(batch_size=param["Lots"]),LogisticRegression()],
#               [load_mnist(batch_size=param["Lots"],model = "MLP"),  MLP()],
#               [load_mnist(batch_size=param["Lots"],model = "CNN"),  MnistCNN()],
#               [load_CIFAR10(batch_size=param["Lots"]) ,CIAFARCNN()]
#               ]

# Data_Model = [
#               [load_eplision_torch(batch_size=param["Lots"]),LogisticRegression()],
#             #   [load_mnist(batch_size=param["Lots"],model = "MLP"),  MLP()],
#             #   [load_mnist(batch_size=param["Lots"],model = "CNN"),  MnistCNN()],
#             #   [load_CIFAR10(batch_size=param["Lots"]) ,CIAFARCNN()]
#               ]

# for dm in Data_Model:
#     data, model = dm[0], dm[1]
#     print("数据加载完成")
#     KF_Grad = KF_grad(Q=10,R=10,P=10, Net_Piror=copy.deepcopy(model))
#     KF_Param = KF_param(Q=10 ,R=1 ,P=10, alfa=param["Q_decent"],threshold= param["threshold"], Net_Piror =copy.deepcopy(model))
#     kf_dpsgd =KFDPSGD(param)
#     kf_dpsgd.DPSGD_Momentum_KFGrad_KFParam(model, data, KF_Param,KF_Grad)  #KFGrad

# #####################################测试KFParam#####################################
Data_Model = [
              # [load_eplision_torch(batch_size=param["Lots"]),LogisticRegression()],
              # [load_mnist(batch_size=param["Lots"],model = "MLP"),  MLP()],
              [load_mnist(batch_size=param["Lots"],model = "CNN"),  MnistCNN()],
             # [load_CIFAR10(batch_size=param["Lots"]) ,CIAFARCNN()]
              ]



for dm in Data_Model:
    data, model = dm[0], dm[1] 
    KF_Grad = KF_grad(Q=10,R=10,P=10, Net_Piror=copy.deepcopy(model))
    KF_Param = KF_param(Q=10 ,R=10 ,P=10, alfa=param["Q_decent"],threshold= param["threshold"], Net_Piror =copy.deepcopy(model))
    kf_dpsgd =KFDPSGD(param)

    kf_dpsgd.test1(model,data,KF_Param )  #KFParam

# #####################################测试KFParam+KF_Param#####################################
# Data_Model = [[load_eplision_torch(),LogisticRegression()],
#               [load_mnist("MLP"),  MLP( )],
#               [load_mnist("CNN"),  MnistCNN()],
#               [load_CIFAR10() ,CIAFARCNN()]]
# for dm in Data_Model:
#     data, model = dm[0], dm[1]
#     KF_Grad = KF_grad(Q=10,R=1,P=10, Net_Piror=copy.deepcopy(model))
#     KF_Param = KF_param(Q=10 ,R=1 ,P=10, alfa=param["Q_decent"],threshold= param["threshold"], Net_Piror =copy.deepcopy(model))
#     kf_dpsgd =KFDPSGD(param)
#     kf_dpsgd.DPSGD_Momentum_KFGrad_KFParam(model,data,KF_Param,KF_Grad) #KFGrad 