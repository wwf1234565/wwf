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
#from data_process import load_eplision_torch,load_mnist,load_CIFAR10
import pandas as pd
from opacus import PrivacyEngine
from opacus.privacy_analysis import compute_rdp, get_privacy_spent
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def load_eplision_torch(data_path="eplision_cen.csv"):
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
    return train_data.float() ,train_label.float(), test_data.float(),test_label.float()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def check_privacy(sample_rate, data_size ,noise_multiplier ,c_epochs ,num_epochs):
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    delta = 1.0 / data_size
    noise_multiplier = args.noise_multiplier
    rdps = compute_rdp(sample_rate, noise_multiplier, steps, orders)
    epsilon, alpha = get_privacy_spent(orders, rdps, delta)

    return epsilon, alpha

class KFDPSGD():
    def __init__(self ,param):
        self.param = param    #{"lr":0.01,"clip":20,"sigma":2,"epochs":epochs,"Ada":3,"Momentum":0.9,"PID_I":1,"PID_D":0}

    #测试DPSGD-KFGrad
    def DPSGD_eplision(self, model, data, noise_multiplier, sample_rate, max_grad_norm):
        LR = False
        if type(model) == LogisticRegression:
            LR = True
        train_data ,train_label ,test_data ,test_label = data[0], data[1], data[2], data[3] 
        if LR==False:
            criterion = nn.CrossEntropyLoss()
            train_label = train_label.long()
        else:
            criterion= nn.BCELoss()
        optimizer= torch.optim.SGD(model.parameters(), lr=self.param["lr"])
        ACC = []

        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=sample_rate,
            alphas=orders,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm
        )
        privacy_engine.attach(optimizer)


        torch_dataset = Data.TensorDataset(train_data,train_label)
        loader = Data.DataLoader(
            dataset=torch_dataset,
            batch_size=int(train_data.size()[0]*sample_rate) 
            )

        for epoch in range(self.param["epochs"]):
            for batch_x,batch_y in loader:
                y_pred = model(batch_x)
                loss = criterion(y_pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #测试
            y_pred_test = model(test_data)
            if LR == False:
                _,mask = torch.max(y_pred_test,1)
            else:
                mask= y_pred_test.ge(0.5).float()
            correct = (mask == test_label).sum()
            acc = correct.item() / test_label.size(0)
            ACC.append(acc)
            print(epoch,"次迭代,DPSGD的Loss:",loss.item())
            delta = 0.002  #1/train_data.size()[0]
            optimal_epsilon, optimal_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
            print('Now privacy use is ε={:.2f} and δ={:.4f} at a={:.2f}'.format(optimal_epsilon,delta,optimal_alpha))
        return


param = {"lr":0.01,"clip":0.5,"sigma": 0.01,"epochs":3000,
        "Ada":2,        #  KFGrad方法的参数
        "Momentum":0.4, "threshold":1,"Q_decent": 0.95, #KFParam方法的参数
        "PID_I":1,"PID_D":4}    #PID方法的参数
 

 
# #####################################测试KFGrad#####################################
# Data_Model = [[load_eplision_torch(),LogisticRegression()],
#               [load_mnist("MLP"),  MLP( )],
#               [load_mnist("CNN"),  MnistCNN()],
#               [load_CIFAR10() ,CIAFARCNN()]]
Data_Model = [[load_eplision_torch(),LogisticRegression()] ]

noise_multiplier = 1
sample_rate = 16/500
max_grad_norm = 0.1

for dm in Data_Model:
    data, model = dm[0], dm[1]
    kf_dpsgd =KFDPSGD(param)
    kf_dpsgd.DPSGD_eplision(model, data, noise_multiplier, sample_rate, max_grad_norm)