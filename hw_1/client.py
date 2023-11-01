import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from load import *

class ClientsGroup(object):
    def __init__(self,dev,class_num,data):
        self.dev=dev
        self.clients_set={}
        self.class_num=class_num
        self.datasetBalanceAllocation(data)

    def datasetBalanceAllocation(self,data):
        for i in range(self.class_num):
            feature,label=getdata(data,i)
            someone=client(TensorDataset(torch.tensor(feature,dtype=torch.float,requires_grad=True),torch.tensor(label,dtype=torch.float,requires_grad=True)),torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),20)
            self.clients_set['client{}'.format(i)]=someone

class client(object):
    def __init__(self,trainDataSet,dev,num_example):
        self.train_ds=trainDataSet
        self.dev=dev
        self.train_dl=None
        self.num_example=num_example
        self.state={}

    def localUpdate(self,localBatchSize,localepoch,Net,lossFun,opti,global_parameters):
        Net.load_state_dict(global_parameters,strict=True)
        self.train_dl=DataLoader(self.train_ds,batch_size=localBatchSize,shuffle=True)
        for epoch in range(localepoch):
            for data,label in self.train_dl:
                data,label=data.to(self.dev),label.to(self.dev)
                preds=Net(data)
                loss=lossFun(preds,label)
                loss.backward()
                opti.step()
                opti.zero_grad()
        return Net.state_dict()