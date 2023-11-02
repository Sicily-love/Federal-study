import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from load import *

'''
datasetBalanceAllocation分到了ClientGroup类外面，用法见main.py
client类被字典CG.clients_set索引
client.num_example不知道是干什么的，照抄过来了
'''

def datasetBalanceAllocation(class_num, data):
    clients_set = {}
    for i in range(class_num):
        feature, label = getdata(data, i, class_num)
        someone = client(TensorDataset(torch.tensor(feature, dtype=torch.float, requires_grad=True), torch.tensor(label, dtype=torch.float, requires_grad=True)), torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 20)
        clients_set['client{}'.format(i)] = someone
    return clients_set

class ClientsGroup(object):
    def __init__(self,dev,class_num):
        self.dev=dev
        self.clients_set={}
        self.class_num=class_num

class client(object):
    def __init__(self, trainDataSet, dev, num_example):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.num_example = num_example
        self.state = {}

    def localUpdate(self, localBatchSize, localepoch, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localepoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()
        return Net.state_dict()
