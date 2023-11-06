import load
import torch
import config_logger
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

logger = config_logger.logger


"""
datasetBalanceAllocation分到了ClientGroup类外面,用法见main.py
client类被字典CG.clients_set索引
client.num_example不知道是干什么的,照抄过来了
"""


def datasetBalanceAllocation(class_num, data):
    clients_set = {}
    for i in range(class_num):
        feature, label = load.getdata(data, i, class_num)
        someone = client(
            TensorDataset(torch.tensor(feature, dtype=torch.float, requires_grad=True), torch.tensor(label, dtype=torch.float, requires_grad=True)),
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        clients_set["client{}".format(i)] = someone
    return clients_set


class ClientsGroup(object):
    def __init__(self, dev, class_num):
        self.dev = dev
        self.clients_set = {}
        self.class_num = class_num


class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.state = {}

    def localUpdate(self, localBatchSize, localepoch, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        running_loss = 0.0
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True, drop_last=True)
        for epoch in range(localepoch):
            progress_bar = tqdm(self.train_dl, desc=f"Epoch {epoch + 1}/{localepoch}", ncols=100, dynamic_ncols=True)
            for data, label in progress_bar:
                data, label = data.to(self.dev), label.to(self.dev)
                opti.zero_grad()
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                running_loss += loss.item()

                progress_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=True)
                progress_bar.update(1)

        logger.info(f"Average Loss: {running_loss / len(self.train_dl)}")
        Net.zero_grad()

        return Net.state_dict()
