import load
import torch
import model
import numpy as np
import config_logger
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

logger = config_logger.logger

loss_functions = {
    "mse": nn.MSELoss(),
    "cross_entropy": nn.CrossEntropyLoss(),
    "huber_loss": torch.nn.SmoothL1Loss(),
    "negative_log_likelihood": nn.NLLLoss(),
}

optimizers = {
    "sgd": optim.SGD,
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
    "adagrad": optim.Adagrad,
    "lbfgs": optim.LBFGS,
}

lr_scheduler = {
    "exp": optim.lr_scheduler.ExponentialLR,
    "step": optim.lr_scheduler.StepLR,
    "muti": optim.lr_scheduler.MultiStepLR,
    "cos": optim.lr_scheduler.CosineAnnealingLR,
}


def datasetBalanceAllocation(class_num, train_data, test_data):
    clients_set = {}
    s = 3001
    while s > 2900:
        avgnum = 3000 / class_num
        weights_list = [avgnum] * (class_num - 1)
        weights_list = np.sum([weights_list, np.random.randint(-0.5 * avgnum, 0.5 * avgnum, class_num - 1)], axis=0).tolist()
        weights_list.append(3000 - sum(weights_list))
        s = sum(weights_list[: class_num - 1])

    for i in range(class_num):
        train_feature, train_label, test_feature, test_label = load.getdata(train_data, test_data, i, weights_list)
        trainDataSet = TensorDataset(
            torch.tensor(train_feature, dtype=torch.float, requires_grad=True),
            torch.tensor(train_label, dtype=torch.float, requires_grad=True),
        )
        someone = client(trainDataSet, i + 1, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        clients_set["client{}".format(i)] = someone
    return clients_set, weights_list, test_feature, test_label


class ClientsGroup(object):
    def __init__(self, dev, class_num):
        self.dev = dev
        self.clients_set = {}
        self.class_num = class_num


class client(object):
    def __init__(self, trainDataSet, client_id, dev):
        self.dataset = trainDataSet
        self.id = client_id
        self.dev = dev
        self.dataloader = None
        self.model = None

    def model_config(self, hidden_dim, loss, optimizer, learning_rate, scheduler):
        self.model = model.SimpleModel(13, hidden_dim, 1)
        self.criterion = loss_functions[loss]
        self.optimizer = optimizers[optimizer](self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.scheduler = lr_scheduler[scheduler](self.optimizer, gamma=0.95)

    def forward(self, batch_size, epochs):
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.model.zero_grad()
        for epoch in range(epochs):
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100, dynamic_ncols=True)

            average_loss = 0
            for data, label in progress_bar:
                data, label = data.to(self.dev), label.to(self.dev)
                self.optimizer.zero_grad()
                preds = self.model(data)
                loss = self.criterion(preds, label)
                loss.backward()
                self.optimizer.step()
                average_loss += loss.item()

                progress_bar.set_postfix(loss=f"{loss.item():.4f}", refresh=True)
                progress_bar.update(1)

            self.scheduler.step()
            logger.info(f"client {self.id}'s {epoch}th epoch average loss: {average_loss / len(self.dataloader)}")

        return self.model.state_dict()

    def update(self, global_epochs, global_parameters):
        for param_tensor in self.model.state_dict():
            self.model.state_dict()[param_tensor] = global_parameters[param_tensor]
        # new_lr = 0.001
        # self.optimizer.param_groups[0]["lr"] = new_lr
        pass
