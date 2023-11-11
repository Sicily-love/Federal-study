import load
import model
import torch
import client
import argparse
import numpy as np
import config_logger
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

logger = config_logger.logger

loss_functions = ["mse", "cross_entropy", "huber_loss", "negative_log_likelihood"]
optimizers = ["sgd", "adam", "rmsprop", "adagrad", "lbfgs"]
lr_scheduler = ["exp", "step", "muti", "cos"]

parser = argparse.ArgumentParser(description="Distributed Learning Parameters")
parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
parser.add_argument("--global_epochs", type=int, default=10, help="Number of global epochs")
parser.add_argument("--local_epochs", type=int, default=5, help="Number of local epochs")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--loss", choices=loss_functions, default="mse", help="Loss function (default: MSE)")
parser.add_argument("--optimizer", choices=optimizers, default="sgd", help="Optimizer (default: SGD)")
parser.add_argument("--scheduler", choices=lr_scheduler, default="exp", help="learning rate scheduler (default: exp)")
parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer dimension (default: 64)")
lr_scheduler

args = parser.parse_args()

# initialize clients set
train_data = load.trainfile()
test_data = load.testfile()
CG = client.ClientsGroup(dev="cuda:0" if torch.cuda.is_available() else "cpu", class_num=args.num_clients)
CG.clients_set, weights, _ = client.datasetBalanceAllocation(args.num_clients, train_data, test_data)
weights = np.array(weights) / sum(weights)

Net = model.SimpleModel(13, args.hidden_dim, 1)

global_parameters = Net.state_dict()

for client_id in range(args.num_clients):
    client_name = "client" + str(client_id)
    CG.clients_set[client_name].model_config(args.hidden_dim, args.loss, args.optimizer, args.learning_rate,
                                             args.scheduler)

for epoch in range(args.global_epochs):
    # redistribute traindata
    train_data = load.trainfile()
    _, weights, tdslist = client.datasetBalanceAllocation(args.num_clients, train_data, test_data)
    weights = np.array(weights) / sum(weights)
    for i in range(args.num_clients):
        CG.clients_set["client{}".format(i)].trainDataSet = tdslist[i]

    for param_tensor in global_parameters:
        global_parameters[param_tensor] -= global_parameters[param_tensor]
    states = []
    for i in range(args.num_clients):
        client_name = "client" + str(i)
        client_state = CG.clients_set[client_name].forward(args.batch_size, args.local_epochs)
        states.append(client_state)
    logger.info(f"{epoch + 1}th distribute training finished")

    for i, state in enumerate(states):
        for param_tensor in state:
            global_parameters[param_tensor] += state[param_tensor] * weights[i]

    for i in range(args.num_clients):
        client_name = "client" + str(i)
        CG.clients_set[client_name].update(epoch, global_parameters)

Net.load_state_dict(global_parameters)
torch.save(Net.state_dict(), "model.pth")


def validate(model, dataloader, dev):
    Net.eval()

    average_loss = 0.0
    criterion = nn.MSELoss()
    labels_all = []
    predicted_all = []

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(dev), labels.to(dev)

            outputs = model(data)
            loss = criterion(outputs, labels)
            print(loss)
            average_loss += loss.item()
            for row in labels.tolist():
                labels_all.append(row[0])
            for row in outputs.tolist():
                predicted_all.append(row[0])

    average_loss = average_loss / len(dataloader)

    print(f"Validation Loss: {average_loss:.4f}")

    plt.scatter(range(len(labels_all)), labels_all, c='b', marker='+', label='actual temp')
    plt.scatter(range(len(predicted_all)), predicted_all, c='r', marker='*', label='estimated temp')
    plt.xlabel('day')
    plt.ylabel('temp')
    plt.legend()
    plt.show()

    return average_loss


# validate test data
test_data = load.testfile()
_, _, test_feature, test_label = load.getdata(train_data, test_data, 0, [3000], True)
testdataset = TensorDataset(
    torch.tensor(test_feature, dtype=torch.float, requires_grad=False),
    torch.tensor(test_label, dtype=torch.float, requires_grad=False),
)
testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
validate(Net, testdataloader, "cuda:0" if torch.cuda.is_available() else "cpu")

# validate train data
train_data = load.trainfile()
test_feature, test_label, _, _ = load.getdata(train_data, test_data, 0, [3000], False)
testdataset = TensorDataset(
    torch.tensor(test_feature, dtype=torch.float, requires_grad=False),
    torch.tensor(test_label, dtype=torch.float, requires_grad=False),
)
testdataloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
validate(Net, testdataloader, "cuda:0" if torch.cuda.is_available() else "cpu")
