import load
import model
import torch
import client
import argparse
import config_logger
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

parser = argparse.ArgumentParser(description="Distributed Learning Parameters")
parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
parser.add_argument(
    "--global_epochs", type=int, default=1, help="Number of global epochs"
)
parser.add_argument(
    "--local_epochs", type=int, default=20, help="Number of local epochs"
)
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument(
    "--loss",
    choices=loss_functions.keys(),
    default="mse",
    help="Loss function (default: MSE)",
)
parser.add_argument(
    "--optimizer",
    choices=optimizers.keys(),
    default="sgd",
    help="Optimizer (default: SGD)",
)
parser.add_argument(
    "--hidden_dim", type=int, default=64, help="Hidden layer dimension (default: 128)"
)


args = parser.parse_args()

Net = model.SimpleModel(13, args.hidden_dim, 1)
global_parameters = Net.state_dict()
criterion = loss_functions[args.loss]
optimizer = optimizers[args.optimizer](Net.parameters(), lr=args.learning_rate)

train_data = load.trainfile()
test_data = load.testfile()
CG = client.ClientsGroup(
    dev="cuda:0" if torch.cuda.is_available() else "cpu", class_num=args.num_clients
)
CG.clients_set, weights, test_feature, test_label= client.datasetBalanceAllocation(args.num_clients, train_data, test_data)


for epoch in range(args.global_epochs):
    states = []
    for i in range(args.num_clients):
        client_name = "client" + str(i)
        client_state = CG.clients_set[client_name].localUpdate(
            args.batch_size,
            args.local_epochs,
            Net,
            criterion,
            optimizer,
            global_parameters,
        )
        states.append(client_state)
    logger.info(f"{epoch + 1}th training finished")

    for param_tensor in global_parameters:
        global_parameters[param_tensor] -= global_parameters[param_tensor]

    for state in states:
        for param_tensor in global_parameters:
            global_parameters[param_tensor] += state[param_tensor] / len(states)

Net.load_state_dict(global_parameters)


def validate(model, val_dataloader, dev):
    Net.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in val_dataloader:
            data, labels = data.to(dev), labels.to(dev)

            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            correct += (outputs == labels).sum().item()
            print((outputs == labels).shape)
            print(outputs == labels)
            total_samples += labels.size(0)

    avg_loss = total_loss / len(val_dataloader)
    accuracy = (correct / total_samples) * 100.0

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


test_ds = TensorDataset(
    torch.tensor(test_feature, dtype=torch.float, requires_grad=False),
    torch.tensor(test_label, dtype=torch.float, requires_grad=False),
)
test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
validate(Net, test_dl, "cuda:0" if torch.cuda.is_available() else "cpu")
