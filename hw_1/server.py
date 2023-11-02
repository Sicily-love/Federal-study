import load
import model
import torch
import client
import argparse
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim

parser = argparse.ArgumentParser(description="Distributed Learning Parameters")

# 添加命令行参数
parser.add_argument("--num_clients", type=int, default=20, help="Number of clients")
parser.add_argument("--global_epochs", type=int, default=15, help="Number of global epochs")
parser.add_argument("--local_epochs", type=int, default=3, help="Number of local epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")

# 解析命令行参数
args = parser.parse_args()

global_model = model.SimpleModel(13, 64, 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(global_model.parameters(), lr=args.learning_rate)
data = load.getdata()

ClientsGroup = client.ClientsGroup("cuda:0" if torch.cuda.is_available() else "cpu", args.num_clients, data)

dataloader = []
# 边缘计算与本地聚合
for epoch in range(args.global_epochs):
    for i in range(args.num_clients):
        for j in range(args.local_epochs):
            running_loss = 0.0
            for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.local_epoch}", ncols=100, dynamic_ncols=True):
                # 将梯度置零
                optimizer.zero_grad()

                # 前向传播
                outputs = model(inputs)

                # 计算损失
                loss = criterion(outputs, labels)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # 每个epoch结束后打印损失
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

            print("Training finished")
    # TODO Model aggregation
    # # 聚合本地模型到全局模型
    # global_model.zero_grad()
    # for client in clients:
    #     for global_param, local_param in zip(global_model.parameters(), client.model.parameters()):
    #         global_param.data += local_param.data / num_clients
    # TODO parallel


# 验证拟合和预测
def validate(model, data):
    # 这里可以使用验证数据来评估模型性能
    pass


validation_data = []
validate(global_model, validation_data)


# 使用全局模型进行预测
def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        return model(input_data)


input_data = torch.randn(10, 13)  # 替换为实际输入数据
predictions = predict(global_model, input_data)
print("Predictions:", predictions)
