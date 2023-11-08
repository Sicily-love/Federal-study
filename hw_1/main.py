import load
import client
import random

my_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

random.shuffle(my_list)

for row in my_list:
    print(row)


data = load.trainfile()

CG = client.ClientsGroup(dev="cuda", class_num=2)
CG.clients_set, w = client.datasetBalanceAllocation(50, data)

print(len(CG.clients_set["client1"].train_ds))
print(CG.clients_set["client2"].dev)
