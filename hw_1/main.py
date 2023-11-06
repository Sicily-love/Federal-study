import client
import load

"""
a simple example
"""

data = load.trainfile()
CG = client.ClientsGroup(dev="cuda", class_num=50)
CG.clients_set = client.datasetBalanceAllocation(50, data)

print(len(CG.clients_set["client1"].train_ds))
print(CG.clients_set["client2"].dev)
