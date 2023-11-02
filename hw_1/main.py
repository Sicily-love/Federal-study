from client import *

'''
a simple example
'''

data=trainfile()
CG=ClientsGroup(dev='cuda', class_num=50)
CG.clients_set=datasetBalanceAllocation(50,data)

print(CG.clients_set['client1'].train_ds)
print(CG.clients_set['client2'].dev)