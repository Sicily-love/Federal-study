import csv
import random
from sklearn import preprocessing

'''
testfile():读取测试数据，以list形式返回打乱后的数据
trainfile():读取训练数据，以list形式返回打乱后的数据
actualsplit(data):返回两个list，分别是feature和label
getdata(data,i,class_num):输入参数data为testfile的输出，class_num为客户端数量，i为客户端编号（0:class_num-1）
                            返回划分给第i个客户端的标准化后的feature和label
'''

def testfile():
    filename='./test.csv'
    data=[]
    replacedict={'Sun':[1,0,0,0,0,0,0],'Sat':[0,1,0,0,0,0,0],'Fri':[0,0,1,0,0,0,0],'Thurs':[0,0,0,1,0,0,0],'Wed':[0,0,0,0,1,0,0],'Tues':[0,0,0,0,0,1,0],'Mon':[0,0,0,0,0,0,1]}
    with open(filename) as csvfile:
        csv_reader=csv.reader(csvfile)
        for row in csv_reader:
            if row[3] in replacedict:
                rownew=row+replacedict[row[3]]
                del rownew[3]
            else:
                rownew=row
            data.append(rownew)
    data.pop(0)
    data_int = [list(map(float, row)) for row in data]
    random.shuffle(data_int)
    return data_int

def trainfile():
    filename='./train.csv'
    data=[]
    replacedict={'Sun':[1,0,0,0,0,0,0],'Sat':[0,1,0,0,0,0,0],'Fri':[0,0,1,0,0,0,0],'Thurs':[0,0,0,1,0,0,0],'Wed':[0,0,0,0,1,0,0],'Tues':[0,0,0,0,0,1,0],'Mon':[0,0,0,0,0,0,1]}
    with open(filename) as csvfile:
        csv_reader=csv.reader(csvfile)
        for row in csv_reader:
            if row[3] in replacedict:
                rownew=replacedict[row[3]]+row
                del rownew[10]
            else:
                rownew=row
            data.append(rownew)
    data.pop(0)
    data_int=[list(map(float,row)) for row in data]
    random.shuffle(data_int)
    return data_int

def actualsplit(data):
    label=[]
    feature=[]
    for row in data:
        label.append([row[-1]])
        row.pop(-1)
        feature.append(row)
    return feature,label

def getdata(data,i,class_num):
    num_per_client=3000/class_num
    data_new=data[int(i*num_per_client):int(i*num_per_client+num_per_client)]
    feature,label=actualsplit(data_new)
    feature_std=preprocessing.StandardScaler().fit_transform(feature)
    label_std=preprocessing.StandardScaler().fit_transform(label)
    return feature_std,label_std