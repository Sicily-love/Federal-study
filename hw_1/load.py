import csv
import random
from sklearn import preprocessing

"""
testfile():读取测试数据,以list形式返回打乱后的数据
trainfile():读取训练数据,以list形式返回打乱后的数据
actualsplit(data):返回两个list,分别是feature和label
getdata(data,i,class_num):输入参数data为testfile的输出,class_num为客户端数量,i为客户端编号(0:class_num-1)
                            返回划分给第i个客户端的标准化后的feature和label
"""


def testfile():
    filename = "./test.csv"
    data = []
    replacedict = {
        "Sun": [1, 0, 0, 0, 0, 0, 0],
        "Sat": [0, 1, 0, 0, 0, 0, 0],
        "Fri": [0, 0, 1, 0, 0, 0, 0],
        "Thurs": [0, 0, 0, 1, 0, 0, 0],
        "Wed": [0, 0, 0, 0, 1, 0, 0],
        "Tues": [0, 0, 0, 0, 0, 1, 0],
        "Mon": [0, 0, 0, 0, 0, 0, 1],
    }
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row[3] in replacedict:
                rownew = replacedict[row[3]] + row
                del rownew[10]
            else:
                rownew = row
            data.append(rownew)
    data.pop(0)
    data_int = [list(map(float, row)) for row in data]
    random.shuffle(data_int)
    return data_int


def trainfile():
    filename = "./train.csv"
    data = []
    replacedict = {
        "Sun": [1, 0, 0, 0, 0, 0, 0],
        "Sat": [0, 1, 0, 0, 0, 0, 0],
        "Fri": [0, 0, 1, 0, 0, 0, 0],
        "Thurs": [0, 0, 0, 1, 0, 0, 0],
        "Wed": [0, 0, 0, 0, 1, 0, 0],
        "Tues": [0, 0, 0, 0, 0, 1, 0],
        "Mon": [0, 0, 0, 0, 0, 0, 1],
    }
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if row[3] in replacedict:
                rownew = replacedict[row[3]] + row
                del rownew[10]
            else:
                rownew = row
            data.append(rownew)
    data.pop(0)
    data_int = [list(map(float, row)) for row in data]
    random.shuffle(data_int)
    return data_int


def actualsplit(data):
    label = []
    feature = []
    for row in data:
        label.append([row[13]])
        feature.append(row[:13])
    return feature, label


def getdata(train_data, test_data, i, weights):
    A=preprocessing.StandardScaler()
    train_feature , train_label = actualsplit(train_data)
    train_label = train_label[int(sum(weights[:i])) : int(sum(weights[: i + 1]))]
    train_feature = A.fit_transform(train_feature)
    train_feature = train_feature[int(sum(weights[:i])) : int(sum(weights[: i + 1]))]

    test_feature , test_label = actualsplit(test_data)
    test_feature = A.transform(test_feature)
    return train_feature, train_label , test_feature , test_label
