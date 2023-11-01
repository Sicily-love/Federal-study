import csv
import random
from sklearn import preprocessing

'''
testfile():读取测试数据，以list形式返回打乱后的数据
trainfile():读取训练数据，以list形式返回打乱后的数据



'''

def testfile():
    filename='./test.csv'
    data=[]
    replacedict={'Sun':'1000000','Sat':'0100000','Fri':'0010000','Thurs':'0001000','Wed':'0000100','Tues':'0000010','Mon':'0000001'}
    with open(filename) as csvfile:
        csv_reader=csv.reader(csvfile)
        for row in csv_reader:
            rownew=[replacedict[i] if i in replacedict else i for i in row]
            data.append(rownew)
    data.pop(0)
    return random.shuffle(data)

def trainfile():
    filename='./train.csv'
    data=[]
    replacedict={'Sun':'1000000','Sat':'0100000','Fri':'0010000','Thurs':'0001000','Wed':'0000100','Tues':'0000010','Mon':'0000001'}
    with open(filename) as csvfile:
        csv_reader=csv.reader(csvfile)
        for row in csv_reader:
            rownew=[replacedict[i] if i in replacedict else i for i in row]
            data.append(rownew)
    data.pop(0)
    return random.shuffle(data)

def actualsplit(data):
    label=[]
    feature=[]
    for row in data:
        label.append=row[-1]
        feature.append(row.pop(-1))
    return feature,label

def getdata(data,class_num):
    data_new=data[(class_num*150):(class_num*150+149)]
    feature,label=actualsplit(data_new)
    feature_std=preprocessing.StandardScaler().fit_transform(feature)
    label_std=preprocessing.StandardScaler().fit_transform(label)
    return feature_std,label_std