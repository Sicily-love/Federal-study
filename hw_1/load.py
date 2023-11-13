import csv
import random
from sklearn import preprocessing


def testfile():
    random.seed()
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
    return data_int


def trainfile():
    random.seed()
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
    return data_int


def actualsplit(data):
    label = []
    feature = []
    for row in data:
        label.append([row[13]])
        feature.append(row[:13])
    return feature, label


def getdata(data, id, mode, weights):
    """get train/test data

    Args:
        data (list): data
        id (int): id of client
        mode (string): "train" or "test"
        weights (list): allocation weights

    Returns:
        tuple: (Standard feature, label)
    """
    A = preprocessing.StandardScaler()
    if mode == "train":
        feature, label = actualsplit(data)
        label = label[int(sum(weights[:id])) : int(sum(weights[: id + 1]))]
        feature = A.fit_transform(feature)
        feature = feature[int(sum(weights[:id])) : int(sum(weights[: id + 1]))]
    elif mode == "test":
        feature, label = actualsplit(data)
        feature = A.fit_transform(feature)

    return feature, label
