import torch
import numpy as np
import csv
import pickle
import os
from config import *
from torch.utils.data import Dataset

class HealthData(Dataset):
    def __init__(self, x, gender_t, age_t):
        assert type(x) is np.ndarray
        assert type(gender_t) is np.ndarray
        assert type(age_t) is np.ndarray
        self.x = torch.from_numpy(x)
        self.gender_t = torch.from_numpy(gender_t)
        self.age_t = torch.from_numpy(age_t)

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
        x = self.x
        gender_t = self.gender_t
        age_t = self.age_t
        return x[idx], gender_t[idx], age_t[idx]

def readFile(filename):
    data = []
    f = open(filename, 'r', encoding='cp949')
    reader = csv.reader(f)
    index = next(reader)
    for row in reader:
        data.append(row)
    f.close()
    data = np.array(data)
    return index, data

#나중에 빈데이터랑 치석 데이터도 해결해야 함
#현재는 꽉 찬 데이터만 가져와서 학습
def extractDataset(data, cmp):
    assert type(data) is np.ndarray
    attr = np.concatenate((ATTR_FEATURE_ARRAY, ATTR_TARGET_ARRY), axis=None)
    x = []
    gender_t = []
    age_t = []
    for row in data:
        snum = int(row[ATTR_SERIAL_IDX])
        if cmp(snum)==False:
            continue
        row[ATTR_AGE_IDX] = int(float(row[ATTR_AGE_IDX]) / 2) / 20
        row[ATTR_GENDER_IDX] = int(float(row[ATTR_GENDER_IDX])) - 1
        if row[ATTR_TARTAR_IDX] == "": row[ATTR_TARTAR_IDX] = -1
        skip = False
        for idx in attr:
            if row[idx] == "": skip=True; break
            #if row[idx] == "": row[idx]=float('nan')
            else: row[idx] = float(row[idx])

        if skip==False:
            x.append(row[ATTR_FEATURE_ARRAY])
            gender_t.append(row[ATTR_GENDER_IDX])
            age_t.append(row[ATTR_AGE_IDX])

    x = np.float32(x)
    gender_t = np.float32(gender_t)
    age_t = np.float32(age_t)
    return x, gender_t, age_t

def createAndSaveDataset(filename, dstFilename):
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    index, data = readFile(filename)
    x_train, t_gender_train, t_age_train = extractDataset(data, lambda s: s%10!=5)
    x_test, t_gender_test, t_age_test = extractDataset(data, lambda s: s%10==5)
    trainset = HealthData(x_train, t_gender_train, t_age_train)
    testset = HealthData(x_test, t_gender_test, t_age_test)
    with open(dstFilename, 'wb') as f:
        pickle.dump([ trainset, testset ], f)
    return trainset, testset

def loadDataset(filename):
    try:
        with open(filename, 'rb') as f:
            [ trainset, testset ] = pickle.load(f)
            return trainset, testset
    except:
        print('Fail (check MISMATCH or NOTFOUND) {0}'.format(filename))
        print('---> TODO: run create_dataset.py first')
        exit(1)
