import torch
from torch.utils.data import DataLoader
from lib.dataset import *

def initDataset():
    print('Dataset Process ...')
    trainset, testset = loadDataset(DATASET_PICKLE_FILE)
    print('Dataset Succesfully Loaded')
    print('Train Sets : {0}'.format(trainset.__len__()))
    print('Test Sets : {0}'.format(testset.__len__()))
    return trainset, testset

def initDataLoader(trainset, testset):
    assert type(trainset) is HealthData
    assert type(testset) is HealthData
    bs = BATCH_SIZE # batch_size
    trainset_loader = DataLoader(trainset, bs, num_workers=8, shuffle=True)
    testset_loader = DataLoader(testset, 4096, num_workers=8, shuffle=True)
    return trainset_loader, testset_loader

def initTensors(gpu):
    x_batch = torch.FloatTensor(1)
    t_gender_batch = torch.LongTensor(1)
    t_age_batch = torch.FloatTensor(1)
    if gpu:
        x_batch = x_batch.cuda()
        t_gender_batch = t_gender_batch.cuda()
        t_age_batch = t_age_batch.cuda()
    return x_batch, t_gender_batch, t_age_batch
