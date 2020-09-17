import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.dataset import *
from config import *
from lib.train_func import *
from lib.init import *

counter = 0
TARGET = None

if __name__=='__main__':
    print('KaKao Internship Applicant : Lim Hwangyu')
    print('GPU Mode (CUDA) : ' + str(CUDA))
    print('GPU Availablity : ' + str(torch.cuda.is_available()))

    if len(sys.argv) <= 1 or sys.argv[1].upper() != 'AGE' and sys.argv[1].upper() != 'GENDER':
        print('사용법1 : python train.py age')
        print('사용법2 : python train.py gender')
        exit(1)
    TARGET = sys.argv[1].upper()

    trainset, testset = initDataset()
    print('Dataset Sample X : {0}'.format(str(trainset.x[0])))
    print('Dataset Sample T : ({0} {1})'.format(trainset.gender_t[0], trainset.age_t[0]))
    trainset_loader, testset_loader = initDataLoader(trainset, testset)
    x_batch, t_gender_batch, t_age_batch = initTensors(CUDA)
    network, optimizer, criterion = createNetwork(TARGET, CUDA)
    assert criterion != None

    logger = SummaryWriter()
    epochs = range(CHECK_POINT, MAX_EPOCH+1)
    for epoch in epochs:
        print('Train epoch {0}'.format(epoch))
        trainNetEpoch(TARGET, network, epoch, optimizer, criterion, x_batch, t_gender_batch, t_age_batch, trainset_loader, logger)
        testNet(TARGET, network, epoch, criterion, x_batch, t_gender_batch, t_age_batch, testset_loader, logger, True)
