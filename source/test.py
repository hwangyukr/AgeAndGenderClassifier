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
    print('Test Network')
    print('KaKao Internship Applicant : Lim Hwangyu')
    print('GPU Mode (CUDA) : ' + str(CUDA))
    print('GPU Availablity : ' + str(torch.cuda.is_available()))
    if len(sys.argv) <= 1 or sys.argv[1].upper() != 'AGE' and sys.argv[1].upper() != 'GENDER':
        print('사용법1 : python test.py age')
        print('사용법2 : python test.py gender')
        exit(1)
    TARGET = sys.argv[1].upper()

    trainset, testset = initDataset()
    trainset_loader, testset_loader = initDataLoader(trainset, testset)
    x_batch, t_gender_batch, t_age_batch = initTensors(CUDA)
    network, optimizer, criterion = createNetwork(TARGET, CUDA, './resource/{0}.model'.format(TARGET))
    assert criterion != None

    logger = SummaryWriter()
    epochs = range(CHECK_POINT, MAX_EPOCH+1)
    testNet(TARGET, network, 0, criterion, x_batch, t_gender_batch, t_age_batch, testset_loader, logger, True)
    print('Some Saved at Output Directory !')
