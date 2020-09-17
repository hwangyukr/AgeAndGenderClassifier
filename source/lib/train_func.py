import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lib.dataset import *
from lib.network.AgeRegNet import AgeRegNet
from lib.network.GenderClsNet import GenderClsNet
from lib.network.utils import *
from config import *
counter = 0

def createNetwork(target, isCUDA, checkPointFilename=None):
    network = None
    criterion = None
    if target == 'GENDER':
        network = GenderClsNet(GENDER_INPUT_LAYER_SIZE , GENDER_OUTPUT_LAYER_SIZE, checkPointFilename)
        criterion = nn.CrossEntropyLoss()
    if target == 'AGE':
        network = AgeRegNet(AGE_INPUT_LAYER_SIZE , AGE_OUTPUT_LAYER_SIZE, checkPointFilename)
        criterion = nn.MSELoss()

    if checkPointFilename != None:
        try:
            network = torch.load(checkPointFilename)
            print('Model Loaded ! ==> {0}'.format(checkPointFilename))
        except:
            print('!!! ============================')
            print('!!! Model Load FAIL !!! ==> {0}'.format(checkPointFilename))
            print('!!! ============================')
            exit(1)

    assert network != None
    assert criterion != None

    if isCUDA:
        network.cuda()
    optimizer = optim.Adam(network.parameters(), lr=LR, weight_decay=DECAY)
    return network, optimizer, criterion

def trainNetEpoch(target, network, epoch, optimizer, criterion, x_batch, t_gender_batch, t_age_batch, trainset_loader, logger):
    assert type(trainset_loader) == DataLoader
    network.train()
    global counter
    count_trained = 0

    getAccuarcyFunc = None
    if target=='GENDER': getAccuarcyFunc = getGenderAccuarcy
    if target=='AGE': getAccuarcyFunc = getAgeAccuarcy
    assert getAccuarcyFunc != None

    for idx, item in enumerate(trainset_loader):
        [x, gender_t, age_t] = item

        x_batch.resize_(x.size()).copy_(x)
        t_batch = None
        if target=='GENDER':
            t_gender_batch.resize_(gender_t.size()).copy_(gender_t)
            t_batch = t_gender_batch
        if target=='AGE':
            t_age_batch.resize_(age_t.size()).copy_(age_t)
            t_batch = t_age_batch

        optimizer.zero_grad()
        y_batch = network(x_batch)

        loss = criterion(y_batch, t_batch)
        logger.add_scalar('{0}/Train/Loss'.format(target), loss, counter)
        counter = counter + 1
        count_trained = count_trained + x_batch.size(0)
        if counter % 100 == 0:
            print('{0} : EPOCH( {1} ) => {2}'.format(target, epoch, count_trained))
            acc = getAccuarcyFunc(y_batch, t_batch)
            logger.add_scalar('{0}/Train/Accuarcy'.format(target), acc, counter)
        loss.backward()
        optimizer.step()
    torch.save(network, '{0}/{1}_{2}_{3}.model'.format(OUTPUT_DIRECTORY,target,NAME,epoch))


def testNet(target, network, epoch, criterion, x_batch, t_gender_batch, t_age_batch, testset_loader, logger=None, visual=True):
    assert type(testset_loader) == DataLoader
    network.eval()
    iter_cnt = 0
    acc_sum = 0
    loss_sum = 0
    test_loss_sum = 0
    print('Testing ...')
    y_visual = np.array([])
    t_visual = np.array([])

    getAccuarcyFunc = None
    if target=='GENDER': getAccuarcyFunc = getGenderAccuarcy
    if target=='AGE': getAccuarcyFunc = getAgeAccuarcy
    assert getAccuarcyFunc != None

    with torch.no_grad():
        for test_idx, test_item in enumerate(testset_loader):
            [x, gender_t, age_t] = test_item
            x_batch.resize_(x.size()).copy_(x)

            t_batch = None
            if target=='GENDER':
                t_gender_batch.resize_(gender_t.size()).copy_(gender_t)
                t_batch = t_gender_batch
            if target=='AGE':
                t_age_batch.resize_(age_t.size()).copy_(age_t)
                t_batch = t_age_batch

            y_batch = network(x_batch)
            loss = criterion(y_batch, t_batch)
            loss_sum += loss
            acc_sum += getAccuarcyFunc(y_batch, t_batch)
            iter_cnt = iter_cnt + 1
            if visual:
                y_visual = np.concatenate((y_visual, y_batch[0:10].cpu().detach()), axis=None)
                t_visual = np.concatenate((t_visual, t_batch[0:10].cpu().detach()), axis=None)

    loss_avg = loss_sum/iter_cnt
    acc_avg = acc_sum/iter_cnt
    print(target, 'Test Loss => ', loss_avg)
    print(target, 'Test Acc  => ', acc_avg)

    if logger != None:
        logger.add_scalar('{0}/Test/Loss'.format(target), loss_avg, epoch)
        logger.add_scalar('{0}/Test/Accuarcy'.format(target), acc_avg, epoch)

    if visual:
        attention = getLayerAttention(network.first_layer)
        if target == 'AGE':
            visualizeAge(epoch, y_visual, t_visual, attention, acc_avg)
        if target == 'GENDER':
            visualizeGender(epoch, y_visual, t_visual, attention, acc_avg)

def visualizeAge(epoch, y_visual, t_visual, attention, acc_avg):
    y_visual = np.int32(y_visual * 20 + 0.5)
    t_visual = t_visual * 20
    sort_idx = np.argsort(t_visual*100+y_visual)
    fig, ax = plt.subplots()
    xaxis = 0
    for i in sort_idx:
        ax.plot(xaxis, t_visual[i], marker='o', markersize=1, color = 'blue', alpha=1.0)
        ax.plot(xaxis, y_visual[i], marker='o', markersize=2, color = 'red', alpha=0.3)
        xaxis = xaxis + 1
    plt.title('Scatter: Blue is Ground Truth', fontsize=20)
    plt.savefig('{0}/AGE_Scatter_{1}_{2}.png'.format(OUTPUT_DIRECTORY, epoch, int(acc_avg*10000)))
    plt.close(fig)

    ind = range(len(attention))
    plt.bar(ind, attention)
    plt.xlabel('Feature', fontsize=10)
    plt.ylabel('Attention', fontsize=10)
    plt.xticks(ind, ind, fontsize=10)
    plt.savefig('{0}/AGE_Attention_{1}_{2}.png'.format(OUTPUT_DIRECTORY, epoch, int(acc_avg*10000)))
    plt.close()

def visualizeGender(epoch, y_visual, t_visual, attention, acc_avg):
    y_visual = y_visual.reshape(-1, 2)
    y_visual = y_visual.argmax(axis=1)

    gt_man_idx = np.where(t_visual==0)[0]
    gt_woman_idx = np.where(t_visual==1)[0]

    men_predict = y_visual[gt_man_idx] # 0 is correct
    women_predict = y_visual[gt_woman_idx] #  1 is correct

    m_count = len(np.where(t_visual==0)[0])
    w_count = len(np.where(t_visual==1)[0])
    m_correct = len(np.where(men_predict==0)[0])
    w_correct = len(np.where(women_predict==1)[0])

    bars = [ m_count, m_correct, w_count, w_correct ]
    labels = ['man_gt', 'man_correct', 'woman_gt', 'woman_correct']
    ind = range(len(bars))
    plt.bar((ind), bars)
    plt.xlabel('Gender', fontsize=10)
    plt.ylabel('Caldinality', fontsize=10)
    plt.xticks(ind, labels, fontsize=10)
    plt.savefig('{0}/Gender_Bar_{1}_{2}.png'.format(OUTPUT_DIRECTORY, epoch, int(acc_avg*10000)))
    plt.close()

    ind = range(len(attention))
    plt.bar(ind, attention)
    plt.xlabel('Feature', fontsize=10)
    plt.ylabel('Attention', fontsize=10)
    plt.xticks(ind, ind, fontsize=10)
    plt.savefig('{0}/GENDER_Attention_{1}_{2}.png'.format(OUTPUT_DIRECTORY, epoch, int(acc_avg*10000)))
    plt.close()

def getAgeAccuarcy(y, t):
    bs = t.size(0)
    predict = y.cpu().detach().numpy()
    predict = np.int32(predict*20 + 0.5)
    target = t.cpu().detach().numpy()
    target = np.int32(target*20)
    succ = float((predict==target).sum().item())
    return succ/bs

def getGenderAccuarcy(y, t):
    bs = t.size(0)
    predict = y.cpu().detach().numpy()
    predict = predict.argmax(axis=1)
    target = t.cpu().detach().numpy()
    succ = float((predict==target).sum().item())
    return succ/bs
