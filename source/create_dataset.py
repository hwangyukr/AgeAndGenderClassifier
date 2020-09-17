from lib.dataset import createAndSaveDataset
from config import *

print('Creating Dataset ...')
print('-----------------------------------------')
print('Src File Name : {0}'.format(CSV_DATA_FILE))
print('Dest File Name : {0}'.format(DATASET_PICKLE_FILE))
print('-----------------------------------------')
print('Waiting ...')
trainset, testset = createAndSaveDataset(CSV_DATA_FILE, DATASET_PICKLE_FILE)
print('DataFile Created Successfully')
print('Train Sets : {0}'.format(trainset.__len__()))
print('Test Sets : {0}'.format(testset.__len__()))
print('-----------------------------------------')
