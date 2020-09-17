import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

from torch.autograd import Variable
from lib.network.utils import *

class AgeRegNet(nn.Module):
    def __init__(self, in_num, age_out_num, checkPointFilename=None):
        super(AgeRegNet, self).__init__()
        self.in_num = in_num
        self.age_out_num = age_out_num
        age_layers = []

        self.first_layer = append_fc_layer(age_layers, in_num, 2048)
        append_fc_layer(age_layers, 2048, 128)
        append_fc_layer(age_layers, 128, 1024)
        age_layers.append(nn.Linear(1024, age_out_num))

        self.age_layers = age_layers
        self.age_fc = nn.Sequential(*age_layers)

        if checkPointFilename == None:
            init_weights(self.age_layers)
        #else:
            #self._load_weights(weight_path)

    def forward(self, x):
        batch_size = x.size(0)
        age_y = self.age_fc(x)
        return age_y[:,0]
