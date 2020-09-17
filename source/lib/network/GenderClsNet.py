import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.network.utils import *
from config import *

class GenderClsNet(nn.Module):
    def __init__(self, in_num, gender_out_num, checkPointFilename=None):
        super(GenderClsNet, self).__init__()
        self.in_num = in_num
        self.gender_out_num = gender_out_num
        gender_layers = []

        self.first_layer = append_fc_layer(gender_layers, in_num, 2048)
        append_fc_layer(gender_layers, 2048, 128)
        append_fc_layer(gender_layers, 128, 1024)
        gender_layers.append(nn.Linear(1024, gender_out_num))

        self.gender_layers = gender_layers
        self.gender_fc = nn.Sequential(*gender_layers)

        if checkPointFilename == None:
            init_weights(self.gender_layers)
        #else:
            #self._load_weights(weight_path)

    def forward(self, x):
        batch_size = x.size(0)
        gender_y = self.gender_fc(x)
        gender_y = F.softmax(gender_y, dim=0)
        return gender_y
