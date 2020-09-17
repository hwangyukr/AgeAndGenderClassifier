from torch import nn
from config import *

def init_weights(layer_list):
    for layer in layer_list:
        normal_init(layer, 0, STDDEV)
        layer.requires_grad_(True)

def append_fc_layer(arr, in_num, out_num):
    ret = nn.Linear(in_num, out_num)
    arr.append(ret)
    arr.append(nn.BatchNorm1d(out_num))
    arr.append(nn.ReLU())
    arr.append(nn.Dropout(0.5))
    return ret

def normal_init(m, mean, stddev):
    if(type(m) == nn.ReLU): return
    if(type(m) == nn.Sigmoid): return
    if(type(m) == nn.Dropout): return
    if(type(m) == nn.Softmax): return
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()

def getLayerAttention(layer):
    layer = layer.weight
    in_num = layer.size(1)
    attention = np.zeros(in_num)
    for idx in range(in_num):
        attention[idx] = layer[:,idx].abs().sum()
    return attention
