import math
import torch.nn as nn


def identity_init(module, bias=0):
    nn.init.eye_(module.weight)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, bias=0):
    std = 1/math.sqrt(module.weight.shape[1])
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)
