import torch
from torch import nn

relu = nn.ReLU()
def Prox_Squared_L2_with_inner_product(input, inner_product, hyper_parameter, step_size):
    alpha = hyper_parameter
    sigma = step_size
    coef = 1./(1.+relu(sigma/alpha))
    return coef * (input - sigma * inner_product)

def Prox_Dual_L1(input, inner_product, hyper_parameter, step_size):
    argument = input-step_size*inner_product
    return torch.where(torch.abs(argument)>hyper_parameter, torch.sign(argument)*hyper_parameter, argument)
