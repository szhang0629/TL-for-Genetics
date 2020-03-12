import torch


def penalty_parameters(net):
    """l1 penalty on weight parameters"""
    penalty = 0
    for name, param in net.named_parameters():
        # penalty += torch.sum(torch.mul(param, param))
        if 'bias' not in name:
            penalty += torch.sum(torch.abs(param))
            # penalty += torch.sum(torch.log(torch.abs(param) * 100+1))

    return penalty
