import torch


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x
