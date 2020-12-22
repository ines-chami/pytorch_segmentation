import torch


def reconstruction_loss(x, x_rec):
    x = x.view((x.shape[0], -1))
    x_rec = x_rec.view((x.shape[0], -1))
    return torch.mean(torch.sum((x - x_rec) ** 2, dim=-1), dim=0)


def soft_ncut_loss(x):
    raise NotImplementedError


def hyphc_loss(x):
    raise NotImplementedError
