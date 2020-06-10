"""
Copied from https://github.com/jxhe/vae-lagging-encoder.
"""

import torch
import numpy as np

def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def generate_grid(zmin, zmax, dz, device, ndim=2):
    """generate a 1- or 2-dimensional grid
    Returns: Tensor, int
        Tensor: The grid tensor with shape (k^2, 2),
            where k=(zmax - zmin)/dz
        int: k
    """

    if ndim == 2:
        x = torch.arange(zmin, zmax, dz)
        k = x.size(0)

        x1 = x.unsqueeze(1).repeat(1, k).view(-1)
        x2 = x.repeat(k)

        return torch.cat((x1.unsqueeze(-1), x2.unsqueeze(-1)), dim=-1).to(device), k

    elif ndim == 1:
        return torch.arange(zmin, zmax, dz).unsqueeze(1).to(device)


def word_dropout(G_inp, unk_token, special_tokens, drop=0.5):
    """
    Do word dropout for input sequence (for text vae).
    :param G_inp: (seq_len, batch_size), input sequence.
    :param unk_token: index of unkown token
    :param special_tokens: list of index of special tokens.
    :param drop: dropout keep ratio.
    :return: sequence after dropout.
    """
    r = np.random.rand(G_inp.size(0), G_inp.size(1))
    # Perform word_dropout according to random values (r) generated for each word
    for i in range(len(G_inp)):
        for j in range(1, G_inp.size(1)):
            if r[i, j] < drop and G_inp[i, j] not in special_tokens:
                G_inp[i, j] = unk_token
    return G_inp