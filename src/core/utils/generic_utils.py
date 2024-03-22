'''
Created on Nov, 2018

@author: hugo

'''
import yaml
import numpy as np
import networkx as nx
from collections import defaultdict
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import VERY_SMALL_NUMBER, INF


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x

def create_mask(x, N, device=None):
    if isinstance(x, torch.Tensor):
        x = x.data
    mask = np.zeros((len(x), N))
    for i in range(len(x)):
        mask[i, :x[i]] = 1
    return to_cuda(torch.Tensor(mask), device)

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config

def syn_normalize_adj_torch(mx : torch.FloatTensor):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


def batch_normalize_adj(mx, mask=None):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    # mx: shape: [batch_size, N, N]

    # strategy 1)
    # rowsum = mx.sum(1)
    # r_inv_sqrt = torch.pow(rowsum, -0.5)
    # r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0. # I got this error: copy_if failed to synchronize: device-side assert triggered

    # strategy 2)
    rowsum = torch.clamp(mx.sum(1), min=VERY_SMALL_NUMBER)
    r_inv_sqrt = torch.pow(rowsum, -0.5)
    if mask is not None:
        r_inv_sqrt = r_inv_sqrt * mask

    r_mat_inv_sqrt = []
    for i in range(r_inv_sqrt.size(0)):
        r_mat_inv_sqrt.append(torch.diag(r_inv_sqrt[i]))

    r_mat_inv_sqrt = torch.stack(r_mat_inv_sqrt, 0)
    return torch.matmul(torch.matmul(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)

def sym_normalize_sparse_adj_scipy(mx_sp) -> sp.csr_matrix:
    """Symmetric normalize the given sparse adj. matrix."""

    mx_sp : sp.csr_matrix = mx_sp.tocsr()
    rowsum = np.array(mx_sp.sum(1, dtype=np.float32), dtype=np.float32)

    # ignore "divide by zero" warning for the isolated nodes
    with np.errstate(divide='ignore'):
        r_inv_sqrt = np.power(rowsum, -0.5, dtype=np.float32).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.

    r_mat_inv_sqrt = sp.diags(r_inv_sqrt, format='csr', dtype=np.float32)
    return r_mat_inv_sqrt.dot(mx_sp).dot(r_mat_inv_sqrt)

def to_undirected(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max() + 1
    else:
        num_nodes = max(num_nodes, edge_index.max() + 1)

    data_type = np.int8
    row, col = edge_index

    vec_of_ones = np.ones(edge_index.shape[1], dtype=data_type)
    adj = sp.csr_matrix((vec_of_ones, (row, col)), shape=(num_nodes, num_nodes), 
                                                    dtype=data_type)

    # make the matrix `adj` symmetric
    adj = (adj + adj.transpose()) > 0

    return adj.astype(data_type)          # use int8 instead of float64

def csr_sp_scipy_2_torch(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx : sp.csr_matrix = sparse_mx.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(crow_indices=sparse_mx.indptr, 
                                   col_indices=sparse_mx.indices,
                                   values=sparse_mx.data,
                                   dtype=torch.float32)
