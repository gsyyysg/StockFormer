import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor

from typing import List


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe

def _loss_reduce(loss: Tensor, reduction: str):
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'none':
        return loss
    else:
        raise NotImplementedError(reduction)

def ranking_loss(pred: Tensor, gt: Tensor, reduction: str = 'mean', weight: Tensor = None):
    # assert (batch, num_nodes)
    loss_mat = ranking_loss_matrix(pred, gt)
    loss_mat = F.relu(-loss_mat)
    loss_mat = torch.mean(loss_mat, dim=2)

    if weight is not None:
        loss_mat = loss_mat * weight

    return _loss_reduce(loss_mat, reduction)

def ranking_loss_matrix(pred: Tensor, gt: Tensor) -> Tensor:
    num_nodes = pred.shape[1]
    pred_matrix = pred.reshape(-1, num_nodes, 1)
    gt_matrix = gt.reshape(-1, num_nodes, 1)
    one_vector = torch.ones(num_nodes, 1, device=pred.device, dtype=pred.dtype)
    one_row_vector = one_vector.transpose(0, 1)
    # Mat_\bar{r_i} - Mat_\bar{r_j} (broadcasting in batch dim)
    pred_diff_matrix = torch.matmul(pred_matrix, one_row_vector) \
                       - torch.matmul(one_vector, pred_matrix.transpose(1, 2)).detach()
    # Mat_{r_i} - Mat_{r_j} (broadcasting in batch dim)
    gt_diff_matrix = torch.matmul(gt_matrix, one_row_vector) \
                     - torch.matmul(one_vector, gt_matrix.transpose(1, 2))
    loss_mat = torch.mul(pred_diff_matrix, gt_diff_matrix)
    return loss_mat

def mirr_top_k(prediction: Tensor, gt: Tensor, k: int) -> float:
    pred_topk_index = torch.topk(prediction, k, dim=1)[1]
    mirr = torch.mean(gt[torch.arange(prediction.shape[0]).unsqueeze(1), pred_topk_index]).item()
    return mirr


def mirr_top1(prediction: Tensor, gt: Tensor) -> float:
    pred_top1_index = torch.argmax(prediction, dim=1)
    mirr = torch.mean(gt[torch.arange(prediction.shape[0]), pred_top1_index]).item()
    return mirr

def mae(pred, true):
    return torch.mean(torch.abs(pred-true)).item()

def mse(pred, true):
    return torch.mean((pred-true)**2).item()

def rank_ic(prediction: Tensor, gt: Tensor) -> List[float]:
    rank_gt = torch.argsort(gt, dim=1, descending=True).float()
    rank_pred = torch.argsort(prediction, dim=1, descending=True).float()
    return _compute_rank_ic(rank_pred, rank_gt)

def correlation(a: Tensor, b: Tensor) -> float:
    return covariance(a, b) / (a.std(unbiased=False).item() * b.std(unbiased=False).item())

def covariance(a: Tensor, b: Tensor) -> float:
    ab = torch.mul(a, b)
    return ab.mean().item() - a.mean().item() * b.mean().item()

def _compute_rank_ic(pred: Tensor, gt: Tensor):

    rank_ic_list = []
    for i in range(pred.shape[0]):
        rank_ic = correlation(pred[i], gt[i])
        rank_ic_list.append(rank_ic)
    return rank_ic_list