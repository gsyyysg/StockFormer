from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils
from utils.metrics import mirr_top1, mirr_top_k, rank_ic, mae, mse


class MetricMeter:

    def __init__(self, name: str, func=None):
        self.name = name
        self.data = []
        self.func = func if func is not None else lambda x: x

    def sync_between_processes(self):
        if not utils.is_dist_available_and_initialized():
            return
        world_size = torch.distributed.get_world_size()
        buf = [[] for _ in range(world_size)]
        for chunk in utils.chunks(self.data, 200):  
            t = torch.tensor(chunk, dtype=torch.float64, device='cuda')
            dist.barrier()
            output_tensors = [t.clone() for _ in range(world_size)]
            torch.distributed.all_gather(output_tensors, t)
            for i in range(world_size):
                buf[i].extend(output_tensors[i].tolist())

        concat = buf[0]
        for i in range(1, world_size):
            concat.extend(buf[i])
        self.data = concat

    @property
    def value(self):
        return self.mean()

    def latest(self):
        return self.data[-1] if len(self.data) > 0 else None

    def reset(self):
        self.data = []

    def update(self, *args, **kwargs):
        self.data.append(self.func(*args, **kwargs))

    def mean(self):
        return np.mean(self.data)

    def max(self):
        return np.max(self.data)

    def min(self):
        return np.min(self.data)

    def argmax(self):
        return np.argmax(self.data)

    def argmin(self):
        return np.argmin(self.data)

    def std(self):
        return np.std(self.data)

    def __str__(self):
        return f"{self.mean():.4f}"


class MIRRTop1(MetricMeter):
    def __init__(self, postfix=''):
        if postfix != '':
            postfix = '/' + postfix
        super(MIRRTop1, self).__init__(f"mirr_top1{postfix}", mirr_top1)

    def __str__(self):
        return f"{self.mean():.6f}"

class MAE(MetricMeter):
    def __init__(self, postfix=''):
        if postfix != '':
            postfix = '/' + postfix
        super(MAE, self).__init__(f"mae{postfix}", mae)

    def __str__(self):
        return f"{self.mean():.6f}"

class MSE(MetricMeter):
    def __init__(self, postfix=''):
        if postfix != '':
            postfix = '/' + postfix
        super(MSE, self).__init__(f"mse{postfix}", mse)

    def __str__(self):
        return f"{self.mean():.6f}"


class MIRRTopK(MetricMeter):
    def __init__(self, k: int, postfix=''):
        if postfix != '':
            postfix = '/' + postfix
        super(MIRRTopK, self).__init__(f"mirr_top{k}{postfix}", mirr_top_k)
        self.k = k

    def update(self, *args, **kwargs):
        kwargs['k'] = self.k
        super(MIRRTopK, self).update(*args, **kwargs)

    def __str__(self):
        return f"{self.mean():.6f}"


class RankIC(MetricMeter):
    def __init__(self, postfix=''):
        if postfix != '':
            postfix = '/' + postfix
        super(RankIC, self).__init__(f"rank_ic{postfix}", rank_ic)

    def update(self, *args, **kwargs):
        self.data.extend(self.func(*args, **kwargs))

    def __str__(self):
        return f"{self.mean():.4f} ± {self.std()}"


class NeuralSortLoss(torch.nn.Module):
    def __init__(self, tau=1.0, hard=False):
        super(NeuralSortLoss, self).__init__()
        self.hard = hard
        self.tau = tau

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, reduction='mean'):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        pred = pred.unsqueeze(-1)
        bsize = pred.size()[0]
        dim = pred.size()[1]
        one = torch.ones(dim, 1, device=pred.device)

        A_scores = torch.abs(pred - pred.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(
            one, torch.transpose(one, 0, 1)))
        scaling = (dim + 1 - 2 * (torch.arange(dim, device=pred.device) + 1)).float()

        C = torch.matmul(pred, scaling.unsqueeze(0))

        P_max = (C - B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / self.tau)

        if self.hard:
            P = torch.zeros_like(P_hat, device=pred.device)
            b_idx = torch.arange(bsize, device=P.device).repeat([1, dim]).view(dim, bsize).transpose(dim0=1,
                                                                                                     dim1=0).flatten()
            r_idx = torch.arange(dim, device=P.device).repeat([bsize, 1]).flatten()
            c_idx = torch.argmax(P_hat, dim=-1).flatten()
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P - P_hat).detach() + P_hat

        P_true = torch.argsort(gt, dim=1, descending=True)
        batch_size, num_stocks = gt.shape
        input = P_hat.reshape(batch_size * num_stocks, num_stocks)
        target = P_true.reshape(batch_size * num_stocks)
        return F.nll_loss(input, target, reduction=reduction)