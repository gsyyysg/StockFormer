import copy
import torch
from torch import nn
import sys

sys.path.append('StockFormer/Transformer')
import config

import pdb


def clone_module(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, device):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        bs = query.shape[0]

        Q = self.fc_q(query)  # (bs, q_len, d_model)
        K = self.fc_k(key)    # (bs, k_len, d_model)
        V = self.fc_v(value)  # (bs, v_len, d_model)

        Q = Q.view(bs, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (bs, n_heads, src_len, head_dim)
        K = K.view(bs, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (bs, n_heads, src_len, head_dim)
        V = V.view(bs, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (bs, n_heads, src_len, head_dim)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # (bs, n_heads, query_len, key_len)
        energy = self.dropout(energy)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)    # (bs, n_heads, seq_len, seq_len)
        x = torch.matmul(attention, V)  # (bs, n_heads, seq_len, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()    # (bs, seq_len, n_heads, head_dim)
        x = x.view(bs, -1, self.d_model)   # (bs, seq_len, d_model)
        x = self.fc_o(x)     # x: (bs, seq_len, d_model)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim=512, dropout=0.1, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        # x = self.dropout(torch.relu(self.fc_1(x)))
        # x = self.fc_2(x)
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        x = self.conv2(x).transpose(-1, 1)
        return x

class MultiheadFeedForward(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.mhfw = nn.ModuleList([FeedForward(d_model=self.head_dim, ff_dim=ff_dim, dropout=dropout) for i in range(self.n_heads)])

    def forward(self, x): # [bs, seq_len, d_model]
        # pdb.set_trace()
        bs = x.shape[0]
        input = x.reshape(bs, -1, self.n_heads, self.head_dim) # [bs, seq_len, n_heads, head_dim]
        outputs = []
        for i in range(self.n_heads):
            outputs.append(self.mhfw[i](input[:, :, i, :])) # [bs, seq_len, head_dim]
        outputs = torch.cat(outputs, dim=-2).reshape(bs, -1, self.d_model) # [bs, seq_len, n_heads, head_dim]
        return outputs

