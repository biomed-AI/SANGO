import torch
from torch import Tensor
import torch.nn as nn
from typing import Iterable, Optional
from ECA_layer import eca_layer

ONEHOT = torch.cat((
    torch.ones(1, 4) / 4, # 
    torch.eye(4), # A, C, G, T
    torch.zeros(1, 4), # padding
), dim=0).float()

class ConvTower(nn.Module):
    def __init__(self, in_channel, out_channel: int, kernel_size) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size, padding=kernel_size//2, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size, padding=kernel_size//2, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.eca = eca_layer(out_channel)

        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size//2)

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channel),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x

        # conv1
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        # conv2 + eca
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.eca(y)

        residual = self.downsample(residual)

        y += residual
        y = self.maxpool(y)
        y = self.relu(y)

        return y

class CACNN(nn.Module):
    def __init__(self, n_cells: int, batch_ids: Optional[Iterable[int]]=None, use_reg_cell=False, hidden_size=32, seq_len: int=1344):
        super().__init__()
        self.config = {
            "n_cells": n_cells,
            "hidden_size": hidden_size,
            "seq_len": seq_len
        }
        if batch_ids is None:
            self.batch_ids = None
        else:
            self.batch_embedding = nn.Embedding(max(batch_ids) + 1, hidden_size)
            self.batch_ids = nn.Parameter(torch.as_tensor(batch_ids), requires_grad=False)
            assert self.batch_ids.ndim == 1
        self.onehot = nn.Parameter(ONEHOT, requires_grad=False)
        self.seq_len = seq_len
        self.use_reg_cell = use_reg_cell

        # 1
        current_len = seq_len
        self.pre_conv = nn.Sequential( # input: (batch_size, 4, seq_len)
            nn.Conv1d(4, out_channels=288, kernel_size=17, padding=8),
            nn.BatchNorm1d(288),
            nn.MaxPool1d(kernel_size=3), # output: (batch_size, 288, 448)
            nn.ReLU(),
        )
        current_len = current_len // 3

        # 2
        self.conv_towers = []
        self.conv_towers.append(ConvTower(288, 64, 5))
        current_len = current_len // 2
        self.conv_towers.append(ConvTower(64, 128, 5))
        current_len = current_len // 2
        self.conv_towers.append(ConvTower(128, 256, 5))
        current_len = current_len // 2
        self.conv_towers.append(ConvTower(256, 512, 5))
        current_len = current_len // 2
        self.conv_towers = nn.Sequential(*self.conv_towers)

        # 3
        self.post_conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(), # (batch_size, 256, 7)
        )
        current_len = current_len // 1

        # 4
        self.flatten = nn.Flatten() # (batch_size, 1792)

        current_len = current_len * 256

        # 5
        self.dense = nn.Sequential(
            nn.Linear(current_len, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 6 
        self.cell_embedding = nn.Linear(hidden_size, n_cells)
    
    def get_embedding(self):
        return self.cell_embedding.state_dict()["weight"]
    
    
    def forward(self, sequence: Tensor) -> Tensor:
        sequence = self.onehot[sequence.long()].transpose(1, 2)
        sequence = self.pre_conv(sequence)
        sequence = self.conv_towers(sequence)
        sequence = self.post_conv(sequence)
        sequence = self.flatten(sequence)
        sequence = self.dense(sequence) # (B, hidden_size)
        logits = self.cell_embedding(sequence)
        if(self.use_reg_cell):
            lr_reg_cell = torch.norm(self.cell_embedding.weight, p=2) + torch.norm(self.cell_embedding.bias, p=2)
        else:
            lr_reg_cell = None
        return logits, lr_reg_cell
