import torch
from torch import nn


class maxout(nn.Module):
    def __init__(self, k, input_size, output_size):
        super().__init__()
        self.units = nn.ModuleList([
            nn.Linear(input_size, output_size) for _ in range(k)
        ])

    def forward(self, x):
        x = torch.stack([unit(x) for unit in self.units], dim=1)
        x = torch.max(x, dim=1)[0]
        return x
