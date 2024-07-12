import torch
import torch.nn.functional as F


class Steerer(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.nn.Parameter(weights.detach())

    def forward(self, x):
        x = x.clone().detach().requires_grad_(False)
        return F.linear(x, self.weights)
