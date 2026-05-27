import torch
import torch.nn as nn


class MLP(nn.Module):
    """Perceptron multicouche baseline — 3072 → 1024 → 512 → 256 → 10."""

    def __init__(self, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 10)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.flatten(1)
        x = self.drop(torch.relu(self.fc1(x)))
        x = self.drop(torch.relu(self.fc2(x)))
        x = self.drop(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)