import torch
import torch.nn as nn


class CNN(nn.Module):
    """CNN à 3 blocs convolutifs — biais inductif adapté aux images."""

    def __init__(self, dropout=0.0):
        super().__init__()

        # Bloc 1 : 3×32×32 → 64×16×16
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(64)

        # Bloc 2 : 64×16×16 → 128×8×8
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)

        # Bloc 3 : 128×8×8 → 256×4×4
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(256)

        # Classification
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(256 * 4 * 4, 10)

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.bn1(self.conv1(x))), 2)
        x = torch.max_pool2d(torch.relu(self.bn2(self.conv2(x))), 2)
        x = torch.max_pool2d(torch.relu(self.bn3(self.conv3(x))), 2)
        x = self.drop(x.flatten(1))
        x = self.fc(x)
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)