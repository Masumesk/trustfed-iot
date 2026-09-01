import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.gn1 = nn.GroupNorm(8, 32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.gn2 = nn.GroupNorm(8, 64)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.gn3 = nn.GroupNorm(8, 64)

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)          # 32x32 -> 16x16

        x = F.relu(self.gn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)          # 16x16 -> 8x8

        x = F.relu(self.gn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)          # 8x8 -> 4x4

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x