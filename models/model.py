import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1=nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3
        )

        self.conv2=nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3
        )

        self.fc1=nn.Linear(32*5*5,10)

    def forward(self,x):

        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x,2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x=torch.flatten(x,1)

        x=self.fc1(x)

        return  x