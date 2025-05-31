import torch
import torch.nn as nn

class DronePolicy(nn.Module):
    def __init__(self, input_dim=8, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)
