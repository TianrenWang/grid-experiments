import torch.nn.functional as F
import torch.nn as nn
import torch

torch.manual_seed(0)


class NormalizedLatent(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.LayerNorm((num_hidden, game.row_count, game.column_count)),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList(
            [
                LayerResBlock(num_hidden, game.row_count, game.column_count)
                for i in range(num_resBlocks)
            ]
            + [nn.LayerNorm((num_hidden, game.row_count, game.column_count))]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.LayerNorm((32, game.row_count, game.column_count)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.LayerNorm((3, game.row_count, game.column_count)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh(),
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value, x


class LayerResBlock(nn.Module):
    def __init__(self, num_hidden, width, height):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.LayerNorm((num_hidden, width, height))
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.LayerNorm((num_hidden, width, height))

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
