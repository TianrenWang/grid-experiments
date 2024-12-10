import torch.nn.functional as F
import torch.nn as nn
import torch

from .resnet import ResNet, ResBlock

torch.manual_seed(0)


class FCBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.fc = nn.Linear(num_hidden, num_hidden)
        self.bn = nn.BatchNorm1d(num_hidden)

    def forward(self, x):
        return F.relu(self.bn(self.fc(x)))


class PlaceCellResNet(ResNet):
    def __init__(
        self,
        game,
        num_resBlocks,
        num_hidden,
        numCells,
        cellDim,
        field,
        cellLearningRate,
        device,
    ):
        super().__init__(game, num_resBlocks, num_hidden, device)
        self.numCells = numCells
        self.latentNorm = nn.BatchNorm2d(num_hidden)

        self.placeCellsHead = nn.Sequential(
            *nn.ModuleList([ResBlock(num_hidden) for i in range(2)]),
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * game.row_count * game.column_count, numCells),
        )

        self.preValueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.valueHead = nn.Sequential(
            nn.Linear(3 * game.row_count * game.column_count + numCells, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            *nn.ModuleList([FCBlock(256) for i in range(3)]),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        self.prePolicyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.policyHead = nn.Sequential(
            nn.Linear(32 * game.row_count * game.column_count + numCells, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            *nn.ModuleList([FCBlock(256) for i in range(3)]),
            nn.Linear(256, game.action_size),
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        x = self.latentNorm(x)
        place = torch.nn.functional.softmax(self.placeCellsHead(x), dim=1)
        policy = self.policyHead(torch.concat([self.prePolicyHead(x), place], 1))
        value = self.valueHead(torch.concat([self.preValueHead(x), place], 1))
        return policy, value, place, x
