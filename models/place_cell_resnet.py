import torch.nn.functional as F
import torch.nn as nn
import torch

from place_cells import PlaceCells
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

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
            + [nn.BatchNorm2d(num_hidden)]
        )

        self.placeCells = PlaceCells(
            numCells,
            cellDim,
            field,
            cellLearningRate,
            torch.randn(numCells, cellDim),
        )

        self.placeCellsHead = nn.Sequential(
            *nn.ModuleList([ResBlock(num_hidden) for i in range(3)]),
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, numCells),
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
        if self.training:
            self.placeCells.tuneCells(x)

        with torch.no_grad():
            noGradLatentState = torch.Tensor(x)

        place = self.placeCellsHead(noGradLatentState)

        with torch.no_grad():
            noGradPlace = torch.Tensor(place)

        policy = self.policyHead(torch.concat([self.prePolicyHead(x), noGradPlace], 1))
        value = self.valueHead(torch.concat([self.preValueHead(x), noGradPlace], 1))
        return policy, value, place, noGradLatentState
