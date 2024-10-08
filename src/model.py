import torch.nn.functional as F
import torch.nn as nn
import torch

from place_cells import PlaceCells

torch.manual_seed(0)


class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()

        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size),
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
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


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


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

        self.placeCells = PlaceCells(
            numCells,
            cellDim,
            field,
            cellLearningRate,
            torch.randn(numCells, cellDim) * 2.5 + 7.5,
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
        place = torch.nn.functional.softmax(place, dim=1)

        with torch.no_grad():
            noGradPlace = torch.Tensor(place)

        policy = self.policyHead(torch.concat([self.prePolicyHead(x), noGradPlace], 1))
        value = self.valueHead(torch.concat([self.preValueHead(x), noGradPlace], 1))
        return policy, value, place, noGradLatentState
