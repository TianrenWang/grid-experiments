import torch.nn.functional as F
import torch.nn as nn
import torch

from .resnet import ResNet

torch.manual_seed(0)


class FCBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.fc = nn.Linear(num_hidden, num_hidden)
        self.bn = nn.BatchNorm1d(num_hidden)

    def forward(self, x):
        return F.relu(self.bn(self.fc(x)))


class PathIntegrator(ResNet):
    def __init__(self, game, num_resBlocks, num_hidden, device, memorySize):
        super().__init__(game, num_resBlocks, num_hidden, device)

        self.trajectoryMemory = nn.LSTM(
            game.action_size + 1, memorySize, 1, batch_first=True
        )

        self.spatialProjection = nn.Sequential(
            nn.Dropout(), nn.Linear(memorySize, memorySize)
        )

        self.integratorHead = nn.Linear(
            memorySize, num_hidden * game.row_count * game.column_count
        )

        self.preValueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.valueHead = nn.Sequential(
            nn.Linear(3 * game.row_count * game.column_count + memorySize, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
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
            nn.Linear(32 * game.row_count * game.column_count + memorySize, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, game.action_size),
        )

        self.to(device)

    def forward(
        self,
        boardStates: torch.Tensor,
        pastMoves: torch.Tensor,
    ):
        boardStates = self.startBlock(boardStates)
        for resBlock in self.backBone:
            boardStates = resBlock(boardStates)
        pastMovesOneHot = (
            torch.nn.functional.one_hot(pastMoves, 8).float().to(self.device)
        )
        integratedMasks = torch.sum((pastMoves < 7).int(), 1).to(self.device)

        integratedOutputs, (_, __) = self.trajectoryMemory(pastMovesOneHot)
        spatialProjection = self.spatialProjection(
            integratedOutputs[torch.arange(integratedOutputs.shape[0]), integratedMasks]
        )

        with torch.no_grad():
            noGradProjection = torch.Tensor(spatialProjection)

        policy = self.policyHead(
            torch.concat([self.prePolicyHead(boardStates), noGradProjection], 1)
        )
        value = self.valueHead(
            torch.concat([self.preValueHead(boardStates), noGradProjection], 1)
        )
        return policy, value, boardStates, self.integratorHead(spatialProjection)
