import torch
import torch.nn as nn

from device import getTorchDevice


class PlaceCells:
    def __init__(
        self,
        numCells: int,
        cellDim: int,
        fieldSize: float,
        learningRate: float,
        relocationThreshold: int = 200,
    ):
        self.numCells = numCells
        self.cellDim = cellDim
        self.learningRate = learningRate
        self.fieldSize = fieldSize
        self.placeCells = nn.Parameter(
            torch.relu(torch.randn(numCells, cellDim) * 2.5 + 7.5), requires_grad=False
        ).to(getTorchDevice())
        # self.relocationThreshold = relocationThreshold
        # self.fireFrequency = nn.Parameter(
        #     torch.zeros(numCells), requires_grad=False)

    def getActivations(self, states: torch.Tensor):
        states = states.to(getTorchDevice())
        diff = states.unsqueeze(1) - self.placeCells.unsqueeze(0)
        dists_squared = torch.sum(torch.abs(diff), dim=-1) ** 2
        unnormalized_activations = -dists_squared / (2 * self.fieldSize**2)
        return torch.nn.functional.softmax(unnormalized_activations, dim=1)

    def train(self, states: torch.Tensor):
        states = states.to(getTorchDevice())
        bestMatchUnit = torch.matmul(states, self.placeCells.T)
        bestMatchUnit = torch.flatten(torch.argmax(bestMatchUnit, 1))
        # firingCounts = torch.bincount(bestMatchUnit, minlength=self.numCells)
        bestMatchVectors = torch.index_select(self.placeCells, 0, bestMatchUnit)
        distances = states - bestMatchVectors
        placeCellUpdates = torch.zeros(self.numCells, self.cellDim).to(getTorchDevice())
        placeCellUpdates = torch.scatter_add(
            placeCellUpdates,
            0,
            bestMatchUnit.unsqueeze(1).expand_as(distances),
            distances,
        ).to(getTorchDevice())

        with torch.no_grad():
            # self.fireFrequency.data.copy_(self.fireFrequency + firingCounts)
            newPlaceCells = self.placeCells + placeCellUpdates * self.learningRate
            self.placeCells.data.copy_(newPlaceCells.to(getTorchDevice()))

    """
    Practically, this function is useless because all place cells are visited at
    very similar frequencies. Not even a magnitude of difference.
    """

    def recalibrate(self):
        mostVisitedIndex = torch.argmax(self.fireFrequency).item()
        leastVisitedIndex = torch.argmin(self.fireFrequency).item()
        mostVisitedCount = self.fireFrequency[mostVisitedIndex].item()
        leastVisitedCount = self.fireFrequency[leastVisitedIndex].item()
        if (
            leastVisitedCount * self.relocationThreshold < mostVisitedCount
            and mostVisitedCount > self.relocationThreshold
        ):
            with torch.no_grad():
                self.placeCells[leastVisitedIndex] = self.placeCells[mostVisitedIndex]

    def evaluate(self, states: torch.Tensor):
        bestMatchUnit = torch.matmul(states, self.placeCells.T)
        bestMatchUnit = torch.flatten(torch.argmax(bestMatchUnit, 1))
        bestMatchVectors = torch.index_select(self.placeCells, 0, bestMatchUnit)
        distances = torch.sum(torch.abs(states - bestMatchVectors), dim=1)
        return torch.sum((torch.relu(distances - self.fieldSize) > 0).int())
