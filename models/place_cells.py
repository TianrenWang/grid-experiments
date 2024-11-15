import torch
import torch.nn as nn

from device import getTorchDevice


class PlaceCells(nn.Module):
    def __init__(
        self,
        numCells: int,
        cellDim: int,
        fieldSize: float,
        learningRate: float,
        cellPositions: torch.Tensor,
        relocationThreshold: int = 200,
    ):
        super().__init__()
        self.numCells = numCells
        self.cellDim = cellDim
        self.learningRate = learningRate
        self.fieldSize = fieldSize
        self.register_buffer("placeCells", cellPositions.to(getTorchDevice()))
        self.relocationThreshold = relocationThreshold
        self.register_buffer(
            "fireFrequency",
            torch.zeros(numCells, dtype=torch.int16).to(getTorchDevice()),
        )

    def forward(self, x):
        states = torch.reshape(x.to(getTorchDevice()), (-1, self.cellDim))
        diff = states.unsqueeze(1) - self.placeCells.unsqueeze(0)
        dists_squared = torch.sum(torch.abs(diff), dim=-1) ** 2
        unnormalized_activations = -dists_squared / (2 * self.fieldSize**2)
        return torch.nn.functional.softmax(unnormalized_activations, dim=1)

    def tuneCells(self, states: torch.Tensor, droprate: float = 0):
        states = torch.reshape(states.to(getTorchDevice()), (-1, self.cellDim))
        cellMask = torch.where(
            torch.rand(
                self.numCells,
                1,
                device=getTorchDevice(),
                dtype=torch.float32,
            )
            > droprate,
            1,
            0,
        )
        droppedCells = self.placeCells * cellMask
        bestMatchUnit = torch.matmul(states, droppedCells.T)
        bestMatchUnit = torch.flatten(torch.argmax(bestMatchUnit, 1))
        firingCounts = torch.bincount(bestMatchUnit, minlength=self.numCells)
        bestMatchVectors = torch.index_select(self.placeCells, 0, bestMatchUnit)
        displacements = states - bestMatchVectors
        placeCellUpdates = torch.zeros(self.numCells, self.cellDim).to(getTorchDevice())
        placeCellUpdates = torch.scatter_add(
            placeCellUpdates,
            0,
            bestMatchUnit.unsqueeze(1).expand_as(displacements),
            displacements,
        ).to(getTorchDevice())

        with torch.no_grad():
            self.fireFrequency = self.fireFrequency + firingCounts
            newPlaceCells = self.placeCells + placeCellUpdates * self.learningRate
            self.placeCells = newPlaceCells.to(getTorchDevice())

    def calibrate(self):
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
                halfFrequency = mostVisitedCount / 2
                self.fireFrequency[leastVisitedIndex] = halfFrequency
                self.fireFrequency[mostVisitedIndex] = halfFrequency

    def getTotalDistance(self, states: torch.Tensor):
        bestMatchUnit = torch.matmul(states, self.placeCells.T)
        bestMatchUnit = torch.flatten(torch.argmax(bestMatchUnit, 1))
        bestMatchVectors = torch.index_select(self.placeCells, 0, bestMatchUnit)
        return torch.sum(torch.abs(states - bestMatchVectors))
