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
        relocationThreshold: int = 10,
    ):
        super().__init__()
        self.numCells = numCells
        self.cellDim = cellDim
        self.learningRate = learningRate
        self.fieldSize = fieldSize
        self.register_buffer("placeCells", cellPositions.to(getTorchDevice()))
        self.relocationThreshold = relocationThreshold
        self.register_buffer(
            "learningFrequency",
            torch.zeros(numCells, dtype=torch.int16).to(getTorchDevice()),
        )
        self.register_buffer(
            "coverageFrequency",
            torch.zeros(numCells, dtype=torch.int16).to(getTorchDevice()),
        )

    def forward(self, x):
        states = torch.reshape(x.to(getTorchDevice()), (-1, self.cellDim))
        return torch.argmax(torch.matmul(states, self.placeCells.T), 1)

    def learn(self, states: torch.Tensor, useMean: bool = False, droprate: float = 0):
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
        bestMatchVectors = torch.index_select(self.placeCells, 0, bestMatchUnit)
        displacements = states - bestMatchVectors
        placeCellUpdates = torch.zeros(self.numCells, self.cellDim).to(getTorchDevice())
        placeCellUpdates = torch.scatter_reduce(
            placeCellUpdates,
            0,
            bestMatchUnit.unsqueeze(1).expand_as(displacements),
            displacements,
            reduce="mean" if useMean else "sum",
        ).to(getTorchDevice())

        with torch.no_grad():
            self._updateFrequencies(displacements, bestMatchUnit)
            newPlaceCells = self.placeCells + placeCellUpdates * (
                1 if useMean else self.learningRate
            )
            self.placeCells = newPlaceCells.to(getTorchDevice())

    def calibrate(self):
        mostVisitedIndex = torch.argmax(self.learningFrequency).item()
        cellUsefulness = self.learningFrequency + self.coverageFrequency**2
        leastUsedIndex = torch.argmin(cellUsefulness).item()
        mostVisitedFreqCount = self.learningFrequency[mostVisitedIndex].item()
        mostVisitedCovCount = self.coverageFrequency[mostVisitedIndex].item()
        leastUsedCount = cellUsefulness[leastUsedIndex].item()
        if leastUsedCount == 0:
            with torch.no_grad():
                self.placeCells[leastUsedIndex] = (
                    self.placeCells[mostVisitedIndex]
                    + torch.randn(self.cellDim, device=getTorchDevice()) * 1e-7
                )
                halfLearningFrequency = mostVisitedFreqCount / 2
                self.learningFrequency[leastUsedIndex] = halfLearningFrequency
                self.learningFrequency[mostVisitedIndex] = halfLearningFrequency
                halfCoverageFrequency = mostVisitedCovCount / 2
                self.coverageFrequency[leastUsedIndex] = halfCoverageFrequency
                self.coverageFrequency[mostVisitedIndex] = halfCoverageFrequency

    def getTotalDistance(self, states: torch.Tensor):
        bestMatchUnit = torch.matmul(states, self.placeCells.T)
        bestMatchUnit = torch.flatten(torch.argmax(bestMatchUnit, 1))
        bestMatchVectors = torch.index_select(self.placeCells, 0, bestMatchUnit)
        return torch.sum(torch.abs(states - bestMatchVectors))

    def resetFireFrequency(self):
        self.learningFrequency = torch.zeros(self.numCells, dtype=torch.int16).to(
            getTorchDevice()
        )
        self.coverageFrequency = torch.zeros(self.numCells, dtype=torch.int16).to(
            getTorchDevice()
        )

    def getDistances(self, states: torch.Tensor):
        states = torch.reshape(states.to(getTorchDevice()), (-1, self.cellDim))
        diff = states.unsqueeze(1) - self.placeCells.unsqueeze(0)
        return torch.sum(torch.abs(diff), dim=-1)

    def _updateFrequencies(self, distances: torch.Tensor, bestMatchUnit: torch.Tensor):
        withinField = torch.where(
            torch.sum(torch.abs(distances), 1) < self.fieldSize,
            1,
            0,
        )
        learningCounts = torch.bincount(
            bestMatchUnit.to(torch.int32),
            minlength=self.numCells,
            weights=(1 - withinField),
        )
        coverageCounts = torch.bincount(
            bestMatchUnit.to(torch.int32), minlength=self.numCells, weights=withinField
        )

        with torch.no_grad():
            self.learningFrequency = self.learningFrequency + learningCounts
            self.coverageFrequency = self.coverageFrequency + coverageCounts

    def countFrequencies(self, states: torch.Tensor):
        states = torch.reshape(states.to(getTorchDevice()), (-1, self.cellDim))
        bestMatchUnit = torch.matmul(states, self.placeCells.T)
        bestMatchUnit = torch.flatten(torch.argmax(bestMatchUnit, 1))
        bestMatchVectors = torch.index_select(self.placeCells, 0, bestMatchUnit)
        distances = torch.abs(states - bestMatchVectors)
        self._updateFrequencies(distances, bestMatchUnit)
