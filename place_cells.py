import torch
import torch.nn as nn


class PlaceCells:
    def __init__(self, numCells: int, cellDim: int, learningRate: float, relocationThreshold: int):
        self.numCells = numCells
        self.cellDim = cellDim
        self.learningRate = learningRate
        self.relocationThreshold = relocationThreshold
        self.placeCells = nn.Parameter(torch.relu(torch.randn(
            numCells, cellDim) * 2.5 + 7.5), requires_grad=False)
        self.fireFrequency = nn.Parameter(
            torch.zeros(numCells), requires_grad=False)

    def train(self, states: torch.Tensor):
        bestMatchUnit = torch.matmul(states, self.placeCells.T)
        bestMatchUnit = torch.flatten(torch.argmax(bestMatchUnit, 1))
        firingCounts = torch.bincount(bestMatchUnit, minlength=self.numCells)
        bestMatchVectors = torch.gather(self.placeCells, 0, bestMatchUnit)
        distances = states - bestMatchVectors
        placeCellUpdates = torch.zeros(self.numCells, self.cellDim)
        placeCellUpdates = torch.scatter_add(placeCellUpdates,
                                             0, bestMatchUnit.unsqueeze(1).expand_as(distances), distances)

        with torch.no_grad():
            self.fireFrequency.data.copy_(self.fireFrequency + firingCounts)
            self.placeCells.data.copy_(
                self.placeCells + placeCellUpdates * self.learningRate)

    def recalibrate(self):
        mostVisitedIndex = torch.argmax(self.fireFrequency).item()
        leastVisitedIndex = torch.argmin(self.fireFrequency).item()
        mostVisitedCount = self.fireFrequency[mostVisitedIndex].item()
        leastVisitedCount = self.fireFrequency[leastVisitedIndex].item()
        if leastVisitedCount * self.relocationThreshold < mostVisitedCount and mostVisitedCount > self.relocationThreshold:
            with torch.no_grad():
                self.placeCells[leastVisitedIndex] = self.placeCells[mostVisitedIndex]
