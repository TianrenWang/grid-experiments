import numpy as np
import torch
import csv

from models import PlaceCells


def findFiringCells(dataPath: str, numCells: int, cellDim: int, firingField: float):
    states = np.loadtxt(f"{dataPath}/states.tsv", delimiter="\t")
    placeCells = PlaceCells(
        numCells, cellDim, firingField, 0.0001, torch.Tensor(states[:numCells])
    )

    stateIndices = {}
    stateIDs = []
    with open(f"{dataPath}/stateLabels.tsv", newline="", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        rowIndex = -1
        for row in reader:
            if rowIndex >= 0:
                stateIndices[row[1]] = rowIndex
                stateIDs.append(row[1])
            rowIndex += 1

    while True:
        print("*********New Search*********")
        found = False
        while not found:
            id = input("Enter ID of state: ")
            if id not in stateIndices:
                print("ID not found. Enter another one.")
                continue
            found = True
        stateIndex = stateIndices[id]
        stateTensor = np.reshape(states[stateIndex], (1, 5376))
        activation = placeCells.getActivations(torch.Tensor(stateTensor)).cpu().numpy()

        for i in range(numCells):
            if activation[0][i] > 0.01:
                print(f"{stateIDs[i]} - activation: {activation[0][i]}")


if __name__ == "__main__":
    findFiringCells("data/place_cell_result_cleaned", 256, 5376, 50)
