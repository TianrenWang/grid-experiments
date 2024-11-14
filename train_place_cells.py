import numpy as np
import torch
import uuid
from models import PlaceCells
from self_eval import saveGameData
from device import getTorchDevice

if __name__ == "__main__":
    numCells = 256
    cellDim = 5376
    placeCells = PlaceCells(
        numCells,
        cellDim,
        100,
        0.0001,
        torch.randn(numCells, cellDim) * 2.5 + 7.5,
    )

    data = np.loadtxt("data/place_cell_training/states.tsv", delimiter="\t")

    batchSize = 32
    iterations = 100

    print(
        f"Starting distance: {placeCells.getTotalDistance(torch.tensor(data).to(torch.float).to(getTorchDevice()))}"
    )

    for i in range(iterations):
        np.random.shuffle(data)
        for j in range(0, len(data), batchSize):
            actualBatchSize = (
                j + batchSize if j + batchSize < len(data) else len(data) - j
            )
            batch = data[j : j + actualBatchSize]
            placeCells.tuneCells(
                torch.tensor(batch).to(torch.float).to(getTorchDevice())
            )

        print(
            f"{i} - Distance: {placeCells.getTotalDistance(torch.tensor(data).to(torch.float).to(getTorchDevice()))}"
        )

    # activations = (
    #     placeCells.getActivations(torch.tensor(data[:1000]).to(torch.float))
    #     .cpu()
    #     .numpy()
    # )
    # print(np.round(np.max(activations, 1), 2))
    # while True:
    #     index = int(input("Enter index of batch: "))
    #     print(np.round(activations[index], 2))

    states = np.loadtxt("data/place_cell_training/states.tsv", delimiter="\t").tolist()

    encounteredStates = set()
    uniqueStates = []

    for state in states:
        stateStr = str(state)
        if stateStr not in encounteredStates:
            uniqueStates.append(state)
            encounteredStates.add(stateStr)

    overlayedStates = placeCells.placeCells.cpu().numpy().tolist() + uniqueStates

    stateLabels = []
    stateDict = {}
    for i in range(len(overlayedStates)):
        stateId = str(uuid.uuid4())[:8]
        currentLabels = ["normal", stateId]
        if i < 256:
            currentLabels = ["place", stateId]
        stateLabels.append(currentLabels)
        stateDict[stateId] = np.array(overlayedStates[i])

    saveGameData(
        overlayedStates, stateLabels, "place_cell_result_cleaned", ["isPlace", "ID"]
    )

    while True:
        found = False
        while not found:
            firstId = input("Enter ID of first state: ")
            if firstId not in stateDict:
                print("ID not found. Enter another one.")
                continue
            secondId = input("Enter ID of second state: ")
            if secondId not in stateDict:
                print("ID not found. Enter another one.")
                continue
            found = True
        state1 = stateDict[firstId]
        state2 = stateDict[secondId]
        distance = np.sum(np.abs(state1 - state2))
        print("Distance is:", distance)
