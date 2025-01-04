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
        f"Starting distance: {placeCells.getTotalDistance(torch.tensor(data).to(torch.float).to(getTorchDevice())) / len(data)}"
    )

    for i in range(iterations):
        np.random.shuffle(data)
        for j in range(0, len(data), batchSize):
            actualBatchSize = (
                j + batchSize if j + batchSize < len(data) else len(data) - j
            )
            batch = data[j : j + actualBatchSize]
            placeCells.learn(torch.tensor(batch).to(torch.float).to(getTorchDevice()))

        print(
            f"{i} - Distance: {placeCells.getTotalDistance(torch.tensor(data).to(torch.float).to(getTorchDevice())) / len(data)}"
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

    states = np.loadtxt("data/overfitting_testing/states.tsv", delimiter="\t").tolist()

    closestCells = []
    batchSize = 128
    for i in range(0, len(states), batchSize):
        batchEndIndex = i + batchSize if len(states) - i >= batchSize else len(states)
        activations = (
            placeCells.forward(torch.tensor(states[i:batchEndIndex]).to(torch.float))
            .cpu()
            .numpy()
        )
        for activation in activations:
            closestCells.append(activation)

    encounteredStates = set()
    uniqueStates = []

    for state in states:
        stateStr = str(np.sum(state))
        if stateStr not in encounteredStates:
            uniqueStates.append(state)
            encounteredStates.add(stateStr)

    overlayedStates = placeCells.placeCells.cpu().numpy().tolist() + uniqueStates

    stateLabels = []
    stateDict = {}
    for i in range(len(overlayedStates)):
        stateId = str(uuid.uuid4())[:8]
        currentLabels = ["normal", stateId, "", ""]
        if i < 256:
            stateId = str(i)
            frequency = placeCells.learningFrequency[i].item()
            if frequency == 0:
                freqLabel = "zero"
            elif frequency < 10:
                freqLabel = "<10."
            elif frequency < 50:
                freqLabel = "<50."
            elif frequency < 100:
                freqLabel = "<100."
            elif frequency < 500:
                freqLabel = "<500."
            elif frequency < 1000:
                freqLabel = "<1000."
            else:
                freqLabel = ">1000."
            currentLabels = ["place", stateId, freqLabel, "none"]
        else:
            currentLabels[3] = str(closestCells[i - 256])
        stateLabels.append(currentLabels)
        stateDict[stateId] = np.array(overlayedStates[i])

    saveGameData(
        overlayedStates,
        stateLabels,
        "overfitting_overlay",
        ["isPlace", "ID", "freq", "closest"],
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
