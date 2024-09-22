import numpy as np
import torch
import uuid
from place_cells import PlaceCells
from self_eval import saveGameData

if __name__ == "__main__":
    """
    Changing the number of cells doesn't significantly improve coverage. I think
    a strictly linear distance loss is insufficient.
    """
    placeCells = PlaceCells(256, 5376, 50, 0.0001, 200)

    data = np.loadtxt("data/place_cell_training/states.tsv", delimiter="\t")

    batchSize = 32
    iterations = 100

    print(f"Starting miss: {placeCells.evaluate(torch.tensor(data).to(torch.float))}")

    for i in range(iterations):
        np.random.shuffle(data)
        for j in range(0, len(data), batchSize):
            actualBatchSize = (
                j + batchSize if j + batchSize < len(data) else len(data) - j
            )
            batch = data[j : j + actualBatchSize]
            placeCells.train(torch.tensor(batch).to(torch.float))

        print(
            f"{i} - Current miss: {placeCells.evaluate(torch.tensor(data).to(torch.float))}"
        )

    states = np.loadtxt("data/place_cell_training/states.tsv", delimiter="\t").tolist()

    encounteredStates = set()
    uniqueStates = []

    for state in states:
        stateStr = str(state)
        if stateStr not in encounteredStates:
            uniqueStates.append(state)
            encounteredStates.add(stateStr)

    overlayedStates = placeCells.placeCells.numpy().tolist() + uniqueStates

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
        distance = np.linalg.norm(state1 - state2)
        print("Distance is:", distance)
