import numpy as np
import torch
import uuid
from models import PlaceCellResNet, PlaceCells
from self_eval import saveGameData, Agent
from device import getTorchDevice
from connect4 import ConnectFour


def overlayCells(states: np.ndarray, placeCells: PlaceCells, dataName: str):
    closestCells = []
    cellActivations = []
    batchSize = 128
    for i in range(0, len(states), batchSize):
        batchEndIndex = i + batchSize if len(states) - i >= batchSize else len(states)
        activations = (
            placeCells.forward(torch.tensor(states[i:batchEndIndex]).to(torch.float))
            .cpu()
            .numpy()
        )
        for activation in activations:
            cellActivations.append(np.max(activation))
        closestCells.append(np.argmax(activations, 1).tolist())
    closestCells = [cell for cells in closestCells for cell in cells]

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
        currentLabels = ["normal", stateId, "", "", ""]
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
            currentLabels = ["place", stateId, freqLabel, stateId, "none"]
        else:
            currentLabels[3] = str(closestCells[i - 256])
            currentLabels[4] = str(cellActivations[i - 256])
        stateLabels.append(currentLabels)
        stateDict[stateId] = np.array(overlayedStates[i])

    saveGameData(
        overlayedStates,
        stateLabels,
        f"{dataName}_overlay_cells",
        ["isPlace", "ID", "frequency", "closest", "activation"],
    )


if __name__ == "__main__":
    version = 14
    expName = "b82fc76d85_cosine_activation"
    dataName = "b82fc76d85_cosine_activation"
    game = ConnectFour()
    model = PlaceCellResNet(game, 9, 128, 256, 5376, 0.15, 0.0001, getTorchDevice())
    agent = Agent(expName, version, model)
    states = np.loadtxt(f"data/{dataName}/states.tsv", delimiter="\t")
    placeCells = model.placeCells
    print(
        f"Starting distance: {placeCells.getTotalDistance(torch.tensor(states).to(torch.float).to(getTorchDevice()))/len(states)}"
    )
    overlayCells(states, placeCells, dataName)
