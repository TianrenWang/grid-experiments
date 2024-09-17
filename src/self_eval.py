import torch
import numpy as np
import os
import csv
import uuid

from connect4 import ConnectFour
from model import ResNet
from mcts import MCTS
from device import getTorchDevice


def testAgentVSAgent(
    version1: int,
    version2: int,
    randomness: float = 0.1,
    numberOfGamesToPlay: int = 25,
    removeDuplicates: bool = False
):
    print(f"Evaluating Version {version1} VS Version {version2}")
    game = ConnectFour()
    modelPath1 = f"version_{version1}"
    modelPath2 = f"version_{version2}"
    args1 = {
        "C": 2,
        "num_searches": 0,
        "dirichlet_epsilon": randomness,
        "dirichlet_alpha": 0.3,
    }
    args2 = {"C": 2, "num_searches": 0,
             "dirichlet_epsilon": 0, "dirichlet_alpha": 0.3}

    model1 = ResNet(game, 9, 128, getTorchDevice())
    if os.path.exists("results/" + modelPath1):
        model1.load_state_dict(
            torch.load(
                f"results/{modelPath1}/model.pt",
                map_location=getTorchDevice(),
                weights_only=True,
            )
        )
    model1.eval()

    model2 = ResNet(game, 9, 128, getTorchDevice())
    if os.path.exists("results/" + modelPath2):
        model2.load_state_dict(
            torch.load(
                f"results/{modelPath2}/model.pt",
                map_location=getTorchDevice(),
                weights_only=True,
            )
        )
    model2.eval()

    mcts1 = MCTS(game, args1, model1)
    mcts2 = MCTS(game, args2, model2)

    wins = 0
    losses = 0
    numberOfGames = 0

    states = []
    stateLabels = []
    encounteredStates = set()

    while numberOfGames < numberOfGamesToPlay:
        state = game.get_initial_state()
        gameId = uuid.uuid4()
        collectorPlayer = 1
        latentStatesOfCurrentGame = []
        boardStatesOfCurrentGame = []
        encountered = []
        playFirst = []
        if numberOfGames % 2 == 0:
            player = 1
        else:
            player = -1
        playFirstCurrentGame = collectorPlayer == player
        while True:
            if player == collectorPlayer:
                neutral_state = game.change_perspective(state, player)
                mcts_probs, latentState = mcts1.search(neutral_state)
                action = np.argmax(mcts_probs)
                latentStatesOfCurrentGame.append(latentState)
                boardStatesOfCurrentGame.append(state)
                playFirst.append(playFirstCurrentGame)
                if str(state) in encounteredStates:
                    encountered.append(True)
                else:
                    encountered.append(False)
                    encounteredStates.add(str(state))
            else:
                neutral_state = game.change_perspective(state, player)
                mcts_probs, latentState = mcts2.search(neutral_state)
                action = np.argmax(mcts_probs)

            state = game.get_next_state(state, action, player)

            value, is_terminal = game.get_value_and_terminated(state, action)

            if is_terminal:
                if value == 1:
                    if collectorPlayer == player:
                        wins += 1
                    else:
                        losses += 1
                for i in range(len(latentStatesOfCurrentGame)):
                    if encountered[i] and removeDuplicates:
                        continue
                    latentState = latentStatesOfCurrentGame[i]
                    states.append(latentState.numpy().flatten().tolist())
                    if value == 1:
                        if collectorPlayer == player:
                            outcome = "win"
                        else:
                            outcome = "loss"
                    else:
                        outcome = "draw"
                    percentComplete = (i + 1) * \
                        100 // len(latentStatesOfCurrentGame)
                    if percentComplete < 10:
                        percentComplete = "0" + str(percentComplete)
                    stateLabels.append(
                        [
                            f"({i})",
                            f"%{percentComplete}%",
                            str(gameId)[:8],
                            randomness,
                            outcome,
                            playFirst[i],
                        ]
                    )
                break

            player = game.get_opponent(player)
        numberOfGames += 1

    print(
        f"Version {version1} VS Version {version2} wins/losses:",
        str(wins),
        "/",
        str(losses),
    )

    return states, stateLabels


def saveGameData(states, stateLabels, dataName: str):
    folder_path = "data/" + dataName
    os.makedirs(folder_path, exist_ok=True)
    with open(folder_path + "/states.tsv", "a", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows(states)

    hasFirstRow = False
    stateLabelsFilePath = folder_path + "/stateLabels.tsv"
    if os.path.exists(stateLabelsFilePath):
        with open(stateLabelsFilePath, "r") as file:
            firstLine = file.readline()
            hasFirstRow = not firstLine.strip()
    else:
        hasFirstRow = True

    with open(stateLabelsFilePath, "a", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        if hasFirstRow:
            writer.writerow(
                ["move", "progress", "ID", "randomness", "outcome", "first"]
            )
        writer.writerows(stateLabels)


if __name__ == "__main__":
    states, stateLabels = testAgentVSAgent(13, 13, 0.5, 400, True)
    saveGameData(states, stateLabels, "experiment")
