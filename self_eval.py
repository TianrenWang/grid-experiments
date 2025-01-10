import torch
import numpy as np
import os
import csv
import uuid
import shutil

from connect4 import ConnectFour
from models import ResNet, PlaceCellResNet
from mcts import MCTS
from device import getTorchDevice

game = ConnectFour()


class Agent:
    def __init__(
        self,
        expName: str,
        version: int,
        model: torch.nn.Module,
        loadModel: bool = True,
    ):
        self.expName = expName
        self.version = version
        self.model = model
        if loadModel:
            self.modelPath = f"{expName}/version_{version}"
            if os.path.exists("results/" + self.modelPath):
                self.model.load_state_dict(
                    torch.load(
                        f"results/{self.modelPath}/model.pt",
                        map_location=getTorchDevice(),
                        weights_only=True,
                    )
                )
        self.model.eval()
        self.args = {
            "C": 2,
            "num_searches": 0,
            "dirichlet_epsilon": 0,
            "dirichlet_alpha": 0.3,
        }
        self.mcts = MCTS(game, self.args, self.model)


def getFinalizedAction(probs: np.ndarray, temperature: float):
    temperature_action_probs = probs ** (1 / temperature)
    temperature_action_probs /= temperature_action_probs.sum()
    return np.random.choice(game.action_size, p=temperature_action_probs)


def testAgentVSAgent(
    agent1: Agent,
    agent2: Agent = Agent("control", 13, ResNet(game, 9, 128, getTorchDevice())),
    temperature: float = 1,
    numberOfGamesToPlay: int = 25,
    removeDuplicates: bool = False,
):
    global game
    print(
        f"Evaluating {agent1.expName} Version {agent1.version} VS {agent2.expName} Version {agent2.version}"
    )

    wins = 0
    losses = 0
    numberOfGames = 0

    gamePositions = dict()
    latentStates = []
    stateLabels = []
    encounteredStates = set()
    actionConfidence = []

    while numberOfGames < numberOfGamesToPlay:
        currentGameConfidence = []
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
                mcts_probs, latentState = agent1.mcts.search(neutral_state)
                action = getFinalizedAction(mcts_probs, temperature)
                currentGameConfidence.append(mcts_probs[action])
                latentStatesOfCurrentGame.append(latentState)
                boardStatesOfCurrentGame.append(neutral_state)
                playFirst.append(playFirstCurrentGame)
                if str(state) in encounteredStates:
                    encountered.append(True)
                else:
                    encountered.append(False)
                    encounteredStates.add(str(state))
            else:
                neutral_state = game.change_perspective(state, player)
                mcts_probs, latentState = agent2.mcts.search(neutral_state)
                action = getFinalizedAction(mcts_probs, temperature)

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
                    positionId = str(uuid.uuid4())[:8]
                    latentState = latentStatesOfCurrentGame[i]
                    latentStates.append(latentState.cpu().numpy().flatten().tolist())
                    gamePositions[positionId] = boardStatesOfCurrentGame[i]
                    if value == 1:
                        if collectorPlayer == player:
                            outcome = "win"
                        else:
                            outcome = "loss"
                    else:
                        outcome = "draw"
                    percentComplete = (i + 1) * 100 // len(latentStatesOfCurrentGame)
                    if percentComplete < 10:
                        percentComplete = "0" + str(percentComplete)
                    stateLabels.append(
                        [
                            f"({i})",
                            f"%{percentComplete}%",
                            str(gameId)[:8],
                            temperature,
                            outcome,
                            playFirst[i],
                            positionId,
                        ]
                    )
                break

            player = game.get_opponent(player)
        numberOfGames += 1
        actionConfidence.append(currentGameConfidence)

    print(
        f"{agent1.expName} Version {agent1.version} VS {agent2.expName} Version {agent2.version} wins/losses:",
        str(wins),
        "/",
        str(losses),
    )

    totalConfidence = 0
    for moveConfidences in actionConfidence:
        totalGameConfidence = 0
        for i in range(len(moveConfidences)):
            totalGameConfidence += moveConfidences[i]
        totalConfidence += totalGameConfidence / len(moveConfidences)
    print(f"Average Confidence: {totalConfidence / len(actionConfidence)}")

    return gamePositions, latentStates, stateLabels


def saveGameData(
    states,
    stateLabels,
    dataName: str,
    columnNames=[
        "move",
        "progress",
        "game ID",
        "randomness",
        "outcome",
        "first",
        "position ID",
    ],
):
    folder_path = "data/" + dataName
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
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
            writer.writerow(columnNames)
        writer.writerows(stateLabels)


if __name__ == "__main__":
    # model = ResNet(game, 9, 128, getTorchDevice())
    model = PlaceCellResNet(game, 9, 128, 256, 5376, 100, 0.01, getTorchDevice())
    agent = Agent("b82fc76d85_cosine_activation", 14, model)
    gamePositions, latentStates, stateLabels = testAgentVSAgent(
        agent, agent, numberOfGamesToPlay=400, removeDuplicates=True
    )
    saveGameData(latentStates, stateLabels, "b82fc76d85_cosine_activation")

    visualizePositions = False
    while visualizePositions:
        found = False
        while not found:
            id = input("Enter ID of position: ")
            if id not in gamePositions:
                print("ID not found. Enter another one.")
                continue
            found = True
        position = gamePositions[id]

        symbols = {
            -1: "O",  # Opponent
            0: ".",  # Empty
            1: "X",  # Current player
        }

        # Print the board
        print(" +---------------+")
        for row in position:
            print(" | " + " ".join(symbols[val] for val in row) + " |")
        print(" +---------------+")
