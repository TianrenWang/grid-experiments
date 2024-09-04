import torch
import numpy as np
import os

from connect4 import ConnectFour
from model import ResNet
from mcts import MCTS
from device import getTorchDevice


def getHumanReadableState(state):
    return np.where(state == -1, 1, np.where(state == 1, 8, state))


def testAgentVSAgent(version1: int, version2: int):
    print(f"Evaluating Version {version1} VS Version {version2}")
    game = ConnectFour()
    modelPath1 = f"version_{version1}"
    modelPath2 = f"version_{version2}"

    args = {
        'C': 2,
        'num_searches': 0,
        'dirichlet_epsilon': 0.1,
        'dirichlet_alpha': 0.3
    }

    model1 = ResNet(game, 9, 128, getTorchDevice())
    if os.path.exists("results/" + modelPath1):
        model1.load_state_dict(torch.load(
            f"results/{modelPath1}/model.pt", map_location=getTorchDevice(), weights_only=True))
    model1.eval()

    model2 = ResNet(game, 9, 128, getTorchDevice())
    if os.path.exists("results/" + modelPath2):
        model2.load_state_dict(torch.load(
            f"results/{modelPath2}/model.pt", map_location=getTorchDevice(), weights_only=True))
    model2.eval()

    mcts1 = MCTS(game, args, model1)
    mcts2 = MCTS(game, args, model2)

    wins = 0
    losses = 0
    numberOfGames = 0
    numberOfGamesToPlay = 400

    while numberOfGames < numberOfGamesToPlay:
        if numberOfGames < numberOfGamesToPlay / 2:
            player = 1
        else:
            player = -1
        state = game.get_initial_state()
        while True:
            if player == 1:
                neutral_state = game.change_perspective(state, player)
                mcts_probs, _ = mcts1.search(neutral_state)
                action = np.argmax(mcts_probs)
            else:
                neutral_state = game.change_perspective(state, player)
                mcts_probs, _ = mcts2.search(neutral_state)
                action = np.argmax(mcts_probs)

            state = game.get_next_state(state, action, player)

            value, is_terminal = game.get_value_and_terminated(state, action)

            if is_terminal:
                if value == 1:
                    if player == 1:
                        wins += 1
                    else:
                        losses += 1
                break

            player = game.get_opponent(player)
        numberOfGames += 1

    print(f"Version {version1} VS Version {version2} wins/losses:",
          str(wins), "/", str(losses))


if __name__ == "__main__":
    testAgentVSAgent(8, 7)
