import torch
import numpy as np
import os

from connect4 import ConnectFour
from model import ResNet
from mcts import MCTS
from device import getTorchDevice


def getHumanReadableState(state):
    return np.where(state == -1, 1, np.where(state == 1, 8, state))


if __name__ == "__main__":
    game = ConnectFour()
    player = 1

    args = {
        'C': 2,
        'num_searches': 600,
        'dirichlet_epsilon': 0.3,
        'dirichlet_alpha': 0.3
    }

    model1 = ResNet(game, 9, 128, getTorchDevice())
    model1.load_state_dict(torch.load(
        "results/model_7_ConnectFour.pt", map_location=getTorchDevice()))
    model1.eval()

    model2 = ResNet(game, 9, 128, getTorchDevice())
    model2.load_state_dict(torch.load(
        "results/model_1_ConnectFour.pt", map_location=getTorchDevice()))
    model2.eval()

    mcts1 = MCTS(game, args, model1)
    mcts2 = MCTS(game, args, model2)

    wins = 0
    losses = 0
    numberOfGames = 0

    while numberOfGames < 20:
        print("Starting game:", str(numberOfGames))
        print("Player 1 wins/losses:", str(wins), "/", str(losses))
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
