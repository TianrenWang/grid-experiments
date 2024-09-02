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
        'dirichlet_epsilon': 0.,
        'dirichlet_alpha': 0.3
    }

    model = ResNet(game, 9, 128, getTorchDevice())

    modelPath = "alphazero_model.pt"

    if os.path.exists(modelPath):
        model.load_state_dict(torch.load(
            "", map_location=getTorchDevice()))
    model.eval()

    mcts = MCTS(game, args, model)

    state = game.get_initial_state()

    while True:
        print(getHumanReadableState(state))

        if player == 1:
            valid_moves = game.get_valid_moves(state)
            print("valid_moves", [i for i in range(
                game.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("action not valid")
                continue

        else:
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        state = game.get_next_state(state, action, player)

        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break

        player = game.get_opponent(player)
