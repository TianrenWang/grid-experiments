import torch

from connect4 import ConnectFour
from model import ResNet
from alphazero import AlphaZeroParallel
from device import getTorchDevice

if __name__ == "__main__":
    game = ConnectFour()
    model = ResNet(game, 9, 128, getTorchDevice())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0001)
    args = {
        'C': 2,
        'num_searches': 600,
        'num_iterations': 8,
        'num_selfPlay_iterations': 500,
        'num_parallel_games': 100,
        'num_epochs': 4,
        'batch_size': 128,
        'temperature': 1.25,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.3
    }

    alphaZero = AlphaZeroParallel(model, optimizer, game, args)
    alphaZero.learn()