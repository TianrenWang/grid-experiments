import torch
import os

from connect4 import ConnectFour
from models import ResNet, PlaceCellResNet, NormalizedLatent  # noqa: F401
from alphazero import AlphaZeroParallel
from device import getTorchDevice

if __name__ == "__main__":
    game = ConnectFour()

    previousVersion = None
    experimentName = "place_cells"

    # model = ResNet(game, 9, 128, getTorchDevice())
    model = PlaceCellResNet(game, 9, 128, 256, 5376, 100, 0.01, getTorchDevice())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        "C": 2,
        "num_searches": 600,
        "num_iterations": 8,
        "num_selfPlay_iterations": 500,
        "num_parallel_games": 100,
        "num_epochs": 16,
        "batch_size": 128,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3,
        "prev_version": previousVersion,
        "exp_name": experimentName,
        "memory": None,
    }

    modelPath = f"results/{experimentName}/version_{previousVersion}"
    if os.path.exists(modelPath):
        model.load_state_dict(
            torch.load(
                f"{modelPath}/model.pt",
                map_location=getTorchDevice(),
                weights_only=True,
            )
        )
        optimizer.load_state_dict(
            torch.load(
                f"{modelPath}/optimizer.pt",
                map_location=getTorchDevice(),
                weights_only=True,
            )
        )

    alphaZero = AlphaZeroParallel(model, optimizer, game, args)
    alphaZero.learn()
