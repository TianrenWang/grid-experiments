import torch
import torch.nn as nn
import numpy as np
import time

from device import getTorchDevice


class SpeedTestNet(nn.Module):
    def __init__(self, num_hidden, device=torch.device("cpu")):
        super().__init__()

        self.device = device

        self.network = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    device = getTorchDevice()
    print(f"Device being used is:", device)
    tensorSize = 10000
    accelModel = SpeedTestNet(tensorSize, getTorchDevice())
    accelModel.eval()

    cpuModel = SpeedTestNet(tensorSize, torch.device("cpu"))
    cpuModel.eval()

    # Warm-up
    for _ in range(100):
        torch.matmul(torch.rand(500, 500).to(getTorchDevice()),
                     torch.rand(500, 500).to(getTorchDevice()))

    start_time = time.time()
    cpuModel(torch.rand(tensorSize))
    print("CPU : --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    accelModel(torch.rand(tensorSize).to(getTorchDevice()))
    print("DEVICE : --- %s seconds ---" % (time.time() - start_time))
