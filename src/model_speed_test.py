import torch
import torch.nn as nn
import time

from device import getTorchDevice


class SpeedTestNet(nn.Module):
    def __init__(self, num_hidden, num_layers, device=torch.device("cpu")):
        super().__init__()

        self.device = device

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

        self.to(device)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    device = getTorchDevice()
    print(f"Device being used is:", device)
    tensorSize = 256
    layers = 20
    batchSize = 256
    accelModel = SpeedTestNet(tensorSize, layers, getTorchDevice())
    accelModel.eval()

    cpuModel = SpeedTestNet(tensorSize, layers)
    cpuModel.eval()

    # Warm-up
    for _ in range(100):
        torch.matmul(torch.rand(500, 500).to(getTorchDevice()),
                     torch.rand(500, 500).to(getTorchDevice()))

    start_time = time.time()
    cpuModel(torch.rand(batchSize, tensorSize))
    print("CPU : --- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    accelModel(torch.rand(batchSize, tensorSize).to(getTorchDevice()))
    print("DEVICE : --- %s seconds ---" % (time.time() - start_time))
