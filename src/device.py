import torch


def getTorchDevice():
    deviceName = "cpu"
    if torch.cuda.is_available():
        deviceName = "cuda"
    elif torch.mps.device_count() > 0:
        deviceName = "mps"
    return torch.device(deviceName)
