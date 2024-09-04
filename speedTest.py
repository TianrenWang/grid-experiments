import time
import torch

device = "mps"

torch.manual_seed(1234)
TENSOR_A_CPU = torch.rand(5000, 5000)
TENSOR_B_CPU = torch.rand(5000, 5000)

torch.manual_seed(1234)
TENSOR_A_MPS = torch.rand(5000, 5000).to(device)
TENSOR_B_MPS = torch.rand(5000, 5000).to(device)

# Warm-up
for _ in range(100):
    torch.matmul(torch.rand(500, 500).to(device),
                 torch.rand(500, 500).to(device))

start_time = time.time()
torch.matmul(TENSOR_A_CPU, TENSOR_B_CPU)
print("CPU : --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
torch.matmul(TENSOR_A_MPS, TENSOR_B_MPS)
print("MPS : --- %s seconds ---" % (time.time() - start_time))
