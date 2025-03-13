import torch
import torch.nn.functional as F
import ctypes

libc = ctypes.CDLL(None)
RAND_MAX = (1 << 31) - 1

libc.srand(0)

N, M, K = 15, 17, 5

input = torch.tensor([libc.rand() for _ in range(N * M)]).reshape(1, 1, N, M) / RAND_MAX
filter = (
    torch.tensor([libc.rand() for _ in range(5 * 5)]).reshape(1, 1, K, K) / RAND_MAX
)

print(F.conv2d(input, filter, padding='same'))
