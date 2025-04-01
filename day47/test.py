import torch
from torch import nn
import ctypes

cuda = ctypes.CDLL("./rmsnorm.so")

cuda.rmsnorm_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # in
    ctypes.POINTER(ctypes.c_float),  # out
    ctypes.c_size_t,  # B
    ctypes.c_size_t,  # N
]

B, N = 8192, 8192
x = torch.randn(B, N).cuda()
print(x.device)
pred = torch.empty_like(x, device=x.device)
rn = nn.RMSNorm(N).cuda()

y = rn(x)

x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float))
pred_ptr = ctypes.cast(pred.data_ptr(), ctypes.POINTER(ctypes.c_float))
cuda.rmsnorm_gpu(x_ptr, pred_ptr, B, N)

print(y)
print(pred)
print("Match impl:", torch.allclose(y, pred, atol=1e-6))
