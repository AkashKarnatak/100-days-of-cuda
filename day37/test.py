import math
import torch
import torch.nn.functional as F
import numpy as np
import ctypes

libc = ctypes.CDLL(None)

libc.srand(0)

B, nh, N, M = 16, 16, 128, 64
Br, Bc = 24, 24

# qkv = torch.tensor([libc.rand() for _ in range(N * M * 3)]).reshape(N, M, 3) / (
#     (1 << 31) - 1
# )

qkv = torch.randn(B, nh, N, M, 3)

q, k, v = qkv.permute(4, 0, 1, 2, 3)
q, k, v = q.clone(), k.clone(), v.clone()

flash_attn = ctypes.CDLL("./flash_attention2.so")

flash_attn.flash_attention_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # *query
    ctypes.POINTER(ctypes.c_float),  # *key
    ctypes.POINTER(ctypes.c_float),  # *value
    ctypes.POINTER(ctypes.c_float),  # *output
    ctypes.POINTER(ctypes.c_float),  # *sums
    ctypes.POINTER(ctypes.c_float),  # *maxes
    ctypes.c_size_t,  # B
    ctypes.c_size_t,  # nh
    ctypes.c_size_t,  # N
    ctypes.c_size_t,  # d
    ctypes.c_size_t,  # Br
    ctypes.c_size_t,  # Bc
]

o = np.zeros((B, nh, N, M), dtype=np.float32)
l = np.zeros((B * nh * N), dtype=np.float32)
m = np.ones((B * nh * N), dtype=np.float32) * float("-inf")

q_ptr = q.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
k_ptr = k.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
v_ptr = v.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
o_ptr = o.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
l_ptr = l.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
m_ptr = m.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

flash_attn.flash_attention_gpu(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr, m_ptr, B, nh, N, M, Br, Bc
)

spda = F.scaled_dot_product_attention(q, k, v).numpy()

print(o)
print(spda)
print(np.allclose(o, spda, atol=1e-6))
print("====>", np.abs(o - spda).max())
