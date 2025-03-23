import ctypes
import torch
import torch.nn.functional as F

cuda = ctypes.CDLL("./softmax.so")

cuda.softmax.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # in
    ctypes.POINTER(ctypes.c_float),  # out
    ctypes.c_size_t,  # N
    ctypes.c_size_t,  # M
]

N, M = 1024, 1024
inp = torch.randn(N, M).cuda()
# inp = torch.ones(N, M).float().cuda() * -10
out = torch.empty_like(inp).cuda()

inp_ptr = ctypes.cast(inp.data_ptr(), ctypes.POINTER(ctypes.c_float))
out_ptr = ctypes.cast(out.data_ptr(), ctypes.POINTER(ctypes.c_float))

cuda.softmax(inp_ptr, out_ptr, N, M)

y = F.softmax(inp, dim=-1)

print(y)
print(out)

print(torch.allclose(y, out))
