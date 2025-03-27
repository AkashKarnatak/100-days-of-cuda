import math
import ctypes
import torch

q = torch.randn(32, 4, 1024, 256).cuda()
k = torch.randn(32, 4, 1024, 256).cuda()
v = torch.randn(32, 4, 1024, 256).cuda()

torch.manual_seed(42)


def precompute_basis(
    seq_len: int, head_dim: int, theta0: float = 10000
) -> torch.Tensor:
    freq = torch.exp(-torch.arange(0, head_dim, 2) * math.log(theta0) / head_dim)
    pos = torch.arange(seq_len).float()
    theta = torch.outer(pos, freq)

    cos = torch.cos(theta).repeat_interleave(2, dim=-1)
    sin = torch.sin(theta).repeat_interleave(2, dim=-1)

    return cos.cuda(), sin.cuda()


def rope(x: torch.Tensor, basis: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    def rotate(x):
        x1 = x[..., ::2, None]
        x2 = x[..., 1::2, None]
        return torch.cat((-x2, x1), dim=-1).view(*x.shape)

    cos, sin = basis
    return x * cos + rotate(x) * sin


B, nh, seq_len, head_dim = q.shape

basis = precompute_basis(seq_len, head_dim)
rope_q1 = rope(q, basis)

cuda = ctypes.CDLL("./rope.so")

cuda.rope_gpu.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # out
    ctypes.POINTER(ctypes.c_float),  # in
    ctypes.c_size_t,  # N
    ctypes.c_size_t,  # T
    ctypes.c_size_t,  # d
]

rope_q2 = torch.empty_like(q)
q_ptr = ctypes.cast(q.data_ptr(), ctypes.POINTER(ctypes.c_float))
rope_q2_ptr = ctypes.cast(rope_q2.data_ptr(), ctypes.POINTER(ctypes.c_float))

N = B * nh * seq_len
cuda.rope_gpu(rope_q2_ptr, q_ptr, N, seq_len, head_dim)

# print(((rope_q1 - rope_q2) > 1e-5).sum() * 100 / rope_q2.numel())
print("Match impl:", torch.allclose(rope_q1, rope_q2, atol=1e-3))
