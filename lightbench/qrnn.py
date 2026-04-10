import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _fo_pool_fwd(Z, F, H, T: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    b, d = pid // D, pid % D
    base = b * T * D + d
    h = 0.0
    for t in range(T):
        idx = base + t * D
        z = tl.load(Z + idx).to(tl.float32)
        f = tl.load(F + idx).to(tl.float32)
        h = f * h + (1.0 - f) * z
        tl.store(H + idx, h)


@triton.jit
def _fo_pool_bwd(GO, F, H, Z, GZ, GF, T: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    b, d = pid // D, pid % D
    base = b * T * D + d
    g = 0.0
    for i in range(T):
        t = T - 1 - i
        idx = base + t * D
        g += tl.load(GO + idx).to(tl.float32)
        f = tl.load(F + idx).to(tl.float32)
        z = tl.load(Z + idx).to(tl.float32)
        if t > 0:
            hp = tl.load(H + base + (t - 1) * D).to(tl.float32)
        else:
            hp = 0.0
        tl.store(GZ + idx, g * (1.0 - f))
        tl.store(GF + idx, g * (hp - z))
        g *= f


class _FOPool(torch.autograd.Function):
    @staticmethod
    def forward(z, f):
        B, T, D = z.shape
        h = torch.empty_like(z)
        _fo_pool_fwd[(B * D,)](z, f, h, T=T, D=D)
        return h

    @staticmethod
    def setup_context(ctx, inputs, output):
        z, f = inputs
        ctx.save_for_backward(f, output, z)

    @staticmethod
    def backward(ctx, grad_output):
        f, h, z = ctx.saved_tensors
        B, T, D = f.shape
        go = grad_output.contiguous()
        gz = torch.empty_like(z)
        gf = torch.empty_like(f)
        _fo_pool_bwd[(B * D,)](go, f, h, z, gz, gf, T=T, D=D)
        return gz, gf


fo_pool = _FOPool.apply


class QRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.gates = nn.Linear(input_size, hidden_size * 2)
        self.hidden_size = hidden_size

    def forward(self, x):
        g = self.gates(x)
        z, f = g.split(self.hidden_size, dim=-1)
        return fo_pool(torch.tanh(z).contiguous(), torch.sigmoid(f).contiguous())
