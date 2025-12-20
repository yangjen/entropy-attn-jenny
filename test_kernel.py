import torch
import numpy as np
from models.entropy_attn_triton import attention

def test_prefill(Z=1, H=8, N_CTX=512, HEAD_DIM=128, temp=None, dtype=torch.float16):
    device = "cuda:0"

    if temp is None:
        temp = torch.ones(Z, H, N_CTX, device=device, dtype=dtype)

    q = 0.5 * torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    k = 0.5 * torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    v = 0.5 * torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    scale = 1 / np.sqrt(HEAD_DIM)

    qk = torch.einsum("bhqd,bhkd->bhqk", q * scale, k)
    mask = torch.triu(torch.ones(N_CTX, N_CTX, device=device, dtype=torch.bool), 1)
    qk = (qk / temp.unsqueeze(-1)) - (mask * 1e8).to(dtype)
    A = qk.softmax(dim=-1)

    logA = torch.log_softmax(qk, dim=-1).float()
    entropy = -(A * logA).masked_fill(mask, 0.0).sum(dim=-1)

    out = torch.einsum("bhqk,bhkd->bhqd", A, v)

    triton_out, triton_entropy = attention(q, k, v, True, scale, temp)

    diff = (out - triton_out).abs()
    ent_diff = (entropy - triton_entropy).abs()

    print(f"attn out diffmax: {diff.amax()=} ent diffmax: {ent_diff.amax()=}")
    torch.testing.assert_close(out, triton_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(entropy, triton_entropy, atol=1e-2, rtol=0)


def test_decode(Z=1, H=8, N_CTX=512, HEAD_DIM=128, temp=None, dtype=torch.float16):
    device = "cuda:0"

    if temp is None:
        temp = torch.ones(Z, H, 1, device=device, dtype=dtype)

    q = 0.5 * torch.randn(Z, H, 1, HEAD_DIM, device=device, dtype=dtype)
    k = 0.5 * torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    v = 0.5 * torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    scale = 1 / np.sqrt(HEAD_DIM)

    qk = torch.einsum("bhqd,bhkd->bhqk", q * scale, k) / temp.unsqueeze(-1)
    A = qk.softmax(dim=-1)

    logA = torch.log_softmax(qk, dim=-1).float()
    entropy = -(A * logA).sum(dim=-1)

    out = torch.einsum("bhqk,bhkd->bhqd", A, v)

    triton_out, triton_entropy = attention(q, k, v, True, scale, temp)

    torch.testing.assert_close(out, triton_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(entropy, triton_entropy, atol=1e-2, rtol=0)

if __name__ == "__main__":
    for i in range(10):
        device = "cuda:0"
        dtype = torch.float16
        Z, H, N_CTX = 4, 8, 512
        temp = torch.rand(Z, H, N_CTX, device=device, dtype=dtype).clamp(min=0.1)
        test_prefill(Z=Z, H=H, N_CTX=N_CTX, temp=temp, dtype=dtype)
        test_decode(Z=Z, H=H, N_CTX=N_CTX, temp=temp[:, :, :1], dtype=dtype)

        print(f"{i} entropy random test ok!")
