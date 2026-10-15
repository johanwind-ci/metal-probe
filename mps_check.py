"""Check whether PyTorch's MPS backend works here and time an fp32/fp16 matmul."""
import time

import torch

print("torch", torch.__version__)
print("mps built:", torch.backends.mps.is_built(), "available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    n = 4096
    for dtype in (torch.float32, torch.float16):
        a = torch.randn(n, n, device="mps", dtype=dtype)
        b = torch.randn(n, n, device="mps", dtype=dtype)
        ref = (a[:64].float().cpu() @ b.float().cpu())
        err = ((a @ b)[:64].float().cpu() - ref).abs().max().item() / ref.abs().max().item()
        best = float("inf")
        for _ in range(5):
            torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                a @ b
            torch.mps.synchronize()
            best = min(best, (time.perf_counter() - t0) / 5)
        print(f"{dtype}: {2 * n**3 / best / 1e12:.2f} TFLOPS, rel err {err:.1e}")
