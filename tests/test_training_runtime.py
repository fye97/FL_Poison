import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flpoison.fl.training import configure_torch_runtime


def test_configure_torch_runtime_enables_cuda_fast_paths(monkeypatch):
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    try:
        args = SimpleNamespace(
            device=torch.device("cuda:0"),
            cudnn_benchmark=True,
            allow_tf32=True,
        )

        configure_torch_runtime(args)

        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cudnn.benchmark = prev_benchmark
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32


def test_configure_torch_runtime_disables_cuda_fast_paths_without_cuda(monkeypatch):
    prev_benchmark = torch.backends.cudnn.benchmark
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    try:
        args = SimpleNamespace(
            device=torch.device("cpu"),
            cudnn_benchmark=True,
            allow_tf32=True,
        )

        configure_torch_runtime(args)

        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
    finally:
        torch.backends.cudnn.benchmark = prev_benchmark
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
