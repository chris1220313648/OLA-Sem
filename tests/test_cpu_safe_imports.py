import importlib
import sys
from pathlib import Path

import torch


def test_importing_t5_module_does_not_query_cuda_device(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root / "bak"))

    sys.modules.pop("wan.modules.t5", None)

    def fail_if_queried():
        raise AssertionError("CUDA device was queried while importing the T5 module")

    monkeypatch.setattr(torch.cuda, "current_device", fail_if_queried)
    module = importlib.import_module("wan.modules.t5")

    assert module.T5EncoderModel is not None
