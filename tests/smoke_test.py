"""Smoke test: every released checkpoint loads and runs dummy inference.

Run from the repository root:

    python tests/smoke_test.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowavatar.models import load_checkpoint
from flowavatar.config import paths

CASES = [
    # (model_type, checkpoint relative path, seq_len)
    ("gru", "GRU/baseline.pt", 40),
    ("lstm", "LSTM/baseline.pt", 40),
    # benchmark models tracked in the repository (SMPL+H, 16 betas)
    ("gru", "benchmark/gru_smplh.pt", 40),
    ("gru", "benchmark/gru_smplh_no_temporal.pt", 40),
    ("gru", "benchmark/gru_smplh_no_hand.pt", 40),
    ("gru", "benchmark/gru_smplh_no_ground.pt", 40),
    ("lstm", "benchmark/lstm_smplh.pt", 20),
]


def main():
    failures = []
    for model_type, rel_path, seq_len in CASES:
        ckpt = paths.checkpoint_dir / rel_path
        if not ckpt.exists():
            print(f"[SKIP] {model_type}: checkpoint missing at {ckpt}")
            continue
        try:
            model = load_checkpoint(str(ckpt), model_type=model_type, device="cpu")
            dummy = torch.zeros((1, seq_len, 90))
            out = model.forward_offline(dummy)
            shapes = {k: tuple(v.shape) for k, v in out.items()}
            assert out["root_rot"].shape[-1] == 6
            assert out["betas"].shape[-1] in (10, 16)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[OK]   {model_type}: {n_params:,} params, outputs {shapes}")
        except Exception as e:
            failures.append((model_type, e))
            print(f"[FAIL] {model_type}: {e}")

    if failures:
        sys.exit(f"{len(failures)} model(s) failed")
    print("All available checkpoints load and run.")


if __name__ == "__main__":
    main()
