"""End-to-end smoke test of the training pipeline on synthetic motions.

Run from the repository root::

    python tests/training_smoke_test.py

Requires the SMPL-X body model (see README); everything runs on the CPU in a
few minutes. It writes a handful of random AMASS-like sequences, runs
``training.prepare_data``, trains the LSTM config for two tiny epochs,
evaluates the checkpoint in both protocols and checks that the checkpoint
loads with ``flowavatar.models.load_checkpoint``.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flowavatar.config import paths  # noqa: E402


def make_fake_amass(root, subsets, num_seqs=2, num_frames=150, fps=60, seed=0):
    """Write random, slowly varying SMPL-X style motions as AMASS npz files."""
    rng = np.random.default_rng(seed)
    kernel = np.ones(21) / 21
    for subset in subsets:
        subject_dir = root / subset / "subject"
        subject_dir.mkdir(parents=True)
        for i in range(num_seqs):
            noise = rng.standard_normal((num_frames + 20, 165))
            poses = np.stack([np.convolve(noise[:, j], kernel, mode="valid") for j in range(165)], axis=1) * 0.4
            poses[:, 66:] = 0.0  # hand / face joints are unused
            trans = np.cumsum(rng.standard_normal((num_frames, 3)) * 0.01, axis=0)
            np.savez(
                subject_dir / f"motion_{i}.npz",
                poses=poses,
                trans=trans,
                betas=rng.standard_normal(16) * 0.3,
                gender="neutral",
                mocap_frame_rate=float(fps),
            )


def main():
    if not Path(paths.smplx_file).exists():
        print(f"[SKIP] SMPL-X model not found at {paths.smplx_file}")
        return

    from flowavatar.models import load_checkpoint
    from training import evaluate, prepare_data, train
    from training.model import FlowAvatarPoseModel, save_checkpoint

    tmp = Path(tempfile.mkdtemp(prefix="flowavatar_smoke_"))
    try:
        amass_dir, data_dir = tmp / "amass", tmp / "processed"
        make_fake_amass(amass_dir, ["ACCAD", "HumanEva"])
        meta = prepare_data.main(
            [
                "--amass_dir",
                str(amass_dir),
                "--save_dir",
                str(data_dir),
                "--fps",
                "60",
                "--device",
                "cpu",
                "--body_model",
                "smplx",
                "--smplx",
                str(paths.smplx_file),
            ]
        )
        assert set(meta["subsets"]) == {"ACCAD", "HumanEva"}, meta["subsets"]
        sample = torch.load(data_dir / "train" / "ACCAD.pt", weights_only=False)[0]
        assert sample["input"].shape == (149, 90) and sample["local_pose"].shape == (149, 132)
        assert sample["joints"].shape == (149, 22, 3) and sample["betas"].shape == (10,)

        cfg = {
            "base": str(REPO_ROOT / "training" / "configs" / "lstm_smplx.yaml"),
            "name": "smoke",
            "data": {"root": str(data_dir), "fps": 60, "seq_len": 20, "repeat": 4},
            "train": {
                "epochs": 2,
                "batch_size": 4,
                "num_workers": 0,
                "val_interval": 1,
                "log_interval": 1,
                "multi_gpu": False,
            },
        }
        cfg_path = tmp / "smoke.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)
        run_dir = train.main(["--config", str(cfg_path), "--out_dir", str(tmp / "run"), "--device", "cpu"])
        for name in ("last.pt", "best_train.pt", "best_val.pt"):
            assert (run_dir / name).exists(), name
        # resume for one more epoch from last.pt
        train.main(
            ["--config", str(cfg_path), "--resume", str(run_dir / "last.pt"), "--device", "cpu", "--epochs", "3"]
        )

        ckpt = str(run_dir / "best_val.pt")
        offline = evaluate.main(["--config", str(cfg_path), "--checkpoint", ckpt, "--device", "cpu"])
        realtime = evaluate.main(["--config", str(cfg_path), "--checkpoint", ckpt, "--device", "cpu", "--window", "20"])
        masked = evaluate.main(
            [
                "--config",
                str(cfg_path),
                "--checkpoint",
                ckpt,
                "--device",
                "cpu",
                "--input_mask",
                "no_hands",
                "--noise_sigma",
                "0.01",
                "--dropout_rate",
                "0.1",
            ]
        )
        for result in (offline, realtime, masked):
            assert "mpjpe" in result and all(np.isfinite(v) for v in result.values()), result

        # The checkpoint must load with the shared package (LSTM preset)
        model = load_checkpoint(ckpt, model_type="lstm", device="cpu")
        out = model.forward_offline(torch.zeros(1, 20, 90))
        assert out["root_rot"].shape == (1, 6) and out["betas"].shape == (1, 10)

        # GRU preset compatibility of the training wrapper
        gru = FlowAvatarPoseModel({"hidden_dim": 256, "num_layers": 3, "use_lstm": False}, paths.smplx_file)
        save_checkpoint(gru, tmp / "gru.pt", epoch=0)
        load_checkpoint(str(tmp / "gru.pt"), model_type="gru", device="cpu")
        print("\n[OK] training smoke test passed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
