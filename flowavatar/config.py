import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    """Filesystem locations used across FlowAvatar.

    The SMPL-X body model cannot be redistributed with this repository.
    Register at https://smpl-x.is.tue.mpg.de, download SMPLX_NEUTRAL.npz, and
    place it at body_models/smplx/SMPLX_NEUTRAL.npz (or point the
    SMPLX_MODEL_PATH environment variable at it).
    """

    repo_root: Path = REPO_ROOT
    body_models_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("FLOWAVATAR_BODY_MODELS_DIR", REPO_ROOT / "body_models")))
    smplx_file: Path = field(default_factory=lambda: Path(
        os.environ.get("SMPLX_MODEL_PATH",
                       REPO_ROOT / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz")))
    checkpoint_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("FLOWAVATAR_CHECKPOINT_DIR", REPO_ROOT / "checkpoints")))
    onnx_dir: Path = field(default_factory=lambda: REPO_ROOT / "converted_onnx")
    # Training: preprocessed AMASS data and experiment outputs (see training/README.md)
    data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("FLOWAVATAR_DATA_DIR", REPO_ROOT / "data")))
    experiments_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("FLOWAVATAR_EXPERIMENTS_DIR", REPO_ROOT / "experiments")))


paths = Paths()

# SMPL-X body-joint groups used by the region-specific predictors
joint_sets = {
    'hand': [20, 21],
    'lower_body': [1, 2, 4, 5, 7, 8, 10, 11],
    'upper_body': [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
}

# First 22 SMPL-X joints (body only; hands/face excluded)
BODY_JOINT_NAMES = [
    "pelvis",          # 0
    "left_hip",        # 1
    "right_hip",       # 2
    "spine1",          # 3
    "left_knee",       # 4
    "right_knee",      # 5
    "spine2",          # 6
    "left_ankle",      # 7
    "right_ankle",     # 8
    "spine3",          # 9
    "left_foot",       # 10
    "right_foot",      # 11
    "neck",            # 12
    "left_collar",     # 13
    "right_collar",    # 14
    "head",            # 15
    "left_shoulder",   # 16
    "right_shoulder",  # 17
    "left_elbow",      # 18
    "right_elbow",     # 19
    "left_wrist",      # 20
    "right_wrist",     # 21
]
