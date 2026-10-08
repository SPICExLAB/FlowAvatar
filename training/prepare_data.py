"""Preprocess AMASS into FlowAvatar training data.

Split: the HMD-Poser / AvatarPoser "protocol 2" split used in the paper.
Twelve AMASS subsets are used for training and HumanEva + Transitions are
held out for testing (see ``TRAIN_SUBSETS`` / ``TEST_SUBSETS``).

Body model (``--body_model``):

* ``smplh`` (default, benchmark): every sequence is replayed through the
  gender-specific SMPL+H model with 16 shape parameters, exactly as
  AvatarPoser, AGRoL and HMD-Poser prepare AMASS. Use this for numbers that
  are comparable with prior work and with the paper tables.
* ``smplx`` (deployment): neutral SMPL-X with 10 shape parameters, the body
  the released streaming checkpoints and the Unity avatar use.

Run from the repository root::

    # benchmark data: 60 fps (GRU, window 40) and 30 fps (on-device LSTM, window 20)
    python -m training.prepare_data --amass_dir data/AMASS --save_dir data/amass_smplh_60fps --fps 60
    python -m training.prepare_data --amass_dir data/AMASS --save_dir data/amass_smplh_30fps --fps 30
    # deployment data
    python -m training.prepare_data --amass_dir data/AMASS --save_dir data/amass_smplx_60fps --fps 60 --body_model smplx

AMASS has to be downloaded from https://amass.is.tue.mpg.de (it cannot be
redistributed) and extracted so that every subset is a folder directly under
``--amass_dir``. Both the SMPL+H and the SMPL-X packages are read (only the
22 body joints are used and the folder names of either release are
recognised); prior work and the paper use the SMPL+H package.

Output: ``<save_dir>/<split>/<subset>.pt`` holds a list of sequence dicts
(see ``training/features.py``) and ``<save_dir>/meta.json`` summarises the run.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowavatar.config import paths
from training.features import compute_sequence_features

BODY_MODELS = {  # name -> (number of shape parameters, genders)
    "smplh": (16, ("male", "female", "neutral")),
    "smplx": (10, ("neutral",)),
}

# Canonical subset name -> folder names used by the SMPL-X / SMPL+H releases
TRAIN_SUBSETS = {
    "ACCAD": ("ACCAD",),
    "BMLmovi": ("BMLmovi",),
    "BMLrub": ("BMLrub", "BioMotionLab_NTroje"),
    "CMU": ("CMU",),
    "EKUT": ("EKUT",),
    "EyesJapanDataset": ("EyesJapanDataset", "Eyes_Japan_Dataset"),
    "KIT": ("KIT",),
    "HDM05": ("HDM05", "MPI_HDM05"),
    "PosePrior": ("PosePrior", "MPI_Limits"),
    "MoSh": ("MoSh", "MPI_mosh"),
    "SFU": ("SFU",),
    "TotalCapture": ("TotalCapture",),
}
TEST_SUBSETS = {
    "HumanEva": ("HumanEva",),
    "Transitions": ("Transitions", "Transitions_mocap"),
}
SPLITS = {"train": TRAIN_SUBSETS, "test": TEST_SUBSETS}

# Per-subject shape / calibration files that are not motions
SKIP_SUFFIXES = ("stagei.npz", "shape.npz")
MIN_FRAMES = 10


def find_subset_dir(amass_dir, aliases):
    for alias in aliases:
        candidate = Path(amass_dir) / alias
        if candidate.is_dir():
            return candidate
    return None


def _as_str(value):
    value = np.asarray(value)
    if value.size == 0:
        return "neutral"
    item = value.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode()
    return str(item)


def load_amass_sequence(npz_path, target_fps, num_betas):
    """Load one AMASS motion and resample it to ``target_fps`` by striding.

    Returns None when the file is not a motion or is slower than the target
    frame rate.
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)
    if not {"poses", "trans", "betas"} <= keys:
        return None
    if "mocap_frame_rate" in keys:  # SMPL-X release
        src_fps = float(data["mocap_frame_rate"])
    elif "mocap_framerate" in keys:  # SMPL+H release
        src_fps = float(data["mocap_framerate"])
    else:
        return None
    stride = int(round(src_fps / target_fps))
    if stride < 1:
        return None

    poses = np.asarray(data["poses"], dtype=np.float32)[::stride, :66]
    trans = np.asarray(data["trans"], dtype=np.float32)[::stride]
    betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)[:num_betas]
    gender = _as_str(data["gender"]) if "gender" in keys else "neutral"
    return {
        "root_orient": poses[:, :3],
        "pose_body": poses[:, 3:66],
        "trans": trans,
        "betas": betas,
        "gender": gender,
        "source_fps": src_fps,
        "stride": stride,
    }


def process_subset(name, subset_dir, body_models, num_betas, fps, device, log=print):
    """``body_models`` maps gender -> BodyModel; unknown genders use the neutral model."""
    files = sorted(p for p in subset_dir.rglob("*.npz") if not p.name.endswith(SKIP_SUFFIXES))
    sequences = []
    stats = {"files": len(files), "sequences": 0, "frames": 0, "skipped": 0, "fps_mismatch": 0}
    for path in tqdm(files, desc=name, leave=False):
        try:
            motion = load_amass_sequence(path, fps, num_betas)
        except Exception as exc:  # corrupt / unexpected file: report and continue
            log(f"  ! {path}: {exc}")
            motion = None
        if motion is None or motion["root_orient"].shape[0] < MIN_FRAMES:
            stats["skipped"] += 1
            continue
        if abs(motion["source_fps"] / motion["stride"] - fps) > 0.5:
            stats["fps_mismatch"] += 1
        body_model = body_models.get(motion["gender"], body_models["neutral"])
        seq = compute_sequence_features(
            motion["root_orient"],
            motion["pose_body"],
            motion["trans"],
            motion["betas"],
            body_model,
            fps,
            device=device,
            num_betas=num_betas,
        )
        seq["gender"] = motion["gender"]
        seq["source_fps"] = motion["source_fps"]
        seq["subset"] = name
        seq["source"] = str(path.relative_to(subset_dir.parent))
        sequences.append(seq)
        stats["sequences"] += 1
        stats["frames"] += int(seq["input"].shape[0])
    return sequences, stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--amass_dir", required=True, help="Directory containing the AMASS subset folders")
    parser.add_argument("--save_dir", required=True, help="Output directory")
    parser.add_argument(
        "--fps", type=int, default=60, help="Target frame rate (60 for the GRU model, 30 for the on-device LSTM)"
    )
    parser.add_argument(
        "--body_model",
        default="smplh",
        choices=list(BODY_MODELS),
        help="smplh: gendered SMPL+H, 16 betas (benchmark, as prior work); "
        "smplx: neutral SMPL-X, 10 betas (deployment)",
    )
    parser.add_argument(
        "--body_models_dir",
        default=None,
        help="Directory with smplh/{male,female,neutral}/model.npz (default: body_models/)",
    )
    parser.add_argument("--smplx", default=None, help="SMPLX_NEUTRAL.npz (default: flowavatar.config.paths.smplx_file)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--subsets", nargs="*", default=None, help="Only process these subsets (canonical names)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute subsets whose output already exists")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from human_body_prior.body_model.body_model import BodyModel

    amass_dir, save_dir = Path(args.amass_dir), Path(args.save_dir)
    if not amass_dir.is_dir():
        sys.exit(f"AMASS directory not found: {amass_dir}")
    device = torch.device(args.device)

    num_betas, genders = BODY_MODELS[args.body_model]
    if args.body_model == "smplx":
        model_files = {"neutral": Path(args.smplx) if args.smplx else Path(paths.smplx_file)}
    else:
        smplh_dir = Path(args.body_models_dir or paths.body_models_dir) / "smplh"
        model_files = {gender: smplh_dir / gender / "model.npz" for gender in genders}
    body_models = {}
    for gender, model_file in model_files.items():
        if not model_file.exists():
            sys.exit(f"{args.body_model} body model not found at {model_file} (see training/README.md, Setup)")
        body_models[gender] = BodyModel(bm_fname=str(model_file), num_betas=num_betas).to(device)
    print(f"body model: {args.body_model} ({num_betas} betas), genders: {', '.join(body_models)}")

    meta = {
        "fps": args.fps,
        "body_model": args.body_model,
        "num_betas": num_betas,
        "body_model_files": {gender: str(path) for gender, path in model_files.items()},
        "amass_dir": str(amass_dir),
        "subsets": {},
    }
    # Keep the entries of subsets processed by an earlier run (their files are
    # skipped below), so meta.json always describes everything in save_dir.
    meta_path = save_dir / "meta.json"
    if meta_path.exists() and not args.overwrite:
        with open(meta_path) as f:
            previous = json.load(f)
        if previous.get("fps") == args.fps and previous.get("body_model") == args.body_model:
            meta["subsets"] = {
                name: stats
                for name, stats in previous.get("subsets", {}).items()
                if Path(stats.get("path", "")).exists()
            }
        else:
            print(
                f"warning: {meta_path} was written for {previous.get('body_model')} at {previous.get('fps')} fps; "
                f"existing subset files will be skipped but not re-described (use --overwrite)"
            )
    start = time.time()
    for split, subsets in SPLITS.items():
        out_dir = save_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, aliases in subsets.items():
            if args.subsets and name not in args.subsets:
                continue
            out_path = out_dir / f"{name}.pt"
            if out_path.exists() and not args.overwrite:
                print(f"[{split}] {name}: exists, skipping ({out_path})")
                continue
            subset_dir = find_subset_dir(amass_dir, aliases)
            if subset_dir is None:
                print(f"[{split}] {name}: not found under {amass_dir} (looked for {', '.join(aliases)}), skipping")
                continue
            sequences, stats = process_subset(name, subset_dir, body_models, num_betas, args.fps, device)
            if not sequences:
                print(f"[{split}] {name}: no usable sequences")
                continue
            torch.save(sequences, out_path)
            stats["path"] = str(out_path)
            meta["subsets"][name] = dict(split=split, **stats)
            print(
                f"[{split}] {name}: {stats['sequences']} sequences, {stats['frames']} frames "
                f"({stats['skipped']} files skipped, {stats['fps_mismatch']} with inexact frame rate) -> {out_path}"
            )

    meta["elapsed_sec"] = round(time.time() - start, 1)
    with open(save_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    total_seq = sum(s["sequences"] for s in meta["subsets"].values())
    total_frames = sum(s["frames"] for s in meta["subsets"].values())
    print(
        f"Done: {total_seq} sequences / {total_frames} frames at {args.fps} fps ({args.body_model}, "
        f"{num_betas} betas) in {meta['elapsed_sec']}s -> {save_dir}"
    )
    return meta


if __name__ == "__main__":
    main()
