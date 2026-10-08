"""Evaluate a FlowAvatar checkpoint on the AMASS test split.

Run from the repository root::

    python -m training.evaluate --config training/configs/gru_smplh.yaml \
        --checkpoint experiments/gru_smplh/<run>/best_val.pt

Protocols:

* offline (default): every test sequence is processed in a single pass. This
  is the protocol of the benchmark tables in the paper (and of AvatarPoser /
  AGRoL / HMD-Poser).
* ``--window N``: real-time protocol. Sliding windows of N frames with
  stride 1; the last frame of each window is kept, as in the streaming demo.

Robustness options: ``--input_mask`` (drop one or both hands for the whole
sequence), ``--noise_sigma`` (Gaussian noise on the tracker positions, in
metres) and ``--dropout_rate`` (probability of a frame repeating the previous
tracker input).
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset import INPUT_MASKS, SequenceDataset, load_split
from training.metrics import METRIC_NAMES, compute_metrics, format_table
from training.model import FlowAvatarPoseModel, head_align, load_weights, unwrap
from training.utils import RunningMean, load_config, resolve_body_model, resolve_path, to_device

POSITION_FEATURES = slice(36, 45)  # global head / hand positions in the 90-D input


def perturb_inputs(x, noise_sigma=0.0, dropout_rate=0.0, generator=None):
    """Simulate degraded tracking. x: [B, T, 90] (modified copy is returned)."""
    if noise_sigma <= 0 and dropout_rate <= 0:
        return x
    x = x.clone()
    if noise_sigma > 0:
        noise = torch.randn(x[:, :, POSITION_FEATURES].shape, generator=generator) * noise_sigma
        x[:, :, POSITION_FEATURES] += noise.to(x.device)
    if dropout_rate > 0:
        drop = (torch.rand(x.shape[:2], generator=generator) < dropout_rate).to(x.device)
        for t in range(1, x.shape[1]):
            x[:, t] = torch.where(drop[:, t, None], x[:, t - 1], x[:, t])
    return x


@torch.no_grad()
def predict_sequence(model, x, window=None, chunk=256):
    """Run the model over one sequence x [1, T, 90].

    With ``window`` set, predictions come from sliding windows (stride 1) and
    only the last frame of each window is kept; the first ``window - 1``
    frames are taken from the first window.
    """
    bare = unwrap(model)
    num_frames = x.shape[1]
    if window is None or num_frames <= window:
        return bare(x)

    windows = x.unfold(1, window, 1)[0].permute(0, 2, 1)  # [T - W + 1, W, 90]
    poses, betas = [], []
    for start in range(0, windows.shape[0], chunk):
        out = bare.net(windows[start : start + chunk])
        if start == 0:
            poses.append(out["pose_params"][0, :-1])
            betas.append(out["betas"][0, :-1])
        poses.append(out["pose_params"][:, -1])
        betas.append(out["betas"][:, -1])
    pose = torch.cat(poses, dim=0)[None]  # [1, T, 132]
    beta = torch.cat(betas, dim=0)[None]  # [1, T, 10]
    joints = bare.forward_kinematics(pose.reshape(-1, pose.shape[-1]), beta.reshape(-1, beta.shape[-1]))
    return {
        "local_pose": pose,
        "betas": beta,
        "global_pose": bare.global_rotations(pose),
        "joints": joints.reshape(1, num_frames, -1, 3),
    }


@torch.no_grad()
def evaluate(
    model,
    dataset,
    device,
    fps,
    window=None,
    noise_sigma=0.0,
    dropout_rate=0.0,
    loss_fn=None,
    seed=0,
    progress=True,
    num_workers=0,
):
    """Average the metrics (and optionally the loss terms) over all sequences."""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    generator = torch.Generator().manual_seed(seed)
    running = RunningMean()
    for batch in tqdm(loader, desc="evaluating", disable=not progress, leave=False):
        batch = to_device(batch, device)
        x = perturb_inputs(batch["input"], noise_sigma, dropout_rate, generator)
        pred = predict_sequence(model, x, window)
        pred["joints"] = head_align(pred["joints"], batch["joints"])
        if loss_fn is not None:
            losses = loss_fn(pred, batch)
            running.update({f"loss/{k}": v for k, v in losses.items()})
        running.update(
            compute_metrics(
                pred["joints"][0],
                batch["joints"][0],
                pred["local_pose"][0],
                batch["local_pose"][0],
                fps,
                batch["floor_height"][0],
            )
        )
    return running.means()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Training config (data location, fps, model params)")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint (.pt) written by training.train")
    parser.add_argument("--split", default="test", help="Data split to evaluate (default: test)")
    parser.add_argument(
        "--window", type=int, default=None, help="Sliding-window (real-time) protocol with this window length"
    )
    parser.add_argument("--input_mask", default="none", choices=list(INPUT_MASKS), help="Hide hand features")
    parser.add_argument("--noise_sigma", type=float, default=0.0, help="Std of Gaussian noise on tracker positions (m)")
    parser.add_argument(
        "--dropout_rate", type=float, default=0.0, help="Probability of repeating the previous input frame"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output", default=None, help="Where to write the metrics JSON (default: next to the checkpoint)"
    )
    parser.add_argument("--subsets", nargs="*", default=None, help="Restrict to these subsets")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    body_model_file, num_betas = resolve_body_model(cfg)
    model_params = dict(checkpoint.get("model_params") or cfg["model"])
    model_params.setdefault("num_betas", num_betas)
    if model_params["num_betas"] != num_betas:
        sys.exit(
            f"checkpoint predicts {model_params['num_betas']} betas but the config's body model "
            f"has {num_betas}; use a config with the matching body_model block"
        )
    model = FlowAvatarPoseModel(model_params, body_model_file).to(device)
    load_weights(model, args.checkpoint)
    model.eval()

    data_cfg = cfg["data"]
    sequences = load_split(resolve_path(data_cfg["root"]), args.split, subsets=args.subsets)
    dataset = SequenceDataset(sequences, input_mask=args.input_mask)
    print(
        f"{len(dataset)} sequences from {resolve_path(data_cfg['root'])}/{args.split} "
        f"(fps={data_cfg['fps']}, window={args.window or 'full sequence'}, "
        f"input_mask={args.input_mask}, noise={args.noise_sigma}, dropout={args.dropout_rate})"
    )

    results = evaluate(
        model,
        dataset,
        device,
        data_cfg["fps"],
        window=args.window,
        noise_sigma=args.noise_sigma,
        dropout_rate=args.dropout_rate,
    )
    print(format_table({Path(args.checkpoint).stem: results}, METRIC_NAMES))

    suffix = "".join(
        [
            f"_window{args.window}" if args.window else "",
            f"_{args.input_mask}" if args.input_mask != "none" else "",
            f"_noise{args.noise_sigma:g}" if args.noise_sigma > 0 else "",
            f"_dropout{args.dropout_rate:g}" if args.dropout_rate > 0 else "",
        ]
    )
    output = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).with_name(f"{Path(args.checkpoint).stem}_eval_{args.split}{suffix}.json")
    )
    with open(output, "w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "window": args.window,
                "input_mask": args.input_mask,
                "noise_sigma": args.noise_sigma,
                "dropout_rate": args.dropout_rate,
                "fps": data_cfg["fps"],
                "metrics": results,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to {output}")
    return results


if __name__ == "__main__":
    main()
