"""Train the FlowAvatar pose model on the preprocessed AMASS data.

Run from the repository root::

    python -m training.train --config training/configs/gru_smplh.yaml

Every run writes to ``experiments/<config name>/<timestamp>/``:
``config.yaml`` (resolved config), ``train.log``, ``metrics.jsonl`` (one
line per epoch / validation), ``last.pt`` (resumable), ``best_train.pt`` and
``best_val.pt`` (lowest validation loss). The ``.pt`` files load directly with
``flowavatar.models.load_checkpoint`` and the streaming demo.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flowavatar.config import paths
from training.dataset import SequenceDataset, WindowDataset, load_split
from training.evaluate import evaluate
from training.losses import PoseLoss
from training.model import FlowAvatarPoseModel, head_align, load_weights, save_checkpoint
from training.utils import (
    RunningMean,
    WarmupStepSchedule,
    count_parameters,
    get_logger,
    load_config,
    resolve_body_model,
    resolve_path,
    set_seed,
    to_device,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default=None, help="Run directory (default: experiments/<name>/<timestamp>)")
    parser.add_argument("--resume", default=None, help="Resume from a last.pt written by this script")
    parser.add_argument("--init", default=None, help="Initialise the network weights from a checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument(
        "--limit_batches", type=int, default=None, help="Use only this many batches per epoch (debugging)"
    )
    return parser.parse_args(argv)


def build_dataloaders(cfg, device, log):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    data_root = resolve_path(data_cfg["root"])
    seq_len = int(data_cfg["seq_len"])
    mask_cfg = data_cfg.get("mask_prob", {})
    mask_prob = (mask_cfg.get("left", 0.0), mask_cfg.get("right", 0.0), mask_cfg.get("both", 0.0))

    train_sequences = load_split(data_root, "train", min_frames=seq_len, subsets=data_cfg.get("train_subsets"))
    train_dataset = WindowDataset(train_sequences, seq_len, repeat=data_cfg.get("repeat", 1), mask_prob=mask_prob)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        drop_last=True,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    if len(train_loader) == 0:
        sys.exit(
            f"no training batches: {len(train_dataset)} windows is fewer than batch_size "
            f"{train_cfg['batch_size']} (incomplete batches are dropped)"
        )
    test_sequences = load_split(data_root, "test", subsets=data_cfg.get("test_subsets"))
    test_dataset = SequenceDataset(test_sequences)
    log.info(
        f"train: {len(train_sequences)} sequences -> {len(train_dataset)} windows of {seq_len} frames, "
        f"{len(train_loader)} batches/epoch (hand masking p={mask_prob})"
    )
    log.info(f"test: {len(test_dataset)} sequences at {data_cfg['fps']} fps")
    return train_loader, test_dataset


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, log, epoch, log_interval, limit_batches=None):
    model.train()
    running = RunningMean()
    num_batches = len(loader) if limit_batches is None else min(limit_batches, len(loader))
    tic = time.time()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_batches:
            break
        batch = to_device(batch, device)
        pred = model(batch["input"])
        pred["joints"] = head_align(pred["joints"], batch["joints"])
        losses = loss_fn(pred, batch)

        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        scheduler.step()
        running.update(losses)

        if batch_idx % log_interval == 0 or batch_idx == num_batches - 1:
            lr = optimizer.param_groups[0]["lr"]
            terms = " ".join(f"{k}={float(v):.4f}" for k, v in losses.items() if k != "total")
            log.info(
                f"epoch {epoch + 1} [{batch_idx + 1}/{num_batches}] lr={lr:.2e} "
                f"total={float(losses['total']):.4f} {terms} ({(time.time() - tic) / (batch_idx + 1):.3f}s/batch)"
            )
    return running.means()


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs

    # Run directory and logging
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.resume:
        out_dir = Path(args.resume).parent
    else:
        out_dir = Path(paths.experiments_dir) / cfg["name"] / time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(log_file=out_dir / "train.log")
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log.info(f"config: {args.config} -> {out_dir}")
    log.info(json.dumps(cfg))

    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info(f"device: {device}" + (f" x{torch.cuda.device_count()}" if device.type == "cuda" else ""))

    train_loader, test_dataset = build_dataloaders(cfg, device, log)

    # Model, loss, optimiser and schedule
    body_model_file, num_betas = resolve_body_model(cfg)
    cfg["model"]["num_betas"] = num_betas
    model = FlowAvatarPoseModel(cfg["model"], body_model_file).to(device)
    log.info(f"body model for forward kinematics: {body_model_file} ({num_betas} betas)")
    log.info(f"FlowAvatarNetwork({cfg['model']}): {count_parameters(model.net):,} trainable parameters")
    if args.init:
        load_weights(model, args.init)
        log.info(f"initialised network weights from {args.init}")

    loss_cfg = dict(cfg["loss"])
    loss_fn = PoseLoss(loss_cfg.pop("type", "l1"), **loss_cfg).to(device)
    log.info(f"loss weights: {loss_fn.weights}")

    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(opt_cfg["lr"]),
        betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        eps=float(opt_cfg.get("eps", 1e-8)),
    )
    steps_per_epoch = len(train_loader) if args.limit_batches is None else min(args.limit_batches, len(train_loader))
    sched_cfg = cfg.get("lr_schedule", {})
    schedule = WarmupStepSchedule(
        decay_steps=[e * steps_per_epoch for e in sched_cfg.get("decay_epochs", [])],
        decay_factor=sched_cfg.get("decay_factor", 0.1),
        decay_duration=sched_cfg.get("decay_duration_epochs", 1) * steps_per_epoch,
        warmup_steps=sched_cfg.get("warmup_epochs", 0) * steps_per_epoch,
        warmup_factor=sched_cfg.get("warmup_factor", 1e-3),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    start_epoch, best = 0, {"train": float("inf"), "val": float("inf")}
    if args.resume:
        checkpoint = load_weights(model, args.resume)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = checkpoint.get("best", best)
        log.info(f"resumed from {args.resume} at epoch {start_epoch}")

    if device.type == "cuda" and torch.cuda.device_count() > 1 and train_cfg.get("multi_gpu", True):
        model = nn.DataParallel(model)
        log.info(f"using {torch.cuda.device_count()} GPUs (DataParallel)")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(out_dir))
    except Exception:  # tensorboard is optional
        log.info("tensorboard not available; logging to train.log / metrics.jsonl only")

    def record(kind, epoch, values):
        with open(out_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"type": kind, "epoch": epoch + 1, **values}) + "\n")
        if writer is not None:
            for key, value in values.items():
                writer.add_scalar(f"{kind}/{key}", value, epoch + 1)

    epochs = int(train_cfg["epochs"])
    val_interval = int(train_cfg.get("val_interval", 1))
    fps = cfg["data"]["fps"]
    for epoch in range(start_epoch, epochs):
        tic = time.time()
        train_losses = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            scheduler,
            device,
            log,
            epoch,
            int(train_cfg.get("log_interval", 50)),
            args.limit_batches,
        )
        log.info(
            f"epoch {epoch + 1}/{epochs} train "
            + " ".join(f"{k}={v:.4f}" for k, v in train_losses.items())
            + f" ({time.time() - tic:.0f}s)"
        )
        record("train", epoch, train_losses)

        if train_losses["total"] < best["train"]:
            best["train"] = train_losses["total"]
            save_checkpoint(model, out_dir / "best_train.pt", epoch=epoch, config=cfg)

        if epoch % val_interval == 0 or epoch == epochs - 1:
            tic = time.time()
            val = evaluate(model, test_dataset, device, fps, loss_fn=loss_fn, progress=False)
            log.info(
                f"epoch {epoch + 1} val "
                + " ".join(f"{k}={v:.4f}" for k, v in val.items())
                + f" ({time.time() - tic:.0f}s)"
            )
            record("val", epoch, val)
            if val["loss/total"] < best["val"]:
                best["val"] = val["loss/total"]
                save_checkpoint(model, out_dir / "best_val.pt", epoch=epoch, config=cfg, val=val)
                log.info(
                    f"new best validation loss {best['val']:.4f} at epoch {epoch + 1} "
                    f"(mpjpe {val['mpjpe']:.2f} cm, mpjre {val['mpjre']:.2f} deg, "
                    f"mpjve {val['mpjve']:.2f} cm/s, jitter {val['jitter']:.1f} m/s^3)"
                )

        # Written after the best_* updates so that a resumed run restores the
        # best losses as of this epoch and does not overwrite best_val.pt with a
        # worse model.
        save_checkpoint(
            model,
            out_dir / "last.pt",
            epoch=epoch,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            best=best,
            config=cfg,
        )

    log.info(f"done; checkpoints in {out_dir}")
    if writer is not None:
        writer.close()
    return out_dir


if __name__ == "__main__":
    main()
