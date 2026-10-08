"""Helpers shared by the training entry points: config loading, seeding,
logging, and the learning-rate schedule."""

import copy
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from flowavatar.config import REPO_ROOT


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path):
    """Load a YAML config.

    A top-level ``base`` key names another config file (relative to this one)
    whose values are inherited; keys in the child override the parent.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        parent = load_config(path.parent / base)
        parent.pop("name", None)  # a derived config is named after itself, not its base
        cfg = _deep_update(parent, cfg)
    cfg.setdefault("name", path.stem)
    return cfg


def resolve_path(p):
    """Absolute paths are kept; relative paths are taken from the repo root."""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_body_model(cfg):
    """Return ``(model_file, num_betas)`` from the config's ``body_model`` block.

    ``file`` is taken relative to ``flowavatar.config.paths.body_models_dir``
    (``body_models/`` or ``$FLOWAVATAR_BODY_MODELS_DIR``); absolute paths are
    kept. Without ``file`` the neutral SMPL-X model (``$SMPLX_MODEL_PATH`` or
    ``body_models/smplx/SMPLX_NEUTRAL.npz``) with 10 betas is used.
    """
    from flowavatar.config import paths

    block = cfg.get("body_model") or {}
    if block.get("file"):
        model_file = Path(os.path.expanduser(str(block["file"])))
        if not model_file.is_absolute():
            model_file = Path(paths.body_models_dir) / model_file
    else:
        model_file = Path(paths.smplx_file)
    if not model_file.exists():
        raise FileNotFoundError(
            f"body model not found at {model_file}; download it (see training/README.md, Setup) or set "
            "body_model.file in the config"
        )
    return model_file, int(block.get("num_betas", 10))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name="flowavatar.training", log_file=None):
    """Console logger, optionally mirrored to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


class WarmupStepSchedule:
    """Per-iteration learning-rate multiplier (the HMD-Poser schedule).

    The multiplier rises geometrically from ``warmup_factor`` to 1 over the
    first ``warmup_steps`` iterations, then is multiplied by ``decay_factor``
    at every iteration in ``decay_steps``, each decay being spread
    geometrically over ``decay_duration`` iterations.
    """

    def __init__(self, decay_steps, decay_factor=0.1, decay_duration=1, warmup_steps=0, warmup_factor=1e-3):
        self.decay_steps = [int(s) for s in decay_steps]
        self.decay_factor = float(decay_factor)
        self.decay_duration = max(1, int(decay_duration))
        self.warmup_steps = int(warmup_steps)
        self.warmup_factor = float(warmup_factor)

    def __call__(self, step):
        mult = 1.0
        if self.warmup_steps > 0 and step < self.warmup_steps:
            mult *= self.warmup_factor ** (1.0 - step / self.warmup_steps)
        for decay_step in self.decay_steps:
            if step >= decay_step + self.decay_duration:
                mult *= self.decay_factor
            elif step > decay_step:
                mult *= self.decay_factor ** ((step - decay_step) / self.decay_duration)
        return mult


class RunningMean:
    """Accumulates the mean of named scalars."""

    def __init__(self):
        self.sums, self.counts = {}, {}

    def update(self, values):
        for key, value in values.items():
            value = float(value.item() if torch.is_tensor(value) else value)
            self.sums[key] = self.sums.get(key, 0.0) + value
            self.counts[key] = self.counts.get(key, 0) + 1

    def means(self):
        return {key: self.sums[key] / self.counts[key] for key in self.sums}


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def to_device(batch, device):
    return {
        key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value) for key, value in batch.items()
    }
