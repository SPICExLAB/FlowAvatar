"""Checkpoint loading for the FlowAvatar pose models."""

import gc
import logging

import torch

logger = logging.getLogger("flowavatar.loader")


class _Config:
    """Minimal config object carrying the per-model parameter dicts."""
    pass


def build_model(model_type, model_params=None):
    """Instantiate an untrained model of the given type.

    Args:
        model_type: 'gru' (default, streaming) or 'lstm' (on-device/Sentis).
        model_params: optional overrides of the preset (e.g. the
            ``model_params`` stored in checkpoints written by training/, or
            input-feature masking flags — see FlowAvatarNetwork for the
            recognized keys).
    """
    model_type = model_type.lower()
    configs = _Config()

    from .network import FlowAvatarNetwork
    configs.model_params = {
        'hidden_dim': 256,
        'num_layers': 1 if model_type == 'lstm' else 3,
        'use_lstm': model_type == 'lstm'
    }
    if model_params:
        configs.model_params.update(model_params)
    return FlowAvatarNetwork(configs)


def load_checkpoint(checkpoint_path, model_type='gru', device=None, model_params=None):
    """
    Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint (.pt / .pth)
        model_type: 'gru' or 'lstm'
        device: Optional torch device for map_location (defaults to cuda if
            available, else cpu)
        model_params: Optional dict merged over the preset and over any
            ``model_params`` stored in the checkpoint (see build_model)

    Returns:
        Model instance in eval mode with loaded weights.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_type = model_type.lower()
    # Checkpoints written by training/ carry their architecture parameters;
    # caller-supplied params (e.g. feature flags from configs/streaming.yaml)
    # take precedence
    stored_params = checkpoint.get('model_params') if isinstance(checkpoint, dict) else None
    model = build_model(model_type, {**(stored_params or {}), **(model_params or {})})

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Older FlowAvatar training checkpoints include an auxiliary
    # joint_predictor head that is not used at inference time.
    keys_to_remove = [key for key in state_dict if key.startswith('joint_predictor')]
    for key in keys_to_remove:
        del state_dict[key]

    model.load_state_dict(state_dict)
    model.eval()
    del checkpoint

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model
