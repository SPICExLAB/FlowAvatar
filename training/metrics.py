"""Benchmark metrics (AvatarPoser / AGRoL / HMD-Poser protocol).

All functions take frame-major tensors: positions [N, 22, 3] in metres,
rotations [N, 22, 3] in axis-angle. ``compute_metrics`` returns a dict in the
units reported in the paper: centimetres (positions, velocities, floor
metrics), degrees (rotations) and m/s^3 (jitter; divide by 100 for the
"10^2 m/s^3" column of the paper tables).
"""

import math

import torch

from flowavatar.utils.transforms import sixd2aa

RAD_TO_DEG = 180.0 / math.pi
M_TO_CM = 100.0

UPPER_BODY = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LOWER_BODY = [0, 1, 2, 4, 5, 7, 8, 10, 11]
HANDS = [20, 21]
ROOT = [0]
FOOT_JOINTS = [7, 10, 8, 11]  # ankles and feet
CONTACT_SPEED_THRESH = 0.01  # metres per frame

METRIC_NAMES = [
    "mpjre",
    "mpjpe",
    "mpjve",
    "handpe",
    "upperpe",
    "lowerpe",
    "rootpe",
    "jitter",
    "upper_jitter",
    "lower_jitter",
    "upper_mpjve",
    "lower_mpjve",
    "penetration",
    "floating",
    "skating",
    "gt_jitter",
]
METRIC_UNITS = {
    "mpjre": "deg",
    "mpjpe": "cm",
    "mpjve": "cm/s",
    "handpe": "cm",
    "upperpe": "cm",
    "lowerpe": "cm",
    "rootpe": "cm",
    "jitter": "m/s^3",
    "upper_jitter": "m/s^3",
    "lower_jitter": "m/s^3",
    "upper_mpjve": "cm/s",
    "lower_mpjve": "cm/s",
    "penetration": "cm",
    "floating": "cm",
    "skating": "cm/frame",
    "gt_jitter": "m/s^3",
}


def rotation_error(pred_aa, gt_aa):
    """Mean absolute axis-angle difference (wrapped to [-pi, pi]) in radians."""
    diff = gt_aa - pred_aa
    diff = torch.where(diff > math.pi, diff - 2 * math.pi, diff)
    diff = torch.where(diff < -math.pi, diff + 2 * math.pi, diff)
    return diff.abs().mean()


def position_error(pred, gt, joints=None):
    err = torch.norm(gt - pred, dim=-1)
    return err.mean() if joints is None else err[:, joints].mean()


def velocity_error(pred, gt, fps, joints=None):
    pred_vel = (pred[1:] - pred[:-1]) * fps
    gt_vel = (gt[1:] - gt[:-1]) * fps
    err = torch.norm(gt_vel - pred_vel, dim=-1)
    return err.mean() if joints is None else err[:, joints].mean()


def jitter(pos, fps, joints=None):
    """Mean magnitude of the third derivative of the joint positions (m/s^3)."""
    if pos.shape[0] < 4:
        return pos.new_zeros(())
    jerk = (pos[3:] - 3 * pos[2:-1] + 3 * pos[1:-2] - pos[:-3]) * (fps**3)
    norm = jerk.norm(dim=-1)
    return norm.mean() if joints is None else norm[:, joints].mean()


def penetration(pos, floor_height):
    """Mean depth of the lowest joint below the floor."""
    lowest = pos[..., 2].min(dim=-1).values
    return torch.relu(floor_height - lowest).mean()


def floating(pos, floor_height):
    """Mean height of the lowest joint above the floor."""
    lowest = pos[..., 2].min(dim=-1).values
    return torch.relu(lowest - floor_height).mean()


def skating(pred, gt):
    """Mean per-frame displacement of ankle/foot joints while the ground-truth
    joint is static (metres per frame)."""
    gt_feet, pred_feet = gt[:, FOOT_JOINTS], pred[:, FOOT_JOINTS]
    static = torch.norm(gt_feet[1:] - gt_feet[:-1], dim=-1) <= CONTACT_SPEED_THRESH
    pred_vel = (pred_feet[1:] - pred_feet[:-1]) * static.unsqueeze(-1)
    return pred_vel.abs().mean()


@torch.no_grad()
def compute_metrics(pred_joints, gt_joints, pred_local_6d, gt_local_6d, fps, floor_height):
    """
    Args:
        pred_joints, gt_joints: [N, 22, 3] world-space positions (pred head-aligned to gt)
        pred_local_6d, gt_local_6d: [N, 132] local joint rotations (22 x 6D)
        fps: frame rate used for velocities and jitter
        floor_height: scalar tensor
    Returns:
        dict name -> float, see ``METRIC_NAMES`` / ``METRIC_UNITS``
    """
    pred_aa = sixd2aa(pred_local_6d.reshape(-1, 6)).reshape(-1, 22, 3)
    gt_aa = sixd2aa(gt_local_6d.reshape(-1, 6)).reshape(-1, 22, 3)
    floor_height = torch.as_tensor(floor_height, dtype=pred_joints.dtype, device=pred_joints.device)
    metrics = {
        "mpjre": rotation_error(pred_aa[:, 1:], gt_aa[:, 1:]) * RAD_TO_DEG,  # body joints, root excluded
        "mpjpe": position_error(pred_joints, gt_joints) * M_TO_CM,
        "mpjve": velocity_error(pred_joints, gt_joints, fps) * M_TO_CM,
        "handpe": position_error(pred_joints, gt_joints, HANDS) * M_TO_CM,
        "upperpe": position_error(pred_joints, gt_joints, UPPER_BODY) * M_TO_CM,
        "lowerpe": position_error(pred_joints, gt_joints, LOWER_BODY) * M_TO_CM,
        "rootpe": position_error(pred_joints, gt_joints, ROOT) * M_TO_CM,
        "jitter": jitter(pred_joints, fps),
        "upper_jitter": jitter(pred_joints, fps, UPPER_BODY),
        "lower_jitter": jitter(pred_joints, fps, LOWER_BODY),
        "upper_mpjve": velocity_error(pred_joints, gt_joints, fps, UPPER_BODY) * M_TO_CM,
        "lower_mpjve": velocity_error(pred_joints, gt_joints, fps, LOWER_BODY) * M_TO_CM,
        "penetration": penetration(pred_joints, floor_height) * M_TO_CM,
        "floating": floating(pred_joints, floor_height) * M_TO_CM,
        "skating": skating(pred_joints, gt_joints) * M_TO_CM,
        "gt_jitter": jitter(gt_joints, fps),
    }
    return {name: float(value) for name, value in metrics.items()}


def format_table(rows, columns=None):
    """Render ``{row_name: {metric: value}}`` as an aligned text table."""
    columns = columns or METRIC_NAMES
    header = ["model"] + [f"{c} ({METRIC_UNITS.get(c, '')})" if c in METRIC_UNITS else c for c in columns]
    body = [[name] + [f"{values.get(c, float('nan')):.3f}" for c in columns] for name, values in rows.items()]
    widths = [max(len(str(r[i])) for r in [header] + body) for i in range(len(header))]
    lines = []
    for row in [header] + body:
        lines.append(" | ".join(str(cell).rjust(w) for cell, w in zip(row, widths)))  # noqa: B905
    lines.insert(1, "-+-".join("-" * w for w in widths))
    return "\n".join(lines)
