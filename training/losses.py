"""Training objective (paper Sec. 3.1.3, Eq. 9).

Pose losses: L1 on the root orientation, the local joint rotations, the
global joint rotations and the joint positions (after forward kinematics,
head-aligned to the ground truth).
Temporal losses: joint acceleration and multi-scale joint velocity.
Hand alignment: position error of the two wrists.
Ground contact: zero velocity of ankle/foot joints while the ground-truth
foot is static, and a penalty for joints below the floor.

The velocity, foot-contact and penetration terms follow AvatarJLM
(Zheng et al., ICCV 2023). Shape parameters are supervised through the
joint-position terms (forward kinematics uses the predicted betas); the
optional ``shape`` term compares them with the ground truth directly.
"""

import torch
from torch import nn

LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT, RIGHT_FOOT = 7, 8, 10, 11
FOOT_JOINTS = [LEFT_ANKLE, LEFT_FOOT, RIGHT_ANKLE, RIGHT_FOOT]
HAND_JOINTS = [20, 21]
CONTACT_SPEED_THRESH = 0.01  # metres per frame


def velocity_loss(loss_fn, pred, gt, interval=1):
    """Velocity error at the given frame interval. pred, gt: [B, T, J, 3]."""
    pred_vel = pred[:, interval::interval] - pred[:, :-interval:interval]
    gt_vel = gt[:, interval::interval] - gt[:, :-interval:interval]
    return loss_fn(pred_vel, gt_vel)


def foot_contact_loss(loss_fn, pred, gt):
    """Penalise motion of ankle/foot joints while the ground-truth joint is static."""
    gt_feet, pred_feet = gt[:, :, FOOT_JOINTS], pred[:, :, FOOT_JOINTS]
    static = torch.norm(gt_feet[:, 1:] - gt_feet[:, :-1], dim=-1) <= CONTACT_SPEED_THRESH
    pred_vel = (pred_feet[:, 1:] - pred_feet[:, :-1]) * static.unsqueeze(-1)
    return loss_fn(pred_vel, torch.zeros_like(pred_vel))


def penetration_loss(pred, floor_height):
    """Mean depth of the lowest joint below the floor. pred: [B, T, J, 3], floor_height: [B]."""
    lowest = pred[..., 2].min(dim=-1).values  # [B, T]
    return torch.relu(floor_height.reshape(-1, 1) - lowest).mean()


class PoseLoss(nn.Module):
    TERMS = (
        "root_orientation",
        "local_pose",
        "global_pose",
        "joint_position",
        "acceleration",
        "velocity",
        "hand_alignment",
        "foot_contact",
        "penetration",
        "shape",
    )

    def __init__(self, loss_type="l1", **weights):
        super().__init__()
        unknown = set(weights) - set(self.TERMS)
        if unknown:
            raise ValueError(f"unknown loss terms: {sorted(unknown)} (known: {list(self.TERMS)})")
        self.weights = {term: float(weights.get(term, 0.0)) for term in self.TERMS}
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"loss_type must be 'l1' or 'l2', got {loss_type!r}")

    def forward(self, pred, gt):
        """
        Args:
            pred: dict with local_pose [B, T, 132], global_pose [B, T, 132],
                  joints [B, T, 22, 3] (head-aligned to the ground truth),
                  betas [B, T, 10]
            gt:   dict with the same keys plus floor_height [B]
        Returns:
            dict of unweighted terms plus the weighted ``total``
        """
        fn = self.loss_fn
        pred_local, gt_local = pred["local_pose"], gt["local_pose"]
        pred_joints, gt_joints = pred["joints"], gt["joints"]
        num_frames = pred_joints.shape[1]
        zero = pred_joints.new_zeros(())

        terms = {
            "root_orientation": fn(pred_local[..., :6], gt_local[..., :6]),
            "local_pose": fn(pred_local[..., 6:], gt_local[..., 6:]),
            "global_pose": fn(pred["global_pose"], gt["global_pose"]),
            "joint_position": fn(pred_joints, gt_joints),
            "hand_alignment": fn(pred_joints[:, :, HAND_JOINTS], gt_joints[:, :, HAND_JOINTS]),
            "penetration": penetration_loss(pred_joints, gt["floor_height"]),
            "shape": fn(pred["betas"], gt["betas"]),
        }
        if num_frames > 2:
            pred_acc = pred_joints[:, :-2] - 2 * pred_joints[:, 1:-1] + pred_joints[:, 2:]
            gt_acc = gt_joints[:, :-2] - 2 * gt_joints[:, 1:-1] + gt_joints[:, 2:]
            terms["acceleration"] = fn(pred_acc, gt_acc)
        else:
            terms["acceleration"] = zero

        velocity = zero
        for interval in (1, 3, 5):
            if num_frames > interval:
                velocity = velocity + velocity_loss(fn, pred_joints, gt_joints, interval) / interval
        terms["velocity"] = velocity
        terms["foot_contact"] = foot_contact_loss(fn, pred_joints, gt_joints) if num_frames > 1 else zero

        terms["total"] = sum(self.weights[term] * terms[term] for term in self.TERMS)
        return terms
