"""Training wrapper: ``FlowAvatarNetwork`` + body-model forward kinematics.

The network predicts local joint rotations (22 x 6D) and shape parameters;
this wrapper adds the global rotations (kinematic chain) and the joint
positions (SMPL+H or SMPL-X body model, see the ``body_model`` config block)
that the losses and metrics need. Checkpoints written by ``save_checkpoint``
contain the bare network ``state_dict`` plus its ``model_params`` and load
directly with ``flowavatar.models.load_checkpoint``.
"""

from types import SimpleNamespace

import torch
from torch import nn

from flowavatar.models import FlowAvatarNetwork
from flowavatar.utils.transforms import forward_kinematics_R, matrot2sixd, sixd2aa, sixd2matrot

NUM_JOINTS = 22
HEAD = 15

# Key renames for checkpoints written by the research code base
LEGACY_KEY_RENAMES = {".temporal_layer.": ".lstm."}


class FlowAvatarPoseModel(nn.Module):
    def __init__(self, model_params, body_model_file):
        """
        Args:
            model_params: ``FlowAvatarNetwork`` parameters; ``num_betas`` must
                match the body model (16 for SMPL+H, 10 for SMPL-X)
            body_model_file: ``model.npz`` used for forward kinematics
        """
        super().__init__()
        from human_body_prior.body_model.body_model import BodyModel

        self.model_params = dict(model_params)
        self.num_betas = int(self.model_params.get("num_betas", 10))
        self.net = FlowAvatarNetwork(SimpleNamespace(model_params=self.model_params))
        self.body_model = BodyModel(bm_fname=str(body_model_file), num_betas=self.num_betas)
        for param in self.body_model.parameters():
            param.requires_grad_(False)
        self.parents = self.body_model.kintree_table[0][:NUM_JOINTS].long().tolist()

    def forward_kinematics(self, pose_6d, betas):
        """Joint positions [N, 22, 3] from local rotations [N, 132] and betas [N, num_betas]."""
        num_frames = pose_6d.shape[0]
        pose_aa = sixd2aa(pose_6d.reshape(-1, 6)).reshape(num_frames, NUM_JOINTS * 3).float()
        body = self.body_model(root_orient=pose_aa[:, :3], pose_body=pose_aa[:, 3:], betas=betas.float())
        return body.Jtr[:, :NUM_JOINTS]

    def global_rotations(self, pose_6d):
        """Global joint rotations [B, T, 132] from local rotations [B, T, 132]."""
        batch, num_frames = pose_6d.shape[:2]
        local_mat = sixd2matrot(pose_6d.reshape(-1, 6)).reshape(batch * num_frames, NUM_JOINTS, 3, 3)
        global_mat = forward_kinematics_R(local_mat, self.parents)
        return matrot2sixd(global_mat.reshape(-1, 3, 3)).reshape(batch, num_frames, NUM_JOINTS * 6)

    def forward(self, x, do_fk=True):
        """
        Args:
            x: [B, T, 90] sparse tracking features
        Returns:
            dict with local_pose [B, T, 132], betas [B, T, num_betas],
            global_pose [B, T, 132] and (if do_fk) joints [B, T, 22, 3]
        """
        out = self.net(x)
        pose, betas = out["pose_params"], out["betas"]
        batch, num_frames = pose.shape[:2]
        result = {"local_pose": pose, "betas": betas, "global_pose": self.global_rotations(pose)}
        if do_fk:
            joints = self.forward_kinematics(pose.reshape(-1, NUM_JOINTS * 6), betas.reshape(-1, self.num_betas))
            result["joints"] = joints.reshape(batch, num_frames, NUM_JOINTS, 3)
        return result


def head_align(pred_joints, gt_joints):
    """Translate predicted joints so the head coincides with the ground truth."""
    return pred_joints - pred_joints[..., HEAD : HEAD + 1, :] + gt_joints[..., HEAD : HEAD + 1, :]


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def save_checkpoint(model, path, epoch=None, **extra):
    """Save in the format read by ``flowavatar.models.load_checkpoint``."""
    bare = unwrap(model)
    torch.save({"state_dict": bare.net.state_dict(), "epoch": epoch, "model_params": bare.model_params, **extra}, path)


def load_weights(model, path, strict=True):
    """Load network weights from a checkpoint; returns the checkpoint dict.

    Also accepts checkpoints of the research code base (``temporal_layer``
    key names, auxiliary ``joint_predictor`` head).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    renamed = {}
    for key, value in state_dict.items():
        if key.startswith("joint_predictor"):
            continue
        for old, new in LEGACY_KEY_RENAMES.items():
            key = key.replace(old, new)
        renamed[key] = value
    unwrap(model).net.load_state_dict(renamed, strict=strict)
    return checkpoint
