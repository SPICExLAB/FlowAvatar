"""Feature extraction: SMPL-X body parameters -> FlowAvatar training samples.

For every motion sequence this module computes

* the 90-D sparse tracking input built from the head and the two wrists:
  global 6D rotation, rotation velocity, position and position velocity of
  each tracker (54-D), plus the two hands expressed in the head frame
  (rotation, rotation velocity, position, position velocity; 36-D), and
* the full-body targets: local and global joint rotations (22 x 6D), joint
  positions, the head transform, an estimated floor height and per-joint
  foot-contact labels.

Velocities are per-frame differences, so frame ``t`` of the input depends on
frames ``t-1`` and ``t``; the first frame of every sequence is dropped and all
per-frame outputs have ``T-1`` frames. The layout matches ``FeatureExtractor``
in ``flowavatar/models/blocks.py``::

    0:18   global 6D rotation            (head, left hand, right hand)
    18:36  global rotation velocity      (head, left hand, right hand)
    36:45  global position               (head, left hand, right hand)
    45:54  global position velocity      (head, left hand, right hand)
    54:66  hand rotation in head frame            (left, right)
    66:78  hand rotation velocity in head frame   (left, right)
    78:84  hand position in head frame            (left, right)
    84:90  hand position velocity in head frame   (left, right)
"""

import torch
from human_body_prior.tools.rotation_tools import aa2matrot

from flowavatar.utils.transforms import forward_kinematics_R, matrot2sixd

NUM_BODY_JOINTS = 22
NUM_BETAS = 10
INPUT_DIM = 90

HEAD, LEFT_WRIST, RIGHT_WRIST = 15, 20, 21
LEFT_ANKLE, RIGHT_ANKLE, LEFT_FOOT, RIGHT_FOOT = 7, 8, 10, 11
TRACKED_JOINTS = [HEAD, LEFT_WRIST, RIGHT_WRIST]

# Floor-height and foot-contact heuristics (metres; speeds are per frame)
FLOOR_VEL_THRESH = 0.005
FLOOR_HEIGHT_OFFSET = 0.01
CONTACT_VEL_THRESH = 0.005
CONTACT_TOE_HEIGHT_THRESH = 0.04
CONTACT_ANKLE_HEIGHT_THRESH = 0.08


@torch.no_grad()
def body_joints(body_model, root_orient, pose_body, betas, trans=None, chunk=4096):
    """Run the body model in chunks and return the first 22 joints [T, 22, 3]."""
    joints = []
    for start in range(0, root_orient.shape[0], chunk):
        end = start + chunk
        kwargs = dict(root_orient=root_orient[start:end], pose_body=pose_body[start:end], betas=betas[start:end])
        if trans is not None:
            kwargs["trans"] = trans[start:end]
        joints.append(body_model(**kwargs).Jtr[:, :NUM_BODY_JOINTS])
    return torch.cat(joints, dim=0)


def _speed(positions):
    """Per-frame displacement magnitude [T] (last value repeated)."""
    speed = torch.norm(positions[1:] - positions[:-1], dim=-1)
    return torch.cat([speed, speed[-1:]])


def floor_and_foot_contacts(joints):
    """Estimate the floor height and per-joint foot-contact labels.

    The floor is the lowest height reached by a static toe joint (minus a
    small offset). A toe/ankle joint is in contact when it is nearly static
    and close to the floor.

    Args:
        joints: world-space joint positions [T, 22, 3], z up
    Returns:
        floor_height: scalar tensor
        contacts: [T, 22] with 1 at contact ankle/foot joints, 0 elsewhere
    """
    left_toe, right_toe = joints[:, LEFT_FOOT], joints[:, RIGHT_FOOT]
    left_ankle, right_ankle = joints[:, LEFT_ANKLE], joints[:, RIGHT_ANKLE]
    left_toe_speed, right_toe_speed = _speed(left_toe), _speed(right_toe)
    left_ankle_speed, right_ankle_speed = _speed(left_ankle), _speed(right_ankle)

    static_heights = [left_toe[left_toe_speed < FLOOR_VEL_THRESH, 2], right_toe[right_toe_speed < FLOOR_VEL_THRESH, 2]]
    static_heights = [h for h in static_heights if h.numel() > 0]
    if static_heights:
        floor_height = torch.cat(static_heights).min() - FLOOR_HEIGHT_OFFSET
    else:
        floor_height = torch.cat([left_toe[:, 2], right_toe[:, 2]]).min() - FLOOR_HEIGHT_OFFSET

    contacts = torch.zeros(joints.shape[0], NUM_BODY_JOINTS, dtype=joints.dtype, device=joints.device)
    contacts[:, LEFT_FOOT] = (
        (left_toe_speed < CONTACT_VEL_THRESH) & (left_toe[:, 2] - floor_height < CONTACT_TOE_HEIGHT_THRESH)
    ).to(joints.dtype)
    contacts[:, RIGHT_FOOT] = (
        (right_toe_speed < CONTACT_VEL_THRESH) & (right_toe[:, 2] - floor_height < CONTACT_TOE_HEIGHT_THRESH)
    ).to(joints.dtype)
    contacts[:, LEFT_ANKLE] = (
        (left_ankle_speed < CONTACT_VEL_THRESH) & (left_ankle[:, 2] - floor_height < CONTACT_ANKLE_HEIGHT_THRESH)
    ).to(joints.dtype)
    contacts[:, RIGHT_ANKLE] = (
        (right_ankle_speed < CONTACT_VEL_THRESH) & (right_ankle[:, 2] - floor_height < CONTACT_ANKLE_HEIGHT_THRESH)
    ).to(joints.dtype)
    return floor_height, contacts


@torch.no_grad()
def compute_sequence_features(root_orient, pose_body, trans, betas, body_model, fps, device="cpu", num_betas=NUM_BETAS):
    """Turn one SMPL-X motion into a FlowAvatar training sequence.

    Args:
        root_orient: [T, 3] root orientation (axis-angle)
        pose_body:   [T, 63] body pose, 21 joints (axis-angle)
        trans:       [T, 3] root translation (metres)
        betas:       [10] (or [T, 10]; the first frame is used) shape parameters
        body_model:  ``human_body_prior`` ``BodyModel`` (neutral SMPL-X, 10 betas)
        fps:         frame rate of the sequence (recorded in the output)
        device:      where to run the body model
        num_betas:   shape parameters kept (must match ``body_model``)

    Returns a dict of CPU float32 tensors:
        input [T-1, 90], local_pose [T-1, 132], global_pose [T-1, 132],
        joints [T-1, 22, 3], head_transform [T-1, 4, 4], betas [10],
        floor_height (scalar), foot_contact [T-1, 22], fps (int) and
        body_params {root_orient [T, 3], pose_body [T, 63], trans [T, 3]}.
    """
    device = torch.device(device)
    root_orient = torch.as_tensor(root_orient, dtype=torch.float32, device=device)
    pose_body = torch.as_tensor(pose_body, dtype=torch.float32, device=device)
    trans = torch.as_tensor(trans, dtype=torch.float32, device=device)
    betas = torch.as_tensor(betas, dtype=torch.float32, device=device)
    betas = (betas[0] if betas.dim() > 1 else betas).reshape(-1)[:num_betas]
    if betas.numel() < num_betas:
        betas = torch.cat([betas, betas.new_zeros(num_betas - betas.numel())])
    num_frames = root_orient.shape[0]
    if num_frames < 2:
        raise ValueError("a sequence needs at least two frames")

    # Local and global joint rotations (22 body joints)
    poses = torch.cat([root_orient, pose_body], dim=1)  # [T, 66]
    local_mat = aa2matrot(poses.reshape(-1, 3)).reshape(num_frames, NUM_BODY_JOINTS, 3, 3)
    parents = body_model.kintree_table[0][:NUM_BODY_JOINTS].long().tolist()
    global_mat = forward_kinematics_R(local_mat, parents)  # [T, 22, 3, 3]
    local_6d = matrot2sixd(local_mat.reshape(-1, 3, 3)).reshape(num_frames, NUM_BODY_JOINTS * 6)
    global_6d = matrot2sixd(global_mat.reshape(-1, 3, 3)).reshape(num_frames, NUM_BODY_JOINTS, 6)

    # Rotation velocity R_{t-1}^T R_t (inverse == transpose for rotations)
    rot_vel = torch.matmul(global_mat[:-1].transpose(-1, -2), global_mat[1:])
    rot_vel_6d = matrot2sixd(rot_vel.reshape(-1, 3, 3)).reshape(num_frames - 1, NUM_BODY_JOINTS, 6)

    # World-space joint positions and tracker positions / velocities
    joints = body_joints(body_model, root_orient, pose_body, betas[None].expand(num_frames, -1), trans)
    tracked_pos = joints[:, TRACKED_JOINTS]  # [T, 3, 3]
    tracked_vel = tracked_pos[1:] - tracked_pos[:-1]

    head_mat = global_mat[:, HEAD]  # [T, 3, 3]
    head_transform = torch.eye(4, device=device).repeat(num_frames, 1, 1)
    head_transform[:, :3, :3] = head_mat
    head_transform[:, :3, 3] = joints[:, HEAD]

    # Hands expressed in the head frame
    hand_mat = global_mat[:, [LEFT_WRIST, RIGHT_WRIST]]  # [T, 2, 3, 3]
    hand_in_head = torch.matmul(head_mat[:, None].transpose(-1, -2), hand_mat)
    hand_in_head_6d = matrot2sixd(hand_in_head.reshape(-1, 3, 3)).reshape(num_frames, 2, 6)
    hand_in_head_vel = torch.matmul(hand_in_head[:-1].transpose(-1, -2), hand_in_head[1:])
    hand_in_head_vel_6d = matrot2sixd(hand_in_head_vel.reshape(-1, 3, 3)).reshape(num_frames - 1, 2, 6)
    hand_rel = joints[:, [LEFT_WRIST, RIGHT_WRIST]] - joints[:, HEAD : HEAD + 1]  # [T, 2, 3]
    hand_pos_head = torch.matmul(hand_rel, head_mat)  # row p^T R == (R^T p)^T
    hand_pos_head_vel = hand_pos_head[1:] - hand_pos_head[:-1]

    n = num_frames - 1
    inputs = torch.cat(
        [
            global_6d[1:, TRACKED_JOINTS].reshape(n, -1),  # 0:18
            rot_vel_6d[:, TRACKED_JOINTS].reshape(n, -1),  # 18:36
            tracked_pos[1:].reshape(n, -1),  # 36:45
            tracked_vel.reshape(n, -1),  # 45:54
            hand_in_head_6d[1:].reshape(n, -1),  # 54:66
            hand_in_head_vel_6d.reshape(n, -1),  # 66:78
            hand_pos_head[1:].reshape(n, -1),  # 78:84
            hand_pos_head_vel.reshape(n, -1),  # 84:90
        ],
        dim=1,
    )
    assert inputs.shape[1] == INPUT_DIM

    floor_height, contacts = floor_and_foot_contacts(joints)

    def cpu(t):
        return t.detach().to("cpu").contiguous()

    return {
        "input": cpu(inputs),
        "local_pose": cpu(local_6d[1:]),
        "global_pose": cpu(global_6d[1:].reshape(n, -1)),
        "joints": cpu(joints[1:]),
        "head_transform": cpu(head_transform[1:]),
        "betas": cpu(betas),
        "floor_height": cpu(floor_height),
        "foot_contact": cpu(contacts[1:]),
        "fps": int(fps),
        "body_params": {"root_orient": cpu(root_orient), "pose_body": cpu(pose_body), "trans": cpu(trans)},
    }
