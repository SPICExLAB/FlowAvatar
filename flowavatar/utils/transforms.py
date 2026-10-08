'''
# --------------------------------------------
# utility functions for 3D transformation
# --------------------------------------------
# AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing (ECCV 2022)
# https://github.com/eth-siplab/AvatarPoser
# Jiaxi Jiang (jiaxi.jiang@inf.ethz.ch)
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

from torch.nn import functional as F
from human_body_prior.tools import tgm_conversion as tgm
from human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa
import torch
import torch.nn as nn


# --------------------------------------------
# basic transformation
# --------------------------------------------

def bgs(d6s):
    d6s = d6s.reshape(-1, 2, 3).permute(0, 2, 1)
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:,:,0], p=2, dim=1)
    a2 = d6s[:,:,1]
    c = torch.bmm(b1.view(bsz,1,-1),a2.view(bsz,-1,1)).view(bsz,1)*b1
    b2 = F.normalize(a2-c,p=2,dim=1)
    b3=torch.cross(b1,b2,dim=1)
    return torch.stack([b1,b2,b3],dim=-1)

def matrot2sixd(pose_matrot):
    '''
    :param pose_matrot: Nx3x3
    :return: pose_6d: Nx6
    '''
    pose_6d = torch.cat([pose_matrot[:,:3,0], pose_matrot[:,:3,1]], dim=1)
    return pose_6d


def aa2sixd(pose_aa):
    '''
    :param pose_aa Nx3
    :return: pose_6d: Nx6
    '''
    pose_matrot = aa2matrot(pose_aa)
    pose_6d = matrot2sixd(pose_matrot)
    return pose_6d

def sixd2matrot(pose_6d):
    '''
    :param pose_6d: Nx6
    :return: pose_matrot: Nx3x3
    '''
    rot_vec_1 = pose_6d[:,:3]
    rot_vec_2 = pose_6d[:,3:6]
    rot_vec_3 = torch.linalg.cross(rot_vec_1, rot_vec_2, dim=1)
    pose_matrot = torch.stack([rot_vec_1,rot_vec_2,rot_vec_3],dim=-1)
    return pose_matrot

def sixd2aa(pose_6d, batch = False):
    '''
    :param pose_6d: Nx6
    :return: pose_aa: Nx3
    '''
    if batch:
        B,J,C = pose_6d.shape
        pose_6d = pose_6d.reshape(-1,6)
    pose_matrot = sixd2matrot(pose_6d)
    pose_aa = matrot2aa(pose_matrot)
    if batch:
        pose_aa = pose_aa.reshape(B,J,3)
    return pose_aa

def sixd2quat(pose_6d):
    '''
    :param pose_6d: Nx6
    :return: pose_quaternion: Nx4
    '''
    pose_mat = sixd2matrot(pose_6d)
    pose_mat_34 = torch.cat((pose_mat, torch.zeros(pose_mat.size(0), pose_mat.size(1), 1)), dim=-1)
    pose_quaternion = tgm.rotation_matrix_to_quaternion(pose_mat_34)
    return pose_quaternion

def quat2aa(pose_quat):
    '''
    :param pose_quat: Nx4
    :return: pose_aa: Nx3
    '''
    return tgm.quaternion_to_angle_axis(pose_quat)

def matrot2mat4x4(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: BxJx3x3 array of a batch of rotation matrices
            - t: BxJx3 array of a batch of translation vectors
        Returns:
            - T: BxJx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1, 0, 0]),
                      F.pad(t[:, :, :, None], [0, 0, 0, 1, 0, 0], value=1)], dim=-1)




#* --------------------------------------------
#* FK, IK related
#*--------------------------------------------


def forward_kinematics_R(R_local: torch.Tensor, parent):
    r"""
    :math:`R_global = FK(R_local)`

    Forward kinematics that computes the global rotation of each joint from local rotations. (torch, batch)

    Notes
    -----
    A joint's *local* rotation is expressed in its parent's frame.

    A joint's *global* rotation is expressed in the base (root's parent) frame.

    R_local[:, i], parent[i] should be the local rotation and parent joint id of
    joint i. parent[i] must be smaller than i for any i > 0.

    Args
    -----
    :param R_local: Joint local rotation tensor in shape [batch_size, *] that can reshape to
                    [batch_size, num_joint, 3, 3] (rotation matrices).
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Joint global rotation, in shape [batch_size, num_joint, 3, 3].
    """
    R_local = R_local.view(R_local.shape[0], -1, 3, 3)
    R_global = _forward_tree(R_local, parent, torch.bmm)
    return R_global

def _forward_tree(x_local: torch.Tensor, parent, reduction_fn):
    r"""
    Multiply/Add matrices along the tree branches. x_local [N, J, *]. parent [J].
    """
    x_global = [x_local[:, 0]]
    for i in range(1, len(parent)):
        x_global.append(reduction_fn(x_global[parent[i]], x_local[:, i]))
    x_global = torch.stack(x_global, dim=1)
    return x_global




class ForwardKinematics(nn.Module):
    def __init__(self, body_model):
        super().__init__()
        self.body_model = body_model
    
    def forward(self, body_rot, root_rot, betas=None):
        """
        Convert 6D rotations to joint positions using SMPL model
        Args:
            body_rot: Body rotations in 6D format (B, 21*6)
            root_rot: Global orientation in 6D format (B, 6)
        Returns:
            joint_positions: (B, J, 3) where J is number of joints
        """
        B = root_rot.shape[0]
        
        # Convert root and body rotations separately
        root_rot_aa = sixd2aa(root_rot).reshape(B, 3)              # [B, 3]
        body_rot_aa = sixd2aa(body_rot.reshape(-1, 6)).reshape(B, 63)  # [B, 63]
        
        # Get joint positions through SMPL
        with torch.cuda.amp.autocast(enabled=True):
            body_pose = self.body_model(
                pose_body=body_rot_aa,
                root_orient=root_rot_aa,
                betas=betas
            )
        
        return body_pose.Jtr[:, :22]


def convert_to_global_space(joint_positions, head_transform):
    """
    Convert local joint positions to global space
    Args:
        joint_positions: Local joint positions [B, T, 22, 3] or [B, 22, 3]
        head_transform: Head transformation matrix [B, T, 4, 4] or [B, 4, 4]
    Returns:
        Global joint positions with same shape as input
    """
    # Handle different input shapes
    if len(joint_positions.shape) == 4:  # Sequence data [B, T, 22, 3]
        B, T = joint_positions.shape[0], joint_positions.shape[1]
        # Reshape to [B*T, 22, 3]
        joint_positions_flat = joint_positions.reshape(-1, 22, 3)
        head_transform_flat = head_transform.reshape(-1, 4, 4)
        
        # Extract head positions
        gt_head_pos = head_transform_flat[..., :3, 3]  # [B*T, 3]
        pred_head_pos = joint_positions_flat[:, 15]  # [B*T, 3]
        
        # Calculate and apply offset
        head_offset = gt_head_pos - pred_head_pos  # [B*T, 3]
        global_positions = joint_positions_flat + head_offset.unsqueeze(1)  # [B*T, 22, 3]
        
        # Reshape back to sequence format
        return global_positions.reshape(B, T, 22, 3)
    else:  # Single frame data [B, 22, 3]
        # Extract head positions
        gt_head_pos = head_transform[..., :3, 3]  # [B, 3]
        pred_head_pos = joint_positions[:, 15]  # [B, 3]
        
        # Calculate and apply offset
        head_offset = gt_head_pos - pred_head_pos  # [B, 3]
        return joint_positions + head_offset.unsqueeze(1)  # [B, 22, 3]



