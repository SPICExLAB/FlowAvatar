using System;
using UnityEngine;

/// <summary>
/// The OVR hand rig as data: the bones the FlowAvatar avatar adds to the official
/// SMPL-X neutral FBX, and nothing of SMPL-X itself. Bone positions are stored as
/// offsets from an SMPL-X joint ("anchor") of the user's own SMPL-X file; rotations
/// are the OVR hand-skeleton rest orientations. All vectors are in the Unity mesh
/// space of the avatar (metres). Skin weights are not stored: the SMPL-X finger
/// weights are moved onto the matching OVR bones (see OvrHandRigPostprocessor).
/// </summary>
[Serializable]
public class OvrHandRigData
{
    public const string AssetPath = "Assets/Scripts/FlowAvatar/Editor/OvrHandRig/ovr_hand_rig.json";

    public string description;
    public OvrBoneDef[] bones;              // parents before children
    public OvrRestOverride[] restOverrides; // SMPL-X body bones whose rest orientation changes

    public static readonly string[] SmplxJointNames = {
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle",
        "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw", "left_eye_smplhf",
        "right_eye_smplhf", "left_index1", "left_index2", "left_index3", "left_middle1", "left_middle2",
        "left_middle3", "left_pinky1", "left_pinky2", "left_pinky3", "left_ring1", "left_ring2", "left_ring3",
        "left_thumb1", "left_thumb2", "left_thumb3", "right_index1", "right_index2", "right_index3",
        "right_middle1", "right_middle2", "right_middle3", "right_pinky1", "right_pinky2", "right_pinky3",
        "right_ring1", "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3" };

    public const int FirstFingerJoint = 25;

    public static bool IsSmplxFingerJoint(string name)
    {
        int i = Array.IndexOf(SmplxJointNames, name);
        return i >= FirstFingerJoint;
    }
}

[Serializable]
public class OvrBoneDef
{
    public string name;
    public string parent;
    public string anchor;     // SMPL-X joint the bone is placed relative to
    public Vector3 offset;    // rest position minus the anchor's rest position
    public Quaternion rotation;
}

[Serializable]
public class OvrRestOverride
{
    public string name;
    public Quaternion rotation;
}
