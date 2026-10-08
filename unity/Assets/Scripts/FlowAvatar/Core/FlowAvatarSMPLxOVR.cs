using System;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
#endif

/// <summary>
/// SMPL-X body whose hands are rigged with the Meta OVR hand skeleton, driven by
/// SMPL-X shape (betas), expression and pose-corrective parameters.
///
/// All rest data -- each bone's rest position and rest local rotation -- is read
/// from the mesh bind poses, never from the current transforms. Shape changes and
/// pose correctives are therefore independent of the pose the avatar is in, and
/// bones whose rest frame is not axis-aligned with SMPL-X (the OVR finger bones,
/// and left_wrist, which the OVR rig rotates by 180 degrees about X) need no
/// special cases. A rotation q given in an SMPL-X joint frame (already mirrored to
/// Unity) is applied as localRotation = q * GetRestLocalRotation(bone).
///
/// Shape uses the betas-to-joints regressor of the SMPL-X Unity add-on
/// (Resources/smplx_betas_to_joints_{gender}.json): each bone moves by the
/// displacement of its SMPL-X joint (bones without one follow their parent).
/// </summary>
public class FlowAvatarSMPLxOVR : MonoBehaviour
{
    public const int NUM_BETAS = 10;
    public const int NUM_EXPRESSIONS = 10;
    public const int NUM_JOINTS = 25;
    public const int SMPLX_NUM_JOINTS = 55;

    public enum ModelType { Unknown, Female, Neutral, Male };

    public ModelType modelType = ModelType.Unknown;

    public float[] betas = new float[NUM_BETAS];
    public float[] expressions = new float[NUM_EXPRESSIONS];

    public bool usePoseCorrectives = true;
    public bool showJointPositions = false;

    // The first 25 SMPL-X joints (body, jaw, eyes); they carry the pose correctives
    public static readonly string[] BodyJointNames = {
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle",
        "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "jaw", "left_eye_smplhf",
        "right_eye_smplhf" };

    // OVR hand bones that sit on an SMPL-X finger joint (SMPL-X joint index). The
    // remaining OVR bones (thumb0, pinky0, *_null, forearm_stub) follow their parent.
    public static readonly IReadOnlyDictionary<string, int> OvrBoneToSmplxJoint = new Dictionary<string, int>
    {
        { "b_l_index1", 25 }, { "b_l_index2", 26 }, { "b_l_index3", 27 },
        { "b_l_middle1", 28 }, { "b_l_middle2", 29 }, { "b_l_middle3", 30 },
        { "b_l_pinky1", 31 }, { "b_l_pinky2", 32 }, { "b_l_pinky3", 33 },
        { "b_l_ring1", 34 }, { "b_l_ring2", 35 }, { "b_l_ring3", 36 },
        { "b_l_thumb1", 37 }, { "b_l_thumb2", 38 }, { "b_l_thumb3", 39 },
        { "b_r_index1", 40 }, { "b_r_index2", 41 }, { "b_r_index3", 42 },
        { "b_r_middle1", 43 }, { "b_r_middle2", 44 }, { "b_r_middle3", 45 },
        { "b_r_pinky1", 46 }, { "b_r_pinky2", 47 }, { "b_r_pinky3", 48 },
        { "b_r_ring1", 49 }, { "b_r_ring2", 50 }, { "b_r_ring3", 51 },
        { "b_r_thumb1", 52 }, { "b_r_thumb2", 53 }, { "b_r_thumb3", 54 },
    };

    // The imported mesh asset; kept serialized so it survives editor reloads while
    // the renderer temporarily points at the shaped runtime copy.
    [SerializeField, HideInInspector] private Mesh _sourceMesh;

    private SkinnedMeshRenderer _smr;
    private Mesh _shapedMesh;                 // runtime copy carrying the shaped bind poses
    private Transform[] _bones;
    private Matrix4x4[] _sourceBindWorld;     // rest bone matrices in mesh space
    private Quaternion[] _restLocalRotations; // per bone, from the bind poses
    private int[] _boneParent;                // index into _bones, -1 if the parent is not a bone
    private int[] _boneSmplxJoint;            // SMPL-X joint index, -1 if none
    private int[] _boneOrder;                 // parents before children
    private Dictionary<Transform, int> _boneIndex;
    private int[] _bodyBone;                  // bone index of BodyJointNames[k]
    private Vector3[] _jointPositions;
    private bool _initialized;

    // Blend shape ranges: Shape000-009, Exp000-009, Pose000-...
    private int _shapeStart = -1;
    private int _expStart = -1;
    private int _poseStart = -1;
    private int _poseCount;

    // Betas-to-joints regressors per gender: [joint, axis, beta] in SMPL-X coordinates
    private static readonly Dictionary<string, float[,,]> Regressors = new Dictionary<string, float[,,]>();

    public void Awake()
    {
        if (!Initialize())
            return;

        // At runtime make the skeleton and bind poses consistent with the serialized
        // betas before any other component reads the rig.
        if (Application.isPlaying)
            ApplyShape();
    }

    void LateUpdate()
    {
        // After the motion controller and hand tracking have set this frame's pose
        if (usePoseCorrectives)
            UpdatePoseCorrectives();
    }

    private bool Initialize()
    {
        if (_initialized)
            return true;

        _smr = GetComponentInChildren<SkinnedMeshRenderer>();
        if (_smr == null)
        {
            Debug.LogError("[FlowAvatarSMPLxOVR] No SkinnedMeshRenderer found");
            return false;
        }
        if (_sourceMesh == null)
            _sourceMesh = _smr.sharedMesh;

        _bones = _smr.bones;
        int n = _bones.Length;
        Matrix4x4[] bindposes = _sourceMesh.bindposes;
        _boneIndex = new Dictionary<Transform, int>(n);
        for (int i = 0; i < n; i++)
            _boneIndex[_bones[i]] = i;

        _sourceBindWorld = new Matrix4x4[n];
        _restLocalRotations = new Quaternion[n];
        _boneParent = new int[n];
        _boneSmplxJoint = new int[n];
        for (int i = 0; i < n; i++)
        {
            _sourceBindWorld[i] = bindposes[i].inverse;
            Transform parent = _bones[i].parent;
            _boneParent[i] = parent != null && _boneIndex.TryGetValue(parent, out int p) ? p : -1;
            string name = _bones[i].name;
            int body = Array.IndexOf(BodyJointNames, name);
            _boneSmplxJoint[i] = body >= 0 ? body : OvrBoneToSmplxJoint.TryGetValue(name, out int j) ? j : -1;
        }
        for (int i = 0; i < n; i++)
        {
            int p = _boneParent[i];
            _restLocalRotations[i] = p >= 0 ? RotationOf(bindposes[p] * _sourceBindWorld[i]) : _bones[i].localRotation;
        }
        _boneOrder = HierarchyOrder();

        _bodyBone = new int[BodyJointNames.Length];
        for (int k = 0; k < BodyJointNames.Length; k++)
        {
            _bodyBone[k] = Array.FindIndex(_bones, b => b.name == BodyJointNames[k]);
            if (_bodyBone[k] < 0)
            {
                Debug.LogError($"[FlowAvatarSMPLxOVR] Bone '{BodyJointNames[k]}' not found in the skinned mesh");
                return false;
            }
        }
        _jointPositions = new Vector3[NUM_JOINTS];

        _shapeStart = _sourceMesh.GetBlendShapeIndex("Shape000");
        _expStart = _sourceMesh.GetBlendShapeIndex("Exp000");
        _poseStart = _sourceMesh.GetBlendShapeIndex("Pose000");
        _poseCount = _poseStart >= 0 ? _sourceMesh.blendShapeCount - _poseStart : 0;

        _initialized = true;
        UpdateJointPositions(false);
        return true;
    }

    private int[] HierarchyOrder()
    {
        var order = new List<int>(_bones.Length);
        var done = new bool[_bones.Length];
        void Visit(int i)
        {
            if (done[i]) return;
            if (_boneParent[i] >= 0) Visit(_boneParent[i]);
            done[i] = true;
            order.Add(i);
        }
        for (int i = 0; i < _bones.Length; i++)
            Visit(i);
        return order.ToArray();
    }

    private static Quaternion RotationOf(Matrix4x4 m) => Quaternion.LookRotation(m.GetColumn(2), m.GetColumn(1));

    private static float[,,] Regressor(string gender)
    {
        if (Regressors.TryGetValue(gender, out var cached))
            return cached;

        string name = "smplx_betas_to_joints_" + gender;
        TextAsset asset = Resources.Load<TextAsset>(name);
        if (asset == null)
        {
            Debug.LogError($"[FlowAvatarSMPLxOVR] Resources/{name}.json not found (part of the SMPL-X Unity add-on, see Assets/SMPLX-OVR/README.md)");
            return null;
        }
        SimpleJSON.JSONNode table = SimpleJSON.JSON.Parse(asset.text)["betasJ_regr"];
        var regressor = new float[SMPLX_NUM_JOINTS, 3, NUM_BETAS];
        for (int j = 0; j < SMPLX_NUM_JOINTS; j++)
            for (int axis = 0; axis < 3; axis++)
                for (int b = 0; b < NUM_BETAS; b++)
                    regressor[j, axis, b] = table[j][axis][b].AsFloat;
        Regressors[gender] = regressor;
        return regressor;
    }

    public bool HasBetaShapes() => _shapeStart >= 0;

    public bool HasExpressions() => _expStart >= 0;

    public bool HasPoseCorrectives() => _poseCount > 0;

    public Vector3[] GetJointPositions() => _jointPositions;

    public void GetModelInfo(out int shapes, out int expressionCount, out int poseCorrectives)
    {
        shapes = HasBetaShapes() ? NUM_BETAS : 0;
        expressionCount = HasExpressions() ? NUM_EXPRESSIONS : 0;
        poseCorrectives = _poseCount;
    }

    /// <summary>
    /// Rest local rotation of a bone of this avatar (identity for the SMPL-X body
    /// joints except left_wrist; the OVR rest orientation for the hand bones).
    /// </summary>
    public Quaternion GetRestLocalRotation(Transform bone)
    {
        if (!Initialize())
            return Quaternion.identity;
        return _boneIndex.TryGetValue(bone, out int i) ? _restLocalRotations[i] : Quaternion.identity;
    }

    public void SetBetaShapes()
    {
        if (!Initialize())
            return;
        if (!HasBetaShapes())
        {
            Debug.LogError("[FlowAvatarSMPLxOVR] The mesh has no shape blend shapes");
            return;
        }
        ApplyShape();
    }

    public float[] GetBetaShapes() => (float[])betas.Clone();

    /// <summary>
    /// Applies the betas: shape blend shapes, plus bone rest positions and bind poses
    /// moved by the betas-to-joints regressor. Independent of the current pose.
    /// </summary>
    private void ApplyShape()
    {
        if (!Initialize())
            return;

        bool defaultShape = Array.TrueForAll(betas, b => b == 0f);

        // Joint displacements (SMPL-X frame, mirrored to Unity mesh space: x -> -x)
        var jointOffset = new Vector3[SMPLX_NUM_JOINTS];
        if (!defaultShape)
        {
            string gender = GenderKey();
            float[,,] regressor = gender != null ? Regressor(gender) : null;
            if (regressor == null)
            {
                Debug.LogError("[FlowAvatarSMPLxOVR] Cannot apply betas: model type not set or regressor missing");
                return;
            }
            for (int j = 0; j < SMPLX_NUM_JOINTS; j++)
            {
                var d = Vector3.zero;
                for (int b = 0; b < NUM_BETAS; b++)
                    d += new Vector3(regressor[j, 0, b], regressor[j, 1, b], regressor[j, 2, b]) * betas[b];
                jointOffset[j] = new Vector3(-d.x, d.y, d.z);
            }
        }

        // Each bone moves with its SMPL-X joint; bones without one follow their parent
        int n = _bones.Length;
        var offset = new Vector3[n];
        var bindWorld = new Matrix4x4[n];
        var bindposes = new Matrix4x4[n];
        foreach (int i in _boneOrder)
        {
            int j = _boneSmplxJoint[i];
            int p = _boneParent[i];
            offset[i] = j >= 0 ? jointOffset[j] : (p >= 0 ? offset[p] : Vector3.zero);
            Matrix4x4 m = _sourceBindWorld[i];
            m.SetColumn(3, m.GetColumn(3) + (Vector4)offset[i]);
            bindWorld[i] = m;
            bindposes[i] = m.inverse;
        }

        // A bone's rest offset in its parent's frame does not depend on the current pose
        for (int i = 0; i < n; i++)
        {
            int p = _boneParent[i];
            if (p >= 0)
                _bones[i].localPosition = bindposes[p].MultiplyPoint3x4(bindWorld[i].GetColumn(3));
        }

        if (defaultShape && _shapedMesh == null)
        {
            _smr.sharedMesh = _sourceMesh;
        }
        else
        {
            if (_shapedMesh == null)
            {
                _shapedMesh = Instantiate(_sourceMesh);
                _shapedMesh.name = _sourceMesh.name + " (shaped)";
                _shapedMesh.hideFlags = HideFlags.DontSave;
            }
            _shapedMesh.bindposes = bindposes;
            _smr.sharedMesh = _shapedMesh;
        }

        if (HasBetaShapes())
            for (int i = 0; i < NUM_BETAS; i++)
                _smr.SetBlendShapeWeight(_shapeStart + i, betas[i] * 100f); // blend shape weights are percentages

        UpdateJointPositions(false);
    }

    private string GenderKey()
    {
        switch (modelType)
        {
            case ModelType.Female: return "female";
            case ModelType.Neutral: return "neutral";
            case ModelType.Male: return "male";
            default: return null;
        }
    }

    // Restores the imported mesh and rest positions (used while the editor saves)
    private void RestoreSourceState()
    {
        Matrix4x4[] bindposes = _sourceMesh.bindposes;
        for (int i = 0; i < _bones.Length; i++)
        {
            int p = _boneParent[i];
            if (p >= 0)
                _bones[i].localPosition = bindposes[p].MultiplyPoint3x4(_sourceBindWorld[i].GetColumn(3));
        }
        _smr.sharedMesh = _sourceMesh;
    }

    public void SetExpressions()
    {
        if (!Initialize())
            return;
        if (!HasExpressions())
        {
            Debug.LogError("[FlowAvatarSMPLxOVR] The mesh has no expression blend shapes");
            return;
        }
        for (int i = 0; i < NUM_EXPRESSIONS; i++)
            _smr.SetBlendShapeWeight(_expStart + i, expressions[i] * 100f);
    }

    /// <summary>
    /// Moves the avatar vertically so that its lowest vertex rests on y = 0 of the
    /// avatar's parent space.
    /// </summary>
    public void SnapToGroundPlane()
    {
        if (!Initialize())
            return;
        var baked = new Mesh();
        _smr.BakeMesh(baked, true);
        Matrix4x4 meshToParent = Matrix4x4.TRS(_smr.transform.position, _smr.transform.rotation, Vector3.one);
        if (transform.parent != null)
            meshToParent = transform.parent.worldToLocalMatrix * meshToParent;
        float lowest = float.MaxValue;
        foreach (Vector3 v in baked.vertices)
            lowest = Mathf.Min(lowest, meshToParent.MultiplyPoint3x4(v).y);
        DestroyImmediate(baked);
        transform.localPosition -= new Vector3(0f, lowest, 0f);
        UpdateJointPositions(false);
    }

    /// <summary>
    /// Puts every bone back to its rest orientation.
    /// </summary>
    public void ResetToRestPose()
    {
        if (!Initialize())
            return;
        for (int i = 0; i < _bones.Length; i++)
            if (_boneParent[i] >= 0)
                _bones[i].localRotation = _restLocalRotations[i];
        UpdatePoseCorrectives();
        UpdateJointPositions(false);
    }

    public void EnablePoseCorrectives(bool enabled)
    {
        usePoseCorrectives = enabled;
        if (!Initialize() || !HasPoseCorrectives())
            return;
        if (enabled)
            UpdatePoseCorrectives();
        else
            for (int i = 0; i < _poseCount; i++)
                _smr.SetBlendShapeWeight(_poseStart + i, 0f);
    }

    /// <summary>
    /// SMPL-X pose blend shapes of the body joints: for joint k (1..24) the nine
    /// weights are the entries of (R_k - I), row-major, where R_k is the joint's
    /// rotation relative to its rest orientation, expressed in the SMPL-X frame.
    /// </summary>
    public void UpdatePoseCorrectives()
    {
        if (!usePoseCorrectives || !Initialize() || !HasPoseCorrectives())
            return;

        for (int k = 1; k < BodyJointNames.Length; k++)
        {
            int first = _poseStart + (k - 1) * 9;
            if (first + 9 > _poseStart + _poseCount)
                break;

            int b = _bodyBone[k];
            Quaternion q = _bones[b].localRotation * Quaternion.Inverse(_restLocalRotations[b]);
            // Unity -> SMPL-X frame: the coordinate mirror x -> -x maps (x, y, z, w) to (x, -y, -z, w)
            Matrix4x4 r = Matrix4x4.Rotate(new Quaternion(q.x, -q.y, -q.z, q.w));
            for (int row = 0; row < 3; row++)
                for (int col = 0; col < 3; col++)
                    _smr.SetBlendShapeWeight(first + row * 3 + col, 100f * (r[row, col] - (row == col ? 1f : 0f)));
        }
    }

    /// <summary>
    /// Refreshes the cached body joint positions (editor gizmos). With
    /// recalculateJoints the betas are applied first.
    /// </summary>
    public bool UpdateJointPositions(bool recalculateJoints = false)
    {
        if (!Initialize())
            return false;
        if (recalculateJoints)
        {
            ApplyShape();
            return true;
        }
        for (int k = 0; k < NUM_JOINTS; k++)
            _jointPositions[k] = _bones[_bodyBone[k]].position;
        return true;
    }

#if UNITY_EDITOR
    // A shaped copy of the mesh exists only in memory; swap the imported mesh back
    // while a scene or prefab is written so neither stores the copy.
    private static readonly List<FlowAvatarSMPLxOVR> _swappedForSave = new List<FlowAvatarSMPLxOVR>();

    [InitializeOnLoadMethod]
    private static void RegisterSaveHooks()
    {
        EditorSceneManager.sceneSaving += (scene, path) => SwapForSave(m => m.gameObject.scene == scene);
        EditorSceneManager.sceneSaved += scene => RestoreAfterSave();
        PrefabStage.prefabSaving += root => SwapForSave(m => m.transform.IsChildOf(root.transform));
        PrefabStage.prefabSaved += root => RestoreAfterSave();
    }

    private static void SwapForSave(Func<FlowAvatarSMPLxOVR, bool> inScope)
    {
        foreach (var m in Resources.FindObjectsOfTypeAll<FlowAvatarSMPLxOVR>())
        {
            if (m._initialized && m._shapedMesh != null && m._smr != null &&
                m._smr.sharedMesh == m._shapedMesh && inScope(m))
            {
                m.RestoreSourceState();
                _swappedForSave.Add(m);
            }
        }
    }

    private static void RestoreAfterSave()
    {
        foreach (var m in _swappedForSave)
            if (m != null)
                m.ApplyShape();
        _swappedForSave.Clear();
    }
#endif
}

#if UNITY_EDITOR
/// <summary>
/// Inspector for previewing shape, expression and pose correctives in the editor.
/// </summary>
[CustomEditor(typeof(FlowAvatarSMPLxOVR))]
public class FlowAvatarSMPLxOVR_Editor : Editor
{
    private bool _snapToGround = true;

    public override void OnInspectorGUI()
    {
        var avatar = (FlowAvatarSMPLxOVR)target;
        avatar.Awake();
        avatar.GetModelInfo(out int shapes, out int expressions, out int correctives);
        EditorGUILayout.HelpBox($"{shapes} shape, {expressions} expression and {correctives} pose-corrective blend shapes", MessageType.None);

        serializedObject.Update();
        EditorGUILayout.PropertyField(serializedObject.FindProperty("modelType"));
        serializedObject.ApplyModifiedProperties();

        Undo.RecordObject(avatar, "Edit SMPL-X parameters");

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Shape (betas)", EditorStyles.boldLabel);
        for (int i = 0; i < FlowAvatarSMPLxOVR.NUM_BETAS; i++)
            avatar.betas[i] = EditorGUILayout.Slider($"beta {i}", avatar.betas[i], -5f, 5f);
        _snapToGround = EditorGUILayout.Toggle("Snap to ground after apply", _snapToGround);
        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Apply")) ApplyShape(avatar);
            if (GUILayout.Button("Random"))
            {
                for (int i = 0; i < avatar.betas.Length; i++) avatar.betas[i] = UnityEngine.Random.Range(-2f, 2f);
                ApplyShape(avatar);
            }
            if (GUILayout.Button("Reset"))
            {
                Array.Clear(avatar.betas, 0, avatar.betas.Length);
                ApplyShape(avatar);
            }
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Expression", EditorStyles.boldLabel);
        for (int i = 0; i < FlowAvatarSMPLxOVR.NUM_EXPRESSIONS; i++)
            avatar.expressions[i] = EditorGUILayout.Slider($"expression {i}", avatar.expressions[i], -2f, 2f);
        using (new EditorGUILayout.HorizontalScope())
        {
            if (GUILayout.Button("Apply")) avatar.SetExpressions();
            if (GUILayout.Button("Reset"))
            {
                Array.Clear(avatar.expressions, 0, avatar.expressions.Length);
                avatar.SetExpressions();
            }
        }

        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Pose", EditorStyles.boldLabel);
        bool correctivesOn = EditorGUILayout.Toggle("Pose correctives", avatar.usePoseCorrectives);
        if (correctivesOn != avatar.usePoseCorrectives)
            avatar.EnablePoseCorrectives(correctivesOn);
        if (GUILayout.Button("Reset to rest pose"))
            avatar.ResetToRestPose();
        bool showJoints = EditorGUILayout.Toggle("Show joint positions", avatar.showJointPositions);
        if (showJoints != avatar.showJointPositions)
        {
            avatar.showJointPositions = showJoints;
            avatar.UpdateJointPositions(false);
            SceneView.RepaintAll();
        }
    }

    private void ApplyShape(FlowAvatarSMPLxOVR avatar)
    {
        avatar.SetBetaShapes();
        if (_snapToGround)
            avatar.SnapToGroundPlane();
    }

    private void OnSceneGUI()
    {
        var avatar = (FlowAvatarSMPLxOVR)target;
        if (!avatar.showJointPositions || avatar.GetJointPositions() == null)
            return;
        Handles.color = Color.yellow;
        foreach (Vector3 p in avatar.GetJointPositions())
            Handles.SphereHandleCap(0, p, Quaternion.identity, 0.025f, EventType.Repaint);
    }
}
#endif
