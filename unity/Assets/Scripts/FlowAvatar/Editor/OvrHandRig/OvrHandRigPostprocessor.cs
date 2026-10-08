using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.Collections;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Turns the official SMPL-X neutral FBX (from the SMPL-X Unity add-on, placed at
/// SourcePath) into the FlowAvatar avatar with Meta OVR hand bones, at import time:
/// applies the rest overrides, adds the OVR hand bones, removes the SMPL-X finger
/// joints and rebinds the mesh. Skin weights stay the official SMPL-X weights, with
/// each finger joint's weights moved onto the OVR bone at the same joint (the hand
/// painted in Maya for the original avatar differs from this by at most 0.008 in
/// weight, under 1 mm on curled fingers).
/// </summary>
public class OvrHandRigPostprocessor : AssetPostprocessor
{
    public const string SourcePath = "Assets/SMPLX-OVR/smplx-neutral.fbx";
    const int MaxInfluences = 8;

    public override uint GetVersion() => 3;

    bool IsTarget => assetPath == SourcePath;

    void OnPreprocessModel()
    {
        if (!IsTarget)
            return;
        var importer = (ModelImporter)assetImporter;
        importer.optimizeGameObjects = false;
        importer.skinWeights = ModelImporterSkinWeights.Custom;
        importer.maxBonesPerVertex = MaxInfluences;
        importer.minBoneWeight = 0f; // keep the small SMPL-X weights (the 0.001 default costs ~1.5 mm in posed limbs)
        importer.importNormals = ModelImporterNormals.Import;
        importer.importBlendShapeNormals = ModelImporterNormals.None;
        context.DependsOnSourceAsset(OvrHandRigData.AssetPath);
    }

    void OnPostprocessModel(GameObject root)
    {
        if (!IsTarget)
            return;
        if (!File.Exists(OvrHandRigData.AssetPath))
        {
            Debug.LogWarning($"[OvrHandRig] {OvrHandRigData.AssetPath} not found; {assetPath} imported unchanged");
            return;
        }
        var rig = JsonUtility.FromJson<OvrHandRigData>(File.ReadAllText(OvrHandRigData.AssetPath));

        var smr = root.GetComponentInChildren<SkinnedMeshRenderer>();
        Mesh mesh = smr.sharedMesh;
        Transform[] oldBones = smr.bones;
        Matrix4x4[] oldBind = mesh.bindposes;
        var restPosition = new Dictionary<string, Vector3>();
        for (int i = 0; i < oldBones.Length; i++)
            restPosition[oldBones[i].name] = oldBind[i].inverse.GetColumn(3);
        var byName = root.GetComponentsInChildren<Transform>(true).ToDictionary(t => t.name);
        Matrix4x4 meshToWorld = smr.transform.localToWorldMatrix;
        Quaternion meshRotation = smr.transform.rotation;

        // Rest overrides first (e.g. left_wrist), so OVR bones are placed under the final frames
        foreach (var o in rig.restOverrides)
            byName[o.name].rotation = meshRotation * o.rotation;

        var ovrBones = new List<Transform>();
        foreach (var def in rig.bones)
        {
            var t = new GameObject(def.name).transform;
            t.SetParent(byName[def.parent], false);
            t.position = meshToWorld.MultiplyPoint3x4(restPosition[def.anchor] + def.offset);
            t.rotation = meshRotation * def.rotation;
            byName[def.name] = t;
            ovrBones.Add(t);
        }

        Transform[] newBones = oldBones.Where(b => !OvrHandRigData.IsSmplxFingerJoint(b.name)).Concat(ovrBones).ToArray();
        var boneIndex = new Dictionary<string, int>();
        for (int i = 0; i < newBones.Length; i++)
            boneIndex[newBones[i].name] = i;
        Matrix4x4[] newBind = newBones.Select(b => b.worldToLocalMatrix * meshToWorld).ToArray();

        // Skin weights: SMPL-X finger joint weights move to the OVR bone at the same joint
        var ovrForSmplx = FlowAvatarSMPLxOVR.OvrBoneToSmplxJoint
            .ToDictionary(kv => OvrHandRigData.SmplxJointNames[kv.Value], kv => kv.Key);
        var bonesPerVertex = mesh.GetBonesPerVertex();
        var oldWeights = mesh.GetAllBoneWeights();
        var counts = new NativeArray<byte>(bonesPerVertex.Length, Allocator.Temp);
        var weights = new List<BoneWeight1>(oldWeights.Length);
        int handVertices = 0;

        for (int v = 0, offset = 0; v < bonesPerVertex.Length; offset += bonesPerVertex[v], v++)
        {
            var influences = new Dictionary<string, float>();
            bool hand = false;
            for (int k = 0; k < bonesPerVertex[v]; k++)
            {
                string name = oldBones[oldWeights[offset + k].boneIndex].name;
                if (ovrForSmplx.TryGetValue(name, out string ovr))
                {
                    name = ovr;
                    hand = true;
                }
                influences[name] = influences.TryGetValue(name, out float w) ? w + oldWeights[offset + k].weight : oldWeights[offset + k].weight;
            }
            if (hand) handVertices++;

            var sorted = influences.Where(kv => kv.Value > 0f).OrderByDescending(kv => kv.Value).Take(MaxInfluences).ToArray();
            float sum = sorted.Sum(kv => kv.Value);
            foreach (var kv in sorted)
                weights.Add(new BoneWeight1 { boneIndex = boneIndex[kv.Key], weight = kv.Value / sum });
            counts[v] = (byte)sorted.Length;
        }

        mesh.bindposes = newBind;
        var weightArray = new NativeArray<BoneWeight1>(weights.ToArray(), Allocator.Temp);
        mesh.SetBoneWeights(counts, weightArray);
        counts.Dispose();
        weightArray.Dispose();
        smr.bones = newBones;

        foreach (string name in OvrHandRigData.SmplxJointNames.Where(OvrHandRigData.IsSmplxFingerJoint))
            if (byName.TryGetValue(name, out var t) && t != null)
                Object.DestroyImmediate(t.gameObject);

        Debug.Log($"[OvrHandRig] {Path.GetFileName(assetPath)}: {rig.restOverrides.Length} rest overrides, " +
                  $"{ovrBones.Count} OVR bones, {newBones.Length} bones, {handVertices} hand vertices re-weighted");
    }
}
