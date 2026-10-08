using UnityEngine;
using Oculus.Interaction;

/// <summary>
/// Utility class for VR calibration functions including T-pose detection,
/// pinch detection, and debug visualization.
/// </summary>
public class CalibrationUtils
{
    #region Debug Visualization

    /// <summary>
    /// Creates a debug sphere with optional parent transform
    /// </summary>
    public static GameObject CreateDebugSphere(string name, Color color, float radius, Transform parent = null)
    {
        GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.name = name;
        sphere.transform.localScale = Vector3.one * radius;

        // Set material color
        var renderer = sphere.GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material = new Material(Shader.Find("Universal Render Pipeline/Lit"));
            renderer.material.color = color;
        }

        // If parent is provided, make the sphere a child of that transform
        if (parent != null)
        {
            sphere.transform.SetParent(parent);
            sphere.transform.localPosition = Vector3.zero;
            sphere.transform.localRotation = Quaternion.identity;
        }

        return sphere;
    }

    /// <summary>
    /// Creates debug axis visualizers for a transform
    /// </summary>
    public static LineRenderer[] CreateDebugAxes(GameObject parent, string namePrefix,
        Material forwardMaterial, Material rightMaterial, Material upMaterial, float length = 0.1f)
    {
        LineRenderer[] axes = new LineRenderer[3];

        // Forward axis (blue Z)
        GameObject forwardObj = new GameObject($"{namePrefix}_ForwardAxis");
        forwardObj.transform.SetParent(parent.transform, false);
        axes[0] = SetupLineRenderer(forwardObj, forwardMaterial, Color.blue, length);

        // Right axis (red X)
        GameObject rightObj = new GameObject($"{namePrefix}_RightAxis");
        rightObj.transform.SetParent(parent.transform, false);
        axes[1] = SetupLineRenderer(rightObj, rightMaterial, Color.red, length);

        // Up axis (green Y)
        GameObject upObj = new GameObject($"{namePrefix}_UpAxis");
        upObj.transform.SetParent(parent.transform, false);
        axes[2] = SetupLineRenderer(upObj, upMaterial, Color.green, length);

        // Initialize the axis lines
        UpdateDebugAxisLines(parent.transform, axes, length);

        return axes;
    }

    /// <summary>
    /// Sets up a line renderer for debug visualization
    /// </summary>
    public static LineRenderer SetupLineRenderer(GameObject obj, Material material, Color color, float length = 0.1f)
    {
        var line = obj.AddComponent<LineRenderer>();
        line.material = material;
        line.startWidth = line.endWidth = 0.005f;
        line.positionCount = 2;

        // Set color if the material supports it
        line.startColor = line.endColor = color;

        // Initial positions will be set in UpdateDebugAxisLines
        return line;
    }

    /// <summary>
    /// Updates the debug axis lines for visualization
    /// </summary>
    public static void UpdateDebugAxisLines(Transform parentTransform, LineRenderer[] axes, float length = 0.1f)
    {
        if (parentTransform == null || axes == null || axes.Length < 3) return;

        // Get the world position of the parent (sphere)
        Vector3 origin = parentTransform.position;

        // Calculate world positions for each axis end point using the parent's transform
        Vector3 forwardEnd = origin + parentTransform.forward * length;
        Vector3 rightEnd = origin + parentTransform.right * length;
        Vector3 upEnd = origin + parentTransform.up * length;

        // Update forward axis (Z) - WORLD SPACE coordinates
        axes[0].SetPosition(0, origin);
        axes[0].SetPosition(1, forwardEnd);

        // Update right axis (X) - WORLD SPACE coordinates
        axes[1].SetPosition(0, origin);
        axes[1].SetPosition(1, rightEnd);

        // Update up axis (Y) - WORLD SPACE coordinates
        axes[2].SetPosition(0, origin);
        axes[2].SetPosition(1, upEnd);
    }

    /// <summary>
    /// Toggles visibility of line renderers
    /// </summary>
    public static void ToggleLineRenderers(LineRenderer[] axes, bool show)
    {
        if (axes == null) return;

        foreach (var axis in axes)
        {
            if (axis != null)
            {
                axis.enabled = show;
            }
        }
    }

    #endregion

    #region Pose Detection

    /// <summary>
    /// Checks if the user is in a T-pose
    /// </summary>
    public static bool IsInTPose(
        Transform leftHandSkeleton,
        Transform rightHandSkeleton,
        bool leftHandVisible,
        bool rightHandVisible,
        float heightDiffTolerance)
    {
        if (!leftHandVisible || !rightHandVisible) return false;

        Vector3 leftPos = leftHandSkeleton.position;
        Vector3 rightPos = rightHandSkeleton.position;

        if (leftPos == Vector3.zero || rightPos == Vector3.zero) return false;

        // 1. Verify hands are at similar heights
        float heightDiff = Mathf.Abs(leftPos.y - rightPos.y);
        bool similarHeight = heightDiff < heightDiffTolerance;

        // 2. Check that hands are roughly in opposite directions from body center
        // Get horizontal vector between hands
        Vector3 handToHandVec = Vector3.ProjectOnPlane(rightPos - leftPos, Vector3.up).normalized;

        // Get horizontal distance between hands (should be significant in T-pose)
        float horizontalDistance = Vector3.ProjectOnPlane(rightPos - leftPos, Vector3.up).magnitude;
        bool handsSpreadWide = horizontalDistance > 0.5f; // At least 0.5m apart horizontally

        // Return true if hands are at similar height and spread wide horizontally
        return similarHeight && handsSpreadWide;
    }

    /// <summary>
    /// Checks if the user is pinching with the right hand
    /// </summary>
    public static bool IsPinching(OVRHand rightHandSource, bool rightHandVisible, float pinchThreshold)
    {
        return rightHandSource != null &&
               rightHandVisible &&
               rightHandSource.GetFingerPinchStrength(OVRHand.HandFinger.Index) > pinchThreshold;
    }

    /// <summary>
    /// Checks if hand tracking data is of sufficient quality for calibration
    /// </summary>
    public static bool CheckHandTrackingQuality(
        Transform leftHandSkeleton,
        Transform rightHandSkeleton,
        bool leftHandVisible,
        bool rightHandVisible)
    {
        if (!leftHandVisible || !rightHandVisible) return false;

        Vector3 leftPos = leftHandSkeleton.position;
        Vector3 rightPos = rightHandSkeleton.position;

        if (leftPos == Vector3.zero || rightPos == Vector3.zero) return false;

        // Check if positions seem valid (not at origin and reasonable distance apart)
        float handDistance = Vector3.Distance(leftPos, rightPos);
        return handDistance > 0.2f && handDistance < 2.5f; // Reasonable human arm span range
    }

    #endregion

    #region Hand Utilities

    /// <summary>
    /// Finds hand mesh renderers in the hand visual hierarchy
    /// </summary>
    public static void FindHandMeshRenderers(
        HandVisual leftHand,
        HandVisual rightHand,
        out Renderer leftHandMeshRenderer,
        out Renderer rightHandMeshRenderer,
        out Material[] originalLeftHandMaterials,
        out Material[] originalRightHandMaterials)
    {
        leftHandMeshRenderer = null;
        rightHandMeshRenderer = null;
        originalLeftHandMaterials = null;
        originalRightHandMaterials = null;

        // Find left hand mesh renderer
        if (leftHand != null)
        {
            // Try to find l_handMeshNode under the hand visual hierarchy
            Transform leftHandMesh = leftHand.transform.FindChildRecursive("l_handMeshNode");
            if (leftHandMesh != null)
            {
                leftHandMeshRenderer = leftHandMesh.GetComponent<Renderer>();
                if (leftHandMeshRenderer != null)
                {
                    // Store original materials
                    originalLeftHandMaterials = leftHandMeshRenderer.materials;
                }
            }
        }

        // Find right hand mesh renderer
        if (rightHand != null)
        {
            // Try to find r_handMeshNode under the hand visual hierarchy
            Transform rightHandMesh = rightHand.transform.FindChildRecursive("r_handMeshNode");
            if (rightHandMesh != null)
            {
                rightHandMeshRenderer = rightHandMesh.GetComponent<Renderer>();
                if (rightHandMeshRenderer != null)
                {
                    // Store original materials
                    originalRightHandMaterials = rightHandMeshRenderer.materials;
                }
            }
        }
    }

    /// <summary>
    /// Sets layer recursively for all child game objects
    /// </summary>
    public static void SetLayerRecursively(GameObject obj, int newLayer)
    {
        if (obj == null)
            return;

        obj.layer = newLayer;

        foreach (Transform child in obj.transform)
        {
            SetLayerRecursively(child.gameObject, newLayer);
        }
    }

    #endregion
}