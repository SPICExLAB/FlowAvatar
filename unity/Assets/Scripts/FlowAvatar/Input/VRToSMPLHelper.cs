using UnityEngine;
using Oculus.Interaction;

/// <summary>
/// Helper class for VRToSMPLCalibrator that handles debug visualization
/// and reference transforms management.
/// </summary>
public class VRToSMPLHelper
{
    #region Properties

    // Reference transforms (always active)
    private Transform headJointRef;       // Reference transform for head joint
    private Transform headsetRef;         // Reference transform for headset (child of headJoint)
    private Transform leftHandRef;        // Reference transform for left hand (child of headJoint)
    private Transform rightHandRef;       // Reference transform for right hand (child of headJoint)

    // Debug visualization objects (attached to reference transforms)
    private GameObject headJointSphere;   // Visual sphere for head joint
    private GameObject headsetSphere;     // Visual sphere for headset
    private GameObject leftHandSphere;    // Visual sphere for left hand
    private GameObject rightHandSphere;   // Visual sphere for right hand

    // Line renderers for debug visualization
    private LineRenderer[] headJointAxes = new LineRenderer[3]; // Forward, Right, Up
    private LineRenderer[] headsetAxes = new LineRenderer[3];
    private LineRenderer[] leftHandAxes = new LineRenderer[3];
    private LineRenderer[] rightHandAxes = new LineRenderer[3];

    // Hand visibility control
    private GameObject leftHandParent;   // Parent of left hand mesh for layer control
    private GameObject rightHandParent;  // Parent of right hand mesh for layer control

    // Debug visualization settings
    private bool showDebugSpheres = false;
    private float sphereRadius = 0.05f;
    private float gizmoLength = 0.1f;

    // Colors for visualization (just active and inactive)
    private Color activeColor;        // Magenta - for active objects
    private Color inactiveColor;      // Dark gray - for inactive objects
    private Color headsetColor;       // Cyan - special color for headset

    // Materials for axis visualization with standard Unity axis colors
    private Material forwardAxisMaterial; // Blue Z
    private Material rightAxisMaterial;   // Red X
    private Material upAxisMaterial;      // Green Y

    #endregion

    #region Setup Methods

    /// <summary>
    /// Initializes the helper with visualization settings.
    /// </summary>
    public VRToSMPLHelper(
        bool showDebug = false,
        float radius = 0.05f,
        float length = 0.1f,
        Color activeCol = default,
        Color inactiveCol = default)
    {
        // Set debug parameters
        this.showDebugSpheres = showDebug;
        this.sphereRadius = radius;
        this.gizmoLength = length;

        // Use provided colors or fallback to defaults
        this.activeColor = activeCol == default ? new Color(1f, 0f, 1f) : activeCol; // Magenta default
        this.inactiveColor = inactiveCol == default ? new Color(0.3f, 0.3f, 0.3f, 0.5f) : inactiveCol; // Dark gray default

        // Special color for headset (cyan) to distinguish it
        this.headsetColor = new Color(0f, 1f, 1f); // Cyan

        // Create standard materials for axis visualization
        CreateStandardAxisMaterials();

        // Setup reference transforms and debug visualization
        SetupReferenceTransforms();
        SetupDebugVisuals();
    }

    /// <summary>
    /// Creates standard axis materials with Unity's axis colors
    /// </summary>
    private void CreateStandardAxisMaterials()
    {
        // Create materials with standard Unity axis colors
        forwardAxisMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        rightAxisMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        upAxisMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));

        // Set colors to match Unity's gizmo colors: Z=Blue, X=Red, Y=Green
        forwardAxisMaterial.color = Color.blue;    // Z-axis (forward)
        rightAxisMaterial.color = Color.red;       // X-axis (right)
        upAxisMaterial.color = Color.green;        // Y-axis (up)
    }

    /// <summary>
    /// Sets up reference transforms for head joint, headset, and hands.
    /// </summary>
    public void SetupReferenceTransforms()
    {
        // 1. Create the head joint reference transform (parent)
        GameObject headJointObj = new GameObject("HeadJoint_Reference");
        headJointRef = headJointObj.transform;

        // 2. Create the headset reference transform (child of head joint)
        GameObject headsetObj = new GameObject("Headset_Reference");
        headsetRef = headsetObj.transform;
        headsetRef.SetParent(headJointRef, false);

        // 3. Create hand reference transforms (children of head joint)
        GameObject leftHandObj = new GameObject("LeftHand_Reference");
        leftHandRef = leftHandObj.transform;
        leftHandRef.SetParent(headJointRef, false);

        GameObject rightHandObj = new GameObject("RightHand_Reference");
        rightHandRef = rightHandObj.transform;
        rightHandRef.SetParent(headJointRef, false);
    }

    /// <summary>
    /// Sets up debug visualization spheres and axes.
    /// </summary>
    public void SetupDebugVisuals()
    {
        // Create debug spheres as children of reference transforms

        // 1. Create head joint sphere attached to reference transform
        headJointSphere = CalibrationUtils.CreateDebugSphere("HeadJointSphere",
            activeColor, sphereRadius, headJointRef);
        headJointAxes = CalibrationUtils.CreateDebugAxes(headJointSphere, "HeadJoint",
            forwardAxisMaterial, rightAxisMaterial, upAxisMaterial, gizmoLength);

        // 2. Create headset sphere attached to reference transform (using cyan to distinguish)
        headsetSphere = CalibrationUtils.CreateDebugSphere("HeadsetSphere",
            headsetColor, sphereRadius, headsetRef);
        headsetAxes = CalibrationUtils.CreateDebugAxes(headsetSphere, "Headset",
            forwardAxisMaterial, rightAxisMaterial, upAxisMaterial, gizmoLength);

        // 3. Create left hand sphere attached to reference transform
        leftHandSphere = CalibrationUtils.CreateDebugSphere("LeftHandSphere",
            activeColor, sphereRadius, leftHandRef);
        leftHandAxes = CalibrationUtils.CreateDebugAxes(leftHandSphere, "LeftHand",
            forwardAxisMaterial, rightAxisMaterial, upAxisMaterial, gizmoLength);

        // 4. Create right hand sphere attached to reference transform
        rightHandSphere = CalibrationUtils.CreateDebugSphere("RightHandSphere",
            activeColor, sphereRadius, rightHandRef);
        rightHandAxes = CalibrationUtils.CreateDebugAxes(rightHandSphere, "RightHand",
            forwardAxisMaterial, rightAxisMaterial, upAxisMaterial, gizmoLength);

        // Initially set visibility based on debug setting
        ToggleDebugVisuals(showDebugSpheres);
    }

    /// <summary>
    /// Sets up hand mesh references for visibility control
    /// </summary>
    public void SetupHandMeshes(HandVisual leftHand, HandVisual rightHand)
    {
        if (leftHand != null)
        {
            Transform leftHandMesh = leftHand.transform.FindChildRecursive("l_handMeshNode");
            if (leftHandMesh != null)
            {
                leftHandParent = leftHandMesh.gameObject;
            }
        }

        if (rightHand != null)
        {
            Transform rightHandMesh = rightHand.transform.FindChildRecursive("r_handMeshNode");
            if (rightHandMesh != null)
            {
                rightHandParent = rightHandMesh.gameObject;
            }
        }
    }

    #endregion

    #region Update Methods

    /// <summary>
    /// Updates debug visualization axes for all reference transforms.
    /// </summary>
    public void UpdateDebugVisualization()
    {
        if (!showDebugSpheres) return;

        // Update all debug axes
        CalibrationUtils.UpdateDebugAxisLines(headJointSphere.transform, headJointAxes, gizmoLength);
        CalibrationUtils.UpdateDebugAxisLines(headsetSphere.transform, headsetAxes, gizmoLength);
        CalibrationUtils.UpdateDebugAxisLines(leftHandSphere.transform, leftHandAxes, gizmoLength);
        CalibrationUtils.UpdateDebugAxisLines(rightHandSphere.transform, rightHandAxes, gizmoLength);
    }

    /// <summary>
    /// Updates hand visibility colors in debug visualization.
    /// </summary>
    public void UpdateHandBallVisibility(bool leftHandVisible, bool rightHandVisible)
    {
        if (!showDebugSpheres) return;

        // Update debug sphere colors based on hand visibility
        var leftRenderer = leftHandSphere.GetComponent<Renderer>();
        var rightRenderer = rightHandSphere.GetComponent<Renderer>();

        if (leftRenderer != null)
            leftRenderer.material.color = leftHandVisible ? activeColor : inactiveColor;

        if (rightRenderer != null)
            rightRenderer.material.color = rightHandVisible ? activeColor : inactiveColor;
    }

    /// <summary>
    /// Sets hand visibility via layers for camera culling
    /// </summary>
    public void SetHandsVisible(bool visible)
    {
        // Layer 3 is typically used for objects that should be invisible to certain cameras
        const int invisibleLayer = 3;
        const int defaultLayer = 0;

        // Set layer for left hand
        if (leftHandParent != null)
        {
            CalibrationUtils.SetLayerRecursively(leftHandParent,
                visible ? defaultLayer : invisibleLayer);
        }

        // Set layer for right hand
        if (rightHandParent != null)
        {
            CalibrationUtils.SetLayerRecursively(rightHandParent,
                visible ? defaultLayer : invisibleLayer);
        }
    }

    /// <summary>
    /// Toggles visibility of debug visualization.
    /// </summary>
    public void ToggleDebugVisuals(bool show)
    {
        showDebugSpheres = show;

        // Simply activate/deactivate the spheres - no need to deactivate references
        if (headJointSphere != null) headJointSphere.SetActive(show);
        if (headsetSphere != null) headsetSphere.SetActive(show);
        if (leftHandSphere != null) leftHandSphere.SetActive(show);
        if (rightHandSphere != null) rightHandSphere.SetActive(show);

        // The axis line renderers are children of the spheres, so they will be hidden
        // when the parent is deactivated, but we still want to explicitly disable them
        CalibrationUtils.ToggleLineRenderers(headJointAxes, show);
        CalibrationUtils.ToggleLineRenderers(headsetAxes, show);
        CalibrationUtils.ToggleLineRenderers(leftHandAxes, show);
        CalibrationUtils.ToggleLineRenderers(rightHandAxes, show);
    }

    #endregion

    #region Reference Transform Methods

    /// <summary>
    /// Updates the head joint reference transform.
    /// </summary>
    public void UpdateHeadJointTransform(Vector3 position, Quaternion rotation)
    {
        headJointRef.position = position;
        headJointRef.rotation = rotation;
    }

    /// <summary>
    /// Updates the headset reference transform as a child of head joint.
    /// </summary>
    public void UpdateHeadsetTransform(Vector3 worldPosition, Quaternion worldRotation, Quaternion headRotation)
    {
        // Convert world position to local relative to head joint
        Vector3 headsetLocalPos = headJointRef.InverseTransformPoint(worldPosition);
        headsetRef.localPosition = headsetLocalPos;

        // Set local rotation as relation between headset and head
        headsetRef.localRotation = Quaternion.Inverse(headRotation) * worldRotation;
    }

    /// <summary>
    /// Updates hand reference transforms.
    /// </summary>
    public void UpdateHandTransforms(Vector3 leftPosition, Quaternion leftRotation,
                                    Vector3 rightPosition, Quaternion rightRotation)
    {
        leftHandRef.position = leftPosition;
        leftHandRef.rotation = leftRotation;

        rightHandRef.position = rightPosition;
        rightHandRef.rotation = rightRotation;
    }

    #endregion

    #region Getters

    // Public getter methods for accessing transforms
    public Transform GetHeadTransform() => headJointRef;
    public Transform GetHeadsetTransform() => headsetRef;
    public Transform GetLeftHandTransform() => leftHandRef;
    public Transform GetRightHandTransform() => rightHandRef;
    public bool GetDebugVisualsState() => showDebugSpheres;

    #endregion

    
}