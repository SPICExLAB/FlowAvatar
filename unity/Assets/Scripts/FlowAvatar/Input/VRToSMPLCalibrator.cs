using UnityEngine;
using Oculus.Interaction;

/// <summary>
/// Handles calibration between VR tracking data and SMPL avatar
/// </summary>
public class VRToSMPLCalibrator : MonoBehaviour
{
    [Header("Avatar Setup")]
    [Tooltip("make sure to assign these based on the avatar if not done it yet")]
    public Transform avatarRoot;
    public Transform avatarHead;
    public Transform avatarLeftEye;
    public Transform avatarRightEye;
    public Transform avatarLeftHand;
    public Transform avatarRightHand;

    [Header("VR Setup")]
    [Tooltip("This will be assigned automatically")]
    public OVRCameraRig ovrCameraRig;
    public HandVisual leftHand;
    public HandVisual rightHand;

    private Transform virtualHeadset;  // 6cm forward from eye center 
    private Vector3 eyeToHeadsetOffset = new Vector3(0, 0, 0.06f); // 6cm forward

    [Header("Calibration Settings")]
    [Tooltip("If false, will use raw VR data with only headset->head offset applied")]
    public bool useCalibration = true;
    public float heightDiffTolerance = 0.2f;
    public float pinchThreshold = 0.9f;
    public float tPoseHoldTime = 2.0f;          // How long user needs to hold T-pose
    public bool requirePinchForCalibration = false; // Whether to also require pinching or just T-pose

    [Header("Debug")]
    [SerializeField] private bool showDebugSpheres = false;  // Off by default
    public Color activeColor = new Color(1f, 0f, 1f);       // Magenta for active objects
    public Color inactiveColor = new Color(0.3f, 0.3f, 0.3f, 0.5f); // Dark gray for inactive
    public float sphereRadius = 0.05f;
    public float gizmoLength = 0.1f;

    private OVRHand rightHandSource;
    private Transform leftHandSkeleton;
    private Transform rightHandSkeleton;
    private Quaternion initialVRRotation = Quaternion.identity;

    // Raw tracking data
    private Vector3 lastRawHeadSetPosition;
    private Vector3 lastRawLeftWristPosition;
    private Vector3 lastRawRightWristPosition;
    private Quaternion lastRawHeadSetRotation = Quaternion.identity;
    private Quaternion lastRawLeftWristRotation = Quaternion.identity;
    private Quaternion lastRawRightWristRotation = Quaternion.identity;

    // Calibration data
    private Vector3 avatarInitialFloorPosition;
    private Vector3 vrInitialFloorPosition;
    private Vector3 scaleFactor;

    // Initial reference transforms for calibration
    private Vector3 initialVRHeadPosition;
    private Vector3 initialVRLeftHandPosition;
    private Vector3 initialVRRightHandPosition;

    private Vector3 initialAvatarHeadsetPosition;
    private Vector3 initialAvatarLeftHandPosition;
    private Vector3 initialAvatarRightHandPosition;

    // T-pose detection
    private float tPoseHoldTimer = 0f;
    private bool isInTPose = false;

    private Vector3 initialHeadToVirtualHeadset;  // Store the initial offset vector
    private Quaternion initialVirtualHeadsetRotation;

    public struct CalibrationData
    {
        public Vector3 headPosition;    // Head joint position
        public Vector3 leftWristPosition;
        public Vector3 rightWristPosition;
        public Quaternion headRotation;
        public Quaternion leftWristRotation;
        public Quaternion rightWristRotation;
        public bool leftHandVisible;
        public bool rightHandVisible;
    }
    private CalibrationData calibratedData;
    private bool isCalibrated;

    // Helper for debug visualization and reference transforms
    private VRToSMPLHelper helper;

    private void Awake()
    {
        Transform CameraRig = GameObject.Find("[BuildingBlock] Camera Rig").transform;
        ovrCameraRig = CameraRig.GetComponent<OVRCameraRig>();
        leftHand = CameraRig.Find("[BuildingBlock] Interaction/[BuildingBlock] Hand Interactions/[BuildingBlock] Synthetic Left Hand").GetComponentInChildren<HandVisual>();
        rightHand = CameraRig.Find("[BuildingBlock] Interaction/[BuildingBlock] Hand Interactions/[BuildingBlock] Synthetic Right Hand").GetComponentInChildren<HandVisual>();
    }

    void Start()
    {
        Initialize();

        // Initialize the helper with debug colors
        helper = new VRToSMPLHelper(
            showDebugSpheres,
            sphereRadius,
            gizmoLength,
            activeColor,
            inactiveColor
        );

        // Set up hand meshes for visibility control
        helper.SetupHandMeshes(leftHand, rightHand);

        // Initially set debug visualization state
        helper.ToggleDebugVisuals(showDebugSpheres);

        // If not using calibration, immediately set up with raw data
        if (!useCalibration)
        {
            // Skip traditional calibration process
            isCalibrated = true;

            // Hide VR hands to avoid duplicates
            helper.SetHandsVisible(false);
        }
    }

    void Initialize()
    {
        rightHandSource = ovrCameraRig.rightHandAnchor.GetComponentInChildren<OVRHand>();

        // Calculate initial eye center and virtual headset position
        Vector3 avatarEyeCenter = (avatarLeftEye.position + avatarRightEye.position) / 2f;
        GameObject virtualHeadsetObj = new GameObject("VirtualHeadset");
        virtualHeadset = virtualHeadsetObj.transform;
        virtualHeadset.SetParent(avatarHead);

        // Set initial position and rotation
        virtualHeadset.position = avatarEyeCenter + avatarHead.rotation * eyeToHeadsetOffset;
        virtualHeadset.rotation = avatarHead.rotation;

        // IMPORTANT: Store the initial vector in world space, not local space
        // This avoids scale issues with the avatar root
        initialHeadToVirtualHeadset = virtualHeadset.position - avatarHead.position;
        initialVirtualHeadsetRotation = virtualHeadset.rotation * Quaternion.Inverse(avatarHead.rotation);

        // Store initial position based on virtual headset
        avatarInitialFloorPosition = new Vector3(
            virtualHeadset.position.x,
            0,
            virtualHeadset.position.z
        );

        UpdateHandReferences();
    }

    void Update()
    {
        UpdateHandReferences();

        // Only check for calibration if we're using it and not already calibrated
        if (useCalibration && !isCalibrated)
            CheckCalibrationTrigger();

        if (isCalibrated)
            UpdateCalibration();

        // Update debug visualizations if needed
        helper.UpdateDebugVisualization();

        // Update debug visualization state
        if (helper.GetDebugVisualsState() != showDebugSpheres)
            helper.ToggleDebugVisuals(showDebugSpheres);
    }

    void UpdateHandReferences()
    {
        if (leftHand.IsVisible)
            leftHandSkeleton = leftHand.GetTransformByHandJointId(0);
        if (rightHand.IsVisible)
            rightHandSkeleton = rightHand.GetTransformByHandJointId(0);
    }

    void StoreLastVR(Vector3 left, Vector3 right)
    {
        Vector3 headsetPosition = ovrCameraRig.centerEyeAnchor.position;
        Quaternion headsetRotation = ovrCameraRig.centerEyeAnchor.rotation;
        lastRawHeadSetPosition = headsetPosition;
        lastRawHeadSetRotation = headsetRotation;

        if (leftHand.IsVisible)
        {
            lastRawLeftWristPosition = left;
            lastRawLeftWristRotation = leftHandSkeleton.rotation;
        }

        if (rightHand.IsVisible)
        {
            lastRawRightWristPosition = right;
            lastRawRightWristRotation = rightHandSkeleton.rotation;
        }
    }

    void PerformCalibration()
    {
        // Get initial VR positions
        Vector3 vrHeadSetPosition = ovrCameraRig.centerEyeAnchor.position;
        Vector3 vrLeftHandPosition = leftHandSkeleton.position;
        Vector3 vrRightHandPosition = rightHandSkeleton.position;

        // Store these initial positions for reference
        initialVRHeadPosition = vrHeadSetPosition;
        initialVRLeftHandPosition = vrLeftHandPosition;
        initialVRRightHandPosition = vrRightHandPosition;

        // Store avatar reference positions
        initialAvatarHeadsetPosition = virtualHeadset.position;
        initialAvatarLeftHandPosition = avatarLeftHand.position;
        initialAvatarRightHandPosition = avatarRightHand.position;

        // Get forward direction for rotation calibration
        Vector3 vrForward = Vector3.ProjectOnPlane(ovrCameraRig.centerEyeAnchor.forward, Vector3.up).normalized;
        initialVRRotation = Quaternion.LookRotation(vrForward, Vector3.up);

        // Store floor position at calibration time
        vrInitialFloorPosition = new Vector3(
            vrHeadSetPosition.x,
            0,
            vrHeadSetPosition.z
        );

        // Calculate arm span in VR using T-pose measurements
        float vrArmSpan = Vector3.Distance(initialVRLeftHandPosition, initialVRRightHandPosition);
        // Calculate avatar arm span
        float avatarArmSpan = Vector3.Distance(initialAvatarLeftHandPosition, initialAvatarRightHandPosition);
        // Calculate horizontal scale factor based on arm span
        float horizontalScale = avatarArmSpan / vrArmSpan;

        // Calculate height scale for vertical scaling
        float heightScale = initialAvatarHeadsetPosition.y / initialVRHeadPosition.y;

        // Apply calculated scale factors - use T-pose arm span for horizontal (X) scaling
        scaleFactor = new Vector3(horizontalScale, heightScale, 1.0f);

        // Store last known good positions and rotations at calibration time
        StoreLastVR(vrLeftHandPosition, vrRightHandPosition);

        isCalibrated = true;

        // Make VR hands "invisible" via layer when calibrated
        helper.SetHandsVisible(false);
    }

    void UpdateCalibration()
    {
        // Get current VR positions
        Vector3 vrHeadSetPosition = ovrCameraRig.centerEyeAnchor.position;
        Vector3 vrLeftHandPos = leftHand.IsVisible ? leftHandSkeleton.position : lastRawLeftWristPosition;
        Vector3 vrRightHandPos = rightHand.IsVisible ? rightHandSkeleton.position : lastRawRightWristPosition;

        // Store for future frames
        StoreLastVR(vrLeftHandPos, vrRightHandPos);

        if (useCalibration)
        {
            // FULL CALIBRATION: Use the original calibration method
            UpdateWithFullCalibration(vrHeadSetPosition, vrLeftHandPos, vrRightHandPos);
        }
        else
        {
            // NO CALIBRATION: Just apply head offset to raw VR data
            UpdateWithoutCalibration(vrHeadSetPosition, vrLeftHandPos, vrRightHandPos);
        }
    }

    // Original calibration method
    void UpdateWithFullCalibration(Vector3 vrHeadSetPosition, Vector3 vrLeftHandPos, Vector3 vrRightHandPos)
    {
        // Calculate relative positions from VR floor for the headset
        Vector3 relativeHeadSetPos = lastRawHeadSetPosition - vrInitialFloorPosition;

        // Apply rotation correction to headset
        Quaternion rotationToForward = Quaternion.Inverse(initialVRRotation);
        Vector3 normalizedHeadSetPos = rotationToForward * relativeHeadSetPos;

        // Apply scaling
        Vector3 scaledHeadSetPos = Vector3.Scale(normalizedHeadSetPos, scaleFactor);

        // Calculate position in avatar space - this is the virtual headset position
        Vector3 finalVirtualHeadset = avatarInitialFloorPosition + scaledHeadSetPos;

        // Get the VR headset rotation in avatar space
        Quaternion finalHeadRotation = rotationToForward * lastRawHeadSetRotation;

        // Calculate virtual headset rotation
        Quaternion virtualHeadsetRotation = finalHeadRotation * initialVirtualHeadsetRotation;

        // Calculate head joint position from virtual headset using the initial offset
        // Transform the local offset to world space correctly
        Vector3 rotatedOffset = finalHeadRotation * initialHeadToVirtualHeadset;
        Vector3 finalHeadJoint = finalVirtualHeadset - rotatedOffset;

        // For hands, still use the global space approach to keep them independent from head
        // Calculate hand movements from calibration positions in VR space
        Vector3 leftHandMovementVR = vrLeftHandPos - initialVRLeftHandPosition;
        Vector3 rightHandMovementVR = vrRightHandPos - initialVRRightHandPosition;

        // Apply rotation alignment and scaling to hand movements
        Vector3 scaledLeftHandMovement = Vector3.Scale(rotationToForward * leftHandMovementVR, scaleFactor);
        Vector3 scaledRightHandMovement = Vector3.Scale(rotationToForward * rightHandMovementVR, scaleFactor);

        // Calculate final hand positions in avatar space - these are global positions
        Vector3 finalLeftHand = initialAvatarLeftHandPosition + scaledLeftHandMovement;
        Vector3 finalRightHand = initialAvatarRightHandPosition + scaledRightHandMovement;

        // Calculate hand rotations
        Quaternion finalLeftHandRot = rotationToForward * lastRawLeftWristRotation;
        Quaternion finalRightHandRot = rotationToForward * lastRawRightWristRotation;

        // Store calibrated data
        calibratedData = new CalibrationData
        {
            headPosition = finalHeadJoint,
            leftWristPosition = finalLeftHand,
            rightWristPosition = finalRightHand,
            headRotation = finalHeadRotation,
            leftWristRotation = finalLeftHandRot,
            rightWristRotation = finalRightHandRot,
            leftHandVisible = leftHand.IsVisible,
            rightHandVisible = rightHand.IsVisible
        };

        // Update transforms via helper
        helper.UpdateHeadJointTransform(finalHeadJoint, finalHeadRotation);
        helper.UpdateHeadsetTransform(finalVirtualHeadset, virtualHeadsetRotation, finalHeadRotation);
        helper.UpdateHandTransforms(finalLeftHand, finalLeftHandRot, finalRightHand, finalRightHandRot);

        // Update hand visibility in debug visualization
        helper.UpdateHandBallVisibility(leftHand.IsVisible, rightHand.IsVisible);
    }

    // Simple method that preserves the same head-headset relationship as calibration
    void UpdateWithoutCalibration(Vector3 vrHeadSetPosition, Vector3 vrLeftHandPos, Vector3 vrRightHandPos)
    {
        // Get the current VR rotations
        Quaternion vrHeadRotation = lastRawHeadSetRotation;
        Quaternion vrLeftRotation = lastRawLeftWristRotation;
        Quaternion vrRightRotation = lastRawRightWristRotation;

        // Use the same relationship between headset and head joint as in the calibrated mode
        // First, calculate the position and rotation of the virtual headset (this is the VR headset)
        Vector3 virtualHeadsetPosition = vrHeadSetPosition;
        Quaternion virtualHeadsetRotation = vrHeadRotation;

        // Use the same approach as the full calibration to calculate head joint position
        // Calculate head joint position from virtual headset using the initial offset
        Vector3 rotatedOffset = vrHeadRotation * initialHeadToVirtualHeadset;
        Vector3 headJointPosition = virtualHeadsetPosition - rotatedOffset;

        // Store calibrated data using values with proper head-headset relationship
        calibratedData = new CalibrationData
        {
            headPosition = headJointPosition,  // Use head joint position, not raw headset
            leftWristPosition = vrLeftHandPos,
            rightWristPosition = vrRightHandPos,
            headRotation = vrHeadRotation,
            leftWristRotation = vrLeftRotation,
            rightWristRotation = vrRightRotation,
            leftHandVisible = leftHand.IsVisible,
            rightHandVisible = rightHand.IsVisible
        };

        // Update transforms via helper - head and headset are different positions
        helper.UpdateHeadJointTransform(headJointPosition, vrHeadRotation);
        helper.UpdateHeadsetTransform(virtualHeadsetPosition, virtualHeadsetRotation, vrHeadRotation);
        helper.UpdateHandTransforms(vrLeftHandPos, vrLeftRotation, vrRightHandPos, vrRightRotation);

        // Update hand visibility in debug visualization
        helper.UpdateHandBallVisibility(leftHand.IsVisible, rightHand.IsVisible);
    }

    void CheckCalibrationTrigger()
    {
        if (isCalibrated) return;

        // Check for T-pose
        bool currentTPoseState = CalibrationUtils.IsInTPose(
            leftHandSkeleton, rightHandSkeleton,
            leftHand.IsVisible, rightHand.IsVisible,
            heightDiffTolerance);

        // Update timer based on T-pose state
        if (currentTPoseState)
        {
            if (!isInTPose) // Just entered T-pose
            {
                isInTPose = true;
            }

            tPoseHoldTimer += Time.deltaTime;
        }
        else
        {
            if (isInTPose) // Just exited T-pose
            {
                isInTPose = false;
            }

            tPoseHoldTimer = 0f;
        }

        // Check if we should calibrate
        if (tPoseHoldTimer >= tPoseHoldTime && CalibrationUtils.CheckHandTrackingQuality(
            leftHandSkeleton, rightHandSkeleton,
            leftHand.IsVisible, rightHand.IsVisible))
        {
            if (requirePinchForCalibration)
            {
                // Standard behavior - requires pinch
                if (CalibrationUtils.IsPinching(rightHandSource, rightHand.IsVisible, pinchThreshold))
                {
                    PerformCalibration();
                }
            }
            else
            {
                // Just use T-pose without requiring pinch
                PerformCalibration();
            }
        }
    }

    // Public methods for accessing calibration data and transforms
    public bool IsCalibrated() => isCalibrated;
    public CalibrationData GetCalibrationData() => calibratedData;
    public Transform GetHeadTransform() => helper.GetHeadTransform();
    public Transform GetHeadsetTransform() => helper.GetHeadsetTransform();
    public Transform GetLeftHandTransform() => helper.GetLeftHandTransform();
    public Transform GetRightHandTransform() => helper.GetRightHandTransform();
}