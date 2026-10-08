//using System;
//using UnityEngine;

///// <summary>
///// Input processor for VR tracking data that accurately matches the Python implementation
///// for creating 90D or 92D feature vectors with optimized performance
///// </summary>
//public class InputProcessor : MonoBehaviour
//{
//    [Header("References")]
//    [SerializeField] private VRToSMPLCalibrator calibrator;

//    [Header("Settings")]
//    [SerializeField] private bool enableDebugLogging = false;
//    [SerializeField] private bool usePreallocation = true;
//    [SerializeField] private bool useHandVisibility = true; // Toggle for hand visibility masking and 92D features

//    // Constants
//    private const int BASE_FEATURE_SIZE = 90;
//    private const int VISIBILITY_FEATURE_SIZE = 2;

//    // Pre-allocated arrays for better performance
//    private Vector3[] unityPositions;
//    private Quaternion[] unityRotations;
//    private Vector3[] transformedPositions;
//    private Quaternion[] transformedRotations;
//    private Vector3[] handPositions;
//    private Quaternion[] handRotations;
//    private Vector3[] localHandPositions;
//    private Quaternion[] localHandRotations;
//    private Vector3[] globalRotations6D;
//    private Vector3[] localRotations6D;
//    private float[] featureVector;

//    // State variables for velocity calculation
//    private bool isFirstFrame = true;
//    private Vector3[] previousGlobalPositions = new Vector3[3];  // head, left hand, right hand
//    private Vector3[] previousLocalPositions = new Vector3[2];   // left hand, right hand in head space
//    private Vector3[] previous6DRotations = new Vector3[6];      // 6D rotations for head, left, right hands
//    private Vector3[] previousLocal6DRotations = new Vector3[4]; // 6D local rotations for left, right hands

//    // Latest transformed head position for external access
//    private Vector3 lastTransformedHeadPosition;

//    // Hand visibility state tracking
//    private bool[] lastHandVisibility = new bool[] { false, false }; // [left, right]

//    // Performance tracking
//    private float lastProcessTime;
//    private float avgProcessTime;
//    private int frameCount;

//    void Awake()
//    {
//        if (calibrator == null)
//            calibrator = GetComponent<VRToSMPLCalibrator>();

//        // Pre-allocate arrays for better performance
//        if (usePreallocation)
//        {
//            unityPositions = new Vector3[3];
//            unityRotations = new Quaternion[3];
//            transformedPositions = new Vector3[3];
//            transformedRotations = new Quaternion[3];
//            handPositions = new Vector3[2];
//            handRotations = new Quaternion[2];
//            localHandPositions = new Vector3[2];
//            localHandRotations = new Quaternion[2];
//            globalRotations6D = new Vector3[6];
//            localRotations6D = new Vector3[4];

//            // Allocate the larger size to support both modes
//            featureVector = new float[BASE_FEATURE_SIZE + VISIBILITY_FEATURE_SIZE];
//        }
//    }

//    /// <summary>
//    /// Get the current feature vector size based on settings
//    /// </summary>
//    public int GetFeatureVectorSize()
//    {
//        int size = useHandVisibility ? BASE_FEATURE_SIZE + VISIBILITY_FEATURE_SIZE : BASE_FEATURE_SIZE;
//        return size;
//    }

//    /// <summary>
//    /// Process VR tracking data into a feature vector (90D or 92D) with optimized performance
//    /// </summary>
//    public float[] ProcessTrackingData()
//    {
//        float startTime = Time.realtimeSinceStartup;
//        int featureSize = GetFeatureVectorSize();

//        if (!calibrator.IsCalibrated())
//        {
//            DebugLog("Calibration not completed yet. Returning zero features.");
//            return new float[featureSize];
//        }

//        // Get the reference transforms from the calibrator
//        Transform headTransform = calibrator.GetHeadTransform();
//        Transform leftHandTransform = calibrator.GetLeftHandTransform();
//        Transform rightHandTransform = calibrator.GetRightHandTransform();

//        // Get current hand visibility state from calibrator's data
//        var calibratedData = calibrator.GetCalibrationData();
//        bool[] handVisibility = new bool[] { calibratedData.leftHandVisible, calibratedData.rightHandVisible };

//        // Log visibility state changes for debugging
//        if (enableDebugLogging && (handVisibility[0] != lastHandVisibility[0] || handVisibility[1] != lastHandVisibility[1]))
//        {
//            Debug.Log($"[InputProcessor] Hand visibility changed: Left={handVisibility[0]}, Right={handVisibility[1]}");
//        }

//        // Update hand visibility state
//        lastHandVisibility = handVisibility;

//        // STEP 1: Get raw Unity positions and rotations
//        if (!usePreallocation)
//        {
//            unityPositions = new Vector3[3];
//            unityRotations = new Quaternion[3];
//        }

//        unityPositions[0] = headTransform.position;
//        unityPositions[1] = leftHandTransform.position;
//        unityPositions[2] = rightHandTransform.position;

//        unityRotations[0] = headTransform.rotation;
//        unityRotations[1] = leftHandTransform.rotation;
//        unityRotations[2] = rightHandTransform.rotation;

//        // STEP 2: Convert Unity coordinates to SMPL/Python coordinate system
//        UnityTransformUtils.TransformCoordinatesToAMASS(
//            unityPositions,
//            unityRotations,
//            out var processedPositions,
//            out var processedRotations
//        );

//        if (usePreallocation)
//        {
//            Array.Copy(processedPositions, transformedPositions, 3);
//            Array.Copy(processedRotations, transformedRotations, 3);
//        }
//        else
//        {
//            transformedPositions = processedPositions;
//            transformedRotations = processedRotations;
//        }

//        // Store transformed head position for external access
//        lastTransformedHeadPosition = transformedPositions[0];

//        // STEP 3: Convert rotations to 6D representation
//        if (!usePreallocation)
//        {
//            globalRotations6D = new Vector3[6]; // 2 vectors per rotation
//        }

//        for (int i = 0; i < 3; i++)
//        {
//            Matrix4x4 rotMatrix = Matrix4x4.Rotate(transformedRotations[i]);
//            Vector3[] rot6D = UnityTransformUtils.MatrixToSixD(rotMatrix);
//            globalRotations6D[i * 2] = rot6D[0];     // First column
//            globalRotations6D[i * 2 + 1] = rot6D[1];  // Second column
//        }

//        // STEP 4: Calculate head space transforms for hands
//        if (!usePreallocation)
//        {
//            handPositions = new Vector3[2];
//            handRotations = new Quaternion[2];
//        }

//        handPositions[0] = transformedPositions[1]; // Left hand
//        handPositions[1] = transformedPositions[2]; // Right hand

//        handRotations[0] = transformedRotations[1]; // Left hand
//        handRotations[1] = transformedRotations[2]; // Right hand

//        UnityTransformUtils.TransformToHeadSpace(
//            transformedPositions[0],    // Head position
//            transformedRotations[0],    // Head rotation
//            handPositions,
//            handRotations,
//            out var processedHandPositions,
//            out var processedHandRotations
//        );

//        if (usePreallocation)
//        {
//            Array.Copy(processedHandPositions, localHandPositions, 2);
//            Array.Copy(processedHandRotations, localHandRotations, 2);
//        }
//        else
//        {
//            localHandPositions = processedHandPositions;
//            localHandRotations = processedHandRotations;
//        }

//        // STEP 5: Convert local rotations to 6D representation
//        if (!usePreallocation)
//        {
//            localRotations6D = new Vector3[4]; // 2 vectors per rotation (2 hands)
//        }

//        for (int i = 0; i < 2; i++)
//        {
//            Matrix4x4 rotMatrix = Matrix4x4.Rotate(localHandRotations[i]);
//            Vector3[] rot6D = UnityTransformUtils.MatrixToSixD(rotMatrix);
//            localRotations6D[i * 2] = rot6D[0];     // First column
//            localRotations6D[i * 2 + 1] = rot6D[1];  // Second column
//        }

//        // STEP 6: Calculate velocities (zero on first frame or visibility transitions)
//        Vector3[] globalPosVelocities;
//        Vector3[] globalRotVelocities6D;
//        Vector3[] localPosVelocities;
//        Vector3[] localRotVelocities6D;

//        if (isFirstFrame)
//        {
//            // Initialize zero velocities on first frame
//            globalPosVelocities = new Vector3[3];
//            globalRotVelocities6D = new Vector3[6];
//            localPosVelocities = new Vector3[2];
//            localRotVelocities6D = new Vector3[4];

//            // Store current values for next frame
//            isFirstFrame = false;
//        }
//        else
//        {
//            // Initialize velocity arrays
//            globalPosVelocities = new Vector3[3];
//            globalRotVelocities6D = new Vector3[6];
//            localPosVelocities = new Vector3[2];
//            localRotVelocities6D = new Vector3[4];

//            // Calculate velocities with special handling for visibility transitions

//            // Head velocities (always calculated normally)
//            globalPosVelocities[0] = transformedPositions[0] - previousGlobalPositions[0];
//            globalRotVelocities6D[0] = globalRotations6D[0] - previous6DRotations[0];
//            globalRotVelocities6D[1] = globalRotations6D[1] - previous6DRotations[1];

//            // Hand velocities (need visibility transition handling)
//            for (int i = 0; i < 2; i++) // For each hand (left=0, right=1)
//            {
//                // Check if this hand just became visible (was invisible in previous frame)
//                bool justBecameVisible = handVisibility[i] && !lastHandVisibility[i];

//                // Handle global position velocities for hands
//                if (justBecameVisible)
//                {
//                    // Zero velocity when hand just became visible
//                    globalPosVelocities[i + 1] = Vector3.zero;
//                    DebugLog($"{(i == 0 ? "Left" : "Right")} hand just became visible - zeroing global position velocity");
//                }
//                else
//                {
//                    // Normal velocity calculation
//                    globalPosVelocities[i + 1] = transformedPositions[i + 1] - previousGlobalPositions[i + 1];
//                }

//                // Handle global rotation velocities for hands
//                if (justBecameVisible)
//                {
//                    // Zero rotation velocity when hand just became visible
//                    globalRotVelocities6D[i * 2 + 2] = Vector3.zero;
//                    globalRotVelocities6D[i * 2 + 3] = Vector3.zero;
//                }
//                else
//                {
//                    // Normal rotation velocity calculation
//                    globalRotVelocities6D[i * 2 + 2] = globalRotations6D[i * 2 + 2] - previous6DRotations[i * 2 + 2];
//                    globalRotVelocities6D[i * 2 + 3] = globalRotations6D[i * 2 + 3] - previous6DRotations[i * 2 + 3];
//                }

//                // Handle local position velocities
//                if (justBecameVisible)
//                {
//                    // Zero local position velocity when hand just became visible
//                    localPosVelocities[i] = Vector3.zero;
//                }
//                else
//                {
//                    // Normal local position velocity calculation
//                    localPosVelocities[i] = localHandPositions[i] - previousLocalPositions[i];
//                }

//                // Handle local rotation velocities
//                if (justBecameVisible)
//                {
//                    // Zero local rotation velocity when hand just became visible
//                    localRotVelocities6D[i * 2] = Vector3.zero;
//                    localRotVelocities6D[i * 2 + 1] = Vector3.zero;
//                }
//                else
//                {
//                    // Normal local rotation velocity calculation
//                    localRotVelocities6D[i * 2] = localRotations6D[i * 2] - previousLocal6DRotations[i * 2];
//                    localRotVelocities6D[i * 2 + 1] = localRotations6D[i * 2 + 1] - previousLocal6DRotations[i * 2 + 1];
//                }
//            }
//        }

//        // STEP 7: Store current values for next frame's velocity calculation
//        Array.Copy(transformedPositions, previousGlobalPositions, transformedPositions.Length);
//        Array.Copy(globalRotations6D, previous6DRotations, globalRotations6D.Length);
//        Array.Copy(localHandPositions, previousLocalPositions, localHandPositions.Length);
//        Array.Copy(localRotations6D, previousLocal6DRotations, localRotations6D.Length);

//        // STEP 8: Create flattened feature vector (90D)
//        float[] baseFeatures = UnityTransformUtils.CreateHeadspaceFeatures(
//            transformedPositions,
//            globalPosVelocities,
//            globalRotations6D,
//            globalRotVelocities6D,
//            localHandPositions,
//            localPosVelocities,
//            localRotations6D,
//            localRotVelocities6D
//        );

//        // STEP 9: Apply masking if needed
//        //if (useHandVisibility)
//        //{
//        //    // Apply masking based on hand visibility
//        //    baseFeatures = ApplyHandVisibilityMasking(baseFeatures, handVisibility);
//        //}

//        // STEP 10: Prepare final feature vector (with or without visibility flags)
//        float[] result;

//        if (useHandVisibility)
//        {
//            // Create extended 92D feature vector with visibility flags
//            result = new float[BASE_FEATURE_SIZE + VISIBILITY_FEATURE_SIZE];

//            // Copy base features
//            Array.Copy(baseFeatures, result, BASE_FEATURE_SIZE);

//            // Add visibility flags at the end
//            // 1.0f for visible, 0.0f for invisible
//            result[BASE_FEATURE_SIZE] = handVisibility[0] ? 1.0f : 0.0f; // Left hand
//            result[BASE_FEATURE_SIZE + 1] = handVisibility[1] ? 1.0f : 0.0f; // Right hand

//            if (enableDebugLogging && (frameCount % 100 == 0))
//            {
//                Debug.Log($"[InputProcessor] 92D feature vector with visibility: left={result[BASE_FEATURE_SIZE]}, right={result[BASE_FEATURE_SIZE + 1]}");
//            }
//        }
//        else
//        {
//            // Use the original 90D feature vector
//            result = baseFeatures;
//        }

//        // Track performance
//        lastProcessTime = Time.realtimeSinceStartup - startTime;
//        avgProcessTime = (avgProcessTime * frameCount + lastProcessTime) / (frameCount + 1);
//        frameCount++;

//        // Log performance every 100 frames
//        if (frameCount % 100 == 0 && enableDebugLogging)
//        {
//            Debug.Log($"[InputProcessor] Avg processing time: {avgProcessTime * 1000:F2}ms over {frameCount} frames");
//            Debug.Log($"[InputProcessor] Feature vector size: {result.Length}D");
//        }

//        return result;
//    }

//    /// <summary>
//    /// Applies masking to hand-related features based on hand visibility
//    /// </summary>
//    private float[] ApplyHandVisibilityMasking(float[] features, bool[] handVisibility)
//    {
//        // Apply masking for each visibility state
//        if (!handVisibility[0]) // Left hand not visible
//        {
//            MaskLeftHandFeatures(features);
//        }

//        if (!handVisibility[1]) // Right hand not visible
//        {
//            MaskRightHandFeatures(features);
//        }

//        return features;
//    }

//    /// <summary>
//    /// Masks all left hand related features to zero
//    /// </summary>
//    private void MaskLeftHandFeatures(float[] features)
//    {
//        // Mask left hand global rotation (indices 2-5 in the 6D rotation section)
//        int leftHandRotStartIdx = 2; // index 0-1 is head, 2-3 is left hand, 4-5 is right hand
//        features[leftHandRotStartIdx] = 0;
//        features[leftHandRotStartIdx + 1] = 0;
//        features[leftHandRotStartIdx + 2] = 0;
//        features[leftHandRotStartIdx + 3] = 0;
//        features[leftHandRotStartIdx + 4] = 0;
//        features[leftHandRotStartIdx + 5] = 0;

//        // Mask left hand rotation velocities (indices 20-23)
//        int leftHandRotVelStartIdx = 20; // index 18-19 is head, 20-23 is left hand, 24-29 is right hand
//        features[leftHandRotVelStartIdx] = 0;
//        features[leftHandRotVelStartIdx + 1] = 0;
//        features[leftHandRotVelStartIdx + 2] = 0;
//        features[leftHandRotVelStartIdx + 3] = 0;
//        features[leftHandRotVelStartIdx + 4] = 0;
//        features[leftHandRotVelStartIdx + 5] = 0;

//        // Mask left hand global position (index 39-41)
//        int leftHandPosStartIdx = 39; // index 36-38 is head, 39-41 is left hand, 42-44 is right hand
//        features[leftHandPosStartIdx] = 0;
//        features[leftHandPosStartIdx + 1] = 0;
//        features[leftHandPosStartIdx + 2] = 0;

//        // Mask left hand position velocities (index 48-50)
//        int leftHandPosVelStartIdx = 48; // index 45-47 is head, 48-50 is left hand, 51-53 is right hand
//        features[leftHandPosVelStartIdx] = 0;
//        features[leftHandPosVelStartIdx + 1] = 0;
//        features[leftHandPosVelStartIdx + 2] = 0;

//        // Mask left hand local rotations (indices 54-59)
//        int leftHandLocalRotStartIdx = 54; // left hand is 54-59
//        features[leftHandLocalRotStartIdx] = 0;
//        features[leftHandLocalRotStartIdx + 1] = 0;
//        features[leftHandLocalRotStartIdx + 2] = 0;
//        features[leftHandLocalRotStartIdx + 3] = 0;
//        features[leftHandLocalRotStartIdx + 4] = 0;
//        features[leftHandLocalRotStartIdx + 5] = 0;

//        // Mask left hand local rotation velocities (indices 66-71)
//        int leftHandLocalRotVelStartIdx = 66; // left hand is 66-71
//        features[leftHandLocalRotVelStartIdx] = 0;
//        features[leftHandLocalRotVelStartIdx + 1] = 0;
//        features[leftHandLocalRotVelStartIdx + 2] = 0;
//        features[leftHandLocalRotVelStartIdx + 3] = 0;
//        features[leftHandLocalRotVelStartIdx + 4] = 0;
//        features[leftHandLocalRotVelStartIdx + 5] = 0;

//        // Mask left hand local positions (indices 78-80)
//        int leftHandLocalPosStartIdx = 78; // left hand is 78-80
//        features[leftHandLocalPosStartIdx] = 0;
//        features[leftHandLocalPosStartIdx + 1] = 0;
//        features[leftHandLocalPosStartIdx + 2] = 0;

//        // Mask left hand local position velocities (indices 84-86)
//        int leftHandLocalPosVelStartIdx = 84; // left hand is 84-86
//        features[leftHandLocalPosVelStartIdx] = 0;
//        features[leftHandLocalPosVelStartIdx + 1] = 0;
//        features[leftHandLocalPosVelStartIdx + 2] = 0;
//    }

//    /// <summary>
//    /// Masks all right hand related features to zero
//    /// </summary>
//    private void MaskRightHandFeatures(float[] features)
//    {
//        // Mask right hand global rotation (indices 4-5 in the 6D rotation section)
//        int rightHandRotStartIdx = 4; // index 0-1 is head, 2-3 is left hand, 4-5 is right hand
//        features[rightHandRotStartIdx] = 0;
//        features[rightHandRotStartIdx + 1] = 0;
//        features[rightHandRotStartIdx + 2] = 0;
//        features[rightHandRotStartIdx + 3] = 0;
//        features[rightHandRotStartIdx + 4] = 0;
//        features[rightHandRotStartIdx + 5] = 0;

//        // Mask right hand rotation velocities (indices 24-29)
//        int rightHandRotVelStartIdx = 24; // index 18-19 is head, 20-23 is left hand, 24-29 is right hand
//        features[rightHandRotVelStartIdx] = 0;
//        features[rightHandRotVelStartIdx + 1] = 0;
//        features[rightHandRotVelStartIdx + 2] = 0;
//        features[rightHandRotVelStartIdx + 3] = 0;
//        features[rightHandRotVelStartIdx + 4] = 0;
//        features[rightHandRotVelStartIdx + 5] = 0;

//        // Mask right hand global position (index 42-44)
//        int rightHandPosStartIdx = 42; // index 36-38 is head, 39-41 is left hand, 42-44 is right hand 
//        features[rightHandPosStartIdx] = 0;
//        features[rightHandPosStartIdx + 1] = 0;
//        features[rightHandPosStartIdx + 2] = 0;

//        // Mask right hand position velocities (index 51-53)
//        int rightHandPosVelStartIdx = 51; // index 45-47 is head, 48-50 is left hand, 51-53 is right hand
//        features[rightHandPosVelStartIdx] = 0;
//        features[rightHandPosVelStartIdx + 1] = 0;
//        features[rightHandPosVelStartIdx + 2] = 0;

//        // Mask right hand local rotations (indices 60-65)
//        int rightHandLocalRotStartIdx = 60; // right hand is 60-65
//        features[rightHandLocalRotStartIdx] = 0;
//        features[rightHandLocalRotStartIdx + 1] = 0;
//        features[rightHandLocalRotStartIdx + 2] = 0;
//        features[rightHandLocalRotStartIdx + 3] = 0;
//        features[rightHandLocalRotStartIdx + 4] = 0;
//        features[rightHandLocalRotStartIdx + 5] = 0;

//        // Mask right hand local rotation velocities (indices 72-77)
//        int rightHandLocalRotVelStartIdx = 72; // right hand is 72-77
//        features[rightHandLocalRotVelStartIdx] = 0;
//        features[rightHandLocalRotVelStartIdx + 1] = 0;
//        features[rightHandLocalRotVelStartIdx + 2] = 0;
//        features[rightHandLocalRotVelStartIdx + 3] = 0;
//        features[rightHandLocalRotVelStartIdx + 4] = 0;
//        features[rightHandLocalRotVelStartIdx + 5] = 0;

//        // Mask right hand local positions (indices 81-83)
//        int rightHandLocalPosStartIdx = 81; // right hand is 81-83
//        features[rightHandLocalPosStartIdx] = 0;
//        features[rightHandLocalPosStartIdx + 1] = 0;
//        features[rightHandLocalPosStartIdx + 2] = 0;

//        // Mask right hand local position velocities (indices 87-89)
//        int rightHandLocalPosVelStartIdx = 87; // right hand is 87-89
//        features[rightHandLocalPosVelStartIdx] = 0;
//        features[rightHandLocalPosVelStartIdx + 1] = 0;
//        features[rightHandLocalPosVelStartIdx + 2] = 0;
//    }

//    /// <summary>
//    /// Gets the converted head position
//    /// </summary>
//    public Vector3 GetConvertedHeadPosition()
//    {
//        if (!calibrator.IsCalibrated())
//        {
//            DebugLog("Warning: Requesting head position before calibration is complete");
//            return Vector3.zero;
//        }

//        return lastTransformedHeadPosition;
//    }

//    /// <summary>
//    /// Gets the current hand visibility state
//    /// </summary>
//    public bool[] GetHandVisibility()
//    {
//        if (!calibrator.IsCalibrated())
//            return new bool[] { false, false };

//        return lastHandVisibility;
//    }

//    /// <summary>
//    /// Gets the current processing performance metrics
//    /// </summary>
//    public (float last, float avg, int count) GetPerformanceMetrics()
//    {
//        return (lastProcessTime * 1000, avgProcessTime * 1000, frameCount);
//    }

//    /// <summary>
//    /// Enable or disable hand visibility masking at runtime
//    /// </summary>
//    public void SetHandVisibilityMasking(bool enable)
//    {
//        useHandVisibility = enable;
//        DebugLog($"Hand visibility masking {(enable ? "enabled" : "disabled")}");
//    }

//    /// <summary>
//    /// Returns whether hand visibility masking is currently enabled
//    /// </summary>
//    public bool IsHandVisibilityMaskingEnabled()
//    {
//        return useHandVisibility;
//    }

//    /// <summary>
//    /// Logger method with conditional execution based on debug flag
//    /// </summary>
//    private void DebugLog(string message)
//    {
//        if (enableDebugLogging)
//        {
//            Debug.Log($"[InputProcessor] {message}");
//        }
//    }
//}





using System;
using UnityEngine;

/// <summary>
/// Optimized input processor for VR tracking data that creates 90D feature vectors
/// with improved performance and memory management
/// </summary>
public class InputProcessor : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private VRToSMPLCalibrator calibrator;

    [Header("Settings")]
    [SerializeField] private bool enableDebugLogging = false;

    // Constants
    private const int FEATURE_SIZE = 90;

    // Pre-allocated arrays - all arrays are pre-sized and reused
    private readonly Vector3[] unityPositions = new Vector3[3];
    private readonly Quaternion[] unityRotations = new Quaternion[3];
    private readonly Vector3[] transformedPositions = new Vector3[3];
    private readonly Quaternion[] transformedRotations = new Quaternion[3];
    private readonly Vector3[] handPositions = new Vector3[2];
    private readonly Quaternion[] handRotations = new Quaternion[2];
    private readonly Vector3[] localHandPositions = new Vector3[2];
    private readonly Quaternion[] localHandRotations = new Quaternion[2];
    private readonly Vector3[] globalRotations6D = new Vector3[6];
    private readonly Vector3[] localRotations6D = new Vector3[4];
    private readonly float[] featureVector = new float[FEATURE_SIZE];

    // Pre-allocated velocity arrays
    private readonly Vector3[] globalPosVelocities = new Vector3[3];
    private readonly Vector3[] globalRotVelocities6D = new Vector3[6];
    private readonly Vector3[] localPosVelocities = new Vector3[2];
    private readonly Vector3[] localRotVelocities6D = new Vector3[4];

    // State variables for velocity calculation
    private bool isFirstFrame = true;
    private readonly Vector3[] previousGlobalPositions = new Vector3[3];
    private readonly Vector3[] previousLocalPositions = new Vector3[2];
    private readonly Vector3[] previous6DRotations = new Vector3[6];
    private readonly Vector3[] previousLocal6DRotations = new Vector3[4];

    // Latest transformed head position for external access
    private Vector3 lastTransformedHeadPosition;

    // Performance tracking (minimal overhead)
    private float avgProcessTime;
    private int frameCount;

    void Awake()
    {
        if (calibrator == null)
            calibrator = GetComponent<VRToSMPLCalibrator>();
    }

    /// <summary>
    /// Process VR tracking data into a 90D feature vector with optimized performance
    /// </summary>
    public float[] ProcessTrackingData()
    {
        if (!calibrator.IsCalibrated())
        {
            // Return pre-allocated zero array without new allocation
            Array.Clear(featureVector, 0, FEATURE_SIZE);
            return featureVector;
        }

        float startTime = enableDebugLogging ? Time.realtimeSinceStartup : 0f;

        // Get the reference transforms from the calibrator
        Transform headTransform = calibrator.GetHeadTransform();
        Transform leftHandTransform = calibrator.GetLeftHandTransform();
        Transform rightHandTransform = calibrator.GetRightHandTransform();

        // STEP 1: Get raw Unity positions and rotations (no allocation)
        unityPositions[0] = headTransform.position;
        unityPositions[1] = leftHandTransform.position;
        unityPositions[2] = rightHandTransform.position;

        unityRotations[0] = headTransform.rotation;
        unityRotations[1] = leftHandTransform.rotation;
        unityRotations[2] = rightHandTransform.rotation;

        // STEP 2: Convert Unity coordinates to SMPL/Python coordinate system
        UnityTransformUtils.TransformCoordinatesToAMASS(
            unityPositions,
            unityRotations,
            out var processedPositions,
            out var processedRotations
        );

        // Direct copy without checking
        Array.Copy(processedPositions, transformedPositions, 3);
        Array.Copy(processedRotations, transformedRotations, 3);

        // Store transformed head position
        lastTransformedHeadPosition = transformedPositions[0];

        // STEP 3: Convert rotations to 6D representation
        for (int i = 0; i < 3; i++)
        {
            Matrix4x4 rotMatrix = Matrix4x4.Rotate(transformedRotations[i]);
            Vector3[] rot6D = UnityTransformUtils.MatrixToSixD(rotMatrix);
            globalRotations6D[i * 2] = rot6D[0];
            globalRotations6D[i * 2 + 1] = rot6D[1];
        }

        // STEP 4: Calculate head space transforms for hands
        handPositions[0] = transformedPositions[1];
        handPositions[1] = transformedPositions[2];
        handRotations[0] = transformedRotations[1];
        handRotations[1] = transformedRotations[2];

        UnityTransformUtils.TransformToHeadSpace(
            transformedPositions[0],
            transformedRotations[0],
            handPositions,
            handRotations,
            out var processedHandPositions,
            out var processedHandRotations
        );

        Array.Copy(processedHandPositions, localHandPositions, 2);
        Array.Copy(processedHandRotations, localHandRotations, 2);

        // STEP 5: Convert local rotations to 6D representation
        for (int i = 0; i < 2; i++)
        {
            Matrix4x4 rotMatrix = Matrix4x4.Rotate(localHandRotations[i]);
            Vector3[] rot6D = UnityTransformUtils.MatrixToSixD(rotMatrix);
            localRotations6D[i * 2] = rot6D[0];
            localRotations6D[i * 2 + 1] = rot6D[1];
        }

        // STEP 6: Calculate velocities (reuse pre-allocated arrays)
        if (isFirstFrame)
        {
            // Zero velocities on first frame
            Array.Clear(globalPosVelocities, 0, 3);
            Array.Clear(globalRotVelocities6D, 0, 6);
            Array.Clear(localPosVelocities, 0, 2);
            Array.Clear(localRotVelocities6D, 0, 4);

            isFirstFrame = false;
        }
        else
        {
            // Calculate velocities directly into pre-allocated arrays
            for (int i = 0; i < 3; i++)
            {
                globalPosVelocities[i] = transformedPositions[i] - previousGlobalPositions[i];
            }

            for (int i = 0; i < 6; i++)
            {
                globalRotVelocities6D[i] = globalRotations6D[i] - previous6DRotations[i];
            }

            for (int i = 0; i < 2; i++)
            {
                localPosVelocities[i] = localHandPositions[i] - previousLocalPositions[i];
            }

            for (int i = 0; i < 4; i++)
            {
                localRotVelocities6D[i] = localRotations6D[i] - previousLocal6DRotations[i];
            }
        }

        // STEP 7: Store current values for next frame
        Array.Copy(transformedPositions, previousGlobalPositions, 3);
        Array.Copy(globalRotations6D, previous6DRotations, 6);
        Array.Copy(localHandPositions, previousLocalPositions, 2);
        Array.Copy(localRotations6D, previousLocal6DRotations, 4);

        // STEP 8: Create feature vector directly in pre-allocated array
        CreateFeatureVector();

        // Performance tracking (only if debugging enabled)
        if (enableDebugLogging)
        {
            float processTime = Time.realtimeSinceStartup - startTime;
            avgProcessTime = (avgProcessTime * frameCount + processTime) / (frameCount + 1);
            frameCount++;

            if (frameCount % 100 == 0)
            {
                Debug.Log($"[InputProcessor] Avg time: {avgProcessTime * 1000:F2}ms over {frameCount} frames");
            }
        }

        return featureVector;
    }

    /// <summary>
    /// Efficiently create the 90D feature vector in the pre-allocated array
    /// </summary>
    private void CreateFeatureVector()
    {
        int offset = 0;

        // Global 6D rotations [0:18]
        for (int i = 0; i < 6; i++)
        {
            featureVector[offset++] = globalRotations6D[i].x;
            featureVector[offset++] = globalRotations6D[i].y;
            featureVector[offset++] = globalRotations6D[i].z;
        }

        // Global rotation velocities [18:36]
        for (int i = 0; i < 6; i++)
        {
            featureVector[offset++] = globalRotVelocities6D[i].x;
            featureVector[offset++] = globalRotVelocities6D[i].y;
            featureVector[offset++] = globalRotVelocities6D[i].z;
        }

        // Global positions [36:45]
        for (int i = 0; i < 3; i++)
        {
            featureVector[offset++] = transformedPositions[i].x;
            featureVector[offset++] = transformedPositions[i].y;
            featureVector[offset++] = transformedPositions[i].z;
        }

        // Global position velocities [45:54]
        for (int i = 0; i < 3; i++)
        {
            featureVector[offset++] = globalPosVelocities[i].x;
            featureVector[offset++] = globalPosVelocities[i].y;
            featureVector[offset++] = globalPosVelocities[i].z;
        }

        // Hand rotations in head space [54:66]
        for (int i = 0; i < 4; i++)
        {
            featureVector[offset++] = localRotations6D[i].x;
            featureVector[offset++] = localRotations6D[i].y;
            featureVector[offset++] = localRotations6D[i].z;
        }

        // Hand rotation velocities in head space [66:78]
        for (int i = 0; i < 4; i++)
        {
            featureVector[offset++] = localRotVelocities6D[i].x;
            featureVector[offset++] = localRotVelocities6D[i].y;
            featureVector[offset++] = localRotVelocities6D[i].z;
        }

        // Hand positions in head space [78:84]
        for (int i = 0; i < 2; i++)
        {
            featureVector[offset++] = localHandPositions[i].x;
            featureVector[offset++] = localHandPositions[i].y;
            featureVector[offset++] = localHandPositions[i].z;
        }

        // Hand position velocities in head space [84:90]
        for (int i = 0; i < 2; i++)
        {
            featureVector[offset++] = localPosVelocities[i].x;
            featureVector[offset++] = localPosVelocities[i].y;
            featureVector[offset++] = localPosVelocities[i].z;
        }
    }

    /// <summary>
    /// Gets the converted head position
    /// </summary>
    public Vector3 GetConvertedHeadPosition()
    {
        return lastTransformedHeadPosition;
    }

    /// <summary>
    /// Gets the current processing performance metrics
    /// </summary>
    public (float avg, int count) GetPerformanceMetrics()
    {
        return (avgProcessTime * 1000, frameCount);
    }
}