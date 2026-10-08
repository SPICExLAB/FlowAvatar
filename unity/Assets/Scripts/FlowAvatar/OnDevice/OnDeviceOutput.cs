using System;
using UnityEngine;

/// <summary>
/// Optimized output handler for avatar pose application (on-device inference runs on the CPU)
/// </summary>
public class OnDeviceOutput : MonoBehaviour
{
    [Header("Required Components")]
    [SerializeField] private AvatarInferenceCPU inferenceComponent;
    [SerializeField] private SMPLMotionController motionController;
    [SerializeField] private OutputProcessor outputProcessor;

    [Header("Animation Settings")]
    [SerializeField] private int minInferencesBeforeApply = 5;

    [Header("Debug")]
    [SerializeField] private bool showDebugLog = false;

    // Pre-allocated arrays to avoid GC
    private readonly float[] poseDataFlat = new float[22 * 3]; // 1 root + 21 joints
    private readonly float[] transData = new float[3];

    // Pre-allocated arrays for ProcessTask
    private readonly float[] rootRotation6D = new float[6];
    private readonly float[] bodyRotation6D = new float[126];
    private readonly float[] betas = new float[10];

    // State
    private int inferencesProcessed = 0;
    private bool hasValidData = false;
    private bool firstProcessedDataReceived = false;

    // Latest pose data
    private Vector3[] latestPoses;
    private Vector3 latestTranslation;
    private float[] latestBetas;
    private bool shouldApplyBetas = false;
    private float latestTimestamp;

    // Latency is measured once per inference result, when its pose is first applied
    private int lastMeasuredInference = -1;

    /// <summary>Recent latencies (ms) from sampling the newest input frame to applying its pose.</summary>
    public RollingSamples LatencyMs { get; } = new RollingSamples(60);

    void Start()
    {
        if (!ValidateComponents()) return;

        // Subscribe to events
        outputProcessor.OnOutputProcessed += OnOutputProcessed;

        if (motionController != null)
        {
            motionController.OnBetaApplicationCompleted += OnBetaApplicationCompleted;
        }
    }

    void Update()
    {
        var output = inferenceComponent.GetModelOutput();
        if (output != null)
        {
            ProcessModelOutput(output);
        }

        // Apply pose if ready
        if (latestPoses != null && hasValidData && !motionController.IsApplyingBetas)
        {
            ApplyPoseToAvatar();
        }
    }

    private void ProcessModelOutput(AvatarInferenceCPU.ModelOutput output)
    {
        if (output.PoseData == null || output.ShapesData == null) return;

        // Copy data to pre-allocated arrays
        Array.Copy(output.PoseData, 0, rootRotation6D, 0, 6);
        Array.Copy(output.PoseData, 6, bodyRotation6D, 0, 126);
        Array.Copy(output.ShapesData, betas, 10);

        // Send to output processor
        outputProcessor.ProcessOutputData(
            rootRotation6D,
            bodyRotation6D,
            output.HeadPosition,
            output.FrameIdx,
            output.Timestamp,
            betas
        );

        inferencesProcessed++;
        CheckInitialization();
    }

    private void CheckInitialization()
    {
        if (!hasValidData && inferencesProcessed >= minInferencesBeforeApply && firstProcessedDataReceived)
        {
            hasValidData = true;
            if (showDebugLog)
            {
                Debug.Log($"[OnDeviceOutput] Starting pose application after {inferencesProcessed} inferences");
            }
        }
    }

    private void OnOutputProcessed(OutputProcessor.ProcessedOutputData processedData)
    {
        // Skip if beta application in progress
        if (motionController != null && motionController.IsApplyingBetas)
        {
            return;
        }

        if (!firstProcessedDataReceived)
        {
            firstProcessedDataReceived = true;
        }

        // Store processed data
        latestPoses = processedData.Poses;
        latestTranslation = processedData.Translation;
        latestTimestamp = processedData.Timestamp;

        // Handle one-time beta application
        if (processedData.ApplyBetas && processedData.Betas != null)
        {
            latestBetas = processedData.Betas;
            shouldApplyBetas = true;
        }
    }

    private void ApplyPoseToAvatar()
    {
        // Convert to flat array (reuse pre-allocated)
        for (int i = 0; i < latestPoses.Length; i++)
        {
            int idx = i * 3;
            poseDataFlat[idx] = latestPoses[i].x;
            poseDataFlat[idx + 1] = latestPoses[i].y;
            poseDataFlat[idx + 2] = latestPoses[i].z;
        }

        transData[0] = latestTranslation.x;
        transData[1] = latestTranslation.y;
        transData[2] = latestTranslation.z;

        // Apply to motion controller
        if (shouldApplyBetas && latestBetas != null)
        {
            motionController.SetPoseFromFloats(poseDataFlat, transData, latestBetas);
            shouldApplyBetas = false; // One-time application
            if (showDebugLog)
            {
                Debug.Log("[OnDeviceOutput] Applied betas to avatar shape");
            }
        }
        else
        {
            motionController.SetPoseFromFloats(poseDataFlat, transData);
        }

        // The latest result is re-applied every frame; measure latency only for a new one
        int completed = inferenceComponent.InferencesCompleted;
        if (completed != lastMeasuredInference && latestTimestamp > 0)
        {
            LatencyMs.Add((Time.realtimeSinceStartup - latestTimestamp) * 1000f);
            lastMeasuredInference = completed;
        }

        latestPoses = null; // Prevent reapplying
    }

    private void OnBetaApplicationCompleted()
    {
        if (showDebugLog)
        {
            Debug.Log("[OnDeviceOutput] Beta shape application completed");
        }
    }

    private bool ValidateComponents()
    {
        if (inferenceComponent == null)
        {
            Debug.LogError("[OnDeviceOutput] Missing inference component!");
            return false;
        }

        if (motionController == null)
        {
            Debug.LogError("[OnDeviceOutput] Missing SMPLMotionController!");
            return false;
        }

        if (outputProcessor == null)
        {
            Debug.LogError("[OnDeviceOutput] Missing OutputProcessor!");
            return false;
        }

        return true;
    }

    void OnDisable()
    {
        if (outputProcessor != null)
        {
            outputProcessor.OnOutputProcessed -= OnOutputProcessed;
        }

        if (motionController != null)
        {
            motionController.OnBetaApplicationCompleted -= OnBetaApplicationCompleted;
        }
    }
}