using UnityEngine;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;

/// <summary>
/// Component for receiving motion data from an ML model and applying it to a character
/// </summary>
public class ReceiveMotionData : MonoBehaviour
{
    [Header("Required Components")]
    public DualClient client;
    public SMPLMotionController motionController;
    public OutputProcessor outputProcessor;

    [Header("Frame Rate Settings")]
    [Tooltip("Target frame rate for motion updates - only 30 or 60 FPS supported")]
    [SerializeField] private int targetFPS = 30;

    [Header("Performance Profiling")]
    [SerializeField] private bool enableProfiling = false;
    [SerializeField] private float profilingInterval = 5.0f;

    // Network handler
    private NetworkReceiveUtils networkReceiver;

    // Performance monitoring
    private int framesProcessed = 0;
    private int framesInvalid = 0;
    private int framesDuringBetaApplication = 0;
    private float lastProfilingTime = 0f;

    // Latest processed pose data
    private Vector3[] latestPoses;
    private Vector3 latestTranslation;
    private float[] latestBetas; // Beta values from the model
    private bool shouldApplyBetas = false; // Flag to indicate if betas should be applied
    private int latestFrameCounter;
    private float latestFrameTimestamp;

    // Queue for accumulated data during beta application
    private ConcurrentQueue<(Vector3[], Vector3, float[], int, float)> accumulatedData =
        new ConcurrentQueue<(Vector3[], Vector3, float[], int, float)>();
    private int maxQueueSize = 5; // Maximum number of frames to keep in queue

    void Awake()
    {
        if (outputProcessor == null)
        {
            outputProcessor = GetComponent<OutputProcessor>();
            if (outputProcessor == null)
            {
                outputProcessor = gameObject.AddComponent<OutputProcessor>();
            }
        }
    }

    void Start()
    {
        if (!ValidateComponents()) return;

        // Ensure target FPS is valid
        if (targetFPS != 30 && targetFPS != 60)
        {
            Debug.LogWarning($"[{nameof(ReceiveMotionData)}] Invalid target FPS: {targetFPS}. Setting to 30 FPS.");
            targetFPS = 30;
        }

        // Initialize network receiver
        networkReceiver = new NetworkReceiveUtils(
            client,
            16 * 1024, // 16KB buffer size
            OnModelPredictionReceived,
            enableProfiling
        );

        // Subscribe to processed pose events
        outputProcessor.OnOutputProcessed += OnOutputProcessed;

        // Subscribe to beta application events
        if (motionController != null)
        {
            motionController.OnBetaApplicationStarted += OnBetaApplicationStarted;
            motionController.OnBetaApplicationCompleted += OnBetaApplicationCompleted;
        }

        if (enableProfiling)
        {
            Debug.Log($"[{nameof(ReceiveMotionData)}] Initialized with target FPS: {targetFPS}");
        }
    }

    private void OnDestroy()
    {
        // Unsubscribe from events
        if (outputProcessor != null)
        {
            outputProcessor.OnOutputProcessed -= OnOutputProcessed;
        }

        if (motionController != null)
        {
            motionController.OnBetaApplicationStarted -= OnBetaApplicationStarted;
            motionController.OnBetaApplicationCompleted -= OnBetaApplicationCompleted;
        }

        // Clean up network receiver
        if (networkReceiver != null)
        {
            networkReceiver.Dispose();
        }
    }

    private void OnBetaApplicationStarted()
    {
        // Beta application has started - we'll continue receiving data
        // but handle it differently
    }

    private void OnBetaApplicationCompleted()
    {
        // Process the most recent accumulated data
        ProcessAccumulatedData();

        // Reset counter
        framesDuringBetaApplication = 0;
    }

    void Update()
    {
        // Skip if not connected
        if (!client.isReceiveConnected)
        {
            return;
        }

        try
        {
            // Process incoming data from network
            bool received = networkReceiver.ReceiveAndProcessData();

            // Check if beta application is in progress
            bool betaApplicationInProgress = (motionController != null && motionController.IsApplyingBetas);

            // If beta application is in progress, we'll still receive data
            // but we won't apply poses immediately - they'll be queued and processed after
            if (betaApplicationInProgress)
            {
                framesDuringBetaApplication++;
                return;
            }

            // Apply latest pose and beta values if available
            if (latestPoses != null)
            {
                // Convert to flat array for motion controller
                float[] poseDataFlat = new float[latestPoses.Length * 3];
                for (int i = 0; i < latestPoses.Length; i++)
                {
                    poseDataFlat[i * 3] = latestPoses[i].x;
                    poseDataFlat[i * 3 + 1] = latestPoses[i].y;
                    poseDataFlat[i * 3 + 2] = latestPoses[i].z;
                }

                // Convert translation to array
                float[] transData = new float[] {
                    latestTranslation.x,
                    latestTranslation.y,
                    latestTranslation.z
                };

                // Apply to motion controller - only pass betas if they should be applied
                if (shouldApplyBetas && latestBetas != null)
                {
                    motionController.SetPoseFromFloats(poseDataFlat, transData, latestBetas);
                    if (enableProfiling)
                    {
                        Debug.Log($"[{nameof(ReceiveMotionData)}] Applied beta values to motion controller");
                    }
                    shouldApplyBetas = false; // Reset flag after applying
                }
                else
                {
                    motionController.SetPoseFromFloats(poseDataFlat, transData);
                }

                framesProcessed++;

                // Clear the data to avoid reapplying in next frame
                latestPoses = null;
            }

            // Log performance stats
            if (enableProfiling && Time.time - lastProfilingTime >= profilingInterval)
            {
                LogPerformanceStats();
                lastProfilingTime = Time.time;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[{nameof(ReceiveMotionData)}] Error in Update: {e.Message}");
        }
    }

    /// <summary>
    /// Handles new raw model output from Python
    /// </summary>
    private void OnModelPredictionReceived(byte[] messageData, FrameMetadata metadata)
    {
        try
        {
            // Parse raw model output
            OutputProcessor.ProcessTask output = ParseModelOutput(messageData, metadata);

            // Process data through output processor
            outputProcessor.ProcessOutputData(
                output.RootRotation6D,
                output.BodyRotation6D,
                output.HeadPosition,
                output.FrameCounter,
                output.Timestamp,
                output.Betas // Pass beta values to output processor
            );
        }
        catch (Exception e)
        {
            Debug.LogError($"[{nameof(ReceiveMotionData)}] Error parsing model output: {e.Message}");
            framesInvalid++;
        }
    }

    /// <summary>
    /// Parse raw binary data into model output format
    /// </summary>
    private OutputProcessor.ProcessTask ParseModelOutput(byte[] messageData, FrameMetadata metadata)
    {
        OutputProcessor.ProcessTask output = new OutputProcessor.ProcessTask();
        int offset = 0;

        // Parse root rotation (6 floats)
        output.RootRotation6D = new float[6];
        for (int i = 0; i < 6; i++)
        {
            output.RootRotation6D[i] = BitConverter.ToSingle(messageData, offset);
            offset += 4;
        }

        // Parse body rotations (21 joints * 6 values = 126 floats)
        output.BodyRotation6D = new float[21 * 6];
        for (int i = 0; i < 21 * 6; i++)
        {
            output.BodyRotation6D[i] = BitConverter.ToSingle(messageData, offset);
            offset += 4;
        }

        // Parse beta values (10 floats)
        output.Betas = new float[10];
        for (int i = 0; i < 10; i++)
        {
            output.Betas[i] = BitConverter.ToSingle(messageData, offset);
            offset += 4;
        }

        // Parse head position (3 floats)
        output.HeadPosition = new Vector3(
            BitConverter.ToSingle(messageData, offset),
            BitConverter.ToSingle(messageData, offset + 4),
            BitConverter.ToSingle(messageData, offset + 8)
        );

        // Set frame metadata
        output.FrameCounter = metadata.FrameCounter;
        output.Timestamp = metadata.Timestamp;

        return output;
    }

    /// <summary>
    /// Callback for when output processing is complete
    /// </summary>
    private void OnOutputProcessed(OutputProcessor.ProcessedOutputData processedData)
    {
        // Check if beta application is in progress
        bool betaApplicationInProgress = (motionController != null && motionController.IsApplyingBetas);

        if (betaApplicationInProgress)
        {
            // Queue the data to process after beta application completes
            QueueDataDuringBetaApplication(
                processedData.Poses,
                processedData.Translation,
                processedData.ApplyBetas ? processedData.Betas : null,
                processedData.FrameCounter,
                processedData.Timestamp
            );
            return;
        }

        // Store the latest processed data
        latestPoses = processedData.Poses;
        latestTranslation = processedData.Translation;

        // Only store beta values if they should be applied
        if (processedData.ApplyBetas && processedData.Betas != null)
        {
            latestBetas = processedData.Betas;
            shouldApplyBetas = true;

            if (enableProfiling)
            {
                Debug.Log($"[{nameof(ReceiveMotionData)}] Received batched beta values to apply");
            }
        }

        latestFrameCounter = processedData.FrameCounter;
        latestFrameTimestamp = processedData.Timestamp;

        if (enableProfiling && framesProcessed % 100 == 0)
        {
            Debug.Log($"[{nameof(ReceiveMotionData)}] Processed frame {latestFrameCounter} " +
                     $"in {processedData.ProcessingTime * 1000:F2}ms");
        }
    }

    /// <summary>
    /// Queue data that arrives during beta application
    /// </summary>
    private void QueueDataDuringBetaApplication(Vector3[] poses, Vector3 translation, float[] betas, int frameCounter, float timestamp)
    {
        // Don't store more than maxQueueSize frames to avoid memory buildup
        if (accumulatedData.Count >= maxQueueSize)
        {
            // Try to dequeue an item to make space
            accumulatedData.TryDequeue(out _);
        }

        // Add to the queue
        accumulatedData.Enqueue((poses, translation, betas, frameCounter, timestamp));

        if (enableProfiling && accumulatedData.Count % 2 == 0)
        {
            Debug.Log($"[{nameof(ReceiveMotionData)}] Queued data during beta application, queue size: {accumulatedData.Count}");
        }
    }

    /// <summary>
    /// Process any data accumulated during beta application
    /// </summary>
    private void ProcessAccumulatedData()
    {
        if (accumulatedData.Count == 0)
            return;

        // Find the most recent data point
        (Vector3[] poses, Vector3 translation, float[] betas, int frameCounter, float timestamp) latestData = default;
        int latestFrameIdx = -1;

        // Peek through all accumulated data to find the most recent frame
        while (accumulatedData.TryDequeue(out var data))
        {
            if (data.Item4 > latestFrameIdx)
            {
                latestFrameIdx = data.Item4;
                latestData = data;
            }
        }

        // If we found valid data, apply it
        if (latestFrameIdx >= 0)
        {
            if (enableProfiling)
            {
                Debug.Log($"[{nameof(ReceiveMotionData)}] Applying most recent frame {latestFrameIdx} after beta application");
            }

            // Set the latest data to be applied in the next Update
            latestPoses = latestData.poses;
            latestTranslation = latestData.translation;
            latestBetas = latestData.betas;
            shouldApplyBetas = latestBetas != null;
            latestFrameCounter = latestData.frameCounter;
            latestFrameTimestamp = latestData.timestamp;
        }
    }

    /// <summary>
    /// Clear all accumulated data
    /// </summary>
    public void ClearAccumulatedData()
    {
        int count = accumulatedData.Count;
        while (accumulatedData.TryDequeue(out _)) { }

        if (enableProfiling && count > 0)
        {
            Debug.Log($"[{nameof(ReceiveMotionData)}] Cleared {count} accumulated frames");
        }
    }

    private void LogPerformanceStats()
    {
        float elapsed = Time.time - lastProfilingTime;
        float fps = elapsed > 0 ? framesProcessed / elapsed : 0;

        Debug.Log($"[{nameof(ReceiveMotionData)}] Performance: " +
                 $"Processed {framesProcessed} frames ({fps:F1} FPS), " +
                 $"Invalid: {framesInvalid}, " +
                 $"During beta application: {framesDuringBetaApplication}, " +
                 $"Queue size: {accumulatedData.Count}, " +
                 $"Processing time: {outputProcessor.GetAverageProcessingTime():F2}ms, " +
                 $"Network stats: {networkReceiver.GetStats()}");

        // Reset counters
        framesProcessed = 0;
        framesInvalid = 0;
    }

    private bool ValidateComponents()
    {
        if (client == null)
        {
            Debug.LogError($"[{nameof(ReceiveMotionData)}] Missing DualClient component!");
            enabled = false;
            return false;
        }

        if (motionController == null)
        {
            Debug.LogError($"[{nameof(ReceiveMotionData)}] Missing SMPLMotionController component!");
            enabled = false;
            return false;
        }

        if (outputProcessor == null)
        {
            Debug.LogError($"[{nameof(ReceiveMotionData)}] Missing OutputProcessor component!");
            enabled = false;
            return false;
        }

        return true;
    }
}