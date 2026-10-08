using System;
using UnityEngine;
using System.Runtime.InteropServices;
using System.Threading;

public class SendTrackingData : MonoBehaviour
{
    [SerializeField] private DualClient client;
    [SerializeField] private VRToSMPLCalibrator calibrator;
    [SerializeField] private InputProcessor inputProcessor;
    [SerializeField] private SMPLMotionController motionController;

    [Header("Performance Settings")]
    [SerializeField] private int targetFPS = 60;
    [SerializeField] private bool enableOptimizedSending = true;
    [SerializeField] private int sendBufferSize = 16384; // 16KB buffer

    // Timing control
    private float sendInterval;
    private float timer;
    private int frameCounter = 0;
    private int framesDropped = 0;
    private int framesSkippedDuringBetas = 0;
    private float lastSendTime;

    // Message components
    private NetworkSendUtils.FrameHeader header;
    private NetworkSendUtils.FrameData frameData;
    private NetworkSendUtils networkUtils;

    // Performance monitoring
    private int successfulSends = 0;
    private int totalAttempts = 0;
    private float[] sendTimes = new float[60]; // Keep track of recent send times
    private int sendTimeIndex = 0;
    private float nextStatsReportTime = 0;
    private const float STATS_REPORT_INTERVAL = 5.0f; // Report stats every 5 seconds

    // Feature vector constants
    private const int BASE_FEATURE_VECTOR_SIZE = 90;
    private const int VISIBILITY_FEATURE_SIZE = 2;
    private const int HEAD_POSITION_SIZE = 3;

    // Pre-allocated buffer
    private byte[] sendBuffer;
    private bool isSending = false;
    private AutoResetEvent sendComplete = new AutoResetEvent(true);

    void Start()
    {
        if (!ValidateComponents()) return;

        // Get the feature vector size from the input processor (90D or 92D)
        //int featureVectorSize = inputProcessor.GetFeatureVectorSize();
        int featureVectorSize = 90;

        // Initialize network utils with optimized mode
        networkUtils = new NetworkSendUtils(client, enableOptimizedSending, featureVectorSize, HEAD_POSITION_SIZE);

        // Pre-allocate send buffer
        sendBuffer = new byte[sendBufferSize];

        // Initialize data structures
        frameData = new NetworkSendUtils.FrameData
        {
            Features = new float[featureVectorSize],
            HeadPosition = new float[HEAD_POSITION_SIZE]
        };

        header = new NetworkSendUtils.FrameHeader
        {
            MessageType = 1,
            FrameRate = (byte)targetFPS,
            FrameCounter = 0,
            Timestamp = 0,
            DroppedFrames = 0,
            FeatureSize = (ushort)featureVectorSize // Store feature size in header
        };

        // Set FPS-dependent settings
        SetTargetFPS(targetFPS);

        // Initialize timing
        lastSendTime = Time.realtimeSinceStartup;
        nextStatsReportTime = Time.time + STATS_REPORT_INTERVAL;

        Debug.Log($"[SendTrackingData] Optimized sending at {targetFPS} FPS (interval: {sendInterval * 1000:F1}ms), feature size: {featureVectorSize}D");

        // Subscribe to beta application events
        if (motionController != null)
        {
            motionController.OnBetaApplicationStarted += OnBetaApplicationStarted;
            motionController.OnBetaApplicationCompleted += OnBetaApplicationCompleted;
        }
    }

    private void OnDestroy()
    {
        if (motionController != null)
        {
            motionController.OnBetaApplicationStarted -= OnBetaApplicationStarted;
            motionController.OnBetaApplicationCompleted -= OnBetaApplicationCompleted;
        }

        networkUtils.Dispose();
        sendComplete.Dispose();
    }

    private void OnBetaApplicationStarted()
    {
        // Beta application has started, but we'll keep sending data
        // We just track it for stats purposes
    }

    private void OnBetaApplicationCompleted()
    {
        // Reset counters after beta application
        framesSkippedDuringBetas = 0;
    }

    void Update()
    {
        // Skip if not calibrated
        if (!calibrator.IsCalibrated()) return;

        // Check beta application - but still continue
        bool betaApplicationInProgress = (motionController != null && motionController.IsApplyingBetas);
        if (betaApplicationInProgress)
        {
            framesSkippedDuringBetas++;
        }

        // Report stats periodically
        if (Time.time >= nextStatsReportTime)
        {
            ReportPerformanceStats();
            nextStatsReportTime = Time.time + STATS_REPORT_INTERVAL;
        }

        // Use delta time for more accurate timing
        timer += Time.deltaTime;

        // Check if we're ready to send
        if (timer >= sendInterval)
        {
            // Reset timer based on interval multiples to avoid drift
            int intervalsPassed = Mathf.FloorToInt(timer / sendInterval);
            timer -= sendInterval * intervalsPassed;

            // If we've missed multiple intervals, that means we're dropping frames
            if (intervalsPassed > 1)
            {
                framesDropped += (intervalsPassed - 1);
                header.DroppedFrames = (uint)framesDropped;
            }

            // Capture and send data if not already in progress and we're connected
            if (sendComplete.WaitOne(0) && client.isSendConnected)
            {
                // Only capture new data if we can send it
                CaptureAndSendData();

                // Increment counter for next frame
                frameCounter++;
                totalAttempts++;
            }
        }
    }

    public void SetTargetFPS(int fps)
    {
        // Limit to valid ML model rates (30 or 60)
        if (fps != 30 && fps != 60)
        {
            Debug.LogWarning($"[SendTrackingData] Invalid target FPS: {fps}. Using closest valid value.");
            fps = fps < 45 ? 30 : 60;
        }

        targetFPS = fps;
        sendInterval = 1f / targetFPS;
        header.FrameRate = (byte)targetFPS;
    }

    private void CaptureAndSendData()
    {
        try
        {
            // Process the tracking data (get a direct reference to avoid copies)
            float[] processedFeatures = inputProcessor.ProcessTrackingData();

            // Update the feature size in the header if it changed
            if (header.FeatureSize != processedFeatures.Length)
            {
                Debug.Log($"[SendTrackingData] Feature vector size changed: {header.FeatureSize} -> {processedFeatures.Length}");
                header.FeatureSize = (ushort)processedFeatures.Length;

                // Resize the feature array if needed
                if (frameData.Features.Length != processedFeatures.Length)
                {
                    frameData.Features = new float[processedFeatures.Length];
                }
            }

            // Copy data directly to our frame data
            // Use direct references to avoid array copies
            Array.Copy(processedFeatures, frameData.Features,
                Math.Min(processedFeatures.Length, frameData.Features.Length));

            // Get and add the head position
            Vector3 headPosition = inputProcessor.GetConvertedHeadPosition();
            frameData.HeadPosition[0] = headPosition.x;
            frameData.HeadPosition[1] = headPosition.y;
            frameData.HeadPosition[2] = headPosition.z;

            // Update header with frame info
            header.FrameCounter = (ushort)(frameCounter % ushort.MaxValue);
            header.Timestamp = Time.realtimeSinceStartup;

            // Mark as sending
            isSending = true;
            sendComplete.Reset();

            // Calculate send timing
            float currentTime = Time.realtimeSinceStartup;
            float sendDelta = currentTime - lastSendTime;
            lastSendTime = currentTime;

            // Track send timing
            sendTimes[sendTimeIndex] = sendDelta;
            sendTimeIndex = (sendTimeIndex + 1) % sendTimes.Length;

            // Send the frame using our optimized method
            if (enableOptimizedSending)
            {
                // Direct send using pre-allocated buffer
                networkUtils.SendFrameData(header, frameData, OnSendComplete);
            }
            else
            {
                // Use the original method
                networkUtils.SendFrameData(header, frameData);
                OnSendComplete(true);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[{nameof(SendTrackingData)}] Error sending: {e.Message}");
            OnSendComplete(false);
        }
    }

    private void OnSendComplete(bool success)
    {
        // Track successful sends
        if (success)
        {
            successfulSends++;
        }

        // Reset sending state
        isSending = false;
        sendComplete.Set();
    }

    private void ReportPerformanceStats()
    {
        // Calculate actual sending rate
        float sumTimes = 0;
        int validTimes = 0;

        foreach (float time in sendTimes)
        {
            if (time > 0)
            {
                sumTimes += time;
                validTimes++;
            }
        }

        float avgInterval = validTimes > 0 ? sumTimes / validTimes : 0;
        float actualFps = avgInterval > 0 ? 1.0f / avgInterval : 0;

        Debug.Log($"[SendTrackingData] Performance: " +
                 $"Send rate: {actualFps:F1} FPS (target: {targetFPS}), " +
                 $"Interval: {avgInterval * 1000:F1}ms, " +
                 $"Success rate: {(float)successfulSends / Math.Max(1, totalAttempts) * 100:F1}%, " +
                 $"Frames dropped: {framesDropped}, " +
                 $"Feature vector size: {header.FeatureSize}D");

        // Reset counters for next period
        totalAttempts = 0;
        successfulSends = 0;
    }

    private bool ValidateComponents()
    {
        if (client == null)
        {
            Debug.LogError($"[{nameof(SendTrackingData)}] Missing DualClient component!");
            enabled = false;
            return false;
        }

        if (calibrator == null)
        {
            Debug.LogError($"[{nameof(SendTrackingData)}] Missing VRToSMPLCalibrator component!");
            enabled = false;
            return false;
        }

        if (inputProcessor == null)
        {
            inputProcessor = GetComponent<InputProcessor>();
            if (inputProcessor == null)
            {
                Debug.LogError($"[{nameof(SendTrackingData)}] Missing InputProcessor component!");
                enabled = false;
                return false;
            }
        }

        return true;
    }
}