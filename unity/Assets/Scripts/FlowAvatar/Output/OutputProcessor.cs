using System;
using UnityEngine;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Collections.Generic;

/// <summary>
/// Handles conversion of 6D rotations to axis-angle format for Unity
/// and processes beta shape parameters for SMPL-X
/// </summary>
public class OutputProcessor : MonoBehaviour
{
    [Header("Processing Settings")]
    [SerializeField] private bool useParallelProcessing = true;
    [SerializeField] private int bufferSize = 100;

    [Header("Beta Shape Settings")]
    [SerializeField] private float initialCalibrationTime = 10.0f; // Time to wait before processing betas
    [SerializeField] private int minFramesBeforeBetaApplication = 90; // Minimum frames to collect

    [Header("Performance Profiling")]
    [SerializeField] private bool enableProfiling = false;

    // Queue for processing work
    private ConcurrentQueue<ProcessTask> processingQueue = new ConcurrentQueue<ProcessTask>();
    private ConcurrentQueue<ProcessedOutputData> resultQueue = new ConcurrentQueue<ProcessedOutputData>();
    private Task processingTask;
    private CancellationTokenSource cancellationTokenSource;
    private bool isProcessing = false;

    // Performance tracking
    private float[] processingTimes;
    private int processTimeIndex = 0;
    private int framesProcessed = 0;

    // Beta collection
    private List<float[]> betasBatch = new List<float[]>();
    private float startTime;
    private bool betasApplied = false;
    private float[] calibratedBetas = new float[10];

    // Thread-safe stopwatch for timing
    private readonly Stopwatch threadStopwatch = new Stopwatch();

    // Data structures for processing
    public struct ProcessTask
    {
        public float[] RootRotation6D;  // 6 values
        public float[] BodyRotation6D;  // 21 joints x 6 values = 126 values
        public float[] Betas;           // 10 SMPL-X shape parameters
        public Vector3 HeadPosition;
        public int FrameCounter;
        public float Timestamp;
    }

    public struct ProcessedOutputData
    {
        public Vector3[] Poses;        // 22 joints (1 root + 21 body) in axis-angle format
        public Vector3 Translation;     // Root translation
        public float[] Betas;           // 10 SMPL-X shape parameters
        public bool ApplyBetas;         // Flag indicating if betas should be applied
        public int FrameCounter;
        public float Timestamp;
        public float ProcessingTime;
    }

    // Reference to the motion controller to check if beta application is enabled
    private SMPLMotionController motionController;

    void Awake()
    {
        // Initialize performance tracking
        processingTimes = new float[bufferSize];
        startTime = Time.time;

        // Find the motion controller
        motionController = GetComponent<SMPLMotionController>();
        if (motionController == null)
        {
            motionController = FindObjectOfType<SMPLMotionController>();
        }
    }

    void OnEnable()
    {
        if (useParallelProcessing)
        {
            StartProcessingThread();
        }

        // Reset beta tracking if re-enabled
        betasApplied = false;
        betasBatch.Clear();
        startTime = Time.time;
    }

    void OnDisable()
    {
        if (isProcessing)
        {
            StopProcessingThread();
        }
    }

    void Update()
    {
        // Process results from the worker thread if using parallel processing
        if (useParallelProcessing)
        {
            ProcessResults();
        }

        // Only collect beta calibration if the motion controller has enableApplyBetas set to true
        bool shouldCollectBetas = motionController != null && motionController.EnableApplyBetas;

        // Check if it's time to finalize beta calibration
        if (shouldCollectBetas && !betasApplied &&
            Time.time - startTime >= initialCalibrationTime &&
            betasBatch.Count >= minFramesBeforeBetaApplication)
        {
            FinalizeBetaCalibration();
        }
    }

    private void FinalizeBetaCalibration()
    {
        if (betasBatch.Count == 0) return;

        // Calculate average beta values
        for (int i = 0; i < 10; i++)
        {
            float sum = 0f;
            foreach (var beta in betasBatch)
            {
                sum += beta[i];
            }
            calibratedBetas[i] = sum / betasBatch.Count;
        }

        // Set flag to indicate betas are ready to be applied
        betasApplied = true;

        if (enableProfiling)
        {
            UnityEngine.Debug.Log($"[OutputProcessor] Beta calibration complete after {Time.time - startTime:F1}s with {betasBatch.Count} samples");
        }

        // Clear the batch to free memory
        betasBatch.Clear();
    }

    private void StartProcessingThread()
    {
        if (isProcessing) return;

        cancellationTokenSource = new CancellationTokenSource();
        var token = cancellationTokenSource.Token;

        // Start the worker task
        processingTask = Task.Run(() => ProcessingWorker(token), token);
        isProcessing = true;

        LogInfo("Started parallel output processing worker");
    }

    private void StopProcessingThread()
    {
        if (!isProcessing) return;

        cancellationTokenSource.Cancel();
        try
        {
            processingTask.Wait(1000); // Give it 1 second to complete
        }
        catch (AggregateException ae)
        {
            if (!(ae.InnerException is TaskCanceledException))
            {
                LogInfo($"Error stopping worker: {ae.Message}");
            }
        }

        isProcessing = false;
        LogInfo($"Stopped parallel output processing worker. Processed {framesProcessed} frames.");
    }

    /// <summary>
    /// Processes raw 6D rotation data into axis-angle format for Unity
    /// </summary>
    public void ProcessOutputData(float[] rootRot6D, float[] bodyRot6D, Vector3 headPosition, int frameCounter, float timestamp, float[] betas = null)
    {
        // Only collect beta values if the motion controller has enableApplyBetas set to true
        bool shouldCollectBetas = motionController != null && motionController.EnableApplyBetas;

        // Store beta values during calibration phase
        if (shouldCollectBetas && !betasApplied && betas != null && betas.Length == 10)
        {
            lock (betasBatch)
            {
                betasBatch.Add((float[])betas.Clone());
            }
        }

        if (useParallelProcessing)
        {
            // Enqueue for background processing
            processingQueue.Enqueue(new ProcessTask
            {
                RootRotation6D = rootRot6D,
                BodyRotation6D = bodyRot6D,
                HeadPosition = headPosition,
                Betas = betas ?? new float[10], // Default to empty array if no betas provided
                FrameCounter = frameCounter,
                Timestamp = timestamp
            });
        }
        else
        {
            // Process directly on the main thread
            float startTime = Time.realtimeSinceStartup;

            // Process rotation data
            Vector3[] poses = new Vector3[22]; // 1 root + 21 body joints
            Vector3 translation = Vector3.zero;

            ProcessRotationsAndTranslation(rootRot6D, bodyRot6D, headPosition, poses, ref translation);

            float processTime = Time.realtimeSinceStartup - startTime;

            // Track processing time
            processingTimes[processTimeIndex] = processTime;
            processTimeIndex = (processTimeIndex + 1) % processingTimes.Length;
            framesProcessed++;

            // Check if we need to apply betas
            bool shouldApplyBetas = false;
            float[] betasToApply = null;

            if (shouldCollectBetas && betasApplied && !betasBatchNotified)
            {
                shouldApplyBetas = true;
                betasToApply = calibratedBetas;
                betasBatchNotified = true; // Set flag to only notify once
            }

            // Create result
            ProcessedOutputData result = new ProcessedOutputData
            {
                Poses = poses,
                Translation = translation,
                Betas = betasToApply,
                ApplyBetas = shouldApplyBetas,
                FrameCounter = frameCounter,
                Timestamp = timestamp,
                ProcessingTime = processTime
            };

            // Fire event with processed data
            OnOutputProcessed?.Invoke(result);
        }
    }

    // Flag to track if we've already notified about the betas
    private bool betasBatchNotified = false;

    private void ProcessingWorker(CancellationToken token)
    {
        try
        {
            // Use thread-safe Stopwatch instead of Time.realtimeSinceStartup
            Stopwatch stopwatch = new Stopwatch();

            while (!token.IsCancellationRequested)
            {
                if (processingQueue.TryDequeue(out ProcessTask task))
                {
                    try
                    {
                        // Start timing with thread-safe Stopwatch
                        stopwatch.Reset();
                        stopwatch.Start();

                        // Process rotation data
                        Vector3[] poses = new Vector3[22]; // 1 root + 21 body joints
                        Vector3 translation = Vector3.zero;

                        ProcessRotationsAndTranslation(task.RootRotation6D, task.BodyRotation6D,
                                                    task.HeadPosition, poses, ref translation);

                        // Stop timing
                        stopwatch.Stop();
                        float processTime = stopwatch.ElapsedMilliseconds / 1000f;

                        // Check if we need to apply betas
                        bool shouldApplyBetas = false;
                        float[] betasToApply = null;

                        // Only apply betas if the motion controller has enableApplyBetas set to true
                        bool shouldCollectBetas = motionController != null && motionController.EnableApplyBetas;

                        lock (betasBatch)
                        {
                            // Only send beta values once after calibration is complete
                            if (shouldCollectBetas && betasApplied && !betasBatchNotified)
                            {
                                shouldApplyBetas = true;
                                betasToApply = calibratedBetas;
                                betasBatchNotified = true; // Set flag to only notify once
                            }
                        }

                        // Enqueue the result
                        resultQueue.Enqueue(new ProcessedOutputData
                        {
                            Poses = poses,
                            Translation = translation,
                            Betas = betasToApply,
                            ApplyBetas = shouldApplyBetas,
                            FrameCounter = task.FrameCounter,
                            Timestamp = task.Timestamp,
                            ProcessingTime = processTime
                        });

                        Interlocked.Increment(ref framesProcessed);
                    }
                    catch (Exception e)
                    {
                        UnityEngine.Debug.LogError($"Error in processing worker: {e.Message}\n{e.StackTrace}");
                    }
                }
                else
                {
                    // Sleep to avoid spinning the CPU when queue is empty
                    Thread.Sleep(1);
                }
            }
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError($"Worker thread exception: {e.Message}\n{e.StackTrace}");
        }
    }

    // Delegate for output processed event
    public delegate void OutputProcessedDelegate(ProcessedOutputData processedData);
    public event OutputProcessedDelegate OnOutputProcessed;

    private void ProcessResults()
    {
        // Process all available results
        while (resultQueue.TryDequeue(out ProcessedOutputData result))
        {
            // Track processing time
            processingTimes[processTimeIndex] = result.ProcessingTime;
            processTimeIndex = (processTimeIndex + 1) % processingTimes.Length;

            // Fire event with the processed data
            OnOutputProcessed?.Invoke(result);
        }
    }

    /// <summary>
    /// Processes 6D rotations into axis-angle format and calculates translation
    /// </summary>
    private void ProcessRotationsAndTranslation(
        float[] rootRot6D, float[] bodyRot6D, Vector3 headPosition,
        Vector3[] poses, ref Vector3 translation)
    {
        // 1. Process root rotation (6D to axis-angle)
        Vector3 rootCol1 = new Vector3(rootRot6D[0], rootRot6D[1], rootRot6D[2]);
        Vector3 rootCol2 = new Vector3(rootRot6D[3], rootRot6D[4], rootRot6D[5]);
        poses[0] = UnityTransformUtils.SixDToAxisAngle(rootCol1, rootCol2);

        // 2. Process body joint rotations (6D to axis-angle)
        for (int j = 0; j < 21; j++)
        {
            int baseIdx = j * 6;
            Vector3 col1 = new Vector3(
                bodyRot6D[baseIdx],
                bodyRot6D[baseIdx + 1],
                bodyRot6D[baseIdx + 2]
            );
            Vector3 col2 = new Vector3(
                bodyRot6D[baseIdx + 3],
                bodyRot6D[baseIdx + 4],
                bodyRot6D[baseIdx + 5]
            );

            // Wrists included: the predicted wrist rotations are used whenever the
            // hands are not tracked (EnableHandTracking overrides them otherwise)
            poses[j + 1] = UnityTransformUtils.SixDToAxisAngle(col1, col2);
        }

        // 3. Convert coordinates from SMPL to Unity
        UnityTransformUtils.ConvertSMPLToUnity(poses, headPosition, ref translation);
    }

    private void LogInfo(string message)
    {
        if (enableProfiling)
        {
            UnityEngine.Debug.Log($"[OutputProcessor] {message}");
        }
    }

    /// <summary>
    /// Get the average processing time in milliseconds
    /// </summary>
    public float GetAverageProcessingTime()
    {
        float sum = 0;
        int count = 0;

        foreach (float time in processingTimes)
        {
            if (time > 0)
            {
                sum += time;
                count++;
            }
        }

        return count > 0 ? (sum / count) * 1000 : 0; // Return in milliseconds
    }

    /// <summary>
    /// Get the number of frames processed
    /// </summary>
    public int GetFramesProcessed() => framesProcessed;

    /// <summary>
    /// Get the current queue size
    /// </summary>
    public int GetQueueSize() => processingQueue.Count;

    /// <summary>
    /// Force beta calibration to complete immediately with current samples
    /// </summary>
    public void ForceCompleteBetaCalibration()
    {
        if (!betasApplied && betasBatch.Count > 0)
        {
            FinalizeBetaCalibration();
        }
    }

    /// <summary>
    /// Reset beta calibration to start over
    /// </summary>
    public void ResetBetaCalibration()
    {
        betasApplied = false;
        betasBatchNotified = false;
        betasBatch.Clear();
        startTime = Time.time;

        if (enableProfiling)
        {
            UnityEngine.Debug.Log("[OutputProcessor] Beta calibration reset");
        }
    }
}