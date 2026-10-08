using System;
using UnityEngine;
using Unity.Sentis;

/// <summary>
/// Samples the tracking features at a fixed rate into a ring buffer and hands the
/// most recent window to the on-device model (Unity Sentis).
/// </summary>
public class OnDeviceInput : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private InputProcessor inputProcessor;

    [Header("Buffer Settings")]
    [Tooltip("Frames per inference window; must match the input shape of the model.")]
    [SerializeField] private int sequenceLength = 40;
    [SerializeField] private int bufferSize = 120;
    [Tooltip("Feature sampling rate (Hz); must match the frame rate the model was trained at, " +
             "since the velocity features are per-frame differences.")]
    [SerializeField] private int targetFPS = 30;

    // Events
    public delegate void InferenceWindowReadyHandler(Tensor<float> inputTensor, Vector3 headPosition, int frameIdx, float timestamp);
    public event InferenceWindowReadyHandler OnInferenceWindowReady;

    // Statistics (read by OnDeviceStatsPanel)
    public int FramesSampled => framesReceived;
    public int BufferedFrames => frameCount;
    public int SequenceLength => sequenceLength;
    public int TargetFPS => targetFPS;

    // Buffer state
    private int writePtr = 0;
    private int frameCount = 0;
    private int framesReceived = 0;
    private bool windowFilled = false;
    private bool inferenceInProgress = false;
    private int lastInferredFrame = -1;

    // Frame buffers - pre-allocated
    private float[][] featuresBuffer;
    private Vector3[] headPosBuffer;
    private float[] timestampBuffer;
    private int[] frameIdxBuffer;

    // Pre-allocated window data
    private float[] windowData;
    private Tensor<float> inputTensor;

    // Timing
    private float timePerFrame;
    private float nextFrameTime;

    // Constants
    private const int FEATURE_SIZE = 90;

    void Awake()
    {
        if (inputProcessor == null)
        {
            Debug.LogError("[OnDeviceInput] Missing InputProcessor!");
            enabled = false;
            return;
        }

        InitializeBuffers();

        timePerFrame = 1.0f / targetFPS;
        nextFrameTime = Time.realtimeSinceStartup + timePerFrame;
    }

    void Update()
    {
        float currentTime = Time.realtimeSinceStartup;

        // Sample on a fixed schedule. Samples can only be taken once per display frame, so
        // take the frame closest to each scheduled time and advance the schedule by whole
        // periods: the average rate stays at targetFPS, and when the display rate is a
        // multiple of it (90 or 120 Hz for 30 Hz) every interval is exact. Restarting the
        // period at the current frame instead would round every interval up to whole
        // display frames (30 Hz became 24 Hz on a 72 Hz display).
        if (currentTime >= nextFrameTime - 0.5f * Time.unscaledDeltaTime)
        {
            ProcessFrame();
            nextFrameTime += timePerFrame;
            if (nextFrameTime <= currentTime)
            {
                // More than one period behind (e.g. a hitch): resynchronise
                nextFrameTime = currentTime + timePerFrame;
            }
        }

        // Try to run inference if ready
        if (windowFilled && !inferenceInProgress)
        {
            TryRunInference();
        }
    }

    private void InitializeBuffers()
    {
        // Pre-allocate all buffers
        featuresBuffer = new float[bufferSize][];
        headPosBuffer = new Vector3[bufferSize];
        timestampBuffer = new float[bufferSize];
        frameIdxBuffer = new int[bufferSize];

        for (int i = 0; i < bufferSize; i++)
        {
            featuresBuffer[i] = new float[FEATURE_SIZE];
        }

        windowData = new float[sequenceLength * FEATURE_SIZE];
        inputTensor = new Tensor<float>(new TensorShape(1, sequenceLength, FEATURE_SIZE));
    }

    private void ProcessFrame()
    {
        // Get features from InputProcessor
        float[] features = inputProcessor.ProcessTrackingData();
        Vector3 headPos = inputProcessor.GetConvertedHeadPosition();
        float timestamp = Time.realtimeSinceStartup;

        // Store in circular buffer
        Array.Copy(features, featuresBuffer[writePtr], FEATURE_SIZE);
        headPosBuffer[writePtr] = headPos;
        timestampBuffer[writePtr] = timestamp;
        frameIdxBuffer[writePtr] = framesReceived;

        // Update state
        if (frameCount < bufferSize) frameCount++;
        if (frameCount >= sequenceLength && !windowFilled) windowFilled = true;

        writePtr = (writePtr + 1) % bufferSize;
        framesReceived++;
    }

    private void TryRunInference()
    {
        if (frameCount < sequenceLength) return;

        // Only infer on a window that contains a new frame; the same window again would
        // reproduce the previous result
        if (framesReceived == lastInferredFrame) return;
        lastInferredFrame = framesReceived;

        // Get the most recent frame index
        int startIdx = (writePtr - 1 + bufferSize) % bufferSize;

        // Build window data (oldest to newest)
        int dataOffset = 0;
        for (int i = sequenceLength - 1; i >= 0; i--)
        {
            int bufferIdx = (startIdx - i + bufferSize) % bufferSize;
            Array.Copy(featuresBuffer[bufferIdx], 0, windowData, dataOffset, FEATURE_SIZE);
            dataOffset += FEATURE_SIZE;
        }

        // Upload to tensor
        inputTensor.Upload(windowData);

        // Get metadata for most recent frame
        Vector3 headPosition = headPosBuffer[startIdx];
        int frameIdx = frameIdxBuffer[startIdx];
        float timestamp = timestampBuffer[startIdx];

        // Mark inference in progress
        inferenceInProgress = true;

        // Fire event
        OnInferenceWindowReady?.Invoke(inputTensor, headPosition, frameIdx, timestamp);
    }

    public void NotifyInferenceComplete()
    {
        inferenceInProgress = false;
    }

    void OnDestroy()
    {
        inputTensor?.Dispose();
    }
}
