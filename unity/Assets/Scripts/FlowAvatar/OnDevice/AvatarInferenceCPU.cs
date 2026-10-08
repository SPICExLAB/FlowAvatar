using System;
using UnityEngine;
using Unity.Sentis;

/// <summary>
/// CPU-optimized inference handler for Avatar pose estimation model
/// Simple implementation using just IsReadbackRequestDone() for CPU tensors
/// </summary>
public class AvatarInferenceCPU : MonoBehaviour
{
    [Header("Model Settings")]
    [SerializeField] private ModelAsset modelAsset;

    [Header("References")]
    [SerializeField] private OnDeviceInput inputBuffer;

    [Header("Debug")]
    [SerializeField] private bool showDebugLog = false;

    // Model output structure
    public class ModelOutput
    {
        public float[] PoseData = new float[132];
        public float[] ShapesData = new float[10];
        public Vector3 HeadPosition;
        public int FrameIdx;
        public float Timestamp;
    }

    // Sentis components
    private Worker worker;
    private ModelOutput currentOutput;

    // State
    private bool isProcessing = false;
    private Tensor inputTensor;
    private Tensor<float> outputPoseTensor;
    private Tensor<float> outputShapesTensor;
    private Vector3 currentHeadPosition;
    private int currentFrameIdx;
    private float currentTimestamp;

    // Performance tracking
    private float inferenceStartTime;
    private int inferencesCompleted = 0;

    // Statistics (read by OnDeviceStatsPanel and OnDeviceOutput)
    public int InferencesCompleted => inferencesCompleted;
    /// <summary>Recent inference times (ms), from scheduling to the result being read on the main thread.</summary>
    public RollingSamples InferenceMs { get; } = new RollingSamples(60);
    public string ModelName => modelAsset != null ? modelAsset.name : "";

    void Start()
    {
        if (!ValidateReferences()) return;

        currentOutput = new ModelOutput();
        LoadModel();

        if (inputBuffer != null)
        {
            inputBuffer.OnInferenceWindowReady += OnInferenceWindowReady;
        }
    }

    private bool ValidateReferences()
    {
        if (inputBuffer == null)
        {
            inputBuffer = GetComponent<OnDeviceInput>();
            if (inputBuffer == null)
            {
                Debug.LogError("[AvatarInferenceCPU] Missing OnDeviceInput!");
                enabled = false;
                return false;
            }
        }

        if (modelAsset == null)
        {
            Debug.LogError("[AvatarInferenceCPU] No model asset assigned!");
            enabled = false;
            return false;
        }

        return true;
    }

    private void LoadModel()
    {
        Model model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, BackendType.CPU);

        if (showDebugLog)
        {
            Debug.Log($"[AvatarInferenceCPU] Model loaded with CPU backend");
        }
    }

    void Update()
    {
        if (isProcessing)
        {
            ProcessCPUInference();
        }
    }

    private void ProcessCPUInference()
    {
        try
        {
            // Try to get outputs if we haven't already
            if (outputPoseTensor == null || outputShapesTensor == null)
            {
                outputPoseTensor = worker.PeekOutput("pred_pose") as Tensor<float>;
                outputShapesTensor = worker.PeekOutput("pred_shapes") as Tensor<float>;
            }

            // Check if tensors are available and readable
            if (outputPoseTensor != null && outputShapesTensor != null &&
                outputPoseTensor.IsReadbackRequestDone() &&
                outputShapesTensor.IsReadbackRequestDone())
            {
                // CPU tensors are directly readable when IsReadbackRequestDone() returns true
                var poseSpan = outputPoseTensor.AsReadOnlySpan();
                var shapesSpan = outputShapesTensor.AsReadOnlySpan();

                // Copy to output arrays
                for (int i = 0; i < Math.Min(poseSpan.Length, 132); i++)
                {
                    currentOutput.PoseData[i] = poseSpan[i];
                }

                for (int i = 0; i < Math.Min(shapesSpan.Length, 10); i++)
                {
                    currentOutput.ShapesData[i] = shapesSpan[i];
                }

                // Update metadata
                currentOutput.HeadPosition = currentHeadPosition;
                currentOutput.FrameIdx = currentFrameIdx;
                currentOutput.Timestamp = currentTimestamp;

                // Track performance
                InferenceMs.Add((Time.realtimeSinceStartup - inferenceStartTime) * 1000f);
                inferencesCompleted++;

                // Cleanup tensors
                outputPoseTensor.Dispose();
                outputShapesTensor.Dispose();
                outputPoseTensor = null;
                outputShapesTensor = null;

                // Reset state
                isProcessing = false;
                inputBuffer.NotifyInferenceComplete();
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[AvatarInferenceCPU] Error: {e.Message}");
            CleanupTensors();
            isProcessing = false;
            inputBuffer.NotifyInferenceComplete();
        }
    }

    private void OnInferenceWindowReady(Tensor<float> tensor, Vector3 headPos, int frameIdx, float timestamp)
    {
        if (isProcessing) return;

        inputTensor = tensor;
        currentHeadPosition = headPos;
        currentFrameIdx = frameIdx;
        currentTimestamp = timestamp;

        // Start inference
        inferenceStartTime = Time.realtimeSinceStartup;
        worker.Schedule(inputTensor);
        isProcessing = true;
    }

    public ModelOutput GetModelOutput() => currentOutput;

    private void CleanupTensors()
    {
        outputPoseTensor?.Dispose();
        outputShapesTensor?.Dispose();
        outputPoseTensor = null;
        outputShapesTensor = null;
    }

    void OnDestroy()
    {
        if (inputBuffer != null)
        {
            inputBuffer.OnInferenceWindowReady -= OnInferenceWindowReady;
        }

        CleanupTensors();
        worker?.Dispose();
    }
}