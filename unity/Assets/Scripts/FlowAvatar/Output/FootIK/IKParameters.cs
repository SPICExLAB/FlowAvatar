using UnityEngine;

/// <summary>
/// Parameter container for IK settings passed from foot placer to leg IK
/// </summary>
[System.Serializable]
public class IKParameters
{
    // Basic correction settings
    [Range(0f, 2f)]
    public float correctionStrength = 1.0f;

    // Progressive correction
    [Range(1, 3)]
    public int correctionPasses = 2;

    [Range(0f, 1f)]
    public float progressiveStrengthScale = 0.7f;

    // Joint rotation scales
    [Range(0.1f, 3f)]
    public float hipRotationScale = 1.5f;

    [Range(0.1f, 3f)]
    public float kneeRotationScale = 1.5f;

    [Range(0.1f, 3f)]
    public float ankleRotationScale = 1.0f;

    // Joint angle limits
    public float hipMinAngle = -120f;
    public float hipMaxAngle = 50f;
    public float kneeMinAngle = 0f;
    public float kneeMaxAngle = 150f;
    public float ankleMinAngle = -20f;
    public float ankleMaxAngle = 70f;

    // Stability settings
    [Range(0f, 0.9f)]
    public float smoothing = 0.2f;

    // Ground alignment
    [Range(0f, 1f)]
    public float groundAlignmentStrength = 0f;

    // Previous frame data for smoothing (state)
    [HideInInspector] public float previousHipCorrection = 0f;
    [HideInInspector] public float previousKneeCorrection = 0f;
    [HideInInspector] public float previousAnkleCorrection = 0f;

    /// <summary>
    /// Reset the smoothing state
    /// </summary>
    public void ResetSmoothing()
    {
        previousHipCorrection = 0f;
        previousKneeCorrection = 0f;
        previousAnkleCorrection = 0f;
    }

    /// <summary>
    /// Initialize default joint limits
    /// </summary>
    public void InitializeDefaults()
    {
        // Set default joint limits
        hipMinAngle = -120f;
        hipMaxAngle = 50f;
        kneeMinAngle = 0f;
        kneeMaxAngle = 150f;
        ankleMinAngle = -10f;
        ankleMaxAngle = 40f;

        // Set default correction scales
        correctionStrength = 1.0f;
        correctionPasses = 2;
        progressiveStrengthScale = 0.7f;

        // Simplified joint scales
        hipRotationScale = 1.5f;
        kneeRotationScale = 1.5f;
        ankleRotationScale = 1.0f;

        // Stability
        smoothing = 0.2f;

        // Alignment
        groundAlignmentStrength = 0f;
    }
}