using UnityEngine;

/// <summary>
/// Simplified implementation of leg IK that focuses only on solving
/// penetration through joint rotation adjustments.
/// </summary>
public class SMPLLegIK : MonoBehaviour
{
    // Bone chain references
    [HideInInspector] public Transform hipJoint;
    [HideInInspector] public Transform kneeJoint;
    [HideInInspector] public Transform ankleJoint;
    [HideInInspector] public Transform footJoint;
    [HideInInspector] public Transform kneeHint;

    // Debug display
    public bool showDebugInfo = true;

    // State tracking (for visualization/debugging)
    private float lastAnklePenetrationAmount = 0f;
    private float lastFootPenetrationAmount = 0f;
    private Vector3 lastGroundNormal = Vector3.up;
    private bool targetReached = false;
    private Vector3 targetPosition = Vector3.zero;
    private bool isForwardStepPose = false;

    /// <summary>
    /// Initialize the leg chain from a foot reference
    /// </summary>
    public void InitializeFromFoot(Transform foot, SMPLXFootPlacer placer)
    {
        if (foot == null)
            return;

        footJoint = foot;

        // Assume standard hierarchy: foot -> ankle -> knee -> hip
        if (foot.parent != null)
        {
            ankleJoint = foot.parent;
            if (ankleJoint.parent != null)
            {
                kneeJoint = ankleJoint.parent;
                if (kneeJoint.parent != null)
                {
                    hipJoint = kneeJoint.parent;
                }
            }
        }

        // Create knee hint as child of knee if needed
        if (kneeJoint != null && kneeHint == null)
        {
            GameObject hintObj = new GameObject(kneeJoint.name + "_Hint");
            kneeHint = hintObj.transform;
            kneeHint.parent = kneeJoint;
            kneeHint.localPosition = new Vector3(0, 0, 0.1f); // Place in front of knee
            kneeHint.localRotation = Quaternion.identity;
        }

        // Set this game object to be the hip
        if (hipJoint != null && this.transform != hipJoint)
        {
            this.gameObject.name = hipJoint.name + "_IK";
            this.transform.position = hipJoint.position;
            this.transform.rotation = hipJoint.rotation;
            this.transform.parent = hipJoint.parent;
        }
    }

    /// <summary>
    /// Apply two-stage foot penetration correction
    /// </summary>
    public void ApplyPenetrationCorrection(float anklePenetrationAmount, float footPenetrationAmount, Vector3 groundNormal, IKParameters parameters)
    {
        if (!IsJointChainValid())
            return;

        // Store for visualization
        lastAnklePenetrationAmount = anklePenetrationAmount;
        lastFootPenetrationAmount = footPenetrationAmount;
        lastGroundNormal = groundNormal;

        // Analyze the pose to determine forward/backward
        AnalyzePoseCharacteristics();

        // Skip if no significant penetration
        bool needsHipKneeCorrection = anklePenetrationAmount > 0.001f;
        bool needsAnkleCorrection = footPenetrationAmount > 0.001f;

        if (!needsHipKneeCorrection && !needsAnkleCorrection)
            return;

        // Set target position for visualization
        targetPosition = footJoint.position + Vector3.up * Mathf.Max(anklePenetrationAmount, footPenetrationAmount) * parameters.correctionStrength;

        // First stage: Lift the ankle by adjusting hip and knee
        if (needsHipKneeCorrection)
        {
            AdjustHipAndKnee(anklePenetrationAmount, parameters);
        }

        // Second stage: Adjust ankle rotation to align foot
        if (needsAnkleCorrection)
        {
            AdjustAnkleRotation(footPenetrationAmount, groundNormal, parameters);
        }

        // Evaluate results
        float distance = Vector3.Distance(footJoint.position, targetPosition);
        float maxPenetration = Mathf.Max(anklePenetrationAmount, footPenetrationAmount);
        targetReached = distance < maxPenetration * 0.5f;
    }

    /// <summary>
    /// Analyze the current pose to determine forward/backward step
    /// </summary>
    private void AnalyzePoseCharacteristics()
    {
        if (!IsJointChainValid()) return;

        // Normalize the hip X angle to determine if this is a forward step
        float hipAngleX = NormalizeAngle(hipJoint.localRotation.eulerAngles.x);
        isForwardStepPose = hipAngleX < 0;
    }

    /// <summary>
    /// First stage: Adjust hip and knee to lift ankle
    /// </summary>
    private void AdjustHipAndKnee(float penetrationAmount, IKParameters parameters)
    {
        if (penetrationAmount < 0.001f) return;

        // Get current joint rotations
        Vector3 hipEuler = hipJoint.localRotation.eulerAngles;
        Vector3 kneeEuler = kneeJoint.localRotation.eulerAngles;

        // Normalize angles to -180 to 180 range
        float hipX = NormalizeAngle(hipEuler.x);
        float kneeX = NormalizeAngle(kneeEuler.x);

        // Calculate leg length and correction scale
        float legLength = CalculateLegLength();
        float correctionRatio = Mathf.Min(1.0f, penetrationAmount / legLength);
        float correctionScale = Mathf.Lerp(180f, 350f, Mathf.Pow(correctionRatio, 0.8f));

        // Calculate hip/knee weight ratio based on pose
        float hipWeight, kneeWeight;
        CalculateHipKneeWeights(out hipWeight, out kneeWeight);

        // Calculate rotation corrections
        float hipCorrection, kneeCorrection;

        if (isForwardStepPose)
        {
            hipCorrection = -penetrationAmount * hipWeight * correctionScale * parameters.hipRotationScale;
            kneeCorrection = penetrationAmount * kneeWeight * correctionScale * parameters.kneeRotationScale;
        }
        else
        {
            hipCorrection = penetrationAmount * hipWeight * correctionScale * parameters.hipRotationScale;
            kneeCorrection = penetrationAmount * kneeWeight * correctionScale * parameters.kneeRotationScale;
        }

        // Apply smoothing if needed
        if (parameters.smoothing > 0)
        {
            float reducedSmoothing = parameters.smoothing * 0.7f; // Less smoothing for faster response
            hipCorrection = Mathf.Lerp(hipCorrection, parameters.previousHipCorrection, reducedSmoothing);
            kneeCorrection = Mathf.Lerp(kneeCorrection, parameters.previousKneeCorrection, reducedSmoothing);

            // Store for next frame
            parameters.previousHipCorrection = hipCorrection;
            parameters.previousKneeCorrection = kneeCorrection;
        }

        // Apply soft limits
        float hipLimitFactor = GetLimitFactor(hipX, hipCorrection, parameters.hipMinAngle, parameters.hipMaxAngle);
        float kneeLimitFactor = GetLimitFactor(kneeX, kneeCorrection, parameters.kneeMinAngle, parameters.kneeMaxAngle);

        // Apply clamping to prevent exceeding joint limits
        float newHipX = Mathf.Clamp(hipX + hipCorrection * hipLimitFactor, parameters.hipMinAngle, parameters.hipMaxAngle);
        float newKneeX = Mathf.Clamp(kneeX + kneeCorrection * kneeLimitFactor, parameters.kneeMinAngle, parameters.kneeMaxAngle);

        // Apply new rotations while preserving Y and Z components
        hipJoint.localRotation = Quaternion.Euler(newHipX, hipEuler.y, hipEuler.z);
        kneeJoint.localRotation = Quaternion.Euler(newKneeX, kneeEuler.y, kneeEuler.z);

        // Debug visualization
        if (showDebugInfo)
        {
            Debug.DrawLine(hipJoint.position, kneeJoint.position, Color.magenta, Time.deltaTime);
            Debug.DrawLine(kneeJoint.position, ankleJoint.position, Color.magenta, Time.deltaTime);
        }
    }

    /// <summary>
    /// Second stage: Adjust ankle rotation to align foot with ground
    /// </summary>
    private void AdjustAnkleRotation(float penetrationAmount, Vector3 groundNormal, IKParameters parameters)
    {
        if (penetrationAmount < 0.001f || ankleJoint == null)
            return;

        // Get current ankle rotation
        Vector3 ankleEuler = ankleJoint.localRotation.eulerAngles;
        float ankleX = NormalizeAngle(ankleEuler.x);

        // Calculate desired ankle angle to align with ground
        float desiredAnkleX = -Vector3.SignedAngle(Vector3.up, groundNormal, Vector3.right);

        // Add extra rotation based on penetration amount
        float extraRotation = penetrationAmount * 15f; // 15 degrees per unit of penetration
        desiredAnkleX += (ankleX < 0) ? -extraRotation : extraRotation;

        // Apply stronger alignment for deeper penetration
        float alignmentStrength = Mathf.Min(1.0f, parameters.groundAlignmentStrength * (1f + penetrationAmount));

        // Blend between current angle and desired angle
        float targetAnkleX = Mathf.Lerp(ankleX, desiredAnkleX, alignmentStrength);

        // Apply smoothing
        if (parameters.smoothing > 0)
        {
            float reducedSmoothing = parameters.smoothing * 0.6f; // Less smoothing for ankle for faster response
            targetAnkleX = Mathf.Lerp(targetAnkleX, NormalizeAngle(parameters.previousAnkleCorrection), reducedSmoothing);
            parameters.previousAnkleCorrection = targetAnkleX;
        }

        // Apply constraints
        targetAnkleX = Mathf.Clamp(targetAnkleX, parameters.ankleMinAngle, parameters.ankleMaxAngle);

        // Apply the new rotation while maintaining Y and Z components
        ankleJoint.localRotation = Quaternion.Euler(targetAnkleX, ankleEuler.y, ankleEuler.z);

        // Debug visualization
        if (showDebugInfo)
        {
            Debug.DrawLine(ankleJoint.position, footJoint.position, Color.yellow, Time.deltaTime);
        }
    }

    /// <summary>
    /// Calculate adaptive hip and knee weights based on current pose angles
    /// </summary>
    private void CalculateHipKneeWeights(out float hipWeight, out float kneeWeight)
    {
        // Get current joint angles
        float hipAngleX = NormalizeAngle(hipJoint.localRotation.eulerAngles.x);
        float kneeAngleX = NormalizeAngle(kneeJoint.localRotation.eulerAngles.x);

        // Calculate normalized bend factors (0-1 range)
        float kneeBendFactor = Mathf.Clamp01(kneeAngleX / 120f);
        float hipBendFactor = Mathf.Clamp01(Mathf.Abs(hipAngleX) / 90f);

        // Base weights
        hipWeight = 0.40f;
        kneeWeight = 0.60f;

        // Adjust weights based on current bend factors
        if (kneeBendFactor > 0.5f)
        {
            float adjustment = (kneeBendFactor - 0.5f) * 0.4f;
            hipWeight -= adjustment;
            kneeWeight += adjustment;
        }

        if (hipBendFactor > 0.6f)
        {
            float adjustment = (hipBendFactor - 0.6f) * 0.5f;
            hipWeight -= adjustment;
            kneeWeight += adjustment;
        }

        // Normalize weights
        float totalWeight = hipWeight + kneeWeight;
        hipWeight /= totalWeight;
        kneeWeight /= totalWeight;
    }

    /// <summary>
    /// Calculate limit factor for soft joint limits
    /// </summary>
    private float GetLimitFactor(float currentAngle, float plannedCorrection, float minAngle, float maxAngle)
    {
        // If no significant correction, return 1
        if (Mathf.Abs(plannedCorrection) < 0.1f)
            return 1.0f;

        // Determine which limit is relevant based on correction direction
        if (plannedCorrection > 0)
        {
            // Approaching max limit
            float roomAvailable = maxAngle - currentAngle;
            float normalizedRoom = Mathf.Clamp01(roomAvailable / Mathf.Abs(plannedCorrection));
            return Mathf.Lerp(0.3f, 1.0f, Mathf.Pow(normalizedRoom, 0.7f));
        }
        else
        {
            // Approaching min limit
            float roomAvailable = currentAngle - minAngle;
            float normalizedRoom = Mathf.Clamp01(roomAvailable / Mathf.Abs(plannedCorrection));
            return Mathf.Lerp(0.3f, 1.0f, Mathf.Pow(normalizedRoom, 0.7f));
        }
    }

    /// <summary>
    /// Calculate approximate leg length for proportional corrections
    /// </summary>
    private float CalculateLegLength()
    {
        if (!IsJointChainValid())
            return 1.0f;

        float hipToKnee = Vector3.Distance(hipJoint.position, kneeJoint.position);
        float kneeToAnkle = Vector3.Distance(kneeJoint.position, ankleJoint.position);
        float ankleToFoot = Vector3.Distance(ankleJoint.position, footJoint.position);

        return hipToKnee + kneeToAnkle + ankleToFoot;
    }

    /// <summary>
    /// Helper to normalize angles to -180 to 180 range
    /// </summary>
    private float NormalizeAngle(float angle)
    {
        while (angle > 180f)
            angle -= 360f;
        while (angle < -180f)
            angle += 360f;
        return angle;
    }

    /// <summary>
    /// Check if the full joint chain is properly configured
    /// </summary>
    public bool IsJointChainValid()
    {
        return hipJoint != null && kneeJoint != null && ankleJoint != null && footJoint != null;
    }

    /// <summary>
    /// Get if this leg is in a forward step pose (for visualization)
    /// </summary>
    public bool IsForwardStepPose()
    {
        return isForwardStepPose;
    }

    /// <summary>
    /// Get the last correction amount (for visualization)
    /// </summary>
    public float GetLastCorrectionAmount()
    {
        return Mathf.Max(lastAnklePenetrationAmount, lastFootPenetrationAmount);
    }

    /// <summary>
    /// Get the target position (for visualization)
    /// </summary>
    public Vector3 GetTargetPosition()
    {
        return targetPosition;
    }

    /// <summary>
    /// Get whether target was reached (for visualization)
    /// </summary>
    public bool WasTargetReached()
    {
        return targetReached;
    }

    /// <summary>
    /// Reset leg rotations for testing
    /// </summary>
    public void ResetLegRotations()
    {
        if (hipJoint != null)
            hipJoint.localRotation = Quaternion.identity;

        if (kneeJoint != null)
            kneeJoint.localRotation = Quaternion.identity;

        if (ankleJoint != null)
            ankleJoint.localRotation = Quaternion.identity;

        if (footJoint != null)
            footJoint.localRotation = Quaternion.identity;
    }

    /// <summary>
    /// Draw the target position in scene view
    /// </summary>
    private void OnDrawGizmos()
    {
        if (!showDebugInfo || !Application.isPlaying)
            return;

        // Draw the target position if we have one
        if (targetPosition != Vector3.zero)
        {
            Gizmos.color = targetReached ? Color.green : Color.red;
            Gizmos.DrawSphere(targetPosition, 0.02f);

            if (footJoint != null)
            {
                Gizmos.DrawLine(footJoint.position, targetPosition);
            }
        }
    }
}