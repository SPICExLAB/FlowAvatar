using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Improved foot placer that uses cached lowest points for detection
/// to detect foot penetration and applies IK to correct it
/// </summary>
public class SMPLXFootPlacer : MonoBehaviour
{
    [Header("Foot Transforms")]
    public Transform leftFoot;
    public Transform rightFoot;

    [Header("Ground Detection")]
    public LayerMask groundLayer;
    public float groundHeight = 0f;
    [Range(0.05f, 0.5f)]
    [Tooltip("Search radius for initial point detection")]
    public float proximityRadius = 0.15f;
    [Range(1, 3)]
    public int maxSolveIterations = 2;

    [Header("Penetration Response")]
    public bool enablePenetrationCorrection = true;

    [Header("IK Parameters")]
    public IKParameters ikParameters = new IKParameters();

    [Header("Optimization")]
    [Tooltip("Height threshold to skip processing (when feet clearly above ground)")]
    public float heightThreshold = 0.1f;
    [Tooltip("Skip full detection every N frames")]
    [Range(1, 5)]
    public int detectionInterval = 2;

    [Header("Debug")]
    public bool showDebugInfo = true;
    public bool showEditorVisuals = true;
    public bool showPerformanceStats = false;

    // Internal references
    private SMPLLegIK leftLegIK;
    private SMPLLegIK rightLegIK;
    private bool isInitialized = false;

    // Mesh reference for finding initial contact points
    private SkinnedMeshRenderer _smr;
    private Mesh _bakedMesh;

    // Cached lowest points after beta application
    private Vector3 _leftAnkleLowestPoint = Vector3.zero;
    private Vector3 _rightAnkleLowestPoint = Vector3.zero;
    private Vector3 _leftFootLowestPoint = Vector3.zero;
    private Vector3 _rightFootLowestPoint = Vector3.zero;
    private float _leftAnkleLowestOffset = 0f;  // Y-offset from ankle to lowest point
    private float _rightAnkleLowestOffset = 0f; // Y-offset from ankle to lowest point
    private float _leftFootLowestOffset = 0f;   // Y-offset from foot to lowest point
    private float _rightFootLowestOffset = 0f;  // Y-offset from foot to lowest point
    private bool _lowestPointsInitialized = false;

    // Performance tracking
    private System.Diagnostics.Stopwatch _stopwatch = new System.Diagnostics.Stopwatch();
    private float _lastProcessingTime = 0f;
    private float _avgProcessingTime = 0f;
    private int _processedFrames = 0;
    private int _frameCounter = 0;

    // State tracking
    public enum FootState { NotPenetrating, Penetrating }
    public FootState LeftAnkleState { get; private set; } = FootState.NotPenetrating;
    public FootState RightAnkleState { get; private set; } = FootState.NotPenetrating;
    public FootState LeftFootState { get; private set; } = FootState.NotPenetrating;
    public FootState RightFootState { get; private set; } = FootState.NotPenetrating;
    public Vector3 LeftGroundNormal { get; private set; } = Vector3.up;
    public Vector3 RightGroundNormal { get; private set; } = Vector3.up;
    public float LeftAnklePenetrationDepth { get; private set; } = 0f;
    public float RightAnklePenetrationDepth { get; private set; } = 0f;
    public float LeftFootPenetrationDepth { get; private set; } = 0f;
    public float RightFootPenetrationDepth { get; private set; } = 0f;

    private void Start()
    {
        InitializeSystem();

        // Initialize IK parameters with defaults
        ikParameters.InitializeDefaults();
    }

    private void InitializeSystem()
    {
        // Get skinned mesh renderer for initial analysis
        _smr = GetComponentInChildren<SkinnedMeshRenderer>();
        if (_smr == null)
        {
            Debug.LogWarning("[SMPLXFootPlacer] No SkinnedMeshRenderer found in children. Using fallback detection.");
        }
        else
        {
            _bakedMesh = new Mesh();
        }

        // Find or create leg IK components
        if (leftLegIK == null && leftFoot != null)
        {
            Transform hip = FindHipFromFoot(leftFoot);
            if (hip != null)
            {
                leftLegIK = hip.GetComponent<SMPLLegIK>();
                if (leftLegIK == null)
                {
                    leftLegIK = hip.gameObject.AddComponent<SMPLLegIK>();
                }
                leftLegIK.InitializeFromFoot(leftFoot, this);
                leftLegIK.showDebugInfo = showDebugInfo;
            }
        }

        if (rightLegIK == null && rightFoot != null)
        {
            Transform hip = FindHipFromFoot(rightFoot);
            if (hip != null)
            {
                rightLegIK = hip.GetComponent<SMPLLegIK>();
                if (rightLegIK == null)
                {
                    rightLegIK = hip.gameObject.AddComponent<SMPLLegIK>();
                }
                rightLegIK.InitializeFromFoot(rightFoot, this);
                rightLegIK.showDebugInfo = showDebugInfo;
            }
        }

        isInitialized = true;

        if (showDebugInfo)
        {
            Debug.Log("[SMPLXFootPlacer] System initialized - waiting for beta application to detect lowest points");
        }
    }

    private Transform FindHipFromFoot(Transform foot)
    {
        // Assuming a standard hierarchy: hip -> knee -> ankle -> foot
        // Navigate up 3 levels from foot to find hip
        Transform current = foot;
        for (int i = 0; i < 3; i++)
        {
            if (current == null || current.parent == null)
                return null;
            current = current.parent;
        }
        return current;
    }

    /// <summary>
    /// Update debug visualization settings
    /// </summary>
    private void UpdateDebugSettings()
    {
        if (leftLegIK != null)
            leftLegIK.showDebugInfo = showDebugInfo;

        if (rightLegIK != null)
            rightLegIK.showDebugInfo = showDebugInfo;
    }

    /// <summary>
    /// Process foot penetrations and apply corrections.
    /// Called by the motion controller after applying animations
    /// </summary>
    public void ProcessFootPenetrations()
    {
        if (!isInitialized || !enablePenetrationCorrection)
        {
            return;
        }

        _stopwatch.Reset();
        _stopwatch.Start();
        _frameCounter++;

        // Quick height check to skip processing if both feet are clearly above ground
        if (leftFoot != null && leftFoot.position.y > groundHeight + heightThreshold &&
            rightFoot != null && rightFoot.position.y > groundHeight + heightThreshold)
        {
            // Reset states and skip further processing
            LeftAnkleState = FootState.NotPenetrating;
            RightAnkleState = FootState.NotPenetrating;
            LeftFootState = FootState.NotPenetrating;
            RightFootState = FootState.NotPenetrating;
            LeftAnklePenetrationDepth = 0f;
            RightAnklePenetrationDepth = 0f;
            LeftFootPenetrationDepth = 0f;
            RightFootPenetrationDepth = 0f;

            _stopwatch.Stop();
            return;
        }

        // Iterative solving approach
        for (int iteration = 0; iteration < maxSolveIterations; iteration++)
        {
            // Update penetration state - use full or quick detection based on frame
            if (_frameCounter % detectionInterval == 0 || !_lowestPointsInitialized)
            {
                // Full detection (every N frames or when lowest points not initialized)
                UpdateFootPenetrationState();
            }
            else
            {
                // Quick detection (most frames)
                QuickPenetrationCheck();
            }

            // Check if we still have penetration
            bool stillPenetrating = (LeftAnkleState == FootState.Penetrating || RightAnkleState == FootState.Penetrating ||
                                    LeftFootState == FootState.Penetrating || RightFootState == FootState.Penetrating);

            if (!stillPenetrating)
            {
                // No more penetration, we're done
                if (showDebugInfo && iteration > 0)
                {
                    Debug.Log($"[SMPLXFootPlacer] Penetration solved after {iteration + 1} iterations");
                }
                break;
            }

            // Apply corrections based on penetration state
            if (leftLegIK != null && leftLegIK.IsJointChainValid())
            {
                if (LeftAnkleState == FootState.Penetrating || LeftFootState == FootState.Penetrating)
                {
                    leftLegIK.ApplyPenetrationCorrection(
                        LeftAnklePenetrationDepth,
                        LeftFootPenetrationDepth,
                        LeftGroundNormal,
                        ikParameters);
                }
            }

            if (rightLegIK != null && rightLegIK.IsJointChainValid())
            {
                if (RightAnkleState == FootState.Penetrating || RightFootState == FootState.Penetrating)
                {
                    rightLegIK.ApplyPenetrationCorrection(
                        RightAnklePenetrationDepth,
                        RightFootPenetrationDepth,
                        RightGroundNormal,
                        ikParameters);
                }
            }
        }

        // Track performance
        _stopwatch.Stop();
        _lastProcessingTime = _stopwatch.ElapsedMilliseconds / 1000f;
        _processedFrames++;
        _avgProcessingTime = ((_avgProcessingTime * (_processedFrames - 1)) + _lastProcessingTime) / _processedFrames;
    }

    /// <summary>
    /// Quick penetration check using pre-calculated lowest points
    /// </summary>
    private void QuickPenetrationCheck()
    {
        // Reset states
        LeftAnkleState = FootState.NotPenetrating;
        RightAnkleState = FootState.NotPenetrating;
        LeftFootState = FootState.NotPenetrating;
        RightFootState = FootState.NotPenetrating;
        LeftAnklePenetrationDepth = 0f;
        RightAnklePenetrationDepth = 0f;
        LeftFootPenetrationDepth = 0f;
        RightFootPenetrationDepth = 0f;

        if (!_lowestPointsInitialized)
        {
            // Fall back to full detection if we don't have lowest points yet
            UpdateFootPenetrationState();
            return;
        }

        // Left ankle check
        if (leftFoot != null && leftFoot.parent != null)
        {
            Transform leftAnkle = leftFoot.parent;

            // Check ankle lowest point
            Vector3 ankleLowestPoint = leftAnkle.position;
            ankleLowestPoint.y -= _leftAnkleLowestOffset;

            if (ankleLowestPoint.y < groundHeight)
            {
                LeftAnkleState = FootState.Penetrating;
                LeftAnklePenetrationDepth = groundHeight - ankleLowestPoint.y;
                LeftGroundNormal = Vector3.up;
            }

            // Check foot lowest point
            Vector3 footLowestPoint = leftFoot.position;
            footLowestPoint.y -= _leftFootLowestOffset;

            if (footLowestPoint.y < groundHeight)
            {
                LeftFootState = FootState.Penetrating;
                LeftFootPenetrationDepth = groundHeight - footLowestPoint.y;
                LeftGroundNormal = Vector3.up;
            }
        }

        // Right ankle check
        if (rightFoot != null && rightFoot.parent != null)
        {
            Transform rightAnkle = rightFoot.parent;

            // Check ankle lowest point
            Vector3 ankleLowestPoint = rightAnkle.position;
            ankleLowestPoint.y -= _rightAnkleLowestOffset;

            if (ankleLowestPoint.y < groundHeight)
            {
                RightAnkleState = FootState.Penetrating;
                RightAnklePenetrationDepth = groundHeight - ankleLowestPoint.y;
                RightGroundNormal = Vector3.up;
            }

            // Check foot lowest point
            Vector3 footLowestPoint = rightFoot.position;
            footLowestPoint.y -= _rightFootLowestOffset;

            if (footLowestPoint.y < groundHeight)
            {
                RightFootState = FootState.Penetrating;
                RightFootPenetrationDepth = groundHeight - footLowestPoint.y;
                RightGroundNormal = Vector3.up;
            }
        }
    }

    /// <summary>
    /// Update foot penetration state by finding the lowest points
    /// </summary>
    public void UpdateFootPenetrationState()
    {
        // Reset states
        LeftAnkleState = FootState.NotPenetrating;
        RightAnkleState = FootState.NotPenetrating;
        LeftFootState = FootState.NotPenetrating;
        RightFootState = FootState.NotPenetrating;
        LeftAnklePenetrationDepth = 0f;
        RightAnklePenetrationDepth = 0f;
        LeftFootPenetrationDepth = 0f;
        RightFootPenetrationDepth = 0f;

        if (!isInitialized)
            return;

        // If we have lowest points already initialized, use them
        if (_lowestPointsInitialized)
        {
            QuickPenetrationCheck();
            return;
        }

        // Otherwise, perform full detection to find lowest points
        if (_smr != null)
        {
            try
            {
                // Bake the current mesh - only needed during initialization
                _smr.BakeMesh(_bakedMesh);

                // Find lowest points for each foot and ankle
                FindLowestPointsUnderFeet();

                // Now check for penetration using these points
                if (_leftAnkleLowestPoint != Vector3.zero && leftFoot != null && leftFoot.parent != null)
                {
                    // Check left ankle penetration
                    if (_leftAnkleLowestPoint.y < groundHeight)
                    {
                        LeftAnkleState = FootState.Penetrating;
                        LeftAnklePenetrationDepth = groundHeight - _leftAnkleLowestPoint.y;
                        LeftGroundNormal = Vector3.up;
                    }
                }

                if (_leftFootLowestPoint != Vector3.zero && leftFoot != null)
                {
                    // Check left foot penetration
                    if (_leftFootLowestPoint.y < groundHeight)
                    {
                        LeftFootState = FootState.Penetrating;
                        LeftFootPenetrationDepth = groundHeight - _leftFootLowestPoint.y;
                        LeftGroundNormal = Vector3.up;
                    }
                }

                if (_rightAnkleLowestPoint != Vector3.zero && rightFoot != null && rightFoot.parent != null)
                {
                    // Check right ankle penetration
                    if (_rightAnkleLowestPoint.y < groundHeight)
                    {
                        RightAnkleState = FootState.Penetrating;
                        RightAnklePenetrationDepth = groundHeight - _rightAnkleLowestPoint.y;
                        RightGroundNormal = Vector3.up;
                    }
                }

                if (_rightFootLowestPoint != Vector3.zero && rightFoot != null)
                {
                    // Check right foot penetration
                    if (_rightFootLowestPoint.y < groundHeight)
                    {
                        RightFootState = FootState.Penetrating;
                        RightFootPenetrationDepth = groundHeight - _rightFootLowestPoint.y;
                        RightGroundNormal = Vector3.up;
                    }
                }
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[SMPLXFootPlacer] Exception during mesh analysis: {e.Message}. Falling back to simple detection.");
                CheckFootPenetrationByRaycast(leftFoot, true);
                CheckFootPenetrationByRaycast(rightFoot, false);
            }
        }
        else
        {
            // Fall back to simple raycast detection
            CheckFootPenetrationByRaycast(leftFoot, true);
            CheckFootPenetrationByRaycast(rightFoot, false);
        }
    }

    /// <summary>
    /// Find the lowest points under each foot and ankle for efficient future detection
    /// </summary>
    private void FindLowestPointsUnderFeet()
    {
        if (_smr == null || _bakedMesh == null || leftFoot == null || rightFoot == null)
            return;

        // Get mesh vertices
        Vector3[] vertices = _bakedMesh.vertices;
        if (vertices == null || vertices.Length == 0)
            return;

        // Reset lowest points
        _leftAnkleLowestPoint = Vector3.zero;
        _rightAnkleLowestPoint = Vector3.zero;
        _leftFootLowestPoint = Vector3.zero;
        _rightFootLowestPoint = Vector3.zero;

        float leftAnkleLowestY = float.MaxValue;
        float rightAnkleLowestY = float.MaxValue;
        float leftFootLowestY = float.MaxValue;
        float rightFootLowestY = float.MaxValue;

        // Get ankle and foot positions
        Transform leftAnkle = leftFoot.parent;
        Transform rightAnkle = rightFoot.parent;
        if (leftAnkle == null || rightAnkle == null)
            return;

        // Get positions for XZ plane search
        Vector2 leftAnklePosXZ = new Vector2(leftAnkle.position.x, leftAnkle.position.z);
        Vector2 rightAnklePosXZ = new Vector2(rightAnkle.position.x, rightAnkle.position.z);
        Vector2 leftFootPosXZ = new Vector2(leftFoot.position.x, leftFoot.position.z);
        Vector2 rightFootPosXZ = new Vector2(rightFoot.position.x, rightFoot.position.z);

        // Find the lowest vertices under each ankle and foot
        for (int i = 0; i < vertices.Length; i++)
        {
            // Convert vertex to world space
            Vector3 worldVertex = _smr.transform.TransformPoint(vertices[i]);
            Vector2 vertexPosXZ = new Vector2(worldVertex.x, worldVertex.z);

            // Check if vertex is under left ankle
            float leftAnkleDist = Vector2.Distance(vertexPosXZ, leftAnklePosXZ);
            if (leftAnkleDist < proximityRadius)
            {
                // Check if it's lower than our current lowest point
                if (worldVertex.y < leftAnkleLowestY)
                {
                    leftAnkleLowestY = worldVertex.y;
                    _leftAnkleLowestPoint = worldVertex;
                }
            }

            // Check if vertex is under left foot
            float leftFootDist = Vector2.Distance(vertexPosXZ, leftFootPosXZ);
            if (leftFootDist < proximityRadius)
            {
                // Check if it's lower than our current lowest point
                if (worldVertex.y < leftFootLowestY)
                {
                    leftFootLowestY = worldVertex.y;
                    _leftFootLowestPoint = worldVertex;
                }
            }

            // Check if vertex is under right ankle
            float rightAnkleDist = Vector2.Distance(vertexPosXZ, rightAnklePosXZ);
            if (rightAnkleDist < proximityRadius)
            {
                // Check if it's lower than our current lowest point
                if (worldVertex.y < rightAnkleLowestY)
                {
                    rightAnkleLowestY = worldVertex.y;
                    _rightAnkleLowestPoint = worldVertex;
                }
            }

            // Check if vertex is under right foot
            float rightFootDist = Vector2.Distance(vertexPosXZ, rightFootPosXZ);
            if (rightFootDist < proximityRadius)
            {
                // Check if it's lower than our current lowest point
                if (worldVertex.y < rightFootLowestY)
                {
                    rightFootLowestY = worldVertex.y;
                    _rightFootLowestPoint = worldVertex;
                }
            }
        }

        // Calculate offsets from joints to lowest points (for future use)
        if (_leftAnkleLowestPoint != Vector3.zero)
        {
            _leftAnkleLowestOffset = leftAnkle.position.y - _leftAnkleLowestPoint.y;
        }

        if (_leftFootLowestPoint != Vector3.zero)
        {
            _leftFootLowestOffset = leftFoot.position.y - _leftFootLowestPoint.y;
        }

        if (_rightAnkleLowestPoint != Vector3.zero)
        {
            _rightAnkleLowestOffset = rightAnkle.position.y - _rightAnkleLowestPoint.y;
        }

        if (_rightFootLowestPoint != Vector3.zero)
        {
            _rightFootLowestOffset = rightFoot.position.y - _rightFootLowestPoint.y;
        }

        // Mark as initialized if we found at least one lowest point
        if (_leftAnkleLowestPoint != Vector3.zero || _rightAnkleLowestPoint != Vector3.zero ||
            _leftFootLowestPoint != Vector3.zero || _rightFootLowestPoint != Vector3.zero)
        {
            _lowestPointsInitialized = true;

            if (showDebugInfo)
            {
                Debug.Log($"[SMPLXFootPlacer] Lowest points initialized. " +
                          $"Left ankle offset: {_leftAnkleLowestOffset:F3}, Left foot offset: {_leftFootLowestOffset:F3}, " +
                          $"Right ankle offset: {_rightAnkleLowestOffset:F3}, Right foot offset: {_rightFootLowestOffset:F3}");
            }
        }
    }

    /// <summary>
    /// Last resort fallback - check foot penetration with simple raycasting
    /// </summary>
    private void CheckFootPenetrationByRaycast(Transform foot, bool isLeft)
    {
        if (foot == null)
            return;

        Transform ankle = foot.parent;
        if (ankle == null)
            return;

        // Check if ankle is below ground
        float ankleY = ankle.position.y;
        if (ankleY < groundHeight)
        {
            float penetration = groundHeight - ankleY;

            if (isLeft)
            {
                LeftAnkleState = FootState.Penetrating;
                LeftAnklePenetrationDepth = penetration;
                LeftGroundNormal = Vector3.up;
            }
            else
            {
                RightAnkleState = FootState.Penetrating;
                RightAnklePenetrationDepth = penetration;
                RightGroundNormal = Vector3.up;
            }
        }

        // Check if foot is below ground
        float footY = foot.position.y;
        if (footY < groundHeight)
        {
            float penetration = groundHeight - footY;

            if (isLeft)
            {
                LeftFootState = FootState.Penetrating;
                LeftFootPenetrationDepth = penetration;
                LeftGroundNormal = Vector3.up;
            }
            else
            {
                RightFootState = FootState.Penetrating;
                RightFootPenetrationDepth = penetration;
                RightGroundNormal = Vector3.up;
            }
        }
    }

    /// <summary>
    /// Reset the system and clear penetration states
    /// </summary>
    public void ResetSystem()
    {
        if (leftLegIK != null)
            leftLegIK.ResetLegRotations();

        if (rightLegIK != null)
            rightLegIK.ResetLegRotations();

        LeftAnkleState = FootState.NotPenetrating;
        RightAnkleState = FootState.NotPenetrating;
        LeftFootState = FootState.NotPenetrating;
        RightFootState = FootState.NotPenetrating;
        LeftAnklePenetrationDepth = 0f;
        RightAnklePenetrationDepth = 0f;
        LeftFootPenetrationDepth = 0f;
        RightFootPenetrationDepth = 0f;
        LeftGroundNormal = Vector3.up;
        RightGroundNormal = Vector3.up;

        // Reset smoothing state
        ikParameters.ResetSmoothing();
    }

    /// <summary>
    /// Event called when inspector values change
    /// </summary>
    private void OnValidate()
    {
        if (Application.isPlaying && isInitialized)
        {
            UpdateDebugSettings();
        }
    }

    /// <summary>
    /// Initialize the foot placer system after beta values have been applied
    /// </summary>
    public void InitializeAfterBetas()
    {
        if (!isInitialized)
            InitializeSystem();

        // Get skinned mesh renderer for mesh analysis
        _smr = GetComponentInChildren<SkinnedMeshRenderer>();
        if (_smr == null)
        {
            Debug.LogWarning("[SMPLXFootPlacer] No SkinnedMeshRenderer found after beta application.");
        }
        else
        {
            _bakedMesh = new Mesh();
        }

        // Clear any previous state
        _lowestPointsInitialized = false;
        _leftAnkleLowestPoint = Vector3.zero;
        _rightAnkleLowestPoint = Vector3.zero;
        _leftFootLowestPoint = Vector3.zero;
        _rightFootLowestPoint = Vector3.zero;
        _leftAnkleLowestOffset = 0f;
        _rightAnkleLowestOffset = 0f;
        _leftFootLowestOffset = 0f;
        _rightFootLowestOffset = 0f;

        // Reset penetration state
        LeftAnkleState = FootState.NotPenetrating;
        RightAnkleState = FootState.NotPenetrating;
        LeftFootState = FootState.NotPenetrating;
        RightFootState = FootState.NotPenetrating;
        LeftAnklePenetrationDepth = 0f;
        RightAnklePenetrationDepth = 0f;
        LeftFootPenetrationDepth = 0f;
        RightFootPenetrationDepth = 0f;
        LeftGroundNormal = Vector3.up;
        RightGroundNormal = Vector3.up;

        // Run initial detection pass to find lowest points
        UpdateFootPenetrationState();

        if (showDebugInfo)
        {
            Debug.Log($"[SMPLXFootPlacer] System initialized after beta application. Lowest points detected: {_lowestPointsInitialized}");
        }
    }

    /// <summary>
    /// Initial foot correction for use after beta shapes are applied
    /// </summary>
    public void InitialFootCorrection()
    {
        if (!isInitialized)
            return;

        // Find lowest points if not already done
        if (!_lowestPointsInitialized)
        {
            UpdateFootPenetrationState();
        }

        // Process penetrations with extra iterations for initial setup
        int savedMaxIterations = maxSolveIterations;
        maxSolveIterations = 3; // Use more iterations for initial setup

        ProcessFootPenetrations();

        // Restore original iteration count
        maxSolveIterations = savedMaxIterations;

        if (showDebugInfo)
        {
            Debug.Log($"[SMPLXFootPlacer] Initial foot correction completed. " +
                      $"Left ankle offset: {_leftAnkleLowestOffset:F3}, Left foot offset: {_leftFootLowestOffset:F3}, " +
                      $"Right ankle offset: {_rightAnkleLowestOffset:F3}, Right foot offset: {_rightFootLowestOffset:F3}");
        }
    }

#if UNITY_EDITOR
    /// <summary>
    /// Visualize the foot detection and penetration state
    /// </summary>
    private void OnDrawGizmos()
    {
        if (!showDebugInfo || !showEditorVisuals || !Application.isPlaying)
            return;

        // Draw the ground plane
        Gizmos.color = new Color(0.5f, 0.8f, 0.5f, 0.2f);
        Vector3 center = new Vector3(transform.position.x, groundHeight, transform.position.z);
        Gizmos.DrawCube(center, new Vector3(2f, 0.01f, 2f));

        // Draw lowest points
        if (_lowestPointsInitialized)
        {
            // Draw left ankle lowest point
            if (_leftAnkleLowestPoint != Vector3.zero && leftFoot != null && leftFoot.parent != null)
            {
                // Original lowest point
                Gizmos.color = Color.yellow;
                Gizmos.DrawSphere(_leftAnkleLowestPoint, 0.015f);

                // Current calculated lowest point
                Transform leftAnkle = leftFoot.parent;
                Vector3 currentLowest = leftAnkle.position;
                currentLowest.y -= _leftAnkleLowestOffset;
                Gizmos.color = Color.magenta;
                Gizmos.DrawSphere(currentLowest, 0.01f);
                Gizmos.DrawLine(leftAnkle.position, currentLowest);
            }

            // Draw left foot lowest point
            if (_leftFootLowestPoint != Vector3.zero && leftFoot != null)
            {
                // Original lowest point
                Gizmos.color = Color.cyan;
                Gizmos.DrawSphere(_leftFootLowestPoint, 0.015f);

                // Current calculated lowest point
                Vector3 currentLowest = leftFoot.position;
                currentLowest.y -= _leftFootLowestOffset;
                Gizmos.color = Color.blue;
                Gizmos.DrawSphere(currentLowest, 0.01f);
                Gizmos.DrawLine(leftFoot.position, currentLowest);
            }

            // Draw right ankle lowest point
            if (_rightAnkleLowestPoint != Vector3.zero && rightFoot != null && rightFoot.parent != null)
            {
                // Original lowest point
                Gizmos.color = Color.yellow;
                Gizmos.DrawSphere(_rightAnkleLowestPoint, 0.015f);

                // Current calculated lowest point
                Transform rightAnkle = rightFoot.parent;
                Vector3 currentLowest = rightAnkle.position;
                currentLowest.y -= _rightAnkleLowestOffset;
                Gizmos.color = Color.magenta;
                Gizmos.DrawSphere(currentLowest, 0.01f);
                Gizmos.DrawLine(rightAnkle.position, currentLowest);
            }

            // Draw right foot lowest point
            if (_rightFootLowestPoint != Vector3.zero && rightFoot != null)
            {
                // Original lowest point
                Gizmos.color = Color.cyan;
                Gizmos.DrawSphere(_rightFootLowestPoint, 0.015f);

                // Current calculated lowest point
                Vector3 currentLowest = rightFoot.position;
                currentLowest.y -= _rightFootLowestOffset;
                Gizmos.color = Color.blue;
                Gizmos.DrawSphere(currentLowest, 0.01f);
                Gizmos.DrawLine(rightFoot.position, currentLowest);
            }
        }

        // Draw foot states
        if (leftFoot != null)
        {
            Transform leftAnkle = leftFoot.parent;

            // Ankle state
            if (leftAnkle != null)
            {
                Gizmos.color = LeftAnkleState == FootState.Penetrating ? Color.red : Color.green;
                Gizmos.DrawWireSphere(leftAnkle.position, 0.02f);
            }

            // Foot state
            Gizmos.color = LeftFootState == FootState.Penetrating ? Color.red : Color.green;
            Gizmos.DrawWireSphere(leftFoot.position, 0.02f);

            // Draw detection radius
            Gizmos.color = new Color(1f, 1f, 0f, 0.2f);
            DrawCircle(leftFoot.position, proximityRadius, 16);
        }

        if (rightFoot != null)
        {
            Transform rightAnkle = rightFoot.parent;

            // Ankle state
            if (rightAnkle != null)
            {
                Gizmos.color = RightAnkleState == FootState.Penetrating ? Color.red : Color.green;
                Gizmos.DrawWireSphere(rightAnkle.position, 0.02f);
            }

            // Foot state
            Gizmos.color = RightFootState == FootState.Penetrating ? Color.red : Color.green;
            Gizmos.DrawWireSphere(rightFoot.position, 0.02f);

            // Draw detection radius
            Gizmos.color = new Color(1f, 1f, 0f, 0.2f);
            DrawCircle(rightFoot.position, proximityRadius, 16);
        }

        // Draw stats
        if (UnityEditor.SceneView.currentDrawingSceneView != null)
        {
            UnityEditor.Handles.BeginGUI();
            GUIStyle style = new GUIStyle();
            style.normal.textColor = Color.white;
            style.fontSize = 12;
            style.fontStyle = FontStyle.Bold;
            style.alignment = TextAnchor.UpperLeft;
            style.normal.background = Texture2D.blackTexture;

            string statsText =
                $"L Ankle: {LeftAnkleState} ({LeftAnklePenetrationDepth:F3})\n" +
                $"L Foot: {LeftFootState} ({LeftFootPenetrationDepth:F3})\n" +
                $"R Ankle: {RightAnkleState} ({RightAnklePenetrationDepth:F3})\n" +
                $"R Foot: {RightFootState} ({RightFootPenetrationDepth:F3})";

            if (showPerformanceStats)
            {
                statsText += $"\nDetection: {(_frameCounter % detectionInterval == 0 ? "Full" : "Quick")} (Frame {_frameCounter})" +
                             $"\nLast: {_lastProcessingTime * 1000:F2}ms, Avg: {_avgProcessingTime * 1000:F2}ms";
            }

            UnityEditor.Handles.Label(
                transform.position + Vector3.up * 0.5f,
                statsText,
                style
            );
            UnityEditor.Handles.EndGUI();
        }
    }

    private void DrawCircle(Vector3 center, float radius, int segments)
    {
        // Draw a circle in the XZ plane
        Vector3 prevPoint = center + new Vector3(radius, 0, 0);

        for (int i = 1; i <= segments; i++)
        {
            float angle = i * Mathf.PI * 2f / segments;
            Vector3 nextPoint = center + new Vector3(Mathf.Cos(angle) * radius, 0, Mathf.Sin(angle) * radius);
            Gizmos.DrawLine(prevPoint, nextPoint);
            prevPoint = nextPoint;
        }
    }
#endif
}