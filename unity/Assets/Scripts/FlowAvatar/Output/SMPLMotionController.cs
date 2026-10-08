using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Oculus.Interaction;

public class SMPLMotionController : MonoBehaviour
{
    [System.Serializable]
    public class SMPLJointConfig
    {
        [Tooltip("All rotational joints in the root-to-leaf order")]
        public Transform[] joints;
        public bool initializeFromCurrentPose = true;
    }

    [Header("Configuration")]

    [SerializeField] private SMPLJointConfig jointConfig = new SMPLJointConfig();

    [Header("Beta Initialization")]
    [Tooltip("Time to wait before initializing beta values")]
    [SerializeField] private float betaInitializationDelay = 5.0f;
    [Tooltip("Main toggle to control beta application throughout the system")]
    [SerializeField] private bool enableApplyBetas = true;
    [Tooltip("Whether to wait for initialization before showing poses")]
    [SerializeField] private bool waitForBetaBeforePosing = true;


    [Header("Performance Profiling")]
    [SerializeField] private bool enableProfiling = false;

    // Public property to expose enableApplyBetas
    public bool EnableApplyBetas => enableApplyBetas;

    // State to track beta application
    private bool isApplyingBetas = false;
    public bool IsApplyingBetas => isApplyingBetas;

    // State to track if poses should be applied
    private bool poseApplicationEnabled = false;
    private bool initialBetaApplied = false;
    private float initializationStartTime = 0f;
    public bool IsPoseApplicationEnabled => poseApplicationEnabled;
    public float InitializationProgress =>
        initializationStartTime > 0 ? Mathf.Clamp01((Time.time - initializationStartTime) / betaInitializationDelay) : 0f;

    private int headJointIndex = 15;

    // Performance monitoring
    private int receivedFramesCount = 0;
    private int appliedFramesCount = 0;
    private int queuedPosesCount = 0;
    private float lastFrameProcessingTime = 0f;
    private float averageProcessingTime = 0f;
    private System.Diagnostics.Stopwatch processingStopwatch = new System.Diagnostics.Stopwatch();
    private int totalProcessedFrames = 0;
    private float totalProcessingTime = 0f;

    // Core data
    private Transform rootTransform;      // Translation joint (first joint)
    private Quaternion[] cachedRotations; // Rest local rotations of the joints (from the avatar bind pose)
    private Vector3[] poseAxisAngles;     // Reused buffer for ApplyPoseData

    // Avatar hierarchy references (auto-assigned)
    private Transform avatarRoot;         // The top-level avatar transform (scale 1)
    private Transform smplxNeutral;       // SMPLX-neutral transform (scale 1)
    private Transform rootJoint;          // root transform (scale 100)
    private Transform pelvisJoint;        // Usually jointConfig.joints[0] (scale 1)

    // Position tracking
    private Vector3 initialAvatarPosition; // Initial avatar root position
    private Vector3 initialPelvisPosition; // Initial pelvis position in world space
    private Vector3 avatarToPelvisVector; // Vector from avatar root to pelvis
    private float rootScale = 100f;       // Scale of the root joint

    // Queue for pending updates
    private Queue<float[,]> pendingPoses = new Queue<float[,]>();
    private Queue<Vector3> pendingPositions = new Queue<Vector3>();
    private Queue<float[]> pendingBetas = new Queue<float[]>(); // Queue for beta values
    private bool processingPose = false;  // Flag to prevent re-entrancy
    private bool newPoseApplied = false;  // Flag to track when new pose was applied
    private bool newBetasApplied = false; // Flag to track when new betas were applied

    private bool isInitialized = false;
    private bool hasPoseData = false;  // Tracks whether we have received pose data
    private bool hasBetaData = false;  // Tracks whether we have received beta data
    private bool bodyShapeChanged = false; // Flag to indicate if body shape has changed

    // References to other components
    private EnableHandTracking _handTracking;
    private SMPLXFootPlacer _footArticulationCorrector;
    private FlowAvatarSMPLxOVR smplxModel;

    // Beta application callbacks
    public delegate void BetaApplicationStartedHandler();
    public delegate void BetaApplicationCompletedHandler();
    public delegate void InitializationProgressHandler(float progress);
    public event BetaApplicationStartedHandler OnBetaApplicationStarted;
    public event BetaApplicationCompletedHandler OnBetaApplicationCompleted;
    public event InitializationProgressHandler OnInitializationProgress;

    // Public properties
    public SMPLJointConfig JointConfig => jointConfig;
    public Quaternion[] CachedRotations => cachedRotations;
    public bool IsInitialized => isInitialized;
    public bool HasPoseData => hasPoseData;
    public bool HasBetaData => hasBetaData;

    void Awake()
    {
        _handTracking = GetComponent<EnableHandTracking>();
        _footArticulationCorrector = GetComponent<SMPLXFootPlacer>();
        smplxModel = GetComponent<FlowAvatarSMPLxOVR>();

        // Auto-assign transforms
        AutoAssignTransforms();
    }

    private void AutoAssignTransforms()
    {
        // Avatar root is this transform
        avatarRoot = transform;

        // Auto-detect hierarchy
        smplxNeutral = avatarRoot.Find("SMPLX-neutral");
        if (smplxNeutral != null)
        {
            rootJoint = smplxNeutral.Find("root");
            if (rootJoint != null)
            {
                pelvisJoint = rootJoint.Find("pelvis");

                // Get the root scale
                rootScale = rootJoint.localScale.x;
            }
        }

        // Fallback: if pelvis not found and joints exist, use first joint as pelvis
        if (pelvisJoint == null && jointConfig.joints != null && jointConfig.joints.Length > 0)
        {
            pelvisJoint = jointConfig.joints[0];
        }

        if (enableProfiling)
        {
            if (smplxNeutral == null)
                Debug.LogWarning($"[{nameof(SMPLMotionController)}] Could not find SMPLX-neutral transform");
            if (rootJoint == null)
                Debug.LogWarning($"[{nameof(SMPLMotionController)}] Could not find root transform");
            if (pelvisJoint == null)
                Debug.LogWarning($"[{nameof(SMPLMotionController)}] Could not find pelvis transform");
        }
    }

    void Start()
    {
        InitializeJointConfiguration();
        RecordInitialPositions();

        if (enableProfiling)
        {
            Debug.Log($"[SMPLMotionController] Initialized with root scale: {rootScale}");
        }

        // Initialize beta and pose application
        if (enableApplyBetas && waitForBetaBeforePosing)
        {
            // Disable pose application until beta initialization is complete
            poseApplicationEnabled = false;

            // Start the initialization sequence
            StartCoroutine(InitializeBetaSequence());
        }
        else
        {
            // No beta initialization needed, enable pose application immediately
            poseApplicationEnabled = true;
        }
    }

    private IEnumerator InitializeBetaSequence()
    {
        // Wait for the specified delay to collect beta data
        initializationStartTime = Time.time;

        // Wait for delay period
        while (Time.time - initializationStartTime < betaInitializationDelay)
        {
            // Update progress event
            float progress = (Time.time - initializationStartTime) / betaInitializationDelay;
            if (OnInitializationProgress != null)
            {
                OnInitializationProgress.Invoke(progress);
            }

            yield return null;
        }

        // Get the latest beta values from the queue
        float[] latestBetas = null;
        lock (pendingBetas)
        {
            if (pendingBetas.Count > 0)
            {
                // Find the most recent beta values
                latestBetas = pendingBetas.Dequeue();
                while (pendingBetas.Count > 0)
                {
                    latestBetas = pendingBetas.Dequeue();
                }
            }
        }

        // Apply beta shapes if we have values
        if (latestBetas != null && smplxModel != null)
        {


            // Notify start of beta application
            isApplyingBetas = true;
            if (OnBetaApplicationStarted != null)
                OnBetaApplicationStarted.Invoke();

            // Pause hand tracking during beta application
            if (_handTracking != null)
            {
                _handTracking.PauseHandTracking();
            }

            yield return null;

            // Apply beta values and shapes
            for (int i = 0; i < 10 && i < latestBetas.Length; i++)
            {
                smplxModel.betas[i] = latestBetas[i];
            }

            // Apply shape changes - this is the heavy operation
            float startTime = Time.realtimeSinceStartup;
            smplxModel.SetBetaShapes();
            float elapsedTime = Time.realtimeSinceStartup - startTime;


            yield return null;

            // Additional setup
            RecordInitialPositions();

            yield return null;

            // Resume hand tracking
            if (_handTracking != null)
            {
                _handTracking.ResumeHandTracking();
            }

            // Mark as complete
            initialBetaApplied = true;
            hasBetaData = true;
            newBetasApplied = true;
            bodyShapeChanged = true;
            isApplyingBetas = false;

            // Notify completion
            if (OnBetaApplicationCompleted != null)
            {
                OnBetaApplicationCompleted.Invoke();
            }



        }


        // Enable pose application after beta is applied
        poseApplicationEnabled = true;

        // Final progress update (100%)
        if (OnInitializationProgress != null)
        {
            OnInitializationProgress.Invoke(1.0f);
        }
    }

    void Update()
    {
        // Skip processing if pose application is not enabled yet
        if (!poseApplicationEnabled)
            return;

        // Avoid re-entrant processing
        if (processingPose) return;

        processingPose = true;

        try
        {
            // Start timing the processing
            processingStopwatch.Reset();
            processingStopwatch.Start();

            // Process any pending poses
            ProcessPendingData();

            // Run post-processing if we applied a new pose
            if (newPoseApplied)
            {
                // Run post-processing
                RunPostProcessing();

                // Reset flag after processing
                newPoseApplied = false;

                // Count the applied frame
                appliedFramesCount++;
            }

            // Update performance metrics
            processingStopwatch.Stop();
            lastFrameProcessingTime = processingStopwatch.ElapsedMilliseconds / 1000f;

            if (appliedFramesCount > 0)
            {
                totalProcessedFrames++;
                totalProcessingTime += lastFrameProcessingTime;
                averageProcessingTime = totalProcessingTime / totalProcessedFrames;
            }

            // Update UI counter for queued poses
            queuedPosesCount = pendingPoses.Count;
        }
        finally
        {
            processingPose = false;
        }
    }

    private void ProcessPendingData()
    {
        // Process betas first if we're not waiting for initialization
        if (enableApplyBetas && !waitForBetaBeforePosing)
            ProcessPendingBetas();

        ProcessPendingPoseAndPosition();
    }

    private void ProcessPendingBetas()
    {
        // Don't process new betas if we're already applying some
        if (isApplyingBetas || initialBetaApplied)
            return;

        // Apply latest beta values from the queue
        float[] latestBetas = null;
        bool hasPendingBetas = false;

        lock (pendingBetas)
        {
            if (pendingBetas.Count > 0)
            {
                // If we're falling behind, discard older beta values to catch up
                while (pendingBetas.Count > 1)
                {
                    if (enableProfiling)
                    {
                        Debug.Log($"[SMPLMotionController] Discarding older beta values, queue size: {pendingBetas.Count}");
                    }
                    pendingBetas.Dequeue();
                }

                latestBetas = pendingBetas.Dequeue();
                hasPendingBetas = true;
            }
        }

        // Apply beta values if we have them
        if (hasPendingBetas && latestBetas != null && smplxModel != null)
        {
            // Apply beta shapes directly
            StartCoroutine(ApplyBetaValuesCoroutine(latestBetas));
        }
    }

    private void ProcessPendingPoseAndPosition()
    {
        // Apply latest pose and position from the queues
        float[,] latestPose = null;
        Vector3 latestPelvisPosition = Vector3.zero;
        bool hasPendingPose = false;
        bool hasPendingPosition = false;

        lock (pendingPoses)
        {
            if (pendingPoses.Count > 0)
            {
                // If we're falling behind, discard older poses to catch up
                while (pendingPoses.Count > 1)
                {
                    if (enableProfiling)
                    {
                        Debug.Log($"[SMPLMotionController] Discarding older pose, queue size: {pendingPoses.Count}");
                    }
                    pendingPoses.Dequeue();
                }

                latestPose = pendingPoses.Dequeue();
                hasPendingPose = true;
            }
        }

        lock (pendingPositions)
        {
            if (pendingPositions.Count > 0)
            {
                // If we're falling behind, discard older positions to catch up
                while (pendingPositions.Count > 1)
                {
                    pendingPositions.Dequeue();
                }

                latestPelvisPosition = pendingPositions.Dequeue();
                hasPendingPosition = true;
            }
        }

        // Apply pose if we have one
        if (hasPendingPose)
        {
            ApplyPoseData(latestPose);
            hasPoseData = true;

            // Set the flag to indicate a new pose was applied
            newPoseApplied = true;
        }

        // Apply position if we have one
        if (hasPendingPosition)
        {
            UpdateAvatarPositionFromPelvis(latestPelvisPosition);
        }
    }

    private void InitializeJointConfiguration()
    {
        if (jointConfig.joints == null || jointConfig.joints.Length == 0)
        {
            Debug.LogError("No joints assigned to SMPLMotionController!");
            return;
        }

        // Get root transform (first joint)
        rootTransform = jointConfig.joints[0];
        if (rootTransform == null)
        {
            Debug.LogError("Root joint cannot be null!");
            return;
        }

        // Initialize arrays
        int jointCount = jointConfig.joints.Length;
        cachedRotations = new Quaternion[jointCount];
        poseAxisAngles = new Vector3[jointCount];

        // Store rest rotations. They come from the avatar's bind pose, so they are
        // correct whatever pose the avatar was saved in (left_wrist rests at 180 deg
        // about X in the OVR-hand rig).
        for (int i = 0; i < jointCount; i++)
        {
            cachedRotations[i] = smplxModel != null
                ? smplxModel.GetRestLocalRotation(jointConfig.joints[i])
                : jointConfig.joints[i].localRotation;
        }

        isInitialized = true;
        hasPoseData = false;
    }

    public void RecordInitialPositions()
    {
        if (!isInitialized || pelvisJoint == null || avatarRoot == null) return;

        // Store initial positions
        initialAvatarPosition = avatarRoot.position;
        initialPelvisPosition = pelvisJoint.position;

        // Calculate the vector from avatar root to pelvis in world space
        avatarToPelvisVector = initialPelvisPosition - initialAvatarPosition;

        if (enableProfiling)
        {
            Debug.Log($"[SMPLMotionController] Recorded initial offsets - Vector: {avatarToPelvisVector}");
        }
    }

    private void ApplyPoseData(float[,] pose)
    {
        for (int i = 0; i < poseAxisAngles.Length; i++)
            poseAxisAngles[i] = new Vector3(pose[i, 0], pose[i, 1], pose[i, 2]);

        ApplyJointRotations(jointConfig.joints, cachedRotations, poseAxisAngles);

        // Set the flag to indicate a new pose was applied
        newPoseApplied = true;
    }

    /// <summary>
    /// Applies SMPL-X joint rotations (axis-angle, already mirrored to Unity by
    /// UnityTransformUtils.ConvertSMPLToUnity) relative to each joint's rest
    /// orientation: localRotation = q * rest.
    /// </summary>
    public static void ApplyJointRotations(Transform[] joints, Quaternion[] restRotations, Vector3[] axisAngles)
    {
        for (int i = 0; i < joints.Length; i++)
            joints[i].localRotation = UnityTransformUtils.AxisAngleToQuaternion(axisAngles[i]) * restRotations[i];
    }

    private void UpdateAvatarPositionFromPelvis(Vector3 targetPelvisPosition)
    {
        if (avatarRoot == null || pelvisJoint == null) return;

        // Calculate what the avatar position should be based on the target pelvis position
        // We need to maintain the same relative offset from avatar to pelvis
        Vector3 newAvatarPosition = targetPelvisPosition - avatarToPelvisVector;

        // Apply the new position to the avatar root
        avatarRoot.position = newAvatarPosition;
    }

    /// <summary>
    /// Coroutine for applying beta values
    /// </summary>
    private IEnumerator ApplyBetaValuesCoroutine(float[] betas)
    {
        if (!enableApplyBetas || smplxModel == null || betas == null || betas.Length < 10)
        {
            yield break;
        }

        if (enableProfiling)
        {
            Debug.Log($"[SMPLMotionController] Starting beta application process...");
        }

        // Set flag to indicate we're applying betas and notify listeners
        isApplyingBetas = true;
        if (OnBetaApplicationStarted != null)
        {
            OnBetaApplicationStarted.Invoke();
        }

        // STEP 1: Pause hand tracking to prevent conflicts during beta application
        if (_handTracking != null)
        {
            _handTracking.PauseHandTracking();
        }

        yield return null;

        // STEP 2: Copy beta values and apply to the model
        for (int i = 0; i < 10; i++)
        {
            smplxModel.betas[i] = betas[i];
        }

        yield return null;

        // STEP 3: Apply beta values to the model
        float startTime = Time.realtimeSinceStartup;
        smplxModel.SetBetaShapes();
        float elapsedTime = Time.realtimeSinceStartup - startTime;

        if (enableProfiling)
        {
            Debug.Log($"[SMPLMotionController] Beta shapes applied in {elapsedTime * 1000:F2}ms");
        }

        yield return null;

        // STEP 4: Update state and recalculate offsets
        newBetasApplied = true;
        bodyShapeChanged = true;
        hasBetaData = true;
        initialBetaApplied = true;
        RecordInitialPositions();

        yield return null;

        // STEP 5: Resume hand tracking
        if (_handTracking != null)
        {
            _handTracking.ResumeHandTracking();
        }

        // Reset flags and notify listeners
        isApplyingBetas = false;
        if (OnBetaApplicationCompleted != null)
        {
            OnBetaApplicationCompleted.Invoke();
        }

        if (enableProfiling)
        {
            Debug.Log($"[SMPLMotionController] Beta application completed");
        }
    }

    /// <summary>
    /// Sets pose and beta values from flat float arrays
    /// </summary>
    public void SetPoseFromFloats(float[] poseData, float[] transData, float[] betaData = null)
    {
        if (!isInitialized)
        {
            Debug.LogWarning("SMPLMotionController not initialized!");
            return;
        }

        if (poseData.Length != jointConfig.joints.Length * 3)
        {
            Debug.LogError($"Invalid pose data length. Expected {jointConfig.joints.Length * 3}, got {poseData.Length}");
            return;
        }

        receivedFramesCount++;

        if (enableProfiling && receivedFramesCount % 30 == 0)
        {
            Debug.Log($"[SMPLMotionController] Received frame {receivedFramesCount}, queue size: {pendingPoses.Count}");
        }

        // Always queue beta values even if we're not applying poses yet
        if (enableApplyBetas && betaData != null && betaData.Length > 0 && smplxModel != null)
        {
            // Queue the beta values for processing in Update
            lock (pendingBetas)
            {
                float[] betaCopy = new float[10];
                int copyLength = Mathf.Min(betaData.Length, 10);
                System.Array.Copy(betaData, betaCopy, copyLength);
                pendingBetas.Enqueue(betaCopy);
            }
        }

        // Skip pose processing if pose application is not enabled yet
        if (!poseApplicationEnabled)
            return;

        // Convert flat array to 2D array for pose data
        float[,] poses = new float[jointConfig.joints.Length, 3];
        for (int i = 0; i < jointConfig.joints.Length; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                poses[i, j] = poseData[i * 3 + j];
            }
        }

        // Queue the pose for processing in Update
        lock (pendingPoses)
        {
            pendingPoses.Enqueue(poses);
        }

        // Set root position based on head position
        SetRootPositionFromHeadPos(transData);
    }

    /// <summary>
    /// Calculates the pelvis position based on head position data from ML
    /// </summary>
    public void SetRootPositionFromHeadPos(float[] transData)
    {
        if (!isInitialized) return;
        if (transData.Length != 3)
        {
            Debug.LogError("Invalid position data length. Expected 3 values.");
            return;
        }

        // Skip pose processing if pose application is not enabled yet
        if (!poseApplicationEnabled)
            return;

        // This is the target head position from ML model
        Vector3 targetHeadPosition = new Vector3(
            transData[0],
            transData[1],
            transData[2]
        );

        // Get the current world position of the head joint
        Vector3 currentHeadPosition = jointConfig.joints[headJointIndex].position;

        // Get the current world position of the pelvis
        Vector3 currentPelvisPosition = pelvisJoint.position;

        // Calculate the offset between target head position and current head position
        Vector3 headOffset = targetHeadPosition - currentHeadPosition;

        // Calculate the new pelvis position by applying the same offset
        Vector3 newPelvisPosition = currentPelvisPosition + headOffset;

        // Queue the position for processing in Update
        lock (pendingPositions)
        {
            pendingPositions.Enqueue(newPelvisPosition);
        }
    }

    /// <summary>
    /// Runs post-processing steps after a pose is applied
    /// </summary>
    public void RunPostProcessing()
    {

        if (_footArticulationCorrector != null)
        {
            _footArticulationCorrector.ProcessFootPenetrations();
        }

        // Step 2: Apply hand tracking with hand position correction method
        if (_handTracking != null)
        {
            _handTracking.UpdateHandTracking();
        }

        if (enableProfiling)
        {
            Debug.Log($"[SMPLMotionController] Completed frame {appliedFramesCount} processing in {lastFrameProcessingTime * 1000:F2}ms");
        }
    }
}