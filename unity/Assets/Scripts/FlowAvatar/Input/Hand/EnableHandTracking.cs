using Oculus.Interaction.Input;
using System;
using System.Collections.Generic;
using UnityEngine;

namespace Oculus.Interaction
{
    [Serializable]
    public class HandJointConfig
    {
        public Transform Wrist;
        public Transform ForearmStub;
        public Transform Thumb0;
        public Transform Thumb1;
        public Transform Thumb2;
        public Transform Thumb3;
        public Transform Index1;
        public Transform Index2;
        public Transform Index3;
        public Transform Middle1;
        public Transform Middle2;
        public Transform Middle3;
        public Transform Ring1;
        public Transform Ring2;
        public Transform Ring3;
        public Transform Pinky0;
        public Transform Pinky1;
        public Transform Pinky2;
        public Transform Pinky3;
    }

    public class EnableHandTracking : MonoBehaviour
    {
        private Hand RealLeftHand;
        private Hand RealRightHand;
        [SerializeField] private bool ForStudy = false;
        [SerializeField] private bool manualTracking = false;

        [Header("Tracking Options")]
        [SerializeField] private bool useHandTracking = true;
        [SerializeField] private bool applyWristRotations = true;
        [SerializeField] private bool applyFingerRotations = true;
        [SerializeField] private bool enableHandCorrection = true;
        [Range(0f, 1f)]
        [SerializeField] private float handCorrectionStrength = 0.7f;

        private VRToSMPLCalibrator _calibrator;
        private SMPLMotionController _motionController;

        [Header("Hand Configurations")]
        [SerializeField] private HandJointConfig leftHandJoints = new HandJointConfig();
        [SerializeField] private HandJointConfig rightHandJoints = new HandJointConfig();

        [Header("Arm Joint Indices")]
        [SerializeField] private int leftElbowIndex = 18;
        [SerializeField] private int leftWristIndex = 20;
        [SerializeField] private int rightElbowIndex = 19;
        [SerializeField] private int rightWristIndex = 21;

        private Transform[] _leftJointList = new Transform[Constants.NUM_HAND_JOINTS];
        private Transform[] _rightJointList = new Transform[Constants.NUM_HAND_JOINTS];
        private bool _started = false;

        // Tracking state
        private Quaternion[] _leftHandRotations;
        private Quaternion[] _rightHandRotations;

        // Default rotations storage
        private Quaternion[] _defaultLeftHandRotations;
        private Quaternion[] _defaultRightHandRotations;

        private bool _hasLeftHandData = false;
        private bool _hasRightHandData = false;

        // Beta application status tracking
        private bool _betaApplicationInProgress = false;

        // Hand visibility timeouts - to prevent flickering
        private float _leftHandVisibilityTimeout = 0f;
        private float _rightHandVisibilityTimeout = 0f;
        private const float VISIBILITY_TIMEOUT = 0.3f; // 300ms timeout for transitions

        // References to arm joints for correction
        private Transform _leftElbow;
        private Transform _rightElbow;
        private Transform _leftWrist;
        private Transform _rightWrist;

        private void Awake()
        {
            if (!ForStudy)
            {
                Transform cameraRig = GameObject.Find("[BuildingBlock] Camera Rig").transform;
                RealLeftHand = cameraRig.Find("[BuildingBlock] Interaction/[BuildingBlock] Hand Interactions/[BuildingBlock] Synthetic Left Hand").GetComponent<SyntheticHand>();
                RealRightHand = cameraRig.Find("[BuildingBlock] Interaction/[BuildingBlock] Hand Interactions/[BuildingBlock] Synthetic Right Hand").GetComponent<SyntheticHand>();

                _calibrator = GetComponent<VRToSMPLCalibrator>();
                _motionController = GetComponent<SMPLMotionController>();
            }

            AutoAssignComponents();
            InitializeJointLists();
            InitializeRotationArrays();
            StoreDefaultRotations(); // Store the default rotations on startup
        }

        private void InitializeRotationArrays()
        {
            _leftHandRotations = new Quaternion[Constants.NUM_HAND_JOINTS];
            _rightHandRotations = new Quaternion[Constants.NUM_HAND_JOINTS];
            _defaultLeftHandRotations = new Quaternion[Constants.NUM_HAND_JOINTS];
            _defaultRightHandRotations = new Quaternion[Constants.NUM_HAND_JOINTS];

            for (int i = 0; i < Constants.NUM_HAND_JOINTS; i++)
            {
                _leftHandRotations[i] = Quaternion.identity;
                _rightHandRotations[i] = Quaternion.identity;
                _defaultLeftHandRotations[i] = Quaternion.identity;
                _defaultRightHandRotations[i] = Quaternion.identity;
            }
        }

        private void StoreDefaultRotations()
        {
            // Store default rotations for left hand
            if (leftHandJoints.Wrist != null)
            {
                _defaultLeftHandRotations[0] = leftHandJoints.Wrist.localRotation;
            }

            // Store default rotations for all left finger joints
            for (var i = 1; i < Constants.NUM_HAND_JOINTS; ++i)
            {
                if (_leftJointList[i] != null)
                {
                    _defaultLeftHandRotations[i] = _leftJointList[i].localRotation;
                }
            }

            // Store default rotations for right hand
            if (rightHandJoints.Wrist != null)
            {
                _defaultRightHandRotations[0] = rightHandJoints.Wrist.localRotation;
            }

            // Store default rotations for all right finger joints
            for (var i = 1; i < Constants.NUM_HAND_JOINTS; ++i)
            {
                if (_rightJointList[i] != null)
                {
                    _defaultRightHandRotations[i] = _rightJointList[i].localRotation;
                }
            }
        }

        private void AutoAssignComponents()
        {
            // Find avatar wrists and assign joints
            Transform avatarRoot = transform; // smplx-neutral_bind

            // Find left wrist and its joints
            var leftWrist = avatarRoot.Find("SMPLX-neutral/root/pelvis/spine1/spine2/spine3/left_collar/left_shoulder/left_elbow/left_wrist");
            if (leftWrist != null)
            {
                AssignLeftHandJoints(leftWrist);
            }

            // Find right wrist and its joints
            var rightWrist = avatarRoot.Find("SMPLX-neutral/root/pelvis/spine1/spine2/spine3/right_collar/right_shoulder/right_elbow/right_wrist");
            if (rightWrist != null)
            {
                AssignRightHandJoints(rightWrist);
            }

            // Get arm joint references for hand correction
            if (_motionController != null && _motionController.JointConfig.joints != null)
            {
                var joints = _motionController.JointConfig.joints;

                if (leftElbowIndex >= 0 && leftElbowIndex < joints.Length)
                    _leftElbow = joints[leftElbowIndex];

                if (rightElbowIndex >= 0 && rightElbowIndex < joints.Length)
                    _rightElbow = joints[rightElbowIndex];

                if (leftWristIndex >= 0 && leftWristIndex < joints.Length)
                    _leftWrist = joints[leftWristIndex];

                if (rightWristIndex >= 0 && rightWristIndex < joints.Length)
                    _rightWrist = joints[rightWristIndex];
            }
        }

        private void AssignLeftHandJoints(Transform leftWrist)
        {
            leftHandJoints.Wrist = leftWrist;
            leftHandJoints.ForearmStub = leftWrist.Find("b_l_forearm_stub");
            leftHandJoints.Thumb0 = leftWrist.Find("b_l_thumb0");
            leftHandJoints.Thumb1 = leftHandJoints.Thumb0.Find("b_l_thumb1");
            leftHandJoints.Thumb2 = leftHandJoints.Thumb1.Find("b_l_thumb2");
            leftHandJoints.Thumb3 = leftHandJoints.Thumb2.Find("b_l_thumb3");
            leftHandJoints.Index1 = leftWrist.Find("b_l_index1");
            leftHandJoints.Index2 = leftHandJoints.Index1.Find("b_l_index2");
            leftHandJoints.Index3 = leftHandJoints.Index2.Find("b_l_index3");
            leftHandJoints.Middle1 = leftWrist.Find("b_l_middle1");
            leftHandJoints.Middle2 = leftHandJoints.Middle1.Find("b_l_middle2");
            leftHandJoints.Middle3 = leftHandJoints.Middle2.Find("b_l_middle3");
            leftHandJoints.Ring1 = leftWrist.Find("b_l_ring1");
            leftHandJoints.Ring2 = leftHandJoints.Ring1.Find("b_l_ring2");
            leftHandJoints.Ring3 = leftHandJoints.Ring2.Find("b_l_ring3");
            leftHandJoints.Pinky0 = leftWrist.Find("b_l_pinky0");
            leftHandJoints.Pinky1 = leftHandJoints.Pinky0.Find("b_l_pinky1");
            leftHandJoints.Pinky2 = leftHandJoints.Pinky1.Find("b_l_pinky2");
            leftHandJoints.Pinky3 = leftHandJoints.Pinky2.Find("b_l_pinky3");
        }

        private void AssignRightHandJoints(Transform rightWrist)
        {
            rightHandJoints.Wrist = rightWrist;
            rightHandJoints.ForearmStub = rightWrist.Find("b_r_forearm_stub");

            // Thumb chain
            rightHandJoints.Thumb0 = rightWrist.Find("b_r_thumb0");
            rightHandJoints.Thumb1 = rightHandJoints.Thumb0.Find("b_r_thumb1");
            rightHandJoints.Thumb2 = rightHandJoints.Thumb1.Find("b_r_thumb2");
            rightHandJoints.Thumb3 = rightHandJoints.Thumb2.Find("b_r_thumb3");

            // Index chain
            rightHandJoints.Index1 = rightWrist.Find("b_r_index1");
            rightHandJoints.Index2 = rightHandJoints.Index1.Find("b_r_index2");
            rightHandJoints.Index3 = rightHandJoints.Index2.Find("b_r_index3");

            // Middle chain
            rightHandJoints.Middle1 = rightWrist.Find("b_r_middle1");
            rightHandJoints.Middle2 = rightHandJoints.Middle1.Find("b_r_middle2");
            rightHandJoints.Middle3 = rightHandJoints.Middle2.Find("b_r_middle3");

            // Ring chain
            rightHandJoints.Ring1 = rightWrist.Find("b_r_ring1");
            rightHandJoints.Ring2 = rightHandJoints.Ring1.Find("b_r_ring2");
            rightHandJoints.Ring3 = rightHandJoints.Ring2.Find("b_r_ring3");

            // Pinky chain
            rightHandJoints.Pinky0 = rightWrist.Find("b_r_pinky0");
            rightHandJoints.Pinky1 = rightHandJoints.Pinky0.Find("b_r_pinky1");
            rightHandJoints.Pinky2 = rightHandJoints.Pinky1.Find("b_r_pinky2");
            rightHandJoints.Pinky3 = rightHandJoints.Pinky2.Find("b_r_pinky3");
        }

        private void InitializeJointLists()
        {
            // Initialize left hand joint list
            _leftJointList[0] = leftHandJoints.Wrist;
            _leftJointList[1] = leftHandJoints.ForearmStub;
            _leftJointList[2] = leftHandJoints.Thumb0;
            _leftJointList[3] = leftHandJoints.Thumb1;
            _leftJointList[4] = leftHandJoints.Thumb2;
            _leftJointList[5] = leftHandJoints.Thumb3;
            _leftJointList[6] = leftHandJoints.Index1;
            _leftJointList[7] = leftHandJoints.Index2;
            _leftJointList[8] = leftHandJoints.Index3;
            _leftJointList[9] = leftHandJoints.Middle1;
            _leftJointList[10] = leftHandJoints.Middle2;
            _leftJointList[11] = leftHandJoints.Middle3;
            _leftJointList[12] = leftHandJoints.Ring1;
            _leftJointList[13] = leftHandJoints.Ring2;
            _leftJointList[14] = leftHandJoints.Ring3;
            _leftJointList[15] = leftHandJoints.Pinky0;
            _leftJointList[16] = leftHandJoints.Pinky1;
            _leftJointList[17] = leftHandJoints.Pinky2;
            _leftJointList[18] = leftHandJoints.Pinky3;

            // Initialize right hand joint list
            _rightJointList[0] = rightHandJoints.Wrist;
            _rightJointList[1] = rightHandJoints.ForearmStub;
            _rightJointList[2] = rightHandJoints.Thumb0;
            _rightJointList[3] = rightHandJoints.Thumb1;
            _rightJointList[4] = rightHandJoints.Thumb2;
            _rightJointList[5] = rightHandJoints.Thumb3;
            _rightJointList[6] = rightHandJoints.Index1;
            _rightJointList[7] = rightHandJoints.Index2;
            _rightJointList[8] = rightHandJoints.Index3;
            _rightJointList[9] = rightHandJoints.Middle1;
            _rightJointList[10] = rightHandJoints.Middle2;
            _rightJointList[11] = rightHandJoints.Middle3;
            _rightJointList[12] = rightHandJoints.Ring1;
            _rightJointList[13] = rightHandJoints.Ring2;
            _rightJointList[14] = rightHandJoints.Ring3;
            _rightJointList[15] = rightHandJoints.Pinky0;
            _rightJointList[16] = rightHandJoints.Pinky1;
            _rightJointList[17] = rightHandJoints.Pinky2;
            _rightJointList[18] = rightHandJoints.Pinky3;
        }

        private void Start()
        {
            if (!ForStudy)
            {
                this.BeginStart(ref _started);
                ValidateSetup();
                this.EndStart(ref _started);
            }
        }

        private void Update()
        {
            // Handle visibility timeouts
            if (_leftHandVisibilityTimeout > 0)
            {
                _leftHandVisibilityTimeout -= Time.deltaTime;
            }

            if (_rightHandVisibilityTimeout > 0)
            {
                _rightHandVisibilityTimeout -= Time.deltaTime;
            }
        }

        // Public method for SMPLMotionController to call before beta application
        public void PauseHandTracking()
        {
            _betaApplicationInProgress = true;

            // Reset hands and fingers to default T-pose
            ResetToDefaultPose();
        }

        // Public method for SMPLMotionController to call after beta application
        public void ResumeHandTracking()
        {
            _betaApplicationInProgress = false;

            // Re-apply the latest hand poses
            UpdateHandTracking();
        }

        // Reset to default T-pose for beta application
        private void ResetToDefaultPose()
        {
            // Reset left hand
            if (leftHandJoints.Wrist != null)
            {
                leftHandJoints.Wrist.localRotation = _defaultLeftHandRotations[0];
            }

            // Reset left finger joints
            for (var i = 1; i < Constants.NUM_HAND_JOINTS; ++i)
            {
                if (_leftJointList[i] != null)
                {
                    _leftJointList[i].localRotation = _defaultLeftHandRotations[i];
                }
            }

            // Reset right hand
            if (rightHandJoints.Wrist != null)
            {
                rightHandJoints.Wrist.localRotation = _defaultRightHandRotations[0];
            }

            // Reset right finger joints
            for (var i = 1; i < Constants.NUM_HAND_JOINTS; ++i)
            {
                if (_rightJointList[i] != null)
                {
                    _rightJointList[i].localRotation = _defaultRightHandRotations[i];
                }
            }
        }

        // Public method to be called from SMPLMotionController
        public void UpdateHandTracking()
        {
            if (!_started || _betaApplicationInProgress || !useHandTracking) return;

            // Check hand visibility using calibration data instead of directly checking hand tracking
            bool leftHandVisible = false;
            bool rightHandVisible = false;

            // Get the calibration data which contains hand visibility information
            if (_calibrator != null && _calibrator.IsCalibrated())
            {
                VRToSMPLCalibrator.CalibrationData calibData = _calibrator.GetCalibrationData();
                leftHandVisible = calibData.leftHandVisible;
                rightHandVisible = calibData.rightHandVisible;
            }

            // When hand is lost, set timeout
            if (_hasLeftHandData && !leftHandVisible)
            {
                _leftHandVisibilityTimeout = VISIBILITY_TIMEOUT;
            }

            if (_hasRightHandData && !rightHandVisible)
            {
                _rightHandVisibilityTimeout = VISIBILITY_TIMEOUT;
            }

            // Use the buffered visibility
            bool effectiveLeftHandVisible = leftHandVisible || _leftHandVisibilityTimeout > 0;
            bool effectiveRightHandVisible = rightHandVisible || _rightHandVisibilityTimeout > 0;

            // Update tracking data from hands if visible
            if (leftHandVisible)
            {
                UpdateLeftHandTarget();
            }

            if (rightHandVisible)
            {
                UpdateRightHandTarget();
            }

            // Apply pose data with visibility consideration
            if (_hasLeftHandData && effectiveLeftHandVisible)
            {
                ApplyLeftHandPose();

                // Apply hand correction for left arm if enabled
                if (enableHandCorrection)
                {
                    CorrectLeftArmChain();
                }
            }
            else
            {
                // If hand not visible and we have predicted data, use it for wrist
                ApplyPredictedLeftWristData();
            }

            if (_hasRightHandData && effectiveRightHandVisible)
            {
                ApplyRightHandPose();

                // Apply hand correction for right arm if enabled
                if (enableHandCorrection)
                {
                    CorrectRightArmChain();
                }
            }
            else
            {
                // If hand not visible and we have predicted data, use it for wrist
                ApplyPredictedRightWristData();
            }
        }

        // Method to correct the left arm chain to better match tracked hand position
        private void CorrectLeftArmChain()
        {
            if (_leftElbow == null || _leftWrist == null) return;
            if (!_hasLeftHandData) return;

            // Get the actual hand transform from calibrator
            if (_calibrator == null || !_calibrator.IsCalibrated()) return;

            Transform realHandTransform = _calibrator.GetLeftHandTransform();
            if (realHandTransform == null) return;

            // Get the current positions in world space
            Vector3 elbowPos = _leftElbow.position;
            Vector3 wristPos = _leftWrist.position;
            Vector3 targetWristPos = realHandTransform.position;

            // Calculate the offset between the current wrist and the target wrist
            Vector3 wristOffset = targetWristPos - wristPos;

            // Only apply correction if there's a significant offset
            if (wristOffset.magnitude > 0.01f)
            {
                // Get the current arm direction
                Vector3 currentArmDir = (wristPos - elbowPos).normalized;

                // Get target arm direction
                Vector3 targetArmDir = (targetWristPos - elbowPos).normalized;

                // Calculate rotation from current to target direction
                Quaternion correctionRotation = Quaternion.FromToRotation(currentArmDir, targetArmDir);

                // Apply the correction to the elbow with the configured strength
                Quaternion blendedRotation = Quaternion.Slerp(
                    _leftElbow.rotation,
                    correctionRotation * _leftElbow.rotation,
                    handCorrectionStrength
                );

                // Apply the corrected rotation
                _leftElbow.rotation = blendedRotation;
            }
        }

        // Method to correct the right arm chain to better match tracked hand position
        private void CorrectRightArmChain()
        {
            if (_rightElbow == null || _rightWrist == null) return;
            if (!_hasRightHandData) return;

            // Get the actual hand transform from calibrator
            if (_calibrator == null || !_calibrator.IsCalibrated()) return;

            Transform realHandTransform = _calibrator.GetRightHandTransform();
            if (realHandTransform == null) return;

            // Get the current positions in world space
            Vector3 elbowPos = _rightElbow.position;
            Vector3 wristPos = _rightWrist.position;
            Vector3 targetWristPos = realHandTransform.position;

            // Calculate the offset between the current wrist and the target wrist
            Vector3 wristOffset = targetWristPos - wristPos;

            // Only apply correction if there's a significant offset
            if (wristOffset.magnitude > 0.01f)
            {
                // Get the current arm direction
                Vector3 currentArmDir = (wristPos - elbowPos).normalized;

                // Get target arm direction
                Vector3 targetArmDir = (targetWristPos - elbowPos).normalized;

                // Calculate rotation from current to target direction
                Quaternion correctionRotation = Quaternion.FromToRotation(currentArmDir, targetArmDir);

                // Apply the correction to the elbow with the configured strength
                Quaternion blendedRotation = Quaternion.Slerp(
                    _rightElbow.rotation,
                    correctionRotation * _rightElbow.rotation,
                    handCorrectionStrength
                );

                // Apply the corrected rotation
                _rightElbow.rotation = blendedRotation;
            }
        }

        // Hand not visible: the wrist keeps the rotation predicted by the pose model
        // (already applied, rest-relative, by SMPLMotionController); fingers go to rest
        private void ApplyPredictedLeftWristData()
        {
            for (var i = 2; i < Constants.NUM_HAND_JOINTS; ++i)
            {
                if (_leftJointList[i] != null)
                {
                    _leftJointList[i].localRotation = _defaultLeftHandRotations[i];
                }
            }
        }

        // Hand not visible: see ApplyPredictedLeftWristData
        private void ApplyPredictedRightWristData()
        {
            for (var i = 2; i < Constants.NUM_HAND_JOINTS; ++i)
            {
                if (_rightJointList[i] != null)
                {
                    _rightJointList[i].localRotation = _defaultRightHandRotations[i];
                }
            }
        }

        private void ApplyLeftHandPose()
        {
            if (!_hasLeftHandData) return;

            // Apply wrist rotation
            if (leftHandJoints.Wrist != null && applyWristRotations)
            {
                leftHandJoints.Wrist.rotation = _leftHandRotations[0];
            }

            // Apply finger joint rotations
            if (applyFingerRotations)
            {
                for (var i = 2; i < Constants.NUM_HAND_JOINTS; ++i)
                {
                    if (_leftJointList[i] != null)
                    {
                        _leftJointList[i].localRotation = _leftHandRotations[i];
                    }
                }
            }
        }

        private void ApplyRightHandPose()
        {
            if (!_hasRightHandData) return;

            // Apply wrist rotation
            if (rightHandJoints.Wrist != null && applyWristRotations)
            {
                rightHandJoints.Wrist.rotation = _rightHandRotations[0];
            }

            // Apply finger joint rotations
            if (applyFingerRotations)
            {
                for (var i = 2; i < Constants.NUM_HAND_JOINTS; ++i)
                {
                    if (_rightJointList[i] != null)
                    {
                        _rightJointList[i].localRotation = _rightHandRotations[i];
                    }
                }
            }
        }

        private void OnEnable()
        {
            if (_started)
            {
                if (RealLeftHand != null) RealLeftHand.WhenHandUpdated += UpdateLeftHandTarget;
                if (RealRightHand != null) RealRightHand.WhenHandUpdated += UpdateRightHandTarget;
            }
        }

        private void OnDisable()
        {
            if (_started)
            {
                if (RealLeftHand != null) RealLeftHand.WhenHandUpdated -= UpdateLeftHandTarget;
                if (RealRightHand != null) RealRightHand.WhenHandUpdated -= UpdateRightHandTarget;
            }
        }

        private void ValidateSetup()
        {
            if (RealLeftHand == null) Debug.LogWarning("Left Hand reference not assigned!");
            if (RealRightHand == null) Debug.LogWarning("Right Hand reference not assigned!");
            if (leftHandJoints.Wrist == null) Debug.LogWarning("Left Hand Start (Wrist) not assigned!");
            if (rightHandJoints.Wrist == null) Debug.LogWarning("Right Hand Start (Wrist) not assigned!");
        }

        private void UpdateLeftHandTarget()
        {
            if (!manualTracking && (_calibrator == null || !_calibrator.IsCalibrated())) return;

            // Only update if the motion controller has received pose data
            if (_motionController != null && !_motionController.HasPoseData) return;

            // Don't update during beta application
            if (_betaApplicationInProgress) return;

            if (!RealLeftHand.IsTrackedDataValid) return;

            if (!RealLeftHand.GetJointPosesLocal(out ReadOnlyHandJointPoses localJoints)) return;

            // Update wrist rotation
            if (leftHandJoints.Wrist != null)
            {
                if (RealLeftHand.GetRootPose(out Pose rootPose))
                {
                    // Convert hand rotation with 180-degree correction for left hand
                    _leftHandRotations[0] = ConvertLeftHandRotation(rootPose.rotation);
                }
            }

            // Update finger joint rotations
            for (var i = 2; i < Constants.NUM_HAND_JOINTS; ++i)
            {
                if (_leftJointList[i] != null)
                {
                    _leftHandRotations[i] = localJoints[i].rotation;
                }
            }

            // Set flags
            _hasLeftHandData = true;
        }

        // Convert hand rotation with adjustments for left hand (180 degree Y rotation)
        private Quaternion ConvertLeftHandRotation(Quaternion originalRotation)
        {
            Vector3 euler = originalRotation.eulerAngles;
            // Apply 180 degree rotation on Y axis for left hand (additional)
            euler.x = -euler.x;
            euler.y = euler.y + 180f;
            euler.z = -euler.z;
            return Quaternion.Euler(euler);
        }

        // Original conversion without 180 degree adjustment (for right hand)
        private Quaternion ConvertRightHandRotation(Quaternion originalRotation)
        {
            Vector3 euler = originalRotation.eulerAngles;
            euler.x = -euler.x;
            euler.y = euler.y + 180f;
            euler.z = -euler.z;
            return Quaternion.Euler(euler);
        }

        private void UpdateRightHandTarget()
        {
            if (!manualTracking && (_calibrator == null || !_calibrator.IsCalibrated())) return;

            // Only update if the motion controller has received pose data
            if (_motionController != null && !_motionController.HasPoseData) return;

            // Don't update during beta application
            if (_betaApplicationInProgress) return;

            if (!RealRightHand.IsTrackedDataValid) return;

            if (!RealRightHand.GetJointPosesLocal(out ReadOnlyHandJointPoses localJoints)) return;

            // Update wrist rotation
            if (rightHandJoints.Wrist != null)
            {
                if (RealRightHand.GetRootPose(out Pose rootPose))
                {
                    _rightHandRotations[0] = ConvertRightHandRotation(rootPose.rotation);
                }
            }

            // Update finger joint rotations
            for (var i = 2; i < Constants.NUM_HAND_JOINTS; ++i)
            {
                if (_rightJointList[i] != null)
                {
                    _rightHandRotations[i] = localJoints[i].rotation;
                }
            }

            // Set flags
            _hasRightHandData = true;
        }

        // Public methods
        public bool HasHandData()
        {
            return _hasLeftHandData || _hasRightHandData;
        }

        public void ResetHandData()
        {
            _hasLeftHandData = false;
            _hasRightHandData = false;
        }

        // Add public methods to access hand joints for the recorder
        public Transform[] GetLeftHandJoints()
        {
            return _leftJointList;
        }

        public Transform[] GetRightHandJoints()
        {
            return _rightJointList;
        }

        public bool HasLeftHandData()
        {
            return _hasLeftHandData;
        }

        public bool HasRightHandData()
        {
            return _hasRightHandData;
        }

        public Dictionary<string, Quaternion> GetLeftHandRawRotations()
        {
            Dictionary<string, Quaternion> rotations = new Dictionary<string, Quaternion>();

            if (!_hasLeftHandData) return rotations;

            // Skip indices 0 (wrist) and 1 (forearm stub)
            // Only capture finger joints starting from index 2
            for (var i = 2; i < Constants.NUM_HAND_JOINTS; ++i)
            {
                string jointName = _leftJointList[i]?.name ?? $"Joint_{i}";
                rotations.Add(jointName, _leftHandRotations[i]);
            }

            return rotations;
        }

        public Dictionary<string, Quaternion> GetRightHandRawRotations()
        {
            Dictionary<string, Quaternion> rotations = new Dictionary<string, Quaternion>();

            if (!_hasRightHandData) return rotations;

            // Skip indices 0 (wrist) and 1 (forearm stub)
            // Only capture finger joints starting from index 2
            for (var i = 2; i < Constants.NUM_HAND_JOINTS; ++i)
            {
                string jointName = _rightJointList[i]?.name ?? $"Joint_{i}";
                rotations.Add(jointName, _rightHandRotations[i]);
            }

            return rotations;
        }
    }
}