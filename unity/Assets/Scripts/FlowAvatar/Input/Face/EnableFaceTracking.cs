using Oculus.Interaction;
using Oculus.Interaction.Input;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text.RegularExpressions; // For regex pattern matching in device detection

/// <summary>
/// Manages face tracking enabling/disabling based on device type and user preference.
/// Automatically configures settings based on headset model (Quest Pro vs Quest 3).
/// </summary>
public class EnableFaceTracking : MonoBehaviour
{
    [Header("Tracking Settings")]
    [SerializeField] private bool manualTracking = false;
    [SerializeField] private SMPLXFaceDriver FaceDriver;
    [Tooltip("Enable or disable face tracking. Auto-set based on device if auto-detect is enabled.")]
    [SerializeField] private bool enableFaceTracking = true;

    [Header("Auto-configuration")]
    [Tooltip("Automatically detect headset type and configure face tracking accordingly")]
    [SerializeField] private bool autoDetectHeadset = true;
    [SerializeField] private bool debugLogDeviceInfo = false;

    // Internal variables
    private VRToSMPLCalibrator Calibrator;
    private bool _started = false;
    private bool _isQuestPro = false;
    private bool _deviceDetected = false;

    private void Awake()
    {
        Calibrator = GetComponent<VRToSMPLCalibrator>();

        if (FaceDriver == null)
        {
            FaceDriver = GetComponentInChildren<SMPLXFaceDriver>();
        }

        // Auto-detect headset if enabled
        if (autoDetectHeadset)
        {
            DetectHeadsetType();
        }
    }

    private void Start()
    {
        this.BeginStart(ref _started);
        ValidateSetup();
        this.EndStart(ref _started);

        // Apply initial face tracking state
        UpdateFaceTrackingState();
    }

    private void ValidateSetup()
    {
        if (FaceDriver == null) Debug.LogWarning("Face Driver reference not assigned!");
        if (Calibrator == null) Debug.LogWarning("VRToSMPLCalibrator reference not assigned!");

        // Log device detection results if debugging is enabled
        if (debugLogDeviceInfo)
        {
            if (_deviceDetected)
            {
                Debug.Log($"Headset detected: {(_isQuestPro ? "Quest Pro" : "Not Quest Pro")}");
                Debug.Log($"Initial face tracking state: {(enableFaceTracking ? "Enabled" : "Disabled")}");
            }
            else
            {
                Debug.Log("Failed to detect headset type");
            }
        }
    }

    private void Update()
    {
        if (!_started) return;

        // If manual tracking is enabled, respect that setting
        if (manualTracking)
        {
            UpdateFaceDriverBasedOnToggle();
            return;
        }

        // Otherwise, check calibration state and toggle
        bool shouldTrack = enableFaceTracking && Calibrator != null && Calibrator.IsCalibrated();

        // Enable/disable face tracking based on calibration state and toggle
        if (FaceDriver != null)
        {
            if (shouldTrack && !FaceDriver.enabled)
            {
                FaceDriver.enabled = true;
            }
            else if (!shouldTrack && FaceDriver.enabled)
            {
                FaceDriver.enabled = false;
            }
        }
    }

    /// <summary>
    /// Detects the current headset type and configures face tracking accordingly
    /// </summary>
    private void DetectHeadsetType()
    {
        string deviceName = SystemInfo.deviceModel;

        if (debugLogDeviceInfo)
        {
            Debug.Log($"Device detected: {deviceName}");
        }

        // Check if it's a Quest Pro
        bool isQuestPro = Regex.IsMatch(deviceName, "Quest Pro", RegexOptions.IgnoreCase);

        // Check if it's a Quest 3
        bool isQuest3 = Regex.IsMatch(deviceName, "Quest 3", RegexOptions.IgnoreCase);

        if (isQuestPro || isQuest3)
        {
            _deviceDetected = true;
            _isQuestPro = isQuestPro;

            // Automatically set enableFaceTracking based on device (unless manual override is set)
            if (!manualTracking)
            {
                enableFaceTracking = isQuestPro;

                if (debugLogDeviceInfo)
                {
                    Debug.Log($"Auto-configured: Face tracking {(enableFaceTracking ? "enabled" : "disabled")} for {(isQuestPro ? "Quest Pro" : "Quest 3")}");
                }
            }
        }
    }

    /// <summary>
    /// Updates the face tracking driver state based on the enableFaceTracking value
    /// </summary>
    private void UpdateFaceDriverBasedOnToggle()
    {
        if (FaceDriver != null)
        {
            FaceDriver.enabled = enableFaceTracking;
        }
    }

    /// <summary>
    /// Updates the face tracking state based on current settings
    /// </summary>
    private void UpdateFaceTrackingState()
    {
        if (manualTracking)
        {
            UpdateFaceDriverBasedOnToggle();
        }
        // Otherwise, Update() will handle enabling/disabling based on calibration
    }

    /// <summary>
    /// Public method to toggle face tracking from other scripts
    /// </summary>
    public void ToggleFaceTracking(bool enable)
    {
        enableFaceTracking = enable;
        UpdateFaceTrackingState();
    }

    /// <summary>
    /// Returns whether the device is a Quest Pro
    /// </summary>
    public bool IsQuestPro()
    {
        return _isQuestPro;
    }

    /// <summary>
    /// Returns whether face tracking is currently enabled
    /// </summary>
    public bool IsFaceTrackingEnabled()
    {
        return enableFaceTracking;
    }
}