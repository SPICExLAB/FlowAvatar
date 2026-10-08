using UnityEngine;
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;

/// <summary>
/// Utility class for optimized network receiving operations.
/// Handles buffer management and message parsing with efficient memory usage.
/// </summary>
public class NetworkReceiveUtils : IDisposable
{
    #region Data Structures

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct FrameHeader
    {
        public uint MessageLength;
        public byte MessageType;
        public byte FrameRate;
        public ushort FrameCounter;
        public float Timestamp;
        public uint DroppedFrames;
    }

    #endregion

    #region Delegates and Events

    /// <summary>
    /// Delegate for model prediction data processing
    /// </summary>
    public delegate void ModelPredictionDelegate(byte[] rawData, FrameMetadata metadata);

    #endregion

    #region Constants

    // Message format constants
    private const int FLOAT_SIZE = 4;
    private const int HEADER_SIZE = 16;     // Size of frame header in bytes

    // Revised constants for 6D rotation format
    private const int ROOT_ROT_6D_FLOATS = 6;           // 6 values for root rotation
    private const int BODY_ROT_6D_FLOATS = 21 * 6;      // 21 joints x 6 values = 126 floats
    private const int BETA_FLOATS = 10;                 // 10 values for SMPL-X shape parameters
    private const int HEAD_POS_FLOATS = 3;              // 3 values for head position

    #endregion

    #region Private Fields

    // Network client reference
    private readonly DualClient client;

    // Buffer handling
    private byte[] buffer;
    private int bufferPosition = 0;

    // Message markers
    private readonly byte[] startMarker;
    private readonly byte[] endMarker;

    // Message handling
    private readonly ModelPredictionDelegate modelCallback;
    private readonly bool profilePerformance;

    // Statistics
    private int framesReceived = 0;
    private int bytesReceived = 0;
    private int invalidFrames = 0;
    private int invalidHeaders = 0;

    #endregion

    #region Constructor and Initialization

    /// <summary>
    /// Creates a new NetworkReceiveUtils instance for handling model predictions
    /// </summary>
    /// <param name="client">DualClient reference for receiving data</param>
    /// <param name="bufferSize">Size of receive buffer in bytes</param>
    /// <param name="modelCallback">Callback to invoke when model prediction is received</param>
    /// <param name="profilePerformance">Enable performance profiling</param>
    public NetworkReceiveUtils(DualClient client, int bufferSize, ModelPredictionDelegate modelCallback, bool profilePerformance = false)
    {
        this.client = client ?? throw new ArgumentNullException(nameof(client));
        this.modelCallback = modelCallback ?? throw new ArgumentNullException(nameof(modelCallback));
        this.profilePerformance = profilePerformance;

        // Initialize buffer with minimum size
        buffer = new byte[Math.Max(bufferSize, 8192)];
        bufferPosition = 0;

        // Initialize message markers
        startMarker = Encoding.UTF8.GetBytes("<START>");
        endMarker = Encoding.UTF8.GetBytes("<END>");

        if (profilePerformance)
        {
            Debug.Log($"[NetworkReceiveUtils] Initialized with buffer size: {buffer.Length} bytes");
            Debug.Log($"[NetworkReceiveUtils] Expected data format: Header ({HEADER_SIZE} bytes), " +
                     $"Root 6D ({ROOT_ROT_6D_FLOATS * FLOAT_SIZE} bytes), " +
                     $"Body 6D ({BODY_ROT_6D_FLOATS * FLOAT_SIZE} bytes), " +
                     $"Betas ({BETA_FLOATS * FLOAT_SIZE} bytes), " +
                     $"Head Pos ({HEAD_POS_FLOATS * FLOAT_SIZE} bytes)");
        }
    }

    #endregion

    #region Public API

    /// <summary>
    /// Receives and processes incoming data
    /// </summary>
    /// <returns>True if new data was processed, false otherwise</returns>
    public bool ReceiveAndProcessData()
    {
        if (!client.isReceiveConnected)
        {
            return false;
        }

        // Receive data from network
        byte[] receivedData = client.ReceiveBytes(4096);

        if (receivedData == null || receivedData.Length == 0)
        {
            return false;
        }

        // Add to buffer and update statistics
        AddToBuffer(receivedData);
        bytesReceived += receivedData.Length;

        // Process all complete messages in buffer
        bool processedAny = false;
        while (ProcessNextMessage())
        {
            processedAny = true;
        }

        return processedAny;
    }

    /// <summary>
    /// Gets statistics about the data received
    /// </summary>
    /// <returns>A string containing receive statistics</returns>
    public string GetStats()
    {
        return $"Frames: {framesReceived}, Bytes: {bytesReceived}, " +
               $"Invalid frames: {invalidFrames}, Invalid headers: {invalidHeaders}, " +
               $"Buffer usage: {bufferPosition}/{buffer.Length} bytes";
    }

    /// <summary>
    /// Disposes resources used by the NetworkReceiveUtils
    /// </summary>
    public void Dispose()
    {
        buffer = null;
    }

    #endregion

    #region Processing Methods

    /// <summary>
    /// Processes the next complete message in the buffer
    /// </summary>
    /// <returns>True if a message was processed, false otherwise</returns>
    private bool ProcessNextMessage()
    {
        // Find start and end markers
        int startIdx = FindMarker(startMarker, 0);
        if (startIdx == -1)
        {
            return false;
        }

        int endIdx = FindMarker(endMarker, startIdx + startMarker.Length);
        if (endIdx == -1)
        {
            return false;
        }

        try
        {
            // Extract message data between markers
            int dataStart = startIdx + startMarker.Length;
            int messageLength = endIdx - dataStart;

            // Validate minimum message size
            if (messageLength < HEADER_SIZE)
            {
                if (profilePerformance)
                {
                    Debug.LogWarning($"[NetworkReceiveUtils] Message too short: {messageLength} bytes, expected at least {HEADER_SIZE}");
                }
                return RemoveProcessedMessage(startIdx, endIdx + endMarker.Length);
            }

            // Parse message header
            FrameHeader header = ParseHeader(buffer, dataStart);

            // Create metadata struct
            FrameMetadata metadata = new FrameMetadata(
                header.FrameCounter,
                header.Timestamp,
                header.DroppedFrames,
                header.FrameRate
            );

            // Validate data length
            int expectedDataSize = (ROOT_ROT_6D_FLOATS + BODY_ROT_6D_FLOATS + BETA_FLOATS + HEAD_POS_FLOATS) * FLOAT_SIZE;
            int actualDataSize = messageLength - HEADER_SIZE;

            if (actualDataSize < expectedDataSize)
            {
                if (profilePerformance)
                {
                    Debug.LogWarning($"[NetworkReceiveUtils] Data section too small: {actualDataSize} bytes, expected {expectedDataSize}");
                }
                invalidHeaders++;
                return RemoveProcessedMessage(startIdx, endIdx + endMarker.Length);
            }

            // Extract data section (entire raw data after header)
            byte[] modelData = new byte[actualDataSize];
            Array.Copy(buffer, dataStart + HEADER_SIZE, modelData, 0, actualDataSize);

            // Validate data for NaN values (just check a sample)
            if (ValidateData(modelData))
            {
                // Call model prediction callback with the raw data
                modelCallback(modelData, metadata);
                framesReceived++;
            }
            else
            {
                invalidFrames++;
            }

            // Remove processed message from buffer
            return RemoveProcessedMessage(startIdx, endIdx + endMarker.Length);
        }
        catch (Exception e)
        {
            if (profilePerformance)
            {
                Debug.LogError($"[NetworkReceiveUtils] Error processing message: {e.Message}");
            }

            // Remove corrupted message and continue
            return RemoveProcessedMessage(startIdx, endIdx + endMarker.Length);
        }
    }

    /// <summary>
    /// Parses a frame header from the buffer
    /// </summary>
    private FrameHeader ParseHeader(byte[] data, int offset)
    {
        FrameHeader header = new FrameHeader
        {
            MessageLength = BitConverter.ToUInt32(data, offset),
            MessageType = data[offset + 4],
            FrameRate = data[offset + 5],
            FrameCounter = BitConverter.ToUInt16(data, offset + 6),
            Timestamp = BitConverter.ToSingle(data, offset + 8),
            DroppedFrames = BitConverter.ToUInt32(data, offset + 12)
        };

        if (profilePerformance && framesReceived % 100 == 0)
        {
            Debug.Log($"[NetworkReceiveUtils] Header: Length={header.MessageLength}, " +
                     $"Type={header.MessageType}, Rate={header.FrameRate}, " +
                     $"Frame={header.FrameCounter}, Time={header.Timestamp:F2}, " +
                     $"Dropped={header.DroppedFrames}");
        }

        return header;
    }

    /// <summary>
    /// Validates data for NaN values by sampling a few floats
    /// </summary>
    private bool ValidateData(byte[] data)
    {
        // Sample a few values from the data to check for NaN
        int sampleCount = Math.Min(20, data.Length / FLOAT_SIZE);
        int stepSize = data.Length / (sampleCount * FLOAT_SIZE);

        for (int i = 0; i < sampleCount; i++)
        {
            int offset = i * stepSize * FLOAT_SIZE;
            if (offset + FLOAT_SIZE <= data.Length)
            {
                float value = BitConverter.ToSingle(data, offset);
                if (float.IsNaN(value) || float.IsInfinity(value))
                {
                    if (profilePerformance)
                    {
                        Debug.LogWarning($"[NetworkReceiveUtils] Invalid value in data at offset {offset}: {value}");
                    }
                    return false;
                }
            }
        }

        return true;
    }

    #endregion

    #region Buffer Management

    /// <summary>
    /// Adds received data to the buffer
    /// </summary>
    private void AddToBuffer(byte[] newData)
    {
        // Check if we need to make room in the buffer
        if (bufferPosition + newData.Length > buffer.Length)
        {
            CompactBuffer();

            // If still not enough space, resize buffer
            if (bufferPosition + newData.Length > buffer.Length)
            {
                if (buffer.Length < 32768) // Cap max size at 32KB
                {
                    int newSize = Math.Min(buffer.Length * 2, 32768);
                    Array.Resize(ref buffer, newSize);

                    if (profilePerformance)
                    {
                        Debug.Log($"[NetworkReceiveUtils] Resized buffer to {buffer.Length} bytes");
                    }
                }
                else
                {
                    // Buffer full even after compaction, reset position
                    if (profilePerformance)
                    {
                        Debug.LogWarning("[NetworkReceiveUtils] Buffer overflow, resetting");
                    }
                    bufferPosition = 0;
                }
            }
        }

        // Copy new data to buffer
        Array.Copy(newData, 0, buffer, bufferPosition, newData.Length);
        bufferPosition += newData.Length;
    }

    /// <summary>
    /// Compacts the buffer by removing unneeded data
    /// </summary>
    private void CompactBuffer()
    {
        // Find first start marker
        int startIdx = FindMarker(startMarker, 0);

        if (startIdx > 0)
        {
            // Keep only data from this marker onwards
            int remaining = bufferPosition - startIdx;
            Array.Copy(buffer, startIdx, buffer, 0, remaining);
            bufferPosition = remaining;
        }
        else if (startIdx == -1)
        {
            // No start marker found, reset buffer
            bufferPosition = 0;
        }
    }

    /// <summary>
    /// Removes a processed message from the buffer
    /// </summary>
    private bool RemoveProcessedMessage(int startIdx, int endIdx)
    {
        if (endIdx >= bufferPosition)
        {
            // Message encompasses entire buffer
            bufferPosition = 0;
            return true;
        }

        // Move remaining data to beginning of buffer
        int remainingLength = bufferPosition - endIdx;
        Array.Copy(buffer, endIdx, buffer, 0, remainingLength);
        bufferPosition = remainingLength;
        return true;
    }

    /// <summary>
    /// Finds a marker in the buffer
    /// </summary>
    private int FindMarker(byte[] marker, int startIndex)
    {
        int markerLength = marker.Length;
        int searchEnd = bufferPosition - markerLength + 1;

        for (int i = startIndex; i < searchEnd; i++)
        {
            bool found = true;
            for (int j = 0; j < markerLength; j++)
            {
                if (buffer[i + j] != marker[j])
                {
                    found = false;
                    break;
                }
            }

            if (found)
            {
                return i;
            }
        }

        return -1;
    }

    #endregion
}

/// <summary>
/// Metadata structure for received frames
/// </summary>
public struct FrameMetadata
{
    public ushort FrameCounter;
    public float Timestamp;
    public uint DroppedFrames;
    public byte FrameRate;

    public FrameMetadata(ushort frameCounter, float timestamp, uint droppedFrames, byte frameRate)
    {
        FrameCounter = frameCounter;
        Timestamp = timestamp;
        DroppedFrames = droppedFrames;
        FrameRate = frameRate;
    }
}