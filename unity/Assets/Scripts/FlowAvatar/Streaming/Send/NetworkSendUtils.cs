using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// Utility class for optimized network sending operations.
/// Simplified to send feature vectors and head position values.
/// </summary>
public class NetworkSendUtils : IDisposable
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
        public ushort FeatureSize;    // Size of feature vector (90 or 92)
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct FrameData
    {
        [MarshalAs(UnmanagedType.ByValArray)]
        public float[] Features; // Dynamic feature vector (90D or 92D)
        [MarshalAs(UnmanagedType.ByValArray)]
        public float[] HeadPosition; // 3D head position (converted)
    }

    #endregion

    #region Configuration

    // Constants
    private const int DEFAULT_BUFFER_POOL_SIZE = 3;
    private const int MIN_BUFFER_SIZE = 256;

    // Client reference
    private readonly DualClient client;

    // Buffer pooling
    private readonly List<BufferObject> bufferPool = new List<BufferObject>();
    private readonly int bufferPoolSize;
    private readonly bool useObjectPooling = true;

    // Async sending
    private readonly bool useAsyncSend = true;
    private readonly SemaphoreSlim sendSemaphore = new SemaphoreSlim(1, 1);
    private volatile bool isSending = false;

    // Message components
    private readonly byte[] markerStart;
    private readonly byte[] markerEnd;
    private readonly int headerSize;
    private int frameSize;      // Dynamic based on feature size
    private int totalMessageSize; // Dynamic based on feature size
    private int featureCount;   // Dynamic (90 or 92)
    private readonly int headPosCount;

    // Stats
    private long totalBytesSent = 0;
    private int framesSent = 0;

    #endregion

    #region Buffer Management

    /// <summary>
    /// Represents a reusable buffer in the buffer pool
    /// </summary>
    private class BufferObject
    {
        public byte[] Buffer { get; }
        public bool InUse { get; set; }

        public BufferObject(int size)
        {
            Buffer = new byte[size];
            InUse = false;
        }
    }

    #endregion

    #region Constructor and Initialization

    /// <summary>
    /// Creates a new NetworkSendUtils instance for sending feature vectors
    /// </summary>
    /// <param name="client">DualClient reference for sending data</param>
    /// <param name="useAsyncSend">Whether to use async sending</param>
    /// <param name="featureCount">Number of features in the vector (90 or 92)</param>
    /// <param name="headPosCount">Number of head position components (defaults to 3 for XYZ)</param>
    /// <param name="bufferPoolSize">Size of buffer pool (if using object pooling)</param>
    public NetworkSendUtils(DualClient client, bool useAsyncSend = true, int featureCount = 90, int headPosCount = 3, int bufferPoolSize = DEFAULT_BUFFER_POOL_SIZE)
    {
        this.client = client ?? throw new ArgumentNullException(nameof(client));
        this.useAsyncSend = useAsyncSend;
        this.bufferPoolSize = bufferPoolSize;
        this.featureCount = Math.Max(1, featureCount);
        this.headPosCount = Math.Max(3, headPosCount); // Ensure at least XYZ components

        // Initialize protocol markers
        markerStart = System.Text.Encoding.ASCII.GetBytes("<START>");
        markerEnd = System.Text.Encoding.ASCII.GetBytes("<END>");

        // Calculate sizes based on actual data 
        headerSize = Marshal.SizeOf<FrameHeader>();

        // Calculate frame size
        // feature array + head position array
        UpdateFrameSize(this.featureCount);

        Debug.Log($"[NetworkSendUtils] Initialized with message size: {totalMessageSize} bytes");
        Debug.Log($"[NetworkSendUtils] Header: {headerSize} bytes, Frame: {frameSize} bytes (includes {featureCount} features and {headPosCount} head position components)");

        // Initialize buffer pool
        InitializeBufferPool();
    }

    /// <summary>
    /// Updates the frame size based on feature count
    /// </summary>
    private void UpdateFrameSize(int newFeatureCount)
    {
        // Validate feature count
        if (newFeatureCount != 90 && newFeatureCount != 92)
        {
            Debug.LogWarning($"[NetworkSendUtils] Unusual feature count: {newFeatureCount}. Expected 90 or 92.");
        }

        // Update feature count
        featureCount = newFeatureCount;

        // Update frame size
        int featuresSize = sizeof(float) * featureCount;
        int headPosSize = sizeof(float) * headPosCount;
        frameSize = featuresSize + headPosSize;

        // Update total message size
        totalMessageSize = markerStart.Length + headerSize + frameSize + markerEnd.Length;

        // Ensure minimum buffer size for safety
        if (totalMessageSize < MIN_BUFFER_SIZE)
        {
            totalMessageSize = MIN_BUFFER_SIZE;
        }
    }

    /// <summary>
    /// Initializes the buffer pool
    /// </summary>
    private void InitializeBufferPool()
    {
        if (!useObjectPooling) return;

        // Create initial buffer pool
        for (int i = 0; i < bufferPoolSize; i++)
        {
            // Use a size that can accommodate both 90D and 92D
            int maxBufferSize = markerStart.Length + headerSize +
                                (sizeof(float) * Math.Max(90, 92)) +
                                (sizeof(float) * headPosCount) +
                                markerEnd.Length;
            bufferPool.Add(new BufferObject(maxBufferSize));
        }

        Debug.Log($"[NetworkSendUtils] Buffer pool initialized with {bufferPoolSize} buffers of {totalMessageSize} bytes each");
    }

    #endregion

    #region Public API

    /// <summary>
    /// Sends frame data with header to the connected client
    /// </summary>
    /// <param name="header">Frame header information</param>
    /// <param name="frameData">Frame data to send</param>
    /// <param name="onComplete">Optional callback when send completes</param>
    public void SendFrameData(FrameHeader header, FrameData frameData, Action<bool> onComplete = null)
    {
        if (!client.isSendConnected)
        {
            onComplete?.Invoke(false);
            return;
        }

        // Check if feature size has changed
        if (header.FeatureSize != featureCount)
        {
            UpdateFrameSize(header.FeatureSize);
            Debug.Log($"[NetworkSendUtils] Feature size changed to {header.FeatureSize}D, updated frame size: {frameSize} bytes");
        }

        // Update header message length
        header.MessageLength = (uint)frameSize;

        // Get a buffer (preferably from pool)
        byte[] buffer = GetBufferFromPool();

        try
        {
            // Directly populate buffer without marshal allocations
            int offset = 0;

            // Write start marker
            Array.Copy(markerStart, 0, buffer, offset, markerStart.Length);
            offset += markerStart.Length;

            // Write header directly (without marshal allocation)
            // MessageLength (uint - 4 bytes)
            BitConverter.GetBytes(header.MessageLength).CopyTo(buffer, offset);
            offset += 4;

            // MessageType and FrameRate (byte + byte - 2 bytes)
            buffer[offset++] = header.MessageType;
            buffer[offset++] = header.FrameRate;

            // FrameCounter (ushort - 2 bytes)
            BitConverter.GetBytes(header.FrameCounter).CopyTo(buffer, offset);
            offset += 2;

            // Timestamp (float - 4 bytes)
            BitConverter.GetBytes(header.Timestamp).CopyTo(buffer, offset);
            offset += 4;

            // DroppedFrames (uint - 4 bytes)
            BitConverter.GetBytes(header.DroppedFrames).CopyTo(buffer, offset);
            offset += 4;

            // FeatureSize (ushort - 2 bytes)
            BitConverter.GetBytes(header.FeatureSize).CopyTo(buffer, offset);
            offset += 2;

            // Write features - use the actual feature size from header
            WriteFloatArray(buffer, ref offset, frameData.Features, header.FeatureSize);

            // Write head position
            WriteFloatArray(buffer, ref offset, frameData.HeadPosition, headPosCount);

            // Write end marker
            Array.Copy(markerEnd, 0, buffer, offset, markerEnd.Length);

            if (useAsyncSend)
            {
                SendAsync(buffer, onComplete);
            }
            else
            {
                // Direct send
                client.SendBytes(buffer);

                // Track statistics
                totalBytesSent += buffer.Length;
                framesSent++;

                onComplete?.Invoke(true);
                ReturnBufferToPool(buffer);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[NetworkSendUtils] Error in send: {e.Message}");
            onComplete?.Invoke(false);
            ReturnBufferToPool(buffer);
        }
    }

    /// <summary>
    /// Checks if the sender is currently busy sending data
    /// </summary>
    /// <returns>True if async send is in progress, false otherwise</returns>
    public bool IsBusy()
    {
        return useAsyncSend && isSending;
    }

    /// <summary>
    /// Gets statistics about the data sent
    /// </summary>
    /// <returns>A tuple containing (totalBytesSent, framesSent)</returns>
    public (long totalBytes, int frames) GetSendStats()
    {
        return (totalBytesSent, framesSent);
    }

    /// <summary>
    /// Disposes resources used by the NetworkSendUtils
    /// </summary>
    public void Dispose()
    {
        try
        {
            sendSemaphore?.Dispose();
        }
        catch (Exception) { /* Ignore disposal errors */ }

        bufferPool.Clear();
    }

    #endregion

    #region Buffer Management Methods

    /// <summary>
    /// Gets a buffer from the pool or creates a new one if needed
    /// </summary>
    /// <returns>Byte array buffer for message data</returns>
    private byte[] GetBufferFromPool()
    {
        if (!useObjectPooling)
        {
            return new byte[totalMessageSize];
        }

        lock (bufferPool)
        {
            // Find an available buffer
            foreach (var bufferObj in bufferPool)
            {
                if (!bufferObj.InUse)
                {
                    bufferObj.InUse = true;
                    return bufferObj.Buffer;
                }
            }

            // If all buffers are in use, create a new one
            if (bufferPool.Count < bufferPoolSize * 2)
            {
                var newBuffer = new BufferObject(totalMessageSize);
                newBuffer.InUse = true;
                bufferPool.Add(newBuffer);
                return newBuffer.Buffer;
            }

            // If we've reached our maximum, just create a temporary buffer
            Debug.LogWarning("[NetworkSendUtils] Buffer pool exhausted, creating temporary buffer");
            return new byte[totalMessageSize];
        }
    }

    /// <summary>
    /// Returns a buffer to the pool for reuse
    /// </summary>
    /// <param name="buffer">Buffer to return</param>
    private void ReturnBufferToPool(byte[] buffer)
    {
        if (!useObjectPooling) return;

        lock (bufferPool)
        {
            foreach (var bufferObj in bufferPool)
            {
                if (bufferObj.Buffer == buffer)
                {
                    bufferObj.InUse = false;
                    return;
                }
            }
        }
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Helper method to write float arrays to buffer without redundant allocations
    /// </summary>
    private void WriteFloatArray(byte[] buffer, ref int offset, float[] array, int count)
    {
        int toCopy = array != null ? Math.Min(array.Length, count) : 0;

        // Copy available values
        for (int i = 0; i < toCopy; i++)
        {
            BitConverter.GetBytes(array[i]).CopyTo(buffer, offset);
            offset += 4;
        }

        // Fill remaining with zeros
        for (int i = toCopy; i < count; i++)
        {
            BitConverter.GetBytes(0.0f).CopyTo(buffer, offset);
            offset += 4;
        }
    }

    /// <summary>
    /// Sends data asynchronously
    /// </summary>
    private async void SendAsync(byte[] buffer, Action<bool> onComplete)
    {
        // Prevent multiple concurrent sends
        if (!await TryAcquireSemaphore(0))
        {
            Debug.LogWarning("[NetworkSendUtils] Async send already in progress, dropping frame");
            ReturnBufferToPool(buffer);
            onComplete?.Invoke(false);
            return;
        }

        try
        {
            isSending = true;

            await Task.Run(() => {
                try
                {
                    client.SendBytes(buffer);

                    // Thread-safe increment
                    Interlocked.Add(ref totalBytesSent, buffer.Length);
                    Interlocked.Increment(ref framesSent);

                    onComplete?.Invoke(true);
                }
                catch (Exception e)
                {
                    Debug.LogError($"[NetworkSendUtils] Async send error: {e.Message}");
                    onComplete?.Invoke(false);
                }
                finally
                {
                    ReturnBufferToPool(buffer);
                }
            });
        }
        catch (Exception e)
        {
            Debug.LogError($"[NetworkSendUtils] Error in async send: {e.Message}");
            ReturnBufferToPool(buffer);
            onComplete?.Invoke(false);
        }
        finally
        {
            isSending = false;
            sendSemaphore.Release();
        }
    }

    /// <summary>
    /// Tries to acquire the send semaphore with a timeout
    /// </summary>
    /// <param name="timeoutMs">Timeout in milliseconds (0 = no wait)</param>
    /// <returns>True if semaphore was acquired, false otherwise</returns>
    private async Task<bool> TryAcquireSemaphore(int timeoutMs = 0)
    {
        try
        {
            return await sendSemaphore.WaitAsync(timeoutMs);
        }
        catch (Exception)
        {
            return false;
        }
    }

    #endregion
}