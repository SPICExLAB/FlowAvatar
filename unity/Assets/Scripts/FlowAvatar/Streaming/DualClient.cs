using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class DualClient : MonoBehaviour
{
    [Header("Server Settings")]
    public string serverIp = "127.0.0.1";
    public int sendPort = 8888;
    public int receivePort = 8889;
    public bool connectOnLoad = true;
    public float reconnectDelay = 1f;

    private Socket sendSocket;
    private Socket receiveSocket;
    private int dataSize = 4096;
    private byte[] data;
    private float reconnectTimer;

    private bool _isSendConnected;
    private bool _isReceiveConnected;

    public bool isSendConnected { get { return _isSendConnected && sendSocket != null && sendSocket.Connected; } }
    public bool isReceiveConnected { get { return _isReceiveConnected && receiveSocket != null && receiveSocket.Connected; } }

    void Start()
    {
        data = new byte[dataSize];
        if (connectOnLoad)
        {
            ConnectBoth();
        }
    }

    public void ConnectBoth()
    {
        ConnectSend();
        ConnectReceive();
    }

    public void ConnectSend()
    {
        try
        {
            if (sendSocket != null)
            {
                sendSocket.Close();
                sendSocket = null;
            }

            sendSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            sendSocket.Connect(new IPEndPoint(IPAddress.Parse(serverIp), sendPort));
            _isSendConnected = true;
            Debug.Log($"[{nameof(DualClient)}] Send connection established");
        }
        catch (Exception ex)
        {
            _isSendConnected = false;
            Debug.LogError($"[{nameof(DualClient)}] Send connection failed: {ex.Message}");
        }
    }

    public void ConnectReceive()
    {
        try
        {
            if (receiveSocket != null)
            {
                receiveSocket.Close();
                receiveSocket = null;
            }

            receiveSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
            receiveSocket.Connect(new IPEndPoint(IPAddress.Parse(serverIp), receivePort));
            _isReceiveConnected = true;
            Debug.Log($"[{nameof(DualClient)}] Receive connection established");
        }
        catch (Exception ex)
        {
            _isReceiveConnected = false;
            Debug.LogError($"[{nameof(DualClient)}] Receive connection failed: {ex.Message}");
        }
    }

    public string Receive(int bufferSize = 4096)
    {
        if (!isReceiveConnected) return string.Empty;

        if (bufferSize != dataSize)
        {
            data = new byte[bufferSize];
            dataSize = bufferSize;
        }

        try
        {
            if (receiveSocket.Available == 0) return string.Empty;

            int length = receiveSocket.Receive(data);
            if (length == 0)
            {
                _isReceiveConnected = false;
                return string.Empty;
            }
            return Encoding.UTF8.GetString(data, 0, length);
        }
        catch (SocketException ex)
        {
            Debug.LogError($"[{nameof(DualClient)}] Receive error: {ex.Message}");
            _isReceiveConnected = false;
            return string.Empty;
        }
    }

    public byte[] ReceiveBytes(int bufferSize = 4096)
    {
        if (!isReceiveConnected) return new byte[0];

        try
        {
            if (receiveSocket.Available == 0) return new byte[0];

            byte[] data = new byte[bufferSize];
            int length = receiveSocket.Receive(data);
            if (length == 0)
            {
                _isReceiveConnected = false;
                return new byte[0];
            }

            // Resize array to actual data length
            byte[] actualData = new byte[length];
            Array.Copy(data, actualData, length);
            return actualData;
        }
        catch (SocketException ex)
        {
            Debug.LogError($"[{nameof(DualClient)}] Receive error: {ex.Message}");
            _isReceiveConnected = false;
            return new byte[0];
        }
    }

    public void Send(string message)
    {
        if (!isSendConnected) return;

        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            sendSocket.Send(data);
        }
        catch (SocketException ex)
        {
            Debug.LogError($"[{nameof(DualClient)}] Send error: {ex.Message}");
            _isSendConnected = false;
        }
    }

    public void SendBytes(byte[] bytes)
    {
        if (!isSendConnected) return;

        try
        {
            sendSocket.Send(bytes);
        }
        catch (SocketException ex)
        {
            Debug.LogError($"[{nameof(DualClient)}] Send error: {ex.Message}");
            _isSendConnected = false;
        }
    }

    void Update()
    {
        // Handle reconnection
        if (!isSendConnected || !isReceiveConnected)
        {
            reconnectTimer += Time.deltaTime;
            if (reconnectTimer >= reconnectDelay)
            {
                reconnectTimer = 0f;
                if (!isSendConnected) ConnectSend();
                if (!isReceiveConnected) ConnectReceive();
            }
        }
    }

    void OnDisable()
    {
        Disconnect();
    }

    public void Disconnect()
    {
        DisconnectSocket(ref sendSocket, "Send");
        DisconnectSocket(ref receiveSocket, "Receive");
    }

    private void DisconnectSocket(ref Socket socket, string socketName)
    {
        if (socket != null)
        {
            try
            {
                if (socket.Connected)
                {
                    socket.Shutdown(SocketShutdown.Both);
                }
            }
            catch (Exception ex)
            {
                Debug.LogError($"[{nameof(DualClient)}] {socketName} socket exception on disconnect: {ex.Message}");
            }
            finally
            {
                socket.Close();
                socket = null;
                if (socketName == "Send") _isSendConnected = false;
                else _isReceiveConnected = false;
                Debug.Log($"[{nameof(DualClient)}] {socketName} connection closed");
            }
        }
    }
}