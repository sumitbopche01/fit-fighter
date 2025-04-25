using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;
using Newtonsoft.Json;

/// <summary>
/// Receives motion detection data from the Python application via WebSocket
/// </summary>
public class MotionDataReceiver : MonoBehaviour
{
    [Header("WebSocket Settings")]
    [SerializeField] private string serverUrl = "ws://127.0.0.1:5678";
    [SerializeField] private float reconnectDelay = 3f;
    
    [Header("Debug")]
    [SerializeField] private bool showDebugLogs = true;
    [SerializeField] private GameObject connectionStatusUI;
    [SerializeField] private TMPro.TextMeshProUGUI statusText;
    
    private WebSocket websocket;
    private bool isConnected = false;
    private bool isReconnecting = false;
    
    // Motion data received from Python
    public MotionData CurrentMotionData { get; private set; }
    
    // Event fired when motion data is received
    public event Action<MotionData> OnMotionDataReceived;

    private void Start()
    {
        CurrentMotionData = new MotionData();
        SetupWebSocket();
    }

    private async void SetupWebSocket()
    {
        websocket = new WebSocket(serverUrl);

        websocket.OnOpen += () =>
        {
            Debug.Log("Connection open!");
            isConnected = true;
            UpdateConnectionStatus("Connected", Color.green);
        };

        websocket.OnError += (e) =>
        {
            Debug.LogError("WebSocket Error: " + e);
            UpdateConnectionStatus("Error: " + e, Color.red);
        };

        websocket.OnClose += (e) =>
        {
            Debug.Log("Connection closed!");
            isConnected = false;
            UpdateConnectionStatus("Disconnected", Color.red);
            
            // Try to reconnect
            if (!isReconnecting)
            {
                StartCoroutine(ReconnectAfterDelay());
            }
        };

        websocket.OnMessage += (bytes) =>
        {
            var message = System.Text.Encoding.UTF8.GetString(bytes);
            if (showDebugLogs)
            {
                Debug.Log("OnMessage: " + message);
            }
            
            try
            {
                // Parse the JSON message
                CurrentMotionData = JsonConvert.DeserializeObject<MotionData>(message);
                
                // Fire event to notify subscribers
                OnMotionDataReceived?.Invoke(CurrentMotionData);
            }
            catch (Exception e)
            {
                Debug.LogError("Failed to parse message: " + e.Message);
            }
        };

        // Connect to the server
        await websocket.Connect();
    }

    private void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (websocket != null)
        {
            websocket.DispatchMessageQueue();
        }
#endif
    }

    private IEnumerator ReconnectAfterDelay()
    {
        isReconnecting = true;
        UpdateConnectionStatus("Reconnecting in " + reconnectDelay + "s...", Color.yellow);
        
        yield return new WaitForSeconds(reconnectDelay);
        
        if (websocket != null)
        {
            websocket.Close();
        }
        
        SetupWebSocket();
        isReconnecting = false;
    }

    private void UpdateConnectionStatus(string message, Color color)
    {
        if (connectionStatusUI != null)
        {
            connectionStatusUI.SetActive(true);
            
            if (statusText != null)
            {
                statusText.text = message;
                statusText.color = color;
            }
        }
    }

    private async void OnDestroy()
    {
        if (websocket != null)
        {
            await websocket.Close();
        }
    }

    private void OnApplicationQuit()
    {
        if (websocket != null)
        {
            websocket.Close();
        }
    }
}

/// <summary>
/// Data structure for motion detection data received from Python
/// </summary>
[System.Serializable]
public class MotionData
{
    public long timestamp;
    public ExerciseStates exercises;
    public List<Landmark> landmarks;
    public PerformanceMetrics performance;
    
    public MotionData()
    {
        exercises = new ExerciseStates();
        landmarks = new List<Landmark>();
        performance = new PerformanceMetrics();
    }
}

[System.Serializable]
public class ExerciseStates
{
    public bool punch;
    public bool squat;
    public bool plank;
}

[System.Serializable]
public class Landmark
{
    public string name;
    public float x;
    public float y;
    public float z;
    public float confidence;
}

[System.Serializable]
public class PerformanceMetrics
{
    public float fps;
    public int latency_ms;
} 