using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using TMPro;

/// <summary>
/// World-space panel with live on-device statistics: application frame rate, input
/// sampling rate, model inference time and rate, and latency from input sampling to
/// pose application.
/// Builds its own canvas at runtime and follows the head, or sits at an anchor.
/// </summary>
public class OnDeviceStatsPanel : MonoBehaviour
{
    [Header("Sources (default: components on this GameObject)")]
    [SerializeField] private OnDeviceInput input;
    [SerializeField] private AvatarInferenceCPU inference;
    [SerializeField] private OnDeviceOutput output;

    [Header("Placement")]
    [Tooltip("Optional fixed transform for the panel; when empty the panel follows the head.")]
    [SerializeField] private Transform anchor;
    [Tooltip("Panel position relative to the head (x right, y up, z forward), turning with the head's yaw only.")]
    [SerializeField] private Vector3 headOffset = new Vector3(-0.15f, -0.2f, 0.9f);
    [Tooltip("How quickly the panel catches up with the head (1/s).")]
    [SerializeField] private float followSpeed = 3f;

    [Header("Update")]
    [SerializeField] private float refreshInterval = 0.5f;
    [Tooltip("Also print the statistics to the console (adb logcat -s Unity) every few seconds.")]
    [SerializeField] private bool logToConsole = false;
    [SerializeField] private float logInterval = 5f;

    private const string Warn = "#FFC857";
    private const float PanelWidth = 440f, PanelHeight = 170f, MetersPerUnit = 0.001f;

    private RectTransform panel;
    private TextMeshProUGUI text;
    private Transform head;
    private bool snapToHead = true;
    private readonly StringBuilder sb = new StringBuilder(256);

    private float lastRefresh, lastLog;
    private int lastFrames, lastInferences, appFrames;
    private float inputHz, modelHz, appFps, displayHz;
    private static readonly List<XRDisplaySubsystem> displays = new List<XRDisplaySubsystem>();

    void Awake()
    {
        if (input == null) input = GetComponent<OnDeviceInput>();
        if (inference == null) inference = GetComponent<AvatarInferenceCPU>();
        if (output == null) output = GetComponent<OnDeviceOutput>();
        BuildPanel();
    }

    void OnEnable()
    {
        if (panel != null) panel.gameObject.SetActive(true);
        snapToHead = true;
        lastRefresh = lastLog = Time.realtimeSinceStartup;
        lastFrames = input != null ? input.FramesSampled : 0;
        lastInferences = inference != null ? inference.InferencesCompleted : 0;
        appFrames = 0;
    }

    void OnDisable()
    {
        if (panel != null) panel.gameObject.SetActive(false);
    }

    void OnDestroy()
    {
        if (panel != null) Destroy(panel.gameObject);
    }

    void Update()
    {
        appFrames++;
        float now = Time.realtimeSinceStartup;
        if (now - lastRefresh < refreshInterval) return;

        float elapsed = now - lastRefresh;
        int frames = input != null ? input.FramesSampled : 0;
        int inferences = inference != null ? inference.InferencesCompleted : 0;
        inputHz = (frames - lastFrames) / elapsed;
        modelHz = (inferences - lastInferences) / elapsed;
        appFps = appFrames / elapsed;
        appFrames = 0;
        displayHz = DisplayRefreshRate();
        lastFrames = frames;
        lastInferences = inferences;
        lastRefresh = now;

        text.text = BuildText();

        if (logToConsole && now - lastLog >= logInterval)
        {
            lastLog = now;
            Debug.Log("[OnDeviceStats] " + BuildLogLine());
        }
    }

    void LateUpdate()
    {
        if (anchor != null)
        {
            panel.SetPositionAndRotation(anchor.position, anchor.rotation);
            return;
        }

        if (head == null)
        {
            Camera cam = Camera.main;
            if (cam == null) return;
            head = cam.transform;
        }

        // Follow the head's position and yaw, not its pitch and roll
        Vector3 forward = Vector3.ProjectOnPlane(head.forward, Vector3.up);
        if (forward.sqrMagnitude < 1e-4f) forward = Vector3.ProjectOnPlane(head.up, Vector3.up);
        Vector3 target = head.position + Quaternion.LookRotation(forward.normalized, Vector3.up) * headOffset;

        float t = snapToHead ? 1f : 1f - Mathf.Exp(-followSpeed * Time.deltaTime);
        snapToHead = false;
        Vector3 position = Vector3.Lerp(panel.position, target, t);
        panel.SetPositionAndRotation(position, Quaternion.LookRotation(position - head.position, Vector3.up));
    }

    private string BuildText()
    {
        sb.Clear();
        int target = input != null ? input.TargetFPS : 0;
        int window = input != null ? input.SequenceLength : 0;

        sb.Append("<b>On-device</b>  <size=80%>").Append(inference != null ? inference.ModelName : "")
          .Append("  T=").Append(window).Append("</size>\n");

        // App: frames actually rendered per second; input is sampled at most once per frame
        sb.Append("App<pos=30%>");
        AppendColored(appFps.ToString("F0") + " fps", displayHz > 0f && appFps < 0.9f * displayHz);
        if (displayHz > 0f) sb.Append("<pos=62%><size=80%>display ").Append(displayHz.ToString("F0")).Append(" Hz</size>");
        sb.Append('\n');

        // Input: the model expects features at exactly the rate it was trained at
        sb.Append("Input<pos=30%>");
        AppendColored(inputHz.ToString("F1") + " Hz", inputHz < 0.9f * target);
        sb.Append("<pos=62%><size=80%>target ").Append(target).Append(" Hz</size>\n");

        // Model: inference time and how many results per second reach the avatar
        sb.Append("Model<pos=30%>");
        if (inference == null || inference.InferenceMs.Count == 0)
        {
            int buffered = input != null ? Mathf.Min(input.BufferedFrames, window) : 0;
            sb.Append("<size=80%>filling window ").Append(buffered).Append('/').Append(window).Append("</size>\n");
        }
        else
        {
            float p95 = inference.InferenceMs.Percentile(0.95f);
            // Slower than the input period means inference, not input, limits the update rate
            AppendColored(inference.InferenceMs.Mean().ToString("F1") + " ms", target > 0 && p95 > 1000f / target);
            sb.Append("<pos=62%><size=80%>p95 ").Append(p95.ToString("F1")).Append(" · ")
              .Append(modelHz.ToString("F1")).Append(" Hz</size>\n");
        }

        // Latency: newest input frame sampled -> its pose applied to the avatar
        sb.Append("Latency<pos=30%>");
        if (output == null || output.LatencyMs.Count == 0)
        {
            sb.Append("<size=80%>-</size>");
        }
        else
        {
            sb.Append(output.LatencyMs.Mean().ToString("F0")).Append(" ms<pos=62%><size=80%>p95 ")
              .Append(output.LatencyMs.Percentile(0.95f).ToString("F0")).Append("</size>");
        }
        return sb.ToString();
    }

    private string BuildLogLine()
    {
        string model = inference != null && inference.InferenceMs.Count > 0
            ? $"{inference.InferenceMs.Mean():F1} ms (p95 {inference.InferenceMs.Percentile(0.95f):F1}), {modelHz:F1} Hz"
            : "no results yet";
        string latency = output != null && output.LatencyMs.Count > 0
            ? $"{output.LatencyMs.Mean():F0} ms (p95 {output.LatencyMs.Percentile(0.95f):F0})"
            : "-";
        return $"app {appFps:F0} fps (display {displayHz:F0} Hz), input {inputHz:F1} Hz (target {(input != null ? input.TargetFPS : 0)}), model {model}, latency {latency}";
    }

    private static float DisplayRefreshRate()
    {
        SubsystemManager.GetSubsystems(displays);
        foreach (var display in displays)
        {
            if (display.running && display.TryGetDisplayRefreshRate(out float hz) && hz > 0f) return hz;
        }
        return 0f;
    }

    private void AppendColored(string value, bool warn)
    {
        if (warn) sb.Append("<color=").Append(Warn).Append('>').Append(value).Append("</color>");
        else sb.Append(value);
    }

    private void BuildPanel()
    {
        var root = new GameObject("OnDevice Stats", typeof(RectTransform), typeof(Canvas));
        root.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
        panel = (RectTransform)root.transform;
        panel.sizeDelta = new Vector2(PanelWidth, PanelHeight);
        panel.localScale = Vector3.one * MetersPerUnit;

        var background = new GameObject("Background", typeof(RectTransform), typeof(Image));
        background.transform.SetParent(panel, false);
        Stretch((RectTransform)background.transform, 0f);
        background.GetComponent<Image>().color = new Color(0f, 0f, 0f, 0.65f);

        var label = new GameObject("Text", typeof(RectTransform), typeof(TextMeshProUGUI));
        label.transform.SetParent(panel, false);
        Stretch((RectTransform)label.transform, 14f);
        text = label.GetComponent<TextMeshProUGUI>();
        text.fontSize = 24f;
        text.color = Color.white;
        text.alignment = TextAlignmentOptions.TopLeft;
        text.enableWordWrapping = false;
        text.raycastTarget = false;
        text.text = "";

        // UI layer: rendered by the headset camera but not by the demo scene's mirror camera
        int uiLayer = LayerMask.NameToLayer("UI");
        if (uiLayer >= 0)
        {
            foreach (var t in root.GetComponentsInChildren<Transform>(true)) t.gameObject.layer = uiLayer;
        }
    }

    private static void Stretch(RectTransform rect, float padding)
    {
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.offsetMin = new Vector2(padding, padding);
        rect.offsetMax = new Vector2(-padding, -padding);
    }
}

/// <summary>Ring buffer of the most recent samples, with mean and percentile.</summary>
public class RollingSamples
{
    private readonly float[] samples;
    private readonly float[] sorted;
    private int count, next;

    public RollingSamples(int capacity)
    {
        samples = new float[capacity];
        sorted = new float[capacity];
    }

    public int Count => count;

    public void Add(float value)
    {
        samples[next] = value;
        next = (next + 1) % samples.Length;
        if (count < samples.Length) count++;
    }

    public float Mean()
    {
        if (count == 0) return float.NaN;
        float sum = 0f;
        for (int i = 0; i < count; i++) sum += samples[i];
        return sum / count;
    }

    public float Percentile(float p)
    {
        if (count == 0) return float.NaN;
        Array.Copy(samples, sorted, count);
        Array.Sort(sorted, 0, count);
        return sorted[Mathf.Clamp(Mathf.CeilToInt(p * count) - 1, 0, count - 1)];
    }
}
