# Streaming protocol (Unity ⇄ Python)

The streaming configuration uses two raw TCP connections. The Python server
(`streaming/live_demo.py`) listens on both ports; the Unity client
(`DualClient`) connects to them:

| Port | Direction | Content |
|------|-----------|---------|
| 8888 | Unity → Python | Tracking features per frame |
| 8889 | Python → Unity | Predicted SMPL-X pose per inference |

All values are **little-endian**; all floats are 32-bit.

## Message framing

Every message is wrapped in ASCII markers:

```
<START> [header] [payload] <END>
```

`<START>` is 7 bytes and `<END>` 5 bytes, so a message is 402 bytes Unity → Python and
608 bytes Python → Unity. The Unity client may send zero bytes after `<END>` (its send
buffers are sized for 92 features), so a receiver should look for `<START>` rather than
read fixed-size records.

## Header

| Field | Type | Bytes | Notes |
|-------|------|-------|-------|
| MessageLength | uint32 | 4 | payload size in bytes |
| MessageType | uint8 | 1 | 1 = tracking input, 2 = prediction result |
| FrameRate | uint8 | 1 | Unity: send rate (**SendTrackingData > Target FPS**); Python: the model's `target_fps` in `configs/streaming.yaml` (0 = uncapped) |
| FrameCounter | uint16 | 2 | Unity: send counter (wraps at 65535); Python: the server's count of received frames for the newest input |
| Timestamp | float32 | 4 | Unity: `Time.realtimeSinceStartup` (s); Python: server time when the newest input arrived |
| DroppedFrames | uint32 | 4 | Unity: missed send intervals so far; Python: 0 |
| FeatureSize | uint16 | 2 | **Unity → Python only**: feature-vector length (90) |

The header is asymmetric: Unity → Python messages carry an 18-byte header
ending in FeatureSize (struct format `<IBBHfIH`); Python → Unity messages
carry a 16-byte header without it (struct format `<IBBHfI`).

## Tracking payload (Unity → Python): 93 floats

| Offset (floats) | Content |
|-----------------|---------|
| 0–89 | 90-D feature vector (below) |
| 90–92 | Head position (x, y, z) |

### The 90-D feature vector

Rotations are 6-D (first two columns of the rotation matrix). Velocities are
differences between consecutive frames at the send rate (Unity
`SendTrackingData > Target FPS`, 60 Hz by default); rotation velocities are
differences of the 6-D vectors.

| Indices | Content |
|---------|---------|
| 0–17 | Global rotations: head, left hand, right hand (6D each) |
| 18–35 | Rotation velocities: head, left hand, right hand (6D each) |
| 36–44 | Global positions: head, left hand, right hand (3D each) |
| 45–53 | Position velocities: head, left hand, right hand (3D each) |
| 54–65 | Head-relative hand rotations: left, right (6D each) |
| 66–77 | Head-relative hand rotation velocities: left, right (6D each) |
| 78–83 | Head-relative hand positions: left, right (3D each) |
| 84–89 | Head-relative hand position velocities: left, right (3D each) |

## Prediction payload (Python → Unity): 145 floats

| Offset (floats) | Content |
|-----------------|---------|
| 0–5 | Root rotation (6D) |
| 6–131 | Body joint rotations: 21 joints × 6D (SMPL-X joints 1–21) |
| 132–141 | Shape parameters β (10) |
| 142–144 | Head position echo (x, y, z) |
