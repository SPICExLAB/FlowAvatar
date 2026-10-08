# FlowAvatar Unity client (Meta Quest)

Unity project for the FlowAvatar demo on Meta Quest (developed on Quest Pro): the headset
and the tracked hands drive a full-body SMPL-X avatar, face tracking drives its
expressions. One scene, two modes:

| Mode | Where the pose model runs | Enable |
|------|---------------------------|--------|
| **Streaming** (default) | Python server on a PC ([`streaming/`](../streaming)) | `FlowAvatar/Network` |
| **On-device** | Headset CPU (Unity Sentis, LSTM) | `FlowAvatar/OnDevice` |

Requirements: Unity **2022.3.60f1** with Android Build Support, and a Quest in developer
mode with **hand tracking enabled** (headset settings > Movement tracking; the setting
belongs to the signed-in account and also applies over Quest Link). Unity resolves the
packages (Meta XR SDK 71, URP 14, Unity Sentis 2.1.2) on first open.

## Setup

1. Open this folder in Unity Hub. The avatar shows as missing until step 2.
2. Download the **SMPL-X Unity project** (e.g. `SMPLX_UnityProject_20241205.zip`) from the
   [SMPL-X download page](https://smpl-x.is.tue.mpg.de/download.php) (registration
   required), unzip it, choose **FlowAvatar > Install SMPL-X Files...** and select the
   folder you unzipped it into ([details](Assets/SMPLX-OVR/README.md)).
3. Open `Assets/Scenes/FlowAvatar_Streaming.unity` and set up one of the modes below.
4. **File > Build Settings**: switch to Android, connect the Quest (allow USB debugging in
   the headset) and **Build And Run**.

## Streaming mode

1. On `FlowAvatar/Network`, set **DualClient > Server Ip** to the PC's LAN address
   (the default `127.0.0.1` only works when playing in the editor over Quest Link).
2. On the PC, start the server from the repository root
   ([Python setup](../README.md#streaming-demo)):

   ```bash
   python -m streaming.live_demo --model gru
   ```

   It listens on TCP 8888 and 8889 ([protocol](../docs/streaming_protocol.md)); allow
   Python through the firewall.
3. Start the app on the headset. It retries the connection every second; the server
   exits if no client connects within 30 seconds of loading the model, and after each
   session, so restart the server before restarting the app.

The GRU takes 60 Hz input, the default **SendTrackingData > Target FPS** on
`FlowAvatar/Network`. For `--model lstm` set it to 30, the rate the LSTM is fed on the
headset.

## On-device mode

Enable `FlowAvatar/OnDevice` and disable `FlowAvatar/Network`. The model is the LSTM
(`Assets/Scripts/FlowAvatar/OnDevice/Models/flowavatar_lstm.onnx`, exported from
`checkpoints/LSTM/baseline.pt`; [export](../README.md#on-device-mode)), fed at 30 Hz with a
40-frame window: **OnDeviceInput > Sequence Length** (40) must match the model's input
window, and **Target FPS** (30) is the rate at which the inputs are sampled.

A panel at the lower left of the view shows live statistics:

| Row | Meaning |
|-----|---------|
| App | rendered frames per second, and the display refresh rate |
| Input | input sampling rate and target; at most one sample per rendered frame |
| Model | inference time (mean, p95) and results per second |
| Latency | from sampling the newest input to applying its pose (mean, p95) |

Yellow marks a rate more than 10 % below its target, or a p95 inference time longer than
one input period. On `FlowAvatar/OnDevice`, **OnDeviceStatsPanel > Anchor** (a Transform)
places the panel there instead of in front of the head, and **Log To Console** also
prints the numbers every 5 s (`adb logcat -s Unity`).

After the first few seconds of start-up, the on-device mode runs at a steady 27 fps or
so on Quest Pro, with about 75 ms per model inference. The headset GPU is the limit, so
the Android quality level (`Assets/Settings/URP-Balanced*`) keeps SSAO off and shadows
within 5 m without soft shadows, and the mirror renders at 1024 × 576
(`Assets/Materials/Mirror.renderTexture`).

## Using the demo

- On first launch, allow the eye- and face-tracking permissions; facial expressions need
  Quest Pro face tracking.
- Put the controllers down. Tracked hands drive the avatar's wrists and fingers directly;
  while a hand is out of view, its predicted wrist rotation is used.
- Move freely during the first five seconds: the avatar holds its rest pose while the
  model estimates your body shape (betas), then applies it and follows you; expect a
  short hitch at that moment (**SMPLMotionController > Enable Apply Betas**, on by
  default, and **Beta Initialization Delay**).

## Personalised avatars (optional)

To use an avatar from the [avatar pipeline](../avatar-pipeline/README.md):

1. Import its `texture.png` into the project (e.g. `Assets/Avatar/Textures/`) and set it as
   the **Base Map** of `Assets/Avatar/Materials/FlowAvatarDefault.mat`.
2. On `FlowAvatar`, enter the ten values of its `betas.json` under
   **FlowAvatarSMPLxOVR > Shape (betas)** (then *Apply*). With
   **SMPLMotionController > Enable Apply Betas** on (the default), the shape the model
   estimates at start-up replaces them; untick it only to keep the `betas.json` shape.

`AvatarFetcher` (`Assets/Scripts/FlowAvatar/Avatar/`) can download both from a web
service instead; the repository does not include one, so it is for wiring up your own.

## Project layout

```
Assets/
├── Avatar/                 FlowAvatar.prefab (avatar with the Network / OnDevice modes), material, texture
├── Scenes/                 FlowAvatar_Streaming.unity (both modes)
├── SMPLX-OVR/              SMPL-X files you install (not in the repository)
└── Scripts/FlowAvatar/
    ├── Core/               FlowAvatarSMPLxOVR (shape, expressions, pose correctives), transforms
    ├── Input/              headset/hand/face tracking, VR-to-avatar mapping, input features
    ├── Output/             pose application (SMPLMotionController), foot placement
    ├── Streaming/          TCP client (DualClient) for the Python server
    ├── OnDevice/           Sentis inference on the headset, statistics panel
    ├── Avatar/             optional avatar download from your own web service (AvatarFetcher)
    └── Editor/             SMPL-X installer and the OVR hand-rig importer
```

### How the avatar is built

On import, `Editor/OvrHandRig/OvrHandRigPostprocessor.cs` converts the add-on's
`smplx-neutral.fbx`: the SMPL-X finger joints are replaced by the Meta OVR hand skeleton
described in `ovr_hand_rig.json` (bone names, hierarchy and rest orientations, positioned
relative to the SMPL-X finger joints), and each finger joint's skin weights move to the
OVR bone at the same joint, so hand tracking drives the fingers directly. Up to eight
bone influences per vertex are kept and the Android (Balanced) and PC (High Fidelity)
quality levels use unlimited skin weights, so the body deforms exactly like SMPL-X.

Third-party components are listed in [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
