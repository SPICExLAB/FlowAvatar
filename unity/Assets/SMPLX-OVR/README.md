# SMPLX-OVR: SMPL-X files you provide

The FlowAvatar avatar is the SMPL-X neutral body from the official **SMPL-X Unity
add-on**, with its hands re-rigged to the Meta OVR hand skeleton. The SMPL-X files are
licensed by the Max Planck Institute for Intelligent Systems for non-commercial
research and cannot be redistributed, so this folder ships empty: download them
yourself and install them with the FlowAvatar menu.

1. Register at <https://smpl-x.is.tue.mpg.de>, accept the SMPL-X license, and
   download the **SMPL-X Unity project** from the
   [download page](https://smpl-x.is.tue.mpg.de/download.php) (e.g.
   `SMPLX_UnityProject_20241205.zip`). Unzip it anywhere; do not open it in this project.
2. In Unity choose **FlowAvatar > Install SMPL-X Files...** and select the unzipped
   add-on folder (the tool finds `SMPLX-Unity/Assets/SMPLX` inside it). It copies:

   | From the add-on (`Assets/SMPLX/`) | To |
   |------|-------------|
   | `Models/smplx-neutral.fbx` | `Assets/SMPLX-OVR/smplx-neutral.fbx` |
   | `Resources/smplx_betas_to_joints_female.json`, `_male.json`, `_neutral.json` | `Assets/SMPLX-OVR/Resources/` |
   | `Textures/smplx_texture_f_alb.png`, `smplx_texture_m_alb.png` (optional) | `Assets/SMPLX-OVR/Textures/` |

   Each file is written with a fixed `.meta`, so `smplx-neutral.fbx` gets the GUID and
   import settings `Assets/Avatar/FlowAvatar.prefab` is built on. Copying the FBX by hand
   gives it a new GUID and the avatar stays missing, so use the menu.
   Do not import the add-on's whole `SMPLX` folder: its scripts and its copy of
   SimpleJSON do not compile next to FlowAvatar's.
3. On import, `OvrHandRigPostprocessor` re-rigs the hands of `smplx-neutral.fbx` to
   the Meta OVR hand skeleton (see
   [How the avatar is built](../../README.md#how-the-avatar-is-built)). The console
   prints `[OvrHandRig] smplx-neutral.fbx: ... 46 OVR bones, 72 bones ...` and
   `Assets/Avatar/FlowAvatar.prefab` becomes available; reopen the scene if it was open.

Do not commit the SMPL-X files (they are listed in `.gitignore`).
