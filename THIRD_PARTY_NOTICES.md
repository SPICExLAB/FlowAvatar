# Third-party notices

## Code

### AvatarPoser (ECCV 2022)

The rotation/transformation utilities in `flowavatar/utils/transforms.py`
derive from [eth-siplab/AvatarPoser](https://github.com/eth-siplab/AvatarPoser).

MIT License — Copyright (c) 2022 ETH Sensing, Interaction & Perception Lab.

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

Citation: Jiang et al., "AvatarPoser: Articulated Full-Body Pose Tracking from
Sparse Motion Sensing", ECCV 2022.

### HMD-Poser (CVPR 2024)

The training pipeline in `training/` follows the structure of
[pico-ai-team/HMD-Poser](https://github.com/pico-ai-team/HMD-Poser): the
AMASS *protocol 2* split (twelve training subsets, HumanEva and Transitions
for testing), the whole-sequence evaluation protocol, and the learning-rate
schedule in `training/utils.py`.

MIT License — Copyright (c) 2024 PICO, ByteDance.

Citation: Dai et al., "HMD-Poser: On-Device Real-time Human Motion Tracking
from Scalable Sparse Observations", CVPR 2024.

### Metrics and losses

The benchmark metrics in `training/metrics.py` (MPJRE, MPJPE, MPJVE, jitter,
hand / upper / lower / root position errors) follow the definitions of
AvatarPoser (MIT, ETH Zurich) and AGRoL (Meta Platforms, Inc.). The
multi-scale velocity, foot-contact and floor-penetration losses in
`training/losses.py` follow AvatarJLM (Zheng et al., ICCV 2023). All of this
code was re-implemented for FlowAvatar.

## Unity client (`unity/`)

### EMAGE face mapping

`unity/Assets/Scripts/FlowAvatar/Input/Face/arkit_retarget_a2e_v10.json` (Meta
face-tracking expressions to ARKit blendshapes) and `mat_final.json` (the
"ARKit2FLAME" weights, ARKit blendshapes to FLAME expression and jaw parameters)
come from EMAGE / PantoMatrix: https://pantomatrix.github.io/EMAGE/,
https://github.com/PantoMatrix/PantoMatrix (released on Hugging Face under
Apache-2.0). `SMPLXFaceDriver.cs` uses the FLAME jaw columns of `mat_final.json`
for the jaw, and its SMPL-X expression coefficients were tuned with reference to
the same matrix.

Citation: Liu et al., "EMAGE: Towards Unified Holistic Co-Speech Gesture
Generation via Expressive Masked Audio Gesture Modeling", CVPR 2024.

### SimpleJSON

`unity/Assets/ThirdParty/JSON/SimpleJSON.cs` — MIT License, Copyright (c)
2012-2019 Markus Göbel (Bunny83). The full license text is in the file header.

### Unity packages (not redistributed; resolved by the Unity Package Manager)

- **Meta XR SDK** (`com.meta.xr.sdk.all`) — Oculus SDK License Agreement:
  https://developer.oculus.com/licenses/oculussdk/. The files Unity generates
  from it in the project (`Assets/Oculus`, `Assets/Resources`, `Assets/XR`,
  `Assets/Plugins/Android/AndroidManifest.xml`) are configuration assets.
- **Unity Sentis**, **Universal Render Pipeline**, **TextMesh Pro** (whose
  Essential Resources are included under `Assets/TextMesh Pro`) — Unity
  Companion License. The bundled LiberationSans font is under the SIL Open Font
  License (`Assets/TextMesh Pro/Fonts/LiberationSans - OFL.txt`).

## Avatar pipeline (`avatar-pipeline/`)

`setup.sh` and the conda environments in `envs/` download the following projects at
pinned versions; they are not redistributed here. The files in `avatar-pipeline/patches/` modify HAHA, Semantic
Guided Human Matting, mmpose and mmhuman3d and are provided under the licenses of
those projects.

- **HAHA** — https://github.com/david-svitov/HAHA, BSD-3-Clause. Svitov et al.,
  "HAHA: Highly Articulated Gaussian Human Avatars with Textured Mesh Prior", ACCV 2024.
- **Semantic Guided Human Matting** — https://github.com/cxgincsu/SemanticGuidedHumanMatting,
  MIT. Chen et al., "Robust Human Matting via Semantic Guidance", ACCV 2022.
- **mmpose** and **mmhuman3d** (including its PyMAF-X implementation) — OpenMMLab,
  Apache-2.0. Zhang et al., "PyMAF-X: Towards Well-aligned Full-body Model Regression
  from Monocular Images", TPAMI 2023.
- **STAR** — https://github.com/ZhenglinZhou/STAR (no license file). Used unmodified as a
  library by `avatar-pipeline/scripts/align_faces.py`. Zhou et al., "STAR Loss: Reducing
  Semantic Ambiguity in Facial Landmark Detection", CVPR 2023.
- **nvdiffrast** — NVIDIA Source Code License (non-commercial use).
- **pytorch3d** — BSD-3-Clause.
- **Real-ESRGAN** — https://github.com/xinntao/Real-ESRGAN, BSD-3-Clause (ncnn-vulkan
  build: MIT). Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
  with Pure Synthetic Data", ICCVW 2021.
- **dlib** 68-point face landmark model, **PyMAF-X** model and data — downloaded from
  their authors under their terms.

`avatar-pipeline/assets/` (eye texture and blend masks) were made for FlowAvatar.

## Separately licensed dependencies (NOT redistributed here)

- **SMPL-X body model** — Max Planck Institute for Intelligent Systems,
  research license, registration required: https://smpl-x.is.tue.mpg.de.
  The Unity client uses `smplx-neutral.fbx`, the betas-to-joints regressors and
  textures of the **SMPL-X Unity add-on**, which each user installs (see
  `unity/Assets/SMPLX-OVR/README.md`); `unity/.../OvrHandRig/ovr_hand_rig.json`
  holds only FlowAvatar's OVR hand-bone definitions, no SMPL-X data. The avatar
  pipeline uses the SMPL-X NPZ model and reads the UV layout and a texture from the
  same add-on, which each user provides (`avatar-pipeline/prepare_smplx.py`).
- **SMPL body model** (neutral, from https://smpl.is.tue.mpg.de) — MPI-IS, research
  license; used by PyMAF-X in the avatar pipeline, provided by each user.
- **AMASS dataset** — MPI-IS, academic license: https://amass.is.tue.mpg.de
  (used to train the released checkpoints; not distributed)
- **human_body_prior** — MPI-IS, non-commercial research license:
  https://github.com/nghorbani/human_body_prior (installed via pip)
