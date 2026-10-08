# FlowAvatar avatar pipeline

Turns a short phone video of a person into a personalised avatar for the Unity client:
a 2048 × 2048 texture in the UV layout of the SMPL-X Unity avatar and the ten SMPL-X
shape parameters (betas). It runs offline on a Linux machine with an NVIDIA GPU.

| Step | What | Component |
|------|------|-----------|
| 1 | Video → frames (720 × 1280) | `scripts/extract_frames.py` |
| 2 | 2D keypoints | [mmpose](https://github.com/open-mmlab/mmpose) 0.28.1 (HRNet + Faster R-CNN) |
| 3 | Person masks | [Semantic Guided Human Matting](https://github.com/cxgincsu/SemanticGuidedHumanMatting) |
| 4 | SMPL-X pose, shape and camera per frame | [PyMAF-X](https://github.com/HongwenZhang/PyMAF-X) in [mmhuman3d](https://github.com/open-mmlab/mmhuman3d) 0.11.0 |
| 5 | Render of the initial fit | [HAHA](https://github.com/david-svitov/HAHA) (`configs/texture_init.yaml`) |
| 6 | Face alignment to the render, face / head / body masks | [STAR](https://github.com/ZhenglinZhou/STAR) landmarks on [dlib](http://dlib.net) face boxes, `scripts/align_faces.py` |
| 7 | Face, head and body textures | HAHA, differentiable rendering with [nvdiffrast](https://github.com/NVlabs/nvdiffrast) |
| 8 | Merge the textures, add the eyes | `scripts/blend_textures.py`, `assets/` |
| 9 | Upscale 512 → 2048 | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (ncnn-vulkan) |
| 10 | Betas averaged over the video | `scripts/export_betas.py` |

The third-party code is not part of this repository: `setup.sh` clones each project
at a pinned version, applies our changes from `patches/` (STAR is used unchanged) and
downloads the prebuilt Real-ESRGAN ncnn-vulkan release.

## Requirements

- Linux with an NVIDIA GPU and a driver that supports CUDA 12.1, plus Vulkan for the
  upscaler (`libvulkan1`).
- The CUDA 12 toolkit (`nvcc`): nvdiffrast compiles its CUDA plugin at the first run in
  `haha`. If it is not in `/usr/local/cuda` or on `PATH`, set `CUDA_HOME` for that
  environment only (`conda env config vars set -n haha CUDA_HOME=/usr/local/cuda-12.x`);
  `open-mmlab` builds against its own CUDA 11.7 compiler.
- `gcc`/`g++` (version 11 or older for the CUDA 11.7 builds of mmcv-full and pytorch3d in
  `open-mmlab`), `cmake` (pip builds dlib), conda, git, curl, bzip2, python3.
- About 23 GB of disk: 19 GB for the two environments, 2.5 GB for the code, weights and
  body models, 1.5 GB of models downloaded at the first run; plus about 0.8 GB per
  10-second video, and the conda / pip caches during the installation.

## Setup

Run the commands below from this folder (`cd avatar-pipeline` in a clone of the repository).

1. **Environments** (the versions the pipeline was developed with; creating `open-mmlab`
   compiles mmcv-full and takes a while):

   ```bash
   conda env create -f envs/open-mmlab.yml
   conda env create -f envs/haha.yml
   ```

   They target the GPUs we used (CUDA 12.1 for `haha`, 11.7 for `open-mmlab`). Newer
   GPUs need matching builds, e.g. an RTX 50-series card needs PyTorch 2.7+ with CUDA
   12.8; keep the other package versions and adjust PyTorch / CUDA accordingly. (Tested
   on an RTX 5090 with PyTorch 2.7.1: `open-mmlab` then needs Python 3.10,
   `colormap==1.1.1` and mmcv-full 1.5.3 built from source with `-std=c++17` instead of
   `-std=c++14` in its `setup.py`.)

2. **Code and public weights:**

   ```bash
   ./setup.sh
   ```

   This fills `third_party/` (HAHA, SemanticGuidedHumanMatting, STAR, mmpose,
   mmhuman3d, and the Real-ESRGAN binary in `realesrgan/`) and `weights/`, installs
   mmhuman3d and builds pytorch3d into `open-mmlab`, and downloads the PyMAF-X model and
   data (the OpenMMLab server can be slow; re-running `./setup.sh` resumes interrupted
   downloads). The matting and landmark weights are on Google Drive; if `gdown` cannot
   fetch them (setup.sh prints "Download failed" and continues), download
   [SGHM-ResNet50.pth](https://drive.google.com/drive/folders/15mGzPJQFEchaZHt9vgbmyOy46XxWtEOZ)
   to `weights/SGHM/` and
   [WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl](https://drive.google.com/file/d/1aOx0wYEZUfBndYy_8IYszLPG_D2fhxrT/view)
   to `weights/`, then run `./setup.sh` again.

3. **Body models.** They are licensed by the Max Planck Institute for Intelligent
   Systems and need a (free, research-only) registration. All three are required:

   | File | Download |
   |------|----------|
   | `SMPLX_NEUTRAL.npz` (SMPL-X v1.0 or v1.1, NPZ; identical in the 10 shape and 10 expression components used here) | [smpl-x.is.tue.mpg.de/download.php](https://smpl-x.is.tue.mpg.de/download.php) |
   | SMPL-X Unity project (e.g. `SMPLX_UnityProject_20241205.zip`, unzipped) | same page; also needed by the Unity client |
   | SMPL neutral model `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` (SMPL v1.1.0 for Python) | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) (used by PyMAF-X) |

   ```bash
   conda run -n haha python prepare_smplx.py \
       --smplx_model /path/to/models/smplx/SMPLX_NEUTRAL.npz \
       --unity_project /path/to/SMPLX_UnityProject_20241205 \
       --smpl_model /path/to/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl
   ```

   Besides copying the models (after checking that the SMPL-X model has the Unity
   avatar's mesh), this reads the UV layout of the Unity avatar from the add-on's
   `smplx-neutral.fbx` (`body_models/smplx/smplx-10.obj`; it differs from the UVs in the
   NPZ) and the add-on texture that initialises the fit.

## Capture

One person in a portrait video (phone, 9:16; other aspect ratios are stretched to
720 × 1280), the whole body in view: start facing the camera, then turn slowly on the
spot for a full turn, arms slightly away from the body. The person must be visible in
every frame (trim the start and end), or step 2 stops. Even lighting and a plain
background help the masks.

## Run

```bash
./run_avatar.sh /path/to/session/video.mp4
```

Put each video in its own folder: frames, masks, fits and the intermediate textures
are written next to it. The results are `texture.png` and `betas.json` in that folder.
A run took about 14 minutes on an RTX 5090. It uses the first GPU; choose another with
`CUDA_VISIBLE_DEVICES`.

To use them in the Unity client, see
[Personalised avatars](../unity/README.md#personalised-avatars-optional): `texture.png`
becomes the Base Map of `Assets/Avatar/Materials/FlowAvatarDefault.mat`, the ten values
of `betas.json` go into **FlowAvatarSMPLxOVR > Shape (betas)**.

## Our changes to the third-party code

| Patch | Changes |
|-------|---------|
| `patches/HAHA.patch` | Texture-only fitting (no Gaussians): `--vid_path` (inputs and outputs next to the video), non-square renders, per-frame SMPL-X parameters and cameras from PyMAF-X, face / head / body masks, the Unity UV layout and the add-on texture as the initial texture, SMPL-X pose refinement and a decaying learning rate during the texture stage, saving `texture.png`; the learning-rate scheduler without the `verbose` argument (removed in PyTorch 2.7) |
| `patches/mmhuman3d.patch` | PyMAF-X demo: single person, inputs and outputs next to the video, perspective cameras and stacked per-frame parameters for HAHA, `betas.npy` |
| `patches/mmpose.patch` | Keypoint demo: all frames in `images/` next to the video, first detected person per frame (box threshold 0.5), saved as `keypoints_2d.npy` |
| `patches/SemanticGuidedHumanMatting.patch` | Binary masks (alpha ≥ 0.6) |

The HAHA texture configurations are in `configs/`; STAR is used unchanged, as a
library, by `scripts/align_faces.py`.

## Licenses

The code in this folder is under the repository's license, except `patches/`, which
modify the third-party projects and are provided under their licenses. The third-party
projects keep their own licenses and are downloaded, not redistributed:

| Project | License |
|---------|---------|
| HAHA | BSD-3-Clause |
| Semantic Guided Human Matting | MIT |
| mmpose, mmhuman3d | Apache-2.0 |
| pytorch3d | BSD-3-Clause |
| nvdiffrast | NVIDIA Source Code License (non-commercial) |
| Real-ESRGAN | BSD-3-Clause (ncnn-vulkan build: MIT) |
| STAR | no license file in the repository; cloned by `setup.sh`, not redistributed |
| SMPL, SMPL-X, dlib's 68-point face model, PyMAF-X model and data | research licenses of their authors |
