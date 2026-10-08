# Training and evaluation

Training and benchmark evaluation of the FlowAvatar pose model on AMASS
(paper Sec. 3.1 and Sec. 5.1). The network definition is shared with the
streaming demo (`flowavatar/models`), so checkpoints trained with the SMPL-X
configs (`gru_smplx.yaml`, `lstm_smplx.yaml`) load directly into
`streaming/live_demo.py` and `streaming/export_onnx.py`.

| File | Content |
|------|---------|
| `prepare_data.py` | AMASS → per-sequence training samples (90-D sparse input, SMPL+H or SMPL-X targets) |
| `features.py` | Feature extraction shared by `prepare_data.py` (input layout documented there) |
| `dataset.py` | Random training windows with hand-dropout augmentation; whole sequences for evaluation |
| `model.py` | `FlowAvatarNetwork` + body-model (SMPL+H / SMPL-X) forward kinematics; checkpoint save/load |
| `losses.py` | Pose, temporal, hand-alignment and ground-contact losses (Eq. 9) |
| `metrics.py` | MPJRE, MPJPE, MPJVE, jitter, hand/upper/lower/root errors, floor metrics |
| `train.py`, `evaluate.py` | Entry points |
| `configs/` | `gru_smplh.yaml` / `lstm_smplh.yaml` (benchmark, SMPL+H), `gru_smplx.yaml` / `lstm_smplx.yaml` (deployment, SMPL-X), `ablations/` (Table 3) |

## 1. Setup

Run all commands in this file from the repository root.

```bash
pip install -r requirements.txt          # torch, numpy, pyyaml, tqdm, human_body_prior
pip install tensorboard                  # optional: per-epoch curves in the run directory
```

Two body models are involved, and it matters which one you use:

| | Benchmark (paper tables, prior work) | Deployment (streaming demo, Unity) |
|---|---|---|
| Body model | SMPL+H, gender-specific, **16** betas | SMPL-X neutral, **10** betas |
| Data configs | `gru_smplh.yaml`, `lstm_smplh.yaml`, `ablations/` | `gru_smplx.yaml`, `lstm_smplx.yaml` |
| Pose decoder | single MLP (`unified_predictor: true`) | per-region heads (released checkpoints) |
| Comparable with | AvatarPoser, AGRoL, AvatarJLM, HMD-Poser, paper Tables 2–4 | `checkpoints/GRU/baseline.pt`, `checkpoints/LSTM/baseline.pt` |

AvatarPoser, AGRoL and HMD-Poser all prepare AMASS with the gendered SMPL+H
models and evaluate with the neutral SMPL+H model; the FlowAvatar numbers in
the paper were produced the same way. The released streaming checkpoints
instead predict the 10 SMPL-X betas the Unity avatar needs. Numbers from the
two settings are **not** comparable with each other.

* **SMPL+H body models** (benchmark): download the *Extended SMPL+H model*
  from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) (the AMASS version
  with 16 shape parameters) and place `male/model.npz`, `female/model.npz`
  and `neutral/model.npz` under `body_models/smplh/`.
* **SMPL-X body model** (deployment): register at
  [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de), download the SMPL-X
  model in NPZ format (v1.1 or v1.0) from the
  [download page](https://smpl-x.is.tue.mpg.de/download.php) and place
  `SMPLX_NEUTRAL.npz` at `body_models/smplx/SMPLX_NEUTRAL.npz` (or set
  `SMPLX_MODEL_PATH`). `body_models/` is in the repository root
  (`FLOWAVATAR_BODY_MODELS_DIR` moves the SMPL+H folder).
* **AMASS**: register at [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de) and
  download the subsets below as the *SMPL+H G* packages (what prior work
  uses; the *SMPL-X N* packages are read as well). Extract them so that every
  subset is a folder directly under one directory, e.g. `data/AMASS/CMU/...`.
  AMASS cannot be redistributed.

The paper uses the HMD-Poser train/test split (*protocol 2*):

| Split | AMASS subsets (SMPL-X name / SMPL+H name) |
|-------|-------------------------------------------|
| train | ACCAD, BMLmovi, BMLrub (BioMotionLab_NTroje), CMU, EKUT, EyesJapanDataset (Eyes_Japan_Dataset), KIT, HDM05 (MPI_HDM05), PosePrior (MPI_Limits), MoSh (MPI_mosh), SFU, TotalCapture |
| test  | HumanEva, Transitions (Transitions_mocap) |

## 2. Prepare the data

```bash
# benchmark (SMPL+H): 60 fps / 40-frame windows for the GRU, 30 fps / 20-frame windows for the LSTM
python -m training.prepare_data --amass_dir data/AMASS --save_dir data/amass_smplh_60fps --fps 60
python -m training.prepare_data --amass_dir data/AMASS --save_dir data/amass_smplh_30fps --fps 30
# deployment (SMPL-X neutral, 10 betas)
python -m training.prepare_data --amass_dir data/AMASS --save_dir data/amass_smplx_60fps --fps 60 --body_model smplx
python -m training.prepare_data --amass_dir data/AMASS --save_dir data/amass_smplx_30fps --fps 30 --body_model smplx
```

Every motion is resampled to the target frame rate by an integer stride
(`round(source_fps / fps)`, as in the AvatarPoser / HMD-Poser preprocessing)
and replayed through the chosen body model (22 body joints; the sequence's
gender selects the SMPL+H model). Subsets recorded at 100 fps (KIT, EKUT)
therefore end up at 50 fps with `--fps 60` (33.3 fps with `--fps 30`) but are
treated as running at the target rate, the
same approximation prior work makes; the script reports these as "inexact
frame rate" and keeps the behaviour so that numbers stay comparable. For each
sequence the script stores the 90-D sparse input (head and wrist rotation,
rotation velocity, position and position velocity, plus the hands in the head
frame; see `features.py`), the local and global joint rotations (6D), the
world-space joint positions, the head transform, an estimated floor height
and foot-contact labels, and the body-model parameters. Output:
`<save_dir>/{train,test}/<subset>.pt` plus `meta.json`. Sequences shorter than
10 frames are dropped. The whole 60 fps set is about 20 GB and is loaded into
memory for training. A GPU (`--device cuda`, the default when available)
makes the body-model forward passes fast.

## 3. Train

```bash
python -m training.train --config training/configs/gru_smplh.yaml          # benchmark model
python -m training.train --config training/configs/lstm_smplh.yaml         # benchmark, on-device architecture
python -m training.train --config training/configs/ablations/no_temporal.yaml
python -m training.train --config training/configs/gru_smplx.yaml          # deployment model for the streaming demo
```

| Config | Model | Data | Body model | Notes |
|--------|-------|------|------------|-------|
| `gru_smplh.yaml` | 3-layer BiGRU, unified decoder | 60 fps, T = 40 | SMPL+H, 16 betas | paper Table 2 model |
| `lstm_smplh.yaml` | 1-layer unidirectional LSTM, ReLU, unified decoder | 30 fps, T = 20 | SMPL+H, 16 betas | on-device (live demo) setting, evaluated on the benchmark data |
| `ablations/no_temporal.yaml` | as `gru_smplh` | | | no velocity / acceleration loss (Table 3) |
| `ablations/no_hand.yaml` | as `gru_smplh` | | | no hand-alignment loss (Table 3) |
| `ablations/no_ground.yaml` | as `gru_smplh` | | | no foot-contact / penetration loss (Table 3) |
| `gru_smplx.yaml` | 3-layer BiGRU, per-region heads | 60 fps, T = 40 | SMPL-X, 10 betas | `checkpoints/GRU/baseline.pt` format |
| `lstm_smplx.yaml` | 1-layer unidirectional LSTM, ReLU, per-region heads | 30 fps, T = 20 | SMPL-X, 10 betas | `checkpoints/LSTM/baseline.pt` format (Sentis export) |

Configs inherit through a `base:` key; the `model` block is passed to
`flowavatar.models.FlowAvatarNetwork` (with `num_betas` from the body model) and the
`body_model` block selects the
model used for forward kinematics in the losses and metrics (neutral, as in
HMD-Poser). Training: Adam, batch size 256, 400 epochs, ten random windows
per sequence and epoch, L1 losses (`loss` block), and random hand dropout
(each window hides the left, the right or both hands with probability 0.2
each, over a random span of 10–19 frames for T = 40 and 9 frames for T = 20)
so the model stays usable when hands
leave the field of view. Shape parameters are supervised through the
joint-position terms (forward kinematics uses the predicted betas);
`loss.shape` adds a direct L1 term if wanted.

Each run writes to `experiments/<name>/<timestamp>/` (`<name>` is the config's
`name:`, e.g. `gru_smplh` or `ablation_no_temporal`):
`config.yaml`, `train.log`, `metrics.jsonl`, TensorBoard events (if
installed), `last.pt` (resumable with `--resume`), `best_train.pt` (lowest
training loss) and `best_val.pt` (lowest validation loss; validation runs every
`train.val_interval` epochs on the test split, as in HMD-Poser). Useful flags:
`--epochs`, `--out_dir`, `--device`, `--init <ckpt>` (warm start),
`--limit_batches` (debugging). All visible GPUs are used with `DataParallel`
(`train.multi_gpu: false` to disable). The main model takes roughly one day on
two A40 GPUs.

## 4. Evaluate

```bash
# benchmark protocol (whole test sequences in one pass), as in the paper tables;
# the released checkpoint, or your own run (experiments/gru_smplh/<run>/best_val.pt)
python -m training.evaluate --config training/configs/gru_smplh.yaml --checkpoint checkpoints/benchmark/gru_smplh.pt
# real-time protocol: sliding 40-frame windows, last frame of each window kept
python -m training.evaluate --config training/configs/gru_smplh.yaml --checkpoint checkpoints/benchmark/gru_smplh.pt --window 40
# robustness: hide one/both hands (no_left, no_right, no_hands), add tracker noise (m), repeat input frames
python -m training.evaluate --config training/configs/gru_smplh.yaml --checkpoint checkpoints/benchmark/gru_smplh.pt --input_mask no_left
python -m training.evaluate --config training/configs/gru_smplh.yaml --checkpoint checkpoints/benchmark/gru_smplh.pt --noise_sigma 0.03 --dropout_rate 0.1
```

The script prints a table and writes the metrics next to the checkpoint as
`<checkpoint name>_eval_<split>[_window<N>][_<mask>][_noise<σ>][_dropout<p>].json`
(`--output` to choose the file). Metrics are averaged over test sequences after
aligning the predicted head to the ground truth (all methods in Table 2 are
evaluated this way). The config must match the checkpoint's body model
(`training.evaluate` checks the number of betas for checkpoints that store
`model_params`). Checkpoints of the research code base load as well (the legacy
`temporal_layer` key names are mapped).

| Metric | Unit | Definition |
|--------|------|------------|
| `mpjre` | deg | mean absolute local joint-angle error (21 body joints, axis-angle) |
| `mpjpe`, `handpe`, `upperpe`, `lowerpe`, `rootpe` | cm | mean joint position error (all / wrists / upper body / lower body / pelvis) |
| `mpjve`, `upper_mpjve`, `lower_mpjve` | cm/s | mean joint velocity error |
| `jitter`, `upper_jitter`, `lower_jitter`, `gt_jitter` | m/s³ | mean norm of the third derivative of joint positions (`gt_jitter`: of the ground truth; paper tables report 10² m/s³, i.e. divide by 100) |
| `penetration`, `floating` | cm | mean depth of the lowest joint below / height above the estimated floor |
| `skating` | cm/frame | mean absolute per-axis foot displacement at frames where the ground-truth foot is static (other frames count as zero) |

## 5. Using a trained checkpoint

Checkpoints hold the bare `FlowAvatarNetwork` state dict plus `epoch`,
`model_params` and the training config, and load with
`flowavatar.models.load_checkpoint` (which reads `model_params`, so benchmark
and deployment checkpoints both load). The streaming demo and the Unity
avatar need the SMPL-X deployment models (`gru_smplx.yaml`, `lstm_smplx.yaml`):

* **Streaming demo:** set `models.gru.checkpoint` (or `models.lstm.checkpoint`) in
  `configs/streaming.yaml`, or in a copy passed with `--config`, to your run;
  relative paths are taken from `checkpoints/`, e.g.
  `../experiments/gru_smplx/<run>/best_val.pt`.
* **On-device (Unity Sentis):**

  ```bash
  pip install onnx onnxruntime
  python -m streaming.export_onnx --model_path experiments/lstm_smplx/<run>/best_val.pt
  ```

  then copy `converted_onnx/flowavatar_sentis_final.onnx` over
  `unity/Assets/Scripts/FlowAvatar/OnDevice/Models/flowavatar_lstm.onnx` (keep its `.meta`).

`python tests/training_smoke_test.py` runs the whole pipeline on synthetic
motions in a few CPU minutes (needs only the SMPL-X model; it prints `[SKIP]`
without it).

## 6. Reference results

The models behind paper Tables 2 and 3 are tracked in
[`checkpoints/benchmark/`](../checkpoints/README.md) and `training.evaluate`
reproduces their numbers (the paper truncates to two decimals, e.g. MPJRE
4.196 → 4.19). AMASS protocol 2, SMPL+H, 60 fps,
whole-sequence protocol (`gru_smplh.yaml`). Main model (paper Table 2):

| Model | MPJRE ↓ (deg) | MPJPE ↓ (cm) | MPJVE ↓ (cm/s) | Jitter ↓ (10² m/s³) |
|-------|--------------:|-------------:|---------------:|--------------------:|
| FlowAvatar GRU (paper) | 4.19 | 5.40 | 21.72 | 3.77 |

Loss ablations (paper Table 3; H-PE is the hand position error):

| Config | MPJRE ↓ | MPJVE ↓ | Jitter ↓ | H-PE ↓ (cm) |
|--------|--------:|--------:|---------:|------------:|
| `ablations/no_temporal.yaml` | 4.55 | 25.58 | 5.17 | 2.50 |
| `ablations/no_hand.yaml` | 4.41 | 23.26 | 3.91 | 2.98 |
| `ablations/no_ground.yaml` | 4.31 | 22.55 | 4.05 | 2.25 |
| `gru_smplh.yaml` (all losses) | 4.19 | 21.72 | 3.77 | 2.31 |

Under identical 60 fps / T = 40 training the LSTM backbone reaches MPJRE 4.43,
MPJPE 5.75 and MPJVE 22.95 (paper Table 4; that model is not included). `lstm_smplh.yaml` is the
on-device (live demo) setting: 30 fps data, 20-frame windows and a
unidirectional (causal) LSTM, evaluated on the same benchmark data.

Run-to-run variation is substantial: repeats of the `gru_smplh` setting differ
by several tenths of a centimetre in MPJPE (random windows, hand dropout and
non-deterministic cuDNN GRU kernels), so compare against the spread of
several runs rather than a single number.
