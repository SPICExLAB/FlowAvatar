# Checkpoints

All checkpoints are tracked in this repository: the benchmark checkpoints
reproduce the paper's numbers, and the deployment checkpoints run the streaming
and on-device demos right after cloning. All load with
`flowavatar.models.load_checkpoint` and `training.evaluate`. Benchmark files
carry their `model_params`, `source` experiment and `epoch`; deployment files
hold the state dict and `epoch` and take their architecture from the `gru` /
`lstm` preset.

```
checkpoints/
├── benchmark/                         # SMPL+H, 16 betas, unified decoder: paper models
│   ├── gru_smplh.pt                   # paper Table 2 / 3 "FlowAvatar" (all losses)
│   ├── gru_smplh_no_temporal.pt       # Table 3 ablation
│   ├── gru_smplh_no_hand.pt           # Table 3 ablation
│   ├── gru_smplh_no_ground.pt         # Table 3 ablation
│   └── lstm_smplh.pt                  # on-device architecture, benchmark setting: causal LSTM, 30 fps
├── GRU/baseline.pt                    # deployment: SMPL-X, 10 betas, streaming demo
└── LSTM/baseline.pt                   # deployment: SMPL-X, 10 betas, Unity Sentis export
```

## Benchmark checkpoints (AMASS protocol 2, SMPL+H, whole-sequence protocol)

Evaluate with

```bash
python -m training.evaluate --config training/configs/gru_smplh.yaml  --checkpoint checkpoints/benchmark/gru_smplh.pt
python -m training.evaluate --config training/configs/lstm_smplh.yaml --checkpoint checkpoints/benchmark/lstm_smplh.pt
```

These need the SMPL+H body models and the prepared AMASS test data
(`data/amass_smplh_60fps`, and `data/amass_smplh_30fps` for the LSTM); see
[training/README.md](../training/README.md), sections 1–2. The metrics are also written
next to the checkpoint (`<name>_eval_test.json`; `--output` to change).

| Checkpoint | Config | MPJRE (deg) | MPJPE (cm) | MPJVE (cm/s) | Jitter (m/s³) | Hand PE (cm) | Paper |
|------------|--------|------------:|-----------:|-------------:|--------------:|-------------:|-------|
| `gru_smplh.pt` | `gru_smplh.yaml` | 4.196 | 5.407 | 21.725 | 377.3 | 2.318 | Table 2: 4.19 / 5.40 / 21.72 / 3.77 / 2.31 |
| `gru_smplh_no_temporal.pt` | `ablations/no_temporal.yaml` | 4.555 | 6.083 | 25.588 | 517.4 | 2.506 | Table 3: 4.55 / 25.58 / 5.17 / 2.50 |
| `gru_smplh_no_hand.pt` | `ablations/no_hand.yaml` | 4.416 | 5.962 | 23.260 | 391.6 | 2.981 | Table 3: 4.41 / 23.26 / 3.91 / 2.98 |
| `gru_smplh_no_ground.pt` | `ablations/no_ground.yaml` | 4.313 | 5.558 | 22.551 | 405.3 | 2.256 | Table 3: 4.31 / 22.55 / 4.05 / 2.25 |
| `lstm_smplh.pt` | `lstm_smplh.yaml` (30 fps, T=20) | 4.299 | 5.905 | 25.419 | 199.4 | 2.340 | on-device (live demo) setting |

These are the outputs of `training.evaluate` on data prepared with
`training.prepare_data` (jitter in m/s³; the paper reports 10² m/s³ and truncates to
two decimals; its Table 3 lists MPJRE / MPJVE / Jitter / Hand PE). They
coincide with the research code's evaluation of the same files to the third
decimal. With the real-time protocol (`--window 40`, last frame of each
sliding window, as in the streaming demo) `gru_smplh.pt` gives MPJRE 4.239,
MPJPE 5.627, MPJVE 27.349 and jitter 788; `lstm_smplh.pt` with `--window 20`
gives 4.300 / 5.885 / 25.594 / 200.

## Deployment checkpoints (SMPL-X neutral, 10 betas)

The models of the streaming demo and the Unity client. They predict the
neutral SMPL-X body with 10 shape parameters that drives the Unity avatar;
`training/configs/gru_smplx.yaml` and `lstm_smplx.yaml` train models in this
format.

| Checkpoint | Architecture | Input window | Used by |
|------------|--------------|--------------|---------|
| `GRU/baseline.pt` | GRU | 60 fps, T = 40 | `python -m streaming.live_demo --model gru` |
| `LSTM/baseline.pt` | LSTM | 30 fps, T = 40 | Unity on-device (exported with `streaming/export_onnx.py` to `flowavatar_lstm.onnx`), `python -m streaming.live_demo --model lstm` |
