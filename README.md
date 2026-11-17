# Fixed RGB-D Dinomaly2 Continual Anomaly Detection

This package contains the fixed-fusion RGB-D continual anomaly-detection
method, its tested Python environment, and the processed MVTec3D and
EyeCandies datasets.

The server source directory retains the previously verified checkpoints. The
distributed archive named
`Final-Continual-RGBD-AD-with-data-no-checkpoints.zip` intentionally excludes
the entire `method/checkpoints/` directory. It contains no trained checkpoint
weights and is intended for training from scratch.

No exploratory dynamic-fusion, alternative-head or IBFM implementation is
included.

## Method

- Input: aligned RGB and XYZ. The dataloader renders XYZ online as a replicated
  three-channel depth image.
- Frozen DINOv2 ViT-B/14 RGB/depth encoder.
- Fixed feature fusion: `0.5 RGB + 0.5 depth` at every fused encoder group.
- One shared Dinomaly2 bottleneck/decoder and one fixed ten-output classifier.
- Feature jitter and category-decoupling classification loss during training.
- Replay-free joint CFNS+DSNS consolidation during incremental stages.
- Only compact aggregate null-space bases are stored; no old sample or
  per-example feature is retained.

## Data

Both processed datasets are included:

```text
<package-root>/data/MVTec3D-processed/
<package-root>/data/EyeCandies-processed/
```

The continual scripts use MVTec3D. EyeCandies is included for separate
experiments but is not used by these MVTec3D continual settings.

## Category order and settings

Fixed category order:

```text
bagel, cable_gland, carrot, cookie, dowel, foam,
peach, potato, rope, tire
```

All settings use the same model, optimizer, fusion ratio and losses. Only the
class partition changes:

| Setting | Stage 0 | Incremental stage(s) |
|---|---|---|
| `10-0` | all ten categories | none |
| `9-1` | first nine categories | `tire` |
| `6-4` | first six categories | remaining four together |
| `6-1x4` | first six categories | `peach`, `potato`, `rope`, `tire` one by one |

`6-1x4` is the setting previously trained and verified. The other settings are
runnable definitions; no unrun results are claimed.

## Previously verified `6-1x4` result

The retained source checkpoint produced these stage-4 ten-category macro
means. Checkpoint weights are deliberately not included in this ZIP.

| I-AUROC | P-AUROC | AUPRO | Classification accuracy |
|---:|---:|---:|---:|
| 78.56 | 98.17 | 94.00 | 36.35 |

## Train from scratch

After extracting the ZIP, run from the extracted package root:

```bash
bash method/run_settings.sh 10-0
bash method/run_settings.sh 9-1
bash method/run_settings.sh 6-4
bash method/run_settings.sh 6-1x4
```

Run all settings sequentially:

```bash
bash method/run_settings.sh all
```

Print all commands without creating output or training:

```bash
DRY_RUN=1 bash method/run_settings.sh all
```

Compatibility entry point for `6-1x4`:

```bash
bash method/run_6_1x4.sh
```

Fresh outputs go to `method/runs/<setting>_seed1/`. The launcher refuses to
append to an existing run.

## Monitor

Example for `6-4`:

```bash
tail -f method/runs/6-4_seed1/train.log
cat method/runs/6-4_seed1/metrics.csv
ls -lh method/runs/6-4_seed1/stage_*_checkpoint.pth
nvidia-smi
```

`train.log` records losses, jitter ratio and CFNS/DSNS residuals.
`metrics.csv` is appended after each completed stage with per-class and macro
I-AUROC, P-AUROC, AUPRO and classification accuracy.

## Fixed training configuration

| Item | Value |
|---|---:|
| Stage-0 iterations | 40,000 |
| Incremental-stage iterations | 5,000 |
| Image / crop / metric size | 448 / 392 / 256 |
| Batch size | 16 |
| Learning rate | 0.002 |
| RGB / depth fusion | 0.5 / 0.5 |
| Feature jitter scale / probability | 20 / 1.0 |
| CFNS energy / maximum rank | 0.995 / 128 |
| DSNS energy / maximum rank | 0.97 / 32 |
| Seed | 1 |

## ZIP contents

```text
Final-Continual-RGBD-AD-with-data/
├── README.md
├── method/
│   ├── code/
│   ├── env/
│   ├── run_settings.sh
│   ├── run_6_1x4.sh
│   └── PACKAGE_AUDIT.txt
└── data/
    ├── MVTec3D-processed/
    └── EyeCandies-processed/
```

`method/checkpoints/` is excluded from the ZIP by design.