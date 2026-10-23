# YOLOv8-Seg ablation study

This repository contains the training script, aggregated measurements, and all run artifacts from the five YOLOv8-Seg experiments.

## Dataset

The Roboflow dataset is not included. Put a compatible `data.yaml` beside `ablation_study.py` and make sure the paths inside it resolve before running.

## Outputs

- `ablation_results.csv`: summary metrics for all five configurations.
- `ablation_runs/train/`: per-epoch logs, plots, and best/last checkpoints.
- `ablation_runs/validation/`: validation outputs for each trained model.
- `ablation_runs/artifacts/`: plots renamed by scenario.

The artifacts correspond to the baseline, 320-pixel input, batch size 8, 15 epochs, and YOLOv8s architecture configurations. Repeated 640/16/30 settings were represented by the baseline run.
