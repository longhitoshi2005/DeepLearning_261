"""YOLOv8-seg ablation runner for a Windows/Linux GPU workstation.

Install first: pip install -U ultralytics pandas
Place data.yaml beside this script and ensure its dataset paths resolve.
"""

from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


ROOT_DIR = Path(__file__).resolve().parent
DATA_YAML = ROOT_DIR / "data.yaml"
OUTPUT_DIR = ROOT_DIR / "ablation_runs"
CSV_PATH = ROOT_DIR / "ablation_results.csv"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# The requested comparisons reuse the baseline (640/16/30). Run each unique
# configuration once: five runs cover the baseline and every requested contrast.
SCENARIOS = [
    {"name": "baseline_yolov8n_640_b16_e30", "model": "yolov8n-seg.pt", "imgsz": 640, "batch": 16, "epochs": 30},
    {"name": "imgsz_320_yolov8n_b16_e30", "model": "yolov8n-seg.pt", "imgsz": 320, "batch": 16, "epochs": 30},
    {"name": "batch_8_yolov8n_640_e30", "model": "yolov8n-seg.pt", "imgsz": 640, "batch": 8, "epochs": 30},
    {"name": "epochs_15_yolov8n_640_b16", "model": "yolov8n-seg.pt", "imgsz": 640, "batch": 16, "epochs": 15},
    {"name": "architecture_yolov8s_640_b16_e30", "model": "yolov8s-seg.pt", "imgsz": 640, "batch": 16, "epochs": 30},
]


def scalar(value: Any) -> float:
    """Convert Ultralytics/numpy/tensor scalar values to a CSV-friendly float."""
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def model_complexity(model: YOLO, imgsz: int) -> tuple[float, float]:
    """Return parameter count in millions and Ultralytics FLOPs in GFLOPs."""
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    flops_g = scalar(get_flops(model.model, imgsz=imgsz))
    return params_m, flops_g


def metric_value(metrics: Any, key: str) -> float:
    """Read a named value from the validator's results_dict, tolerating API variants."""
    values = getattr(metrics, "results_dict", {}) or {}
    return scalar(values.get(key, float("nan")))


def copy_run_plots(train_dir: Path, val_dir: Path, scenario_name: str) -> None:
    """Keep commonly useful run/validation plots under unique stable filenames."""
    artifact_dir = OUTPUT_DIR / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        "results.png": train_dir / "results.png",
        "confusion_matrix.png": val_dir / "confusion_matrix.png",
        "val_batch0_pred.jpg": val_dir / "val_batch0_pred.jpg",
    }
    for output_name, source in candidates.items():
        if source.exists():
            shutil.copy2(source, artifact_dir / f"{scenario_name}_{output_name}")
        else:
            print(f"[WARN] Không tìm thấy biểu đồ: {source}")


def run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Train, validate, collect metrics and save this run's plots."""
    name = scenario["name"]
    print(f"\n{'=' * 78}\nBắt đầu: {name}\n{'=' * 78}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(scenario["model"])
    params_m, flops_g = model_complexity(model, scenario["imgsz"])
    train_result = model.train(
        data=DATA_YAML,
        imgsz=scenario["imgsz"],
        batch=scenario["batch"],
        epochs=scenario["epochs"],
        device=DEVICE,
        project=str(OUTPUT_DIR / "train"),
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    # Validate the best checkpoint explicitly to get consistent metrics, speed,
    # confusion matrix and prediction examples for every run.
    best_weights = Path(train_result.save_dir) / "weights" / "best.pt"
    val_model = YOLO(str(best_weights))
    val_result = val_model.val(
        data=DATA_YAML,
        imgsz=scenario["imgsz"],
        batch=scenario["batch"],
        device=DEVICE,
        project=str(OUTPUT_DIR / "validation"),
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    metrics = val_result
    speed = getattr(metrics, "speed", {}) or {}
    preprocess_ms = scalar(speed.get("preprocess", float("nan")))
    inference_ms = scalar(speed.get("inference", float("nan")))
    postprocess_ms = scalar(speed.get("postprocess", float("nan")))
    total_ms = preprocess_ms + inference_ms + postprocess_ms
    fps = 1000.0 / total_ms if total_ms > 0 else float("nan")
    peak_vram_gb = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else float("nan")
    )

    copy_run_plots(Path(train_result.save_dir), Path(val_result.save_dir), name)

    row = {
        "scenario": name,
        "model": scenario["model"],
        "imgsz": scenario["imgsz"],
        "batch": scenario["batch"],
        "epochs": scenario["epochs"],
        "parameters_M": params_m,
        "FLOPs_G": flops_g,
        "mask_precision": metric_value(metrics, "metrics/precision(M)"),
        "mask_recall": metric_value(metrics, "metrics/recall(M)"),
        "mask_mAP50": metric_value(metrics, "metrics/mAP50(M)"),
        "mask_mAP50_95": metric_value(metrics, "metrics/mAP50-95(M)"),
        "box_mAP50": metric_value(metrics, "metrics/mAP50(B)"),
        "box_mAP50_95": metric_value(metrics, "metrics/mAP50-95(B)"),
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "FPS_total": fps,
        "peak_VRAM_GB": peak_vram_gb,
    }

    del model, val_model, train_result, val_result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def print_markdown_table(frame: pd.DataFrame) -> None:
    """Print all result columns as a paste-ready Markdown table."""
    display = frame.copy()
    for col in display.select_dtypes(include="number").columns:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    # Avoid requiring tabulate in Colab environments.
    headers = list(display.columns)
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in display.itertuples(index=False, name=None):
        print("| " + " | ".join(str(value) for value in row) + " |")


def main() -> None:
    if not Path(DATA_YAML).is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_YAML}. Hãy đặt data.yaml trong thư mục làm việc "
            "hoặc chỉnh DATA_YAML thành đường dẫn dataset trên Colab."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        try:
            row = run_scenario(scenario)
            rows.append(row)
            # Persist after every completed run in case a later run is interrupted.
            pd.DataFrame(rows).to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
            print(f"Đã cập nhật {CSV_PATH.resolve()}")
        except Exception as exc:
            print(f"[ERROR] Kịch bản {scenario['name']} thất bại: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Keep earlier completed runs and stop, so missing comparisons are explicit.
            raise

    results = pd.DataFrame(rows)
    results.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print("\nBảng tổng hợp Markdown:")
    print_markdown_table(results)
    print(f"\nCSV: {CSV_PATH.resolve()}")
    print(f"Artifacts: {(OUTPUT_DIR / 'artifacts').resolve()}")


if __name__ == "__main__":
    main()
