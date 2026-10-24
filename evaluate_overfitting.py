import pandas as pd
from pathlib import Path
import json

runs_dir = Path('./ablation_runs/train')
summary_data = []

for exp in sorted(runs_dir.iterdir()):
    if not exp.is_dir():
        continue
    res_file = exp / 'results.csv'
    if not res_file.exists():
        continue
        
    df = pd.read_csv(res_file)
    df.columns = df.columns.str.strip()
    
    # Loss columns
    train_box = df['train/box_loss'] if 'train/box_loss' in df.columns else 0
    val_box = df['val/box_loss'] if 'val/box_loss' in df.columns else 0
    train_seg = df['train/seg_loss'] if 'train/seg_loss' in df.columns else 0
    val_seg = df['val/seg_loss'] if 'val/seg_loss' in df.columns else 0
    train_cls = df['train/cls_loss'] if 'train/cls_loss' in df.columns else 0
    val_cls = df['val/cls_loss'] if 'val/cls_loss' in df.columns else 0
    train_dfl = df['train/dfl_loss'] if 'train/dfl_loss' in df.columns else 0
    val_dfl = df['val/dfl_loss'] if 'val/dfl_loss' in df.columns else 0
    
    val_total = val_box + val_seg + val_cls + val_dfl
    train_total = train_box + train_seg + train_cls + train_dfl
    
    min_val_epoch = int(val_total.idxmin() + 1)
    min_val_loss = float(val_total.min())
    final_val_loss = float(val_total.iloc[-1])
    
    min_train_loss = float(train_total.min())
    final_train_loss = float(train_total.iloc[-1])
    
    val_loss_increase_pct = float(((final_val_loss - min_val_loss) / min_val_loss) * 100)
    
    # Metric columns
    mask_p = float(df['metrics/precision(M)'].iloc[-1]) if 'metrics/precision(M)' in df.columns else None
    mask_r = float(df['metrics/recall(M)'].iloc[-1]) if 'metrics/recall(M)' in df.columns else None
    mask_map50 = float(df['metrics/mAP50(M)'].iloc[-1]) if 'metrics/mAP50(M)' in df.columns else None
    mask_map50_95 = float(df['metrics/mAP50-95(M)'].iloc[-1]) if 'metrics/mAP50-95(M)' in df.columns else None
    
    best_mask_map = float(df['metrics/mAP50-95(M)'].max()) if 'metrics/mAP50-95(M)' in df.columns else None
    best_mask_map_epoch = int(df['metrics/mAP50-95(M)'].idxmax() + 1) if 'metrics/mAP50-95(M)' in df.columns else None
    
    box_map50 = float(df['metrics/mAP50(B)'].iloc[-1]) if 'metrics/mAP50(B)' in df.columns else None
    box_map50_95 = float(df['metrics/mAP50-95(B)'].iloc[-1]) if 'metrics/mAP50-95(B)' in df.columns else None
    
    summary_data.append({
        'experiment': exp.name,
        'epochs': len(df),
        'epoch_first_train_loss': float(train_total.iloc[0]),
        'epoch_first_val_loss': float(val_total.iloc[0]),
        'final_train_loss': round(final_train_loss, 4),
        'final_val_loss': round(final_val_loss, 4),
        'min_val_loss': round(min_val_loss, 4),
        'min_val_loss_epoch': min_val_epoch,
        'val_loss_inc_from_min_pct': round(val_loss_increase_pct, 2),
        'best_mask_mAP50_95': round(best_mask_map, 4),
        'best_mask_mAP_epoch': best_mask_map_epoch,
        'final_mask_mAP50_95': round(mask_map50_95, 4),
        'final_mask_mAP50': round(mask_map50, 4),
        'final_mask_P': round(mask_p, 4),
        'final_mask_R': round(mask_r, 4),
        'final_box_mAP50_95': round(box_map50_95, 4),
        'final_box_mAP50': round(box_map50, 4)
    })

with open('overfitting_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

print("Saved overfitting_summary.json successfully")
