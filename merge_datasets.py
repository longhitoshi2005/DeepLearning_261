"""
Module to merge multiple YOLOv8-Segmentation datasets into a unified dataset with a single class 'durian'.
Supports both already extracted folders and automatic unzipping of .zip files.
"""

import os
import shutil
import zipfile
from pathlib import Path
import yaml

VALID_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

def extract_zips(raw_path: Path):
    """Auto-extract any zip files found in raw_path into separate subdirectories."""
    zip_files = list(raw_path.glob("*.zip"))
    for z in zip_files:
        extract_dir = raw_path / z.stem
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            print(f"Extracting {z.name} -> {extract_dir.name}...")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

def convert_line_to_durian_seg(line: str, warning_stats: dict):
    parts = line.strip().split()
    if not parts:
        return None
    
    # Check if segmentation (at least 3 points = 6 coords + 1 class = 7 tokens)
    # or bounding box (5 tokens)
    if len(parts) == 5:
        # Bounding box: class cx cy w h
        warning_stats['bbox_converted'] += 1
        cx, cy, w, h = map(float, parts[1:5])
        x_min = max(0.0, min(1.0, cx - w / 2))
        x_max = max(0.0, min(1.0, cx + w / 2))
        y_min = max(0.0, min(1.0, cy - h / 2))
        y_max = max(0.0, min(1.0, cy + h / 2))
        # Convert bbox to 4 polygon corners
        return f"0 {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f} {x_min:.6f} {y_max:.6f}\n"
    elif len(parts) >= 7 and len(parts) % 2 == 1:
        # Segmentation polygon
        warning_stats['seg_instances'] += 1
        # Set class index to 0
        return "0 " + " ".join(parts[1:]) + "\n"
    else:
        warning_stats['invalid_lines'] += 1
        return None

def merge_roboflow_datasets(raw_dir: str, output_dir: str):
    raw_path = Path(raw_dir).resolve()
    out_path = Path(output_dir).resolve()
    
    # Extract any .zip archives first
    extract_zips(raw_path)
    
    # Destination split dirs
    splits = ['train', 'valid', 'test']
    for s in splits:
        (out_path / s / 'images').mkdir(parents=True, exist_ok=True)
        (out_path / s / 'labels').mkdir(parents=True, exist_ok=True)
        
    print(f"Scanning for datasets in: {raw_path}")
    # Find all subdirectories that look like dataset splits
    dataset_dirs = [
        d for d in raw_path.iterdir() 
        if d.is_dir() and d.resolve() != out_path and d.name != out_path.name and not d.name.startswith('.')
    ]
    
    stats = {
        'datasets_found': len(dataset_dirs),
        'images_copied': {'train': 0, 'valid': 0, 'test': 0},
        'labels_processed': {'train': 0, 'valid': 0, 'test': 0},
        'seg_instances': 0,
        'bbox_converted': 0,
        'invalid_lines': 0
    }
    
    for d in dataset_dirs:
        ds_name = d.name
        print(f"\nProcessing dataset: {ds_name}...")
        
        # Check subfolders for train, val/valid, test
        for s in splits:
            # Match split names (val or valid)
            candidate_splits = [s]
            if s == 'valid':
                candidate_splits.append('val')
            elif s == 'val':
                candidate_splits.append('valid')
            
            src_split_dir = None
            for cs in candidate_splits:
                if (d / cs).is_dir():
                    src_split_dir = d / cs
                    break
            
            if not src_split_dir:
                continue
                
            src_img_dir = src_split_dir / 'images'
            src_lbl_dir = src_split_dir / 'labels'
            
            # If there is no images/labels subfolder, check if files are directly in split dir
            if not src_img_dir.is_dir():
                src_img_dir = src_split_dir
            if not src_lbl_dir.is_dir():
                src_lbl_dir = src_split_dir
                
            img_files = [f for f in src_img_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMG_EXTS]
            print(f"  [{s}] Found {len(img_files)} images in {src_split_dir.name}")
            
            for img_file in img_files:
                base_stem = img_file.stem
                new_stem = f"{ds_name}_{base_stem}"
                new_img_name = f"{new_stem}{img_file.suffix}"
                dst_img_path = out_path / s / 'images' / new_img_name
                
                # Copy image
                shutil.copy2(img_file, dst_img_path)
                stats['images_copied'][s] += 1
                
                # Process label
                lbl_file = src_lbl_dir / f"{base_stem}.txt"
                dst_lbl_path = out_path / s / 'labels' / f"{new_stem}.txt"
                
                if lbl_file.exists():
                    processed_lines = []
                    with open(lbl_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            new_line = convert_line_to_durian_seg(line, stats)
                            if new_line:
                                processed_lines.append(new_line)
                    with open(dst_lbl_path, 'w', encoding='utf-8') as f:
                        f.writelines(processed_lines)
                    stats['labels_processed'][s] += 1
                else:
                    # Empty label file for background / no object
                    dst_lbl_path.touch()
                    stats['labels_processed'][s] += 1

    # Create data.yaml in unified YOLOv8 format
    yaml_content = {
        'path': str(out_path).replace('\\', '/'),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,
        'names': {0: 'durian'}
    }
    
    yaml_file = out_path / 'data.yaml'
    with open(yaml_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
    # Also write a data.yaml at root directory for convenience
    root_yaml = raw_path / 'data.yaml'
    with open(root_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print("\n" + "="*50)
    print("MERGE COMPLETE SUMMARY:")
    print(f"Datasets processed: {stats['datasets_found']}")
    print(f"Images: Train={stats['images_copied']['train']}, Valid={stats['images_copied']['valid']}, Test={stats['images_copied']['test']}, Total={sum(stats['images_copied'].values())}")
    print(f"Segmentation Instances: {stats['seg_instances']}")
    if stats['bbox_converted'] > 0:
        print(f"Bbox converted to polygons: {stats['bbox_converted']}")
    print(f"Generated YAML configuration: {yaml_file}")
    print("="*50)
    return stats

if __name__ == '__main__':
    merge_roboflow_datasets('.', './durian_dataset_merged')

