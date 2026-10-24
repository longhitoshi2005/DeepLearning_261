# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    model_path = 'ablation_runs/train/baseline_yolov8n_640_b16_e30/weights/best.pt'
    img_path = 'report/sau-rieng2.jpg'
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Running inference on {img_path}...")
    results = model.predict(source=img_path, conf=0.25, verbose=False)
    res = results[0]
    
    # Save raw plotted prediction
    plotted = res.plot()
    pred_path = 'report/sau-rieng2_pred.jpg'
    cv2.imwrite(pred_path, plotted)
    print(f"Saved prediction image to {pred_path}")
    
    # Extract statistics
    orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    pred_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    h, w, _ = orig_img.shape
    
    num_boxes = len(res.boxes)
    print(f"Total detections: {num_boxes}")
    
    detections_info = []
    for i, box in enumerate(res.boxes):
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        xyxy = [round(v, 2) for v in box.xyxy[0].tolist()]
        cx = (xyxy[0] + xyxy[2]) / 2.0
        cy = (xyxy[1] + xyxy[3]) / 2.0
        detections_info.append({
            'id': i + 1,
            'class': cls_name,
            'conf': conf,
            'bbox': xyxy,
            'center': (round(cx, 1), round(cy, 1))
        })
        print(f"Detection #{i+1}: Class={cls_name}, Confidence={conf*100:.2f}%, BBox={xyxy}, Center=({cx:.1f}, {cy:.1f})")
        
    speed = res.speed
    print(f"Preprocess: {speed['preprocess']:.2f} ms")
    print(f"Inference: {speed['inference']:.2f} ms")
    print(f"Postprocess: {speed['postprocess']:.2f} ms")
    
    # Create side-by-side comparison figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), dpi=200)
    
    axes[0].imshow(orig_img)
    axes[0].set_title(f"Anh goc: sau-rieng2.jpg ({w} x {h})", fontsize=12, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    axes[1].imshow(pred_rgb)
    axes[1].set_title("Ket qua YOLOv8n-Seg (Baseline best.pt)", fontsize=12, fontweight='bold', pad=10)
    axes[1].axis('off')
    
    top_det = detections_info[0] if detections_info else None
    if top_det:
        info_str = f"Phat hien: {top_det['class']} | Do tin cay (Confidence): {top_det['conf']*100:.2f}%\n" \
                   f"Hop bao (BBox): [{top_det['bbox'][0]}, {top_det['bbox'][1]}, {top_det['bbox'][2]}, {top_det['bbox'][3]}]\n" \
                   f"Toc do suy luan: {speed['inference']:.1f} ms (~{1000/speed['inference']:.0f} FPS trên GPU RTX 3060)"
    else:
        info_str = "Khong phat hien duoc doi tuong voi nguong conf=0.25"
        
    plt.figtext(0.5, 0.02, info_str, ha='center', fontsize=10.5,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#2E7D32', lw=1.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    summary_path = 'report/sau-rieng2_detection_summary.png'
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved side-by-side summary figure to {summary_path}")

if __name__ == '__main__':
    main()

