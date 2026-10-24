import sys, os
from ultralytics import YOLO
import cv2
import numpy as np

model_path = r"c:\BKHCM\Year4\DL\ablation_runs\train\baseline_yolov8n_640_b16_e30\weights\best.pt"
image_path = r"c:\BKHCM\Year4\DL\report\sau-rieng3.jpg"
output_path = r"c:\BKHCM\Year4\DL\report\sau-rieng3_pred.jpg"

print(f"Loading model from: {model_path}")
model = YOLO(model_path)

print(f"Running inference on: {image_path}")
results = model.predict(source=image_path, conf=0.25, save=False)

res = results[0]
orig_img = cv2.imread(image_path)
h, w = orig_img.shape[:2]
total_pixels = h * w

print(f"\n=== RESULTS FOR sau-rieng3.jpg ({w}x{h}) ===")
boxes = res.boxes
masks = res.masks

num_detections = len(boxes) if boxes is not None else 0
print(f"Total detections: {num_detections}")

if num_detections > 0:
    for i in range(num_detections):
        box = boxes[i]
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = model.names[cls_id]
        
        bw = xyxy[2] - xyxy[0]
        bh = xyxy[3] - xyxy[1]
        box_area = bw * bh
        area_pct = (box_area / total_pixels) * 100
        
        print(f"Detection {i+1}:")
        print(f"  Class: {cls_name} (ID: {cls_id})")
        print(f"  Confidence: {conf:.4f} ({conf*100:.2f}%)")
        print(f"  BBox: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
        print(f"  BBox dimensions: {bw:.1f} x {bh:.1f} ({area_pct:.2f}% of image area)")
        
        if masks is not None:
            mask_data = masks.data[i].cpu().numpy()
            mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_pixels = np.count_nonzero(mask_resized > 0.5)
            mask_pct = (mask_pixels / total_pixels) * 100
            print(f"  Mask pixel area: {mask_pixels} ({mask_pct:.2f}% of image area)")
else:
    print("NO objects detected above confidence threshold 0.25!")

# Plot and save output image
plotted_img = res.plot()
cv2.imwrite(output_path, plotted_img)
print(f"\nSaved plotted prediction to: {output_path}")

