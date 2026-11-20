# AquaVision Methodology: River Surface Waste Detection

This document describes the end-to-end engineering methodology implemented in the **AquaVision** project, covering problem definition, data curation, preprocessing pipelines, model selection, hyperparameter tuning, and verification.

---

## 1. Problem Definition & Context

Riverine plastic transport represents one of the largest contributors to oceanic marine pollution. Surface floating waste—dominated by single-use beverage containers, expanded polystyrene foam, and plastic wrappings—often travels through inland river systems before discharging into coastal waters.

Traditional river waste cleanup workflows rely heavily on manual visual surveillance or scheduled physical barrier clearing, which are reactive, labor-intensive, and non-scalable. **AquaVision** provides an automated, computer vision-based real-time detection solution suitable for deployment on:
- Autonomous surface vessels (ASVs) and river cleanup trash skimmers.
- Fixed river-bank monitoring cameras and smart bridge stations.
- Unmanned aerial vehicles (UAVs) performing aerial river surveys.

---

## 2. Dataset Architecture & Composition

The training dataset unites multi-source water surface waste imagery to maximize generalization:

1. **Roboflow Universe Dataset**: Base domain dataset (`aqua-vision-0whs7 / plastic_waste-tqfvh`) curated for aquatic surface debris.
2. **Kaggle Water Surface Litter Repositories**: Diverse river perspectives, light angles, and debris aggregations.
3. **Custom Real-World River Imagery**: Field-collected images capturing local river bank conditions, heavy siltation, varying water velocities, and organic floaters.

### Dataset Volume
- **Raw Annotated Images**: Approximately 10,000 images.
- **Augmented Dataset**: 18,635 images with standardized bounding box annotations in YOLO format.
- **Data Splits**:
  - Training Set: ~80%
  - Validation Set: ~13%
  - Test Set: ~7%

---

## 3. Preprocessing & Data Augmentation

To address typical river inspection challenges—such as fluctuating sunlight reflections, water ripple distortions, and partial debris submergence—the data pipeline incorporated targeted augmentations:

| Pipeline Step | Parameter / Transformation | Justification |
|---|---|---|
| **EXIF Normalization** | Auto-orientation with orientation tag stripping | Prevents upside-down or sideways camera sensor artifacts |
| **Resolution Rescaling** | 640 × 640 pixels (Fit within aspect ratio) | Balances spatial resolution for small debris with high FPS |
| **Rotational Augmentation** | Random rotation between -10° and +10° | Simulates orientation shifts caused by water currents |
| **Brightness Adjustment** | Random variation between -15% and +15% | Simulates shifting cloud cover and canopy shade |
| **Exposure Adjustment** | Random variation between -10% and +10% | Robustness against camera overexposure on reflective water |
| **Mosaic Augmentation** | Enabled (closed in final 10 epochs) | Teaches the model to recognize multiple small targets simultaneously |

---

## 4. Model Selection: YOLO11n

**YOLO11n** (Nano) from Ultralytics was selected as the optimal architecture for AquaVision based on three criteria:

1. **Computational Efficiency**: At ~2.6M parameters and ~6.5 GFLOPs, it delivers 30+ FPS even on edge computing hardware (such as NVIDIA Jetson Orin Nano, Raspberry Pi 5 with Hailo-8, or standard CPU laptops).
2. **Enhanced Feature Extraction**: Incorporates modern **C3k2** blocks and **C2PSA** (Spatial Attention) mechanisms designed to preserve multi-scale features against noisy water backgrounds.
3. **Anchor-Free Decoupled Head**: Directly predicts bounding box coordinates and class probabilities separately, improving convergence speed on small floating targets.

---

## 5. Training Pipeline & Hyperparameters

The model was trained using the following configuration:

```yaml
task: detect
mode: train
model: yolo11n.pt
epochs: 100
batch: 16
imgsz: 640
optimizer: MuSGD
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
close_mosaic: 10
amp: true
```

### Key Training Strategies
- **Transfer Learning**: Initialized from COCO-pretrained `yolo11n.pt` weights.
- **Automatic Mixed Precision (AMP)**: Enabled FP16 training to accelerate backpropagation and reduce VRAM utilization.
- **Close Mosaic**: Mosaic augmentation was disabled for the last 10 epochs (epochs 91–100) to allow clean gradient stabilization on natural object contours.

---

## 6. Evaluation Protocol

Performance was measured on the validation set using standard COCO-style object detection metrics:
- **Precision (P)**: Ratio of true positive detections over total positive predictions.
- **Recall (R)**: Ratio of true positive detections over ground truth waste annotations.
- **mAP@50**: Mean Average Precision at an Intersection over Union (IoU) threshold of 0.50.
- **mAP@50-95**: Mean Average Precision averaged over 10 IoU thresholds from 0.50 to 0.95 (step 0.05).
- **F1-Score**: Harmonic mean of Precision and Recall:
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

The verified performance metrics are documented in [performance_metrics.md](../evaluation/performance_metrics.md).
