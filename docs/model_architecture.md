# AquaVision YOLO11n - Architecture & Network Design

This document details the architectural composition and design principles of the **YOLO11n** neural network employed by **AquaVision** for river surface waste detection.

---

## 1. Architectural Overview

YOLO11 represents the state-of-the-art iteration in the Ultralytics YOLO real-time vision family. AquaVision utilizes the **Nano (`yolo11n`)** variant, engineered specifically for high frame rates and minimal compute footprint without sacrificing accuracy on multi-scale floating debris.

The architecture comprises three functional stages:
1. **Backbone**: Feature extraction network using optimized CSPNet blocks.
2. **Neck**: Multi-scale feature fusion via FPN (Feature Pyramid Network) + PAN (Path Aggregation Network) enriched with spatial attention.
3. **Head**: Decoupled, anchor-free detection head.

```text
[ Input Image: 640x640x3 ]
           │
           ▼
┌─────────────────────────────────┐
│            BACKBONE             │
│  - Conv layers (stem downsample)│
│  - C3k2 Blocks (gradient split) │
│  - SPPF (Fast Spatial Pyramid)  │
└────────────────┬────────────────┘
                 │ Multi-scale feature maps (P3, P4, P5)
                 ▼
┌─────────────────────────────────┐
│              NECK               │
│  - FPN (Top-down feature path)  │
│  - PAN (Bottom-up feature path) │
│  - C2PSA (Spatial Attention)    │
└────────────────┬────────────────┘
                 │ Enriched feature pyramids
                 ▼
┌─────────────────────────────────┐
│         DETECTION HEAD          │
│  - Decoupled Classification Head│
│  - Decoupled Regression Head    │
│  - Anchor-Free Task Aligner     │
└────────────────┬────────────────┘
                 │
                 ▼
[ Detections: Bounding Boxes + Class Scores (can, foam, plastic, etc.) ]
```

---

## 2. Component Breakdown

### 2.1 C3k2 Block (Cross Stage Partial with 2 Convolutions)
The backbone and neck feature upgraded **C3k2** blocks. Unlike earlier CSP implementations, C3k2 offers dynamic kernel flexibility (toggling between dual $3 \times 3$ convolutions and standard bottleneck modules). This design:
- Enhances feature reusability across deeper layers.
- Mitigates gradient vanishing during the 100-epoch training schedule.
- Reduces parameter count while preserving receptive field sensitivity.

### 2.2 SPPF (Spatial Pyramid Pooling - Fast)
Positioned at the bottleneck layer of the backbone (feature map P5):
- Evaluates spatial pools at multiple receptive scales ($5 \times 5$, $9 \times 9$, $13 \times 13$ equivalent via consecutive small max-pooling operations).
- Ensures that floating debris ranging from tiny bottle caps to large fragmented bags are detected accurately regardless of distance from the camera sensor.

### 2.3 C2PSA (Cross Stage Partial with Spatial Attention)
Integrated into the feature neck:
- Introduces self-attention mechanisms into convolutional feature maps.
- Focuses representation weight on high-salience debris regions while suppressing non-informative background noise (such as repetitive water ripples, sunlight reflections, and minor foam froth).

### 2.4 Decoupled Anchor-Free Head
The detection head separates the learning objectives:
1. **Classification Branch**: Predicts probabilities across the 5 target classes (`can`, `foam`, `plastic`, `plastic bottle`, `unknow`).
2. **Regression Branch**: Predicts bounding box boundary offsets directly without relying on predefined anchor aspect ratios.

This separation prevents conflicts between classification loss and localization loss during backpropagation.

---

## 3. Loss Functions & Optimization

AquaVision YOLO11n uses three composite loss functions:

1. **Complete IoU (CIoU) Loss**: Measures overlap area, center point distance, and aspect ratio alignment between predicted and ground-truth bounding boxes:
   $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

2. **Distribution Focal Loss (DFL)**: Treats box boundaries as continuous probability distributions rather than fixed single values, enabling flexible localization around soft or partially submerged edges:
   $$\mathcal{L}_{\text{DFL}}(S_i, S_{i+1}) = - \left( (y_{i+1} - y)\log(S_i) + (y - y_i)\log(S_{i+1}) \right)$$

3. **Binary Cross-Entropy (BCE) Loss**: Evaluates multi-class classification errors across all classes independently.

The total training loss is computed as:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}}$$

---

## 4. Hardware Footprint & Operational Profile

| Property | Value |
|---|---|
| **Architecture** | YOLO11n |
| **Parameters** | ~2.6 Million |
| **FLOPs (at 640×640)** | ~6.5 GFLOPs |
| **Model Weight Size** | 5.48 MB (`best.pt`) |
| **FP16 Inference Time (GPU)** | ~2.5 ms (CUDA) |
| **INT8 / ONNX Support** | Supported via Ultralytics export |
| **Target Deployments** | Autonomous river boats, surveillance drones, edge gateways |
