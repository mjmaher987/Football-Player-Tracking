# ⚽ Football Player Tracking

> Detecting and segmenting football (soccer) players from broadcast frames using three complementary computer-vision approaches.

**Author:** Mohammad Javad Maheronnaghsh
**Dataset:** [Football Player Segmentation (Kaggle)](https://www.kaggle.com/datasets/ihelon/football-player-segmentation/code)

![Detection sample](https://github.com/mjmaher987/Football-Player-Tracking/assets/77095635/31abc6c8-8aa1-499e-8573-6334bb520252)

---

## 📋 Overview

This project explores **player detection and segmentation** on football match images. After a short literature review (Google Scholar, Kaggle, GitHub), three different methods were implemented and compared, ranging from off-the-shelf object detectors to trained segmentation networks and foundation models.

| # | Method | Type | Framework | Idea |
|---|--------|------|-----------|------|
| 1 | **YOLOv8** (Ultralytics) | Object detection | PyTorch | Bounding boxes around each player + jersey-color / grass-mask analysis |
| 2 | **U-Net & U²-Net** | Semantic segmentation | TensorFlow / Keras | Pixel-level player masks, trained on the dataset annotations |
| 3 | **Segment Anything (SAM)** | Promptable segmentation | PyTorch | Zero-shot automatic mask generation from a foundation model |

---

## 🔄 Process Pipeline

```mermaid
flowchart TD
    A[Football match frames] --> B[Preprocessing<br/>resize · BGR→RGB · COCO masks]
    B --> P1
    B --> P2
    B --> P3

    subgraph P1 [Project 1 · YOLOv8 Detection]
        C1[YOLOv8x pretrained] --> C2[Bounding boxes<br/>conf = 0.7]
        C2 --> C3[Per-player crop<br/>+ jersey color]
        C3 --> C4[Grass mask removal]
    end

    subgraph P2 [Project 2 · U-Net / U²-Net]
        D1[Encoder–Decoder<br/>+ skip connections] --> D2[Train on masks]
        D2 --> D3[Pixel-wise prediction]
    end

    subgraph P3 [Project 3 · Segment Anything]
        E1[SAM ViT-H checkpoint] --> E2[Automatic mask generator]
        E2 --> E3[Filter masks by area]
    end

    P1 --> R[Compare results]
    P2 --> R
    P3 --> R
```

---

## 🧩 Methods in Detail

### 1. YOLOv8 — Detection & Analysis
Uses the pretrained **`yolov8x`** model from [Ultralytics](https://github.com/ultralytics/ultralytics) to detect players (COCO `person` class) with a confidence threshold of `0.7`. Each detection returns two corner points `(x, y)` and `(x2, y2)` plus a class and confidence score. The detected crops are further analysed to estimate **jersey colors** and to build a **grass mask** (background subtraction) so player pixels can be isolated from the pitch.

### 2. U-Net & U²-Net — Segmentation
- **U-Net**: a U-shaped encoder–decoder with skip connections (contracting path → expansive path) producing pixel-level masks.
- **U²-Net**: a nested U-structure built from **Residual U-Blocks (RSU-L)** instead of plain convolutional blocks — a six-stage encoder, five-stage decoder, and a saliency-map fusion module.

COCO-style polygon annotations are converted to binary masks (via `imantics`), images resized to `512×512`, and both models are trained and then compared side-by-side against the ground truth.

### 3. Segment Anything (SAM)
Uses Meta AI's [**Segment Anything Model**](https://arxiv.org/pdf/2304.02643.pdf) with the ViT-H checkpoint and the `SamAutomaticMaskGenerator` to produce zero-shot masks, then filters them by area to keep player-sized regions.

---

## 🖼️ Sample Results

| | |
|---|---|
| ![sample](https://github.com/mjmaher987/Football-Player-Tracking/assets/77095635/d698fd3e-32bc-4cbf-9bc0-6586ab4282db) | ![sample](https://github.com/mjmaher987/Football-Player-Tracking/assets/77095635/a09f649b-9657-49cf-8265-6c6852ce2dc9) |
| ![sample](https://github.com/mjmaher987/Football-Player-Tracking/assets/77095635/b33a7945-5079-49f0-85dc-d9dd2cac7e92) | |

---

## 🚀 Getting Started

All code lives in [`football-tracking-final_release.ipynb`](./football-tracking-final_release.ipynb). The notebook was developed on **Kaggle** (with the dataset mounted under `/kaggle/input`).

```bash
# clone
git clone https://github.com/mjmaher987/Football-Player-Tracking.git
cd Football-Player-Tracking

# key dependencies
pip install ultralytics tensorflow opencv-python imantics
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Then open the notebook and run the section for the approach you want to explore (each project is self-contained).

### Standalone Kaggle notebooks
- [YOLOv8 player detection](https://www.kaggle.com/code/mjmaher987/football-player-detection)
- [Segmentation with U-Net & U²-Net](https://www.kaggle.com/code/mjmaher987/football-players-segmentation-with-tf-unet-u2net)
- [Segment Anything — how to](https://www.kaggle.com/code/mjmaher987/segment-anything-model-how-to)

---

## 📄 License

See [LICENSE](./LICENSE).
