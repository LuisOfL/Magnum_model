# Magnum stock detector model

This is a tool to improve and test computer vision model for magnum ice cream company
---

## 🛠 Project Architecture

This repository is structured into three main component directories:

```text
.
├── 📁 01_yolo_detection_training/     # YOLO Object Detection Training Pipeline
├── 📁 02_yolo_segmentation_training/  # YOLO Instance Segmentation Training Pipeline
└── 📁 03_freezer_audit_app/           # Real-Time Audit Tool & AI Refinement Interface

---

## 📂 Module Descriptions

### 1. `01_yolo_detection_training/`
Contains the scripts, configuration files, and dataset processing utilities required to train and evaluate the **YOLO Object Detection Model**. 
* **Purpose:** Identifies and locates individual product SKUs (bounding boxes) within commercial freezers.

### 2. `02_yolo_segmentation_training/`
Contains the training pipeline for the **YOLO Instance Segmentation Model**.
* **Purpose:** Generates precise pixel-level masks for structural elements inside the freezer (such as grid dividers, baskets, lanes, and shelves). These masks define spatial boundaries used for depth calibration.

### 3. `03_freezer_audit_app/`
The core client-facing application and testing suite built using **FastAPI** (Backend) and **Flet** (Frontend UI).
* **Key Capabilities:**
  * **Multi-Model Inference:** Runs Object Detection (YOLO), Instance Segmentation (YOLO-Seg), and Monocular Depth Estimation (*Depth Anything V2*) in parallel.
  * **k-NN Spatial Depth Estimation:** Uses $k$-Nearest Neighbors ($k$-NN) matching between detected product centers and structural grid coordinates to calculate relative stock depletion per SKU/lane.
  * **Visual Heatmaps:** Renders colorized depth heatmaps (`cv2.COLORMAP_INFERNO`) alongside detection and segmentation outputs.
  * **Model Refinement & Debugging:** Includes detailed per-item evaluation logs (depth values, confidence scores, capacity limits, and missing stock counts) to benchmark and fine-tune model parameters and calibration rules.

---

## 🚀 Key Features

* **3D Depth-Based Inventory Estimation:** Combines 2D object detection with 3D depth maps to estimate hidden or stacked items behind visible stock.
* **Granular SKU Analytics:** Generates both high-level aggregated inventory totals and individual detection metrics.
* **Interactive Dashboard:** Dark-themed UI for uploading imagery, previewing bounding boxes, viewing segmentation polygons, and auditing depth maps in real time.

---

## 💻 Tech Stack

* **Computer Vision & ML:** Ultralytics YOLO (Detection & Segmentation), Hugging Face Transformers (`Depth Anything V2`), OpenCV, PyTorch.
* **Data Processing:** SciPy (`cKDTree` for spatial $k$-NN search), NumPy, Pillow.
* **Backend:** FastAPI, Uvicorn.
* **Frontend UI:** Flet (Flutter for Python).
