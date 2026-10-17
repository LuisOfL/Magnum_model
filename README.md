# Magnum stock detector model

This is a tool to improve and test computer vision model for magnum ice cream company
---

## 🛠 Project Architecture

This repository is structured into three main component directories:

```text
.
├── 📁 01_yolo_detection_training/     # YOLO Object Detection Training Pipeline
├── 📁 02_yolo_segmentation_training/  # YOLO Instance Segmentation Training Pipeline
└── 📁 03_MVP/           # Tool to test and improve model
```
----

## 📂 Module Descriptions

### 1. `01_yolo_detection_training/`
Contains the scripts, configuration files, and dataset processing utilities required to train and evaluate the **YOLO Object Detection Model**. 
* **Purpose:** Identifies and locates individual product SKUs (bounding boxes) within commercial freezers.

### 2. `02_yolo_segmentation_training/`
Contains the training pipeline for the **YOLO Instance Segmentation Model**.
* **Purpose:** Generates precise pixel-level masks for structural elements inside the freezer (such as grid dividers, baskets, lanes, and shelves). These masks define spatial boundaries used for depth calibration.

### 3. `03_MVP/`
The core client-facing application and testing suite built using **FastAPI** (Backend) and **Flet** (Frontend UI).


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
