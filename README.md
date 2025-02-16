# Training YOLOv11 for Traffic Monitoring
---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project focuses on training YOLOv11, a state-of-the-art object detection model, to accurately detect vehicles, traffic signals in real-time. 
This makes it an ideal solution for smart city and traffic management applications.

---

## Installation

### Prerequisites

- Python 3.10 or later
- [PyTorch](https://pytorch.org/) (with CUDA support for GPU acceleration)
- [OpenCV](https://opencv.org/)

### Setup Instructions

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/YOLOv11-for-Traffic-Monitoring.git
    cd YOLOv11-for-Traffic-Monitoring
    ```

2. **Create a virtual environment:**

    ```bash
    python -m venv env
    source env/bin/activate  # Windows: env\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip install ultralytics
    ```

---

## Data Preparation

1. **Dataset Acquisition:**  
   Obtain a traffic monitoring dataset via this [Roboflow project](https://universe.roboflow.com/other/traffic-monitoring-nsn3m/dataset/4) or your own collection. Ensure your dataset includes annotated images in the YOLOv11 format.

2. **Directory Structure:**  
   Organize your dataset as follows:

    ```
    /train
      /images
      /labels
    /valid
      /images
      /labels
    /test
      /images
      /labels
    ```

3. **Configuration File:**  
   Update the `data.yaml` file with:
   - Paths to your training and validation datasets.
   - List of class names (e.g., car, pedestrian, traffic_light, etc.).

---

## Training

Train YOLOv11 using the provided training script. Adjust hyperparameters as needed.

```bash
python training.py
```

## Inference

To run detection on a video:
```bash
python inference.py --input "Road_Traffic.mp4" --output "./videos/TrafficCam_video.mp4" --weights "best.pt" --tracker "botsort.yaml"
```

## References
- [YOLOv11 for Vehicle Detection](https://arxiv.org/html/2410.22898v1)
