# Smart Waste Sorter

A webcam-based AI waste classification and recycling assistant built with Python, OpenCV, YOLO classification, and PySide6.

## Features
- Live webcam preview
- Classification of six waste categories
- Combined waste result with category, confidence score, and recommended bin
- Duplicate-count prevention
- Category statistics and total item count
- Detection history with timestamp, category, and confidence
- CSV files for history and statistics
- Reset statistics

## Supported Categories
- Organic
- Glass
- Can/Metal
- Other
- Paper
- Plastic

## Technologies
- Python
- Ultralytics YOLO classification
- OpenCV
- PySide6

## Installation
Activate the project's virtual environment and install dependencies:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
From the project root folder, run:

```powershell
python main.py
```

## Use the camera sorter
1. Click **Start Camera** and keep the center detection area empty until the app is ready.
2. Show one waste item in the center of the camera.
3. Read the **Classification Result** and **Recommended bin**. Once the same item is recognized steadily, its category count and total count increase once.
4. Remove the item and wait for the app to say it is ready before showing the next item.

## Important Files
- `main.py` - Main GUI application
- `requirements.txt` - Python dependencies
- `waste_history.csv` - Detection history
- `waste_statistics.csv` - Category statistics
- `dataset-classification/` - Classification dataset
- `runs/` - Training results and model weights

## Dataset and Model
The project uses a trained YOLO classification model with six categories. The dataset contains training, validation, and test folders.

The dataset source, license, and image counts still need to be documented in the final report.

## Testing
The GUI, webcam classification, confidence display, sorting instructions, statistics, and duplicate-count prevention have been tested.

Low-confidence predictions and camera/model error handling still need final verification.

## Limitations
- Classifies one waste item at a time.
- Results may vary with lighting, background, and camera distance.
- Bin recommendations are displayed on screen; physical bin movement requires additional hardware, such as a servo motor and controller.
- Physical bins, servo motors, and conveyor belts are not implemented.


## Evaluation Results

The model was evaluated on 1,043 test images.

- Test accuracy: 90.80%
- Macro F1-score: 89.95%
- Weighted F1-score: 90.78%

| Category | F1-score |
|---|---:|
| Can/Metal | 90.97% |
| Glass | 91.91% |
| Organic | 87.36% |
| Other | 89.29% |
| Paper | 93.06% |
| Plastic | 87.11% |

Results may vary depending on lighting, background, camera distance, and the waste item.



## Dataset Image Counts

The classification dataset contains 12,786 images across six categories.

| Category | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Can/Metal | 1644 | 79 | 156 | 1879 |
| Glass | 1929 | 78 | 190 | 2197 |
| Organic | 432 | 23 | 40 | 495 |
| Other | 1404 | 89 | 138 | 1631 |
| Paper | 3786 | 183 | 338 | 4307 |
| Plastic | 1986 | 110 | 181 | 2277 |
| **Total** | **11181** | **562** | **1043** | **12786** |

The dataset is imbalanced, with fewer Organic images than the other categories.


## Dataset Source and License

- Dataset name: Recyclable Materials v2
- Source: https://universe.roboflow.com/ecycle/recyclable-materials-ljuil
- Provider: Roboflow Universe user
- License: CC BY 4.0 (Creative Commons Attribution 4.0)
- Original export date: March 3, 2024
- Total images: 12,786

The original dataset contains 10 categories. These were mapped into six project categories: Organic, Glass, Can/Metal, Other, Paper, and Plastic.

Dataset attribution: Recyclable Materials, provided by a Roboflow Universe user, licensed under CC BY 4.0.
