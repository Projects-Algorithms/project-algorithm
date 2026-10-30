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
From the project root, use Python and PowerShell to create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Model weights

The application requires a trained six-class YOLO classification model at
`model\best.pt`. Model weights are excluded from Git by `.gitignore`, so
cloning this repository alone does not provide the model. Obtain the project
weights from the project maintainer or its published release, then place them
at:

```text
model\best.pt
```

The model must contain these class names: `Can_Metal`, `Glass`, `Organic`,
`Other`, `Paper`, and `Plastic`. If the weights are missing or incompatible,
the app cannot start camera classification. To make setup easier for other
users, publish the weights as a GitHub Release asset and add its download link
here.

## Run
Once the dependencies and model weights are in place, run:

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
- `model/best.pt` - Required trained model weights (not included in Git)
- `evaluate_test.py` - Evaluate the model on the test dataset
- `evaluation/` - Saved evaluation report and confusion matrix

## Dataset and Model
The project uses a trained YOLO classification model with six categories. The
dataset contains training, validation, and test folders. Dataset counts,
source, and license are documented below.

## Testing
The model was evaluated on the held-out test set; results are shown below.
Application startup and model loading have been verified. Webcam behavior,
low-confidence predictions, and camera error handling still need real-device
testing.

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
