from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

MODEL_PATH = Path(
    "runs/classify/runs/classify/smart_waste_six/weights/best.pt"
)
TEST_DIR = Path("dataset-classification/test")
OUTPUT_DIR = Path("evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

categories = sorted(
    folder.name for folder in TEST_DIR.iterdir() if folder.is_dir()
)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
true_labels = []
predicted_labels = []

model = YOLO(str(MODEL_PATH))

for category in categories:
    folder = TEST_DIR / category
    images = [
        path for path in folder.rglob("*")
        if path.suffix.lower() in extensions
    ]

    print(f"{category}: {len(images)} test images")

    for image_path in images:
        result = model.predict(str(image_path), verbose=False)[0]
        predicted_index = int(result.probs.top1)
        predicted_category = model.names[predicted_index]

        true_labels.append(category)
        predicted_labels.append(predicted_category)

if not true_labels:
    raise RuntimeError("No test images were found.")

accuracy = accuracy_score(true_labels, predicted_labels)
report = classification_report(
    true_labels,
    predicted_labels,
    labels=categories,
    zero_division=0,
    digits=4,
)

print("\nTEST ACCURACY:", f"{accuracy * 100:.2f}%")
print("\nCLASSIFICATION REPORT")
print(report)

(OUTPUT_DIR / "classification_report.txt").write_text(
    f"Test accuracy: {accuracy * 100:.2f}%\n\n{report}",
    encoding="utf-8",
)

matrix = confusion_matrix(
    true_labels, predicted_labels, labels=categories
)
display = ConfusionMatrixDisplay(
    confusion_matrix=matrix, display_labels=categories
)
display.plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_test.png", dpi=150)
plt.close()

print("\nSaved report: evaluation/classification_report.txt")
print("Saved confusion matrix: evaluation/confusion_matrix_test.png")
