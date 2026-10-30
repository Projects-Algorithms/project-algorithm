
import sys
import csv
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFrame,
    QMessageBox,
)

MODEL_PATH = (
    "runs/classify/runs/classify/"
    "smart_waste_six/weights/best.pt"
)

CONFIDENCE_THRESHOLD = 60.0
HISTORY_FILE = "waste_history.csv"

STABLE_FRAMES_REQUIRED = 8
REMOVAL_FRAMES_REQUIRED = 8

# Adjust this if the camera's background or lighting changes.
EMPTY_DIFF_THRESHOLD = 18.0

CATEGORIES = [
    "Organic",
    "Glass",
    "Can_Metal",
    "Other",
    "Paper",
    "Plastic",
]

INSTRUCTIONS = {
    "Organic": "Use the organic waste bin.",
    "Glass": "Use the glass recycling bin.",
    "Can_Metal": "Use the metal recycling bin.",
    "Other": "Check local disposal instructions.",
    "Paper": "Use the paper recycling bin.",
    "Plastic": "Use the plastic recycling bin.",
}


class SmartWasteUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Smart Waste Sorter")
        self.resize(1150, 780)

        self.counts = {name: 0 for name in CATEGORIES}
        self.camera = None
        self.model = None

        # Automatic counting state
        self.last_category = None
        self.stable_frames = 0
        self.item_locked = False
        self.removal_frames = 0
        self.last_confidence = 0.0

        # Empty view of the central detection area
        self.empty_roi_gray = None

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f3f7f4;
                color: #20352a;
                font-family: Arial;
                font-size: 14px;
            }
            QLabel#title {
                font-size: 28px;
                font-weight: bold;
                color: #176b45;
            }
            QLabel#subtitle {
                color: #63756a;
                font-size: 14px;
            }
            QFrame#panel {
                background-color: white;
                border: 1px solid #dce7df;
                border-radius: 12px;
            }
            QLabel#section {
                font-size: 18px;
                font-weight: bold;
            }
            QLabel#value {
                font-size: 22px;
                font-weight: bold;
                color: #176b45;
            }
            QPushButton {
                background-color: #19764c;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 11px 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #125c3a;
            }
            QPushButton:disabled {
                background-color: #9eafa4;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header
        header = QHBoxLayout()
        title_area = QVBoxLayout()

        title = QLabel("Smart Waste Sorter")
        title.setObjectName("title")

        subtitle = QLabel(
            "Computer Vision • AI Classification • Recycling Assistant"
        )
        subtitle.setObjectName("subtitle")

        title_area.addWidget(title)
        title_area.addWidget(subtitle)
        header.addLayout(title_area)
        header.addStretch()

        self.status_label = QLabel("Status: Camera stopped")
        header.addWidget(self.status_label)
        main_layout.addLayout(header)

        # Camera panel
        content = QHBoxLayout()
        content.setSpacing(16)

        camera_panel = QFrame()
        camera_panel.setObjectName("panel")
        camera_layout = QVBoxLayout(camera_panel)

        camera_heading = QLabel("Live Camera")
        camera_heading.setObjectName("section")
        camera_layout.addWidget(camera_heading)

        self.camera_label = QLabel("Camera preview will appear here")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(540, 380)
        self.camera_label.setStyleSheet(
            "background-color: #18241d; color: white; "
            "border-radius: 8px;"
        )
        camera_layout.addWidget(self.camera_label, 1)

        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.start_camera)
        camera_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.clicked.connect(self.stop_camera)
        self.stop_button.setEnabled(False)
        camera_layout.addWidget(self.stop_button)

        content.addWidget(camera_panel, 3)

        # Prediction panel
        result_panel = QFrame()
        result_panel.setObjectName("panel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setSpacing(14)

        result_heading = QLabel("Classification Result")
        result_heading.setObjectName("section")
        result_layout.addWidget(result_heading)

        result_layout.addWidget(QLabel("Predicted category"))

        self.category_label = QLabel("Waiting...")
        self.category_label.setObjectName("value")
        result_layout.addWidget(self.category_label)

        result_layout.addWidget(QLabel("Confidence"))

        self.confidence_label = QLabel("--")
        self.confidence_label.setObjectName("value")
        result_layout.addWidget(self.confidence_label)

        result_layout.addWidget(QLabel("Sorting recommendation"))

        self.instruction_label = QLabel(
            "Start the camera to classify an item."
        )
        self.instruction_label.setWordWrap(True)
        result_layout.addWidget(self.instruction_label)

        result_layout.addStretch()
        content.addWidget(result_panel, 2)
        main_layout.addLayout(content, 3)

        # Statistics panel
        stats_panel = QFrame()
        stats_panel.setObjectName("panel")
        stats_layout = QVBoxLayout(stats_panel)

        stats_heading = QLabel("Waste Statistics")
        stats_heading.setObjectName("section")
        stats_layout.addWidget(stats_heading)

        grid = QGridLayout()
        self.count_labels = {}

        for index, name in enumerate(CATEGORIES):
            name_label = QLabel(name.replace("_", "/"))
            value_label = QLabel("0")
            value_label.setObjectName("value")

            row = (index // 3) * 2
            column = index % 3

            grid.addWidget(name_label, row, column)
            grid.addWidget(value_label, row + 1, column)

            self.count_labels[name] = value_label

        stats_layout.addLayout(grid)

        self.total_label = QLabel("Total items: 0")
        self.total_label.setObjectName("value")
        stats_layout.addWidget(self.total_label)

        self.reset_button = QPushButton("Reset Statistics")
        self.reset_button.clicked.connect(self.reset_statistics)
        stats_layout.addWidget(self.reset_button)

        main_layout.addWidget(stats_panel, 1)

        # Load the trained model
        try:
            self.model = YOLO(MODEL_PATH)
            self.status_label.setText("Status: Model ready")
        except Exception as error:
            self.status_label.setText("Status: Model loading failed")
            self.instruction_label.setText(str(error))
            self.start_button.setEnabled(False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.load_history()
        self.update_statistics()

    def get_detection_roi(self, frame):
        """Get a resized grayscale image of the central area."""
        height, width = frame.shape[:2]

        x1 = width // 4
        x2 = 3 * width // 4
        y1 = height // 5
        y2 = 4 * height // 5

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        return cv2.resize(gray, (160, 120))

    def item_is_present(self, frame):
        """
        Compare the central area with the saved empty view.
        This is an approximate background-difference check.
        """
        if self.empty_roi_gray is None:
            return False

        current_roi = self.get_detection_roi(frame)

        difference = cv2.absdiff(
            self.empty_roi_gray,
            current_roi,
        )

        changed_ratio = float((difference > 25).mean())
        mean_difference = float(difference.mean())

        return (
            mean_difference >= EMPTY_DIFF_THRESHOLD
            and changed_ratio > 0.02
        )

    def start_camera(self):
        if self.model is None:
            self.status_label.setText("Status: Model unavailable")
            return

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            self.status_label.setText("Status: Cannot open camera")
            self.instruction_label.setText(
                "Check your webcam and try again."
            )
            return

        success, frame = self.camera.read()

        if not success:
            self.camera.release()
            self.camera = None
            self.status_label.setText("Status: Cannot read camera")
            return

        # Start with the detection area EMPTY.
        self.empty_roi_gray = self.get_detection_roi(frame)

        self.last_category = None
        self.stable_frames = 0
        self.item_locked = False
        self.removal_frames = 0

        self.timer.start(100)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.status_label.setText(
            "Status: Ready — place one item in the center"
        )

    def update_frame(self):
        if self.camera is None:
            return

        success, frame = self.camera.read()

        if not success:
            self.status_label.setText("Status: Camera read error")
            self.stop_camera()
            return

        try:
            item_present = self.item_is_present(frame)

            # Handle removal before allowing another count.
            if self.item_locked:
                if item_present:
                    self.removal_frames = 0
                    self.status_label.setText(
                        "Status: Remove the counted item"
                    )
                else:
                    self.removal_frames += 1

                if self.removal_frames >= REMOVAL_FRAMES_REQUIRED:
                    self.item_locked = False
                    self.last_category = None
                    self.stable_frames = 0
                    self.removal_frames = 0

                    # Learn the empty area again after removal.
                    self.empty_roi_gray = self.get_detection_roi(frame)

                    self.status_label.setText(
                        "Status: Ready for next item"
                    )

            result = self.model.predict(
                frame,
                imgsz=224,
                verbose=False,
            )[0]

            class_id = int(result.probs.top1)
            category = result.names[class_id]
            confidence = float(result.probs.top1conf) * 100
            self.last_confidence = confidence

            if confidence >= CONFIDENCE_THRESHOLD:
                self.category_label.setText(
                    category.replace("_", "/")
                )
                self.confidence_label.setText(
                    f"{confidence:.1f}%"
                )
                self.instruction_label.setText(
                    INSTRUCTIONS.get(
                        category,
                        "Check local disposal instructions.",
                    )
                )

                if not self.item_locked and item_present:
                    if category == self.last_category:
                        self.stable_frames += 1
                    else:
                        self.last_category = category
                        self.stable_frames = 1

                    if self.stable_frames >= STABLE_FRAMES_REQUIRED:
                        self.counts[category] += 1
                        self.item_locked = True
                        self.removal_frames = 0

                        self.save_history(category, confidence)
                        self.update_statistics()

                        self.status_label.setText(
                            "Counted item: "
                            + category.replace("_", "/")
                        )

                elif not self.item_locked:
                    self.last_category = None
                    self.stable_frames = 0

            else:
                self.category_label.setText("UNKNOWN")
                self.confidence_label.setText(
                    f"{confidence:.1f}%"
                )
                self.instruction_label.setText(
                    "Please check this item manually."
                )

                if not self.item_locked:
                    self.last_category = None
                    self.stable_frames = 0

            # Draw the central detection area.
            height, width = frame.shape[:2]

            x1 = width // 4
            x2 = 3 * width // 4
            y1 = height // 5
            y2 = 4 * height // 5

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "Place one item here",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        except Exception as error:
            self.category_label.setText("Prediction error")
            self.instruction_label.setText(str(error))
            self.status_label.setText("Status: Processing error")

        # Display camera preview.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape

        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()

        pixmap = QPixmap.fromImage(image).scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.camera_label.setPixmap(pixmap)

    def load_history(self):
        try:
            with open(
                HISTORY_FILE,
                "r",
                newline="",
                encoding="utf-8",
            ) as file:
                reader = csv.DictReader(file)

                for row in reader:
                    category = row.get("Category", "")
                    if category in self.counts:
                        self.counts[category] += 1

        except FileNotFoundError:
            pass
        except (OSError, csv.Error) as error:
            self.status_label.setText(
                f"Status: History read error: {error}"
            )

    def update_statistics(self):
        total = 0

        for category, count in self.counts.items():
            self.count_labels[category].setText(str(count))
            total += count

        self.total_label.setText(f"Total items: {total}")

    def save_history(self, category, confidence):
        file_path = Path(HISTORY_FILE)
        file_exists = file_path.exists() and file_path.stat().st_size > 0

        with open(
            file_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(
                    ["Timestamp", "Category", "Confidence"]
                )

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                category,
                f"{confidence:.1f}%",
            ])

    def reset_statistics(self):
        answer = QMessageBox.question(
            self,
            "Reset Statistics",
            "Reset all category counts and clear saved history?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        try:
            # Clear saved history so old counts do not return.
            with open(
                HISTORY_FILE,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["Timestamp", "Category", "Confidence"]
                )

        except OSError as error:
            QMessageBox.warning(
                self,
                "Could not reset history",
                str(error),
            )
            return

        self.counts = {name: 0 for name in CATEGORIES}

        # Reset counting state.
        self.last_category = None
        self.stable_frames = 0
        self.removal_frames = 0

        # If an item is still in the detection area, keep the lock.
        # This prevents counting the same item again after reset.
        if self.camera is not None and self.item_locked:
            self.status_label.setText(
                "Statistics reset. Remove the current item first."
            )
        else:
            self.item_locked = False
            self.status_label.setText(
                "Statistics reset. Ready for a new item."
            )

        self.update_statistics()

    def stop_camera(self):
        self.timer.stop()

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.camera_label.clear()
        self.camera_label.setText("Camera stopped")
        self.start_button.setEnabled(self.model is not None)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Camera stopped")

        self.last_category = None
        self.stable_frames = 0
        self.item_locked = False
        self.removal_frames = 0
        self.empty_roi_gray = None

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartWasteUI()
    window.show()
    sys.exit(app.exec())
