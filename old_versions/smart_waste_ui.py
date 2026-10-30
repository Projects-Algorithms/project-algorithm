
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
STABLE_FRAMES_REQUIRED = 8
REMOVAL_FRAMES_REQUIRED = 8

HISTORY_FILE = "waste_history.csv"
STATISTICS_FILE = "waste_statistics.csv"

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
        self.resize(1100, 750)

        self.project_dir = Path(__file__).resolve().parent
        self.history_path = self.project_dir / HISTORY_FILE
        self.statistics_path = self.project_dir / STATISTICS_FILE

        self.model = None
        self.camera = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.counts = {name: 0 for name in CATEGORIES}

        # Counting state
        self.last_category = None
        self.stable_frames = 0
        self.item_locked = False
        self.removal_frames = 0

        # Background and counted-item references
        self.empty_roi_gray = None
        self.counted_roi_gray = None

        self.last_confidence = 0.0

        self.create_ui()
        self.update_statistics()
        self.prepare_csv_files()

        try:
            model_file = self.project_dir / MODEL_PATH
            if not model_file.exists():
                raise FileNotFoundError(
                    f"Model file not found:\n{model_file}"
                )

            self.model = YOLO(str(model_file))
            self.status_label.setText("Status: Model loaded")

        except Exception as error:
            self.status_label.setText("Status: Model loading failed")
            self.instruction_label.setText(str(error))
            self.start_button.setEnabled(False)

    def create_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(12)

        heading = QLabel("Smart Waste Sorter")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading.setStyleSheet(
            "font-size: 26px; font-weight: bold; padding: 8px;"
        )
        main_layout.addWidget(heading)

        subtitle = QLabel(
            "AI Waste Classification and Recycling Assistant"
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, 1)

        # Camera panel
        camera_panel = QVBoxLayout()

        camera_title = QLabel("Live Camera")
        camera_title.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )
        camera_panel.addWidget(camera_title)

        self.camera_label = QLabel(
            "Click Start Camera to begin"
        )
        self.camera_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.camera_label.setMinimumSize(520, 350)
        self.camera_label.setStyleSheet(
            "background-color: #202020; color: white; "
            "border: 1px solid #777;"
        )
        camera_panel.addWidget(self.camera_label, 1)

        self.status_label = QLabel("Status: Waiting")
        self.status_label.setWordWrap(True)
        camera_panel.addWidget(self.status_label)

        content_layout.addLayout(camera_panel, 3)

        # Prediction panel
        result_panel = QVBoxLayout()

        result_title = QLabel("Classification Result")
        result_title.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )
        result_panel.addWidget(result_title)

        self.category_label = QLabel("No item")
        self.category_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; "
            "padding: 10px; background-color: #e8f5e9;"
        )
        self.category_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        result_panel.addWidget(self.category_label)

        self.confidence_label = QLabel("Confidence: --")
        result_panel.addWidget(self.confidence_label)

        self.instruction_label = QLabel(
            "Place one item in the center of the camera."
        )
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setMinimumHeight(55)
        result_panel.addWidget(self.instruction_label)

        result_panel.addSpacing(10)

        stats_title = QLabel("Waste Statistics")
        stats_title.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )
        result_panel.addWidget(stats_title)

        grid = QGridLayout()
        self.count_labels = {}

        for index, name in enumerate(CATEGORIES):
            row = (index // 3) * 2
            column = index % 3

            name_label = QLabel(name.replace("_", "/"))
            value_label = QLabel("0")
            value_label.setObjectName("value")
            value_label.setAlignment(
                Qt.AlignmentFlag.AlignCenter
            )
            value_label.setStyleSheet(
                "font-size: 20px; font-weight: bold; "
                "padding: 5px;"
            )

            grid.addWidget(name_label, row, column)
            grid.addWidget(value_label, row + 1, column)
            self.count_labels[name] = value_label

        result_panel.addLayout(grid)

        total_row = QHBoxLayout()
        total_row.addWidget(QLabel("Total Items Counted:"))
        self.total_label = QLabel("0")
        self.total_label.setStyleSheet(
            "font-size: 20px; font-weight: bold;"
        )
        total_row.addWidget(self.total_label)
        total_row.addStretch()
        result_panel.addLayout(total_row)

        result_panel.addStretch()
        content_layout.addLayout(result_panel, 2)

        # Buttons
        buttons = QHBoxLayout()

        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.reset_button = QPushButton("Reset Statistics")
        self.quit_button = QPushButton("Quit")

        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.reset_button.clicked.connect(self.reset_statistics)
        self.quit_button.clicked.connect(self.close)

        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.reset_button)
        buttons.addWidget(self.quit_button)

        main_layout.addLayout(buttons)

    def prepare_csv_files(self):
        if not self.history_path.exists():
            with open(
                self.history_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Timestamp",
                    "Category",
                    "Confidence",
                ])

        self.save_statistics()

    def get_detection_roi(self, frame):
        height, width = frame.shape[:2]

        x1 = width // 4
        x2 = 3 * width // 4
        y1 = height // 5
        y2 = 4 * height // 5

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        return cv2.resize(gray, (160, 120))

    def roi_difference(self, first, second):
        if first is None or second is None:
            return 0.0

        difference = cv2.absdiff(first, second)
        changed_ratio = float((difference > 25).mean())
        mean_difference = float(difference.mean())

        return mean_difference, changed_ratio

    def item_is_present(self, frame):
        if self.empty_roi_gray is None:
            return False

        current_roi = self.get_detection_roi(frame)
        result = self.roi_difference(
            self.empty_roi_gray,
            current_roi,
        )

        if isinstance(result, float):
            return False

        mean_difference, changed_ratio = result

        return (
            mean_difference >= 12.0
            and changed_ratio > 0.05
        )

    def start_camera(self):
        if self.model is None:
            self.status_label.setText(
                "Status: Model unavailable"
            )
            return

        if self.camera is not None:
            return

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            self.status_label.setText(
                "Status: Cannot open camera"
            )
            self.instruction_label.setText(
                "Check your webcam and try again."
            )
            return

        success, frame = self.camera.read()

        if not success:
            self.camera.release()
            self.camera = None
            self.status_label.setText(
                "Status: Cannot read camera"
            )
            return

        # Start with an empty detection area.
        self.empty_roi_gray = self.get_detection_roi(frame)
        self.counted_roi_gray = None

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
            self.status_label.setText(
                "Status: Camera read error"
            )
            self.stop_camera()
            return

        try:
            current_roi = self.get_detection_roi(frame)
            item_present = self.item_is_present(frame)

            # While locked, compare the current image with the
            # image of the item that was counted.
            if self.item_locked:
                if self.counted_roi_gray is not None:
                    mean_difference, changed_ratio = (
                        self.roi_difference(
                            self.counted_roi_gray,
                            current_roi,
                        )
                    )

                    item_changed = (
                        mean_difference >= 12.0
                        and changed_ratio > 0.05
                    )
                else:
                    item_changed = not item_present

                if item_changed:
                    self.removal_frames += 1
                else:
                    self.removal_frames = 0

                if self.removal_frames >= REMOVAL_FRAMES_REQUIRED:
                    # The counted item was removed or changed.
                    self.item_locked = False
                    self.counted_roi_gray = None
                    self.last_category = None
                    self.stable_frames = 0
                    self.removal_frames = 0

                    # Use the current view as the new empty baseline.
                    self.empty_roi_gray = current_roi.copy()

                    self.status_label.setText(
                        "Status: Ready for next item"
                    )
                else:
                    self.status_label.setText(
                        "Status: Remove the counted item"
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
                display_category = category.replace("_", "/")

                self.category_label.setText(display_category)
                self.confidence_label.setText(
                    f"Confidence: {confidence:.1f}%"
                )
                self.instruction_label.setText(
                    INSTRUCTIONS.get(
                        category,
                        "Check local disposal instructions.",
                    )
                )

                # Count only if unlocked and an item is present.
                if not self.item_locked and item_present:
                    if category == self.last_category:
                        self.stable_frames += 1
                    else:
                        self.last_category = category
                        self.stable_frames = 1

                    if self.stable_frames >= STABLE_FRAMES_REQUIRED:
                        self.counts[category] += 1

                        # Lock immediately after counting.
                        self.item_locked = True
                        self.removal_frames = 0
                        self.counted_roi_gray = current_roi.copy()

                        self.save_history(category, confidence)
                        self.update_statistics()

                        self.status_label.setText(
                            "Counted item: " + display_category
                        )

                elif not self.item_locked:
                    self.last_category = None
                    self.stable_frames = 0

            else:
                self.category_label.setText("UNKNOWN")
                self.confidence_label.setText(
                    f"Confidence: {confidence:.1f}%"
                )
                self.instruction_label.setText(
                    "Please check this item manually."
                )

                if not self.item_locked:
                    self.last_category = None
                    self.stable_frames = 0

            # Draw the detection area.
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
            self.status_label.setText(
                "Status: Processing error"
            )

        # Show camera preview.
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

    def save_history(self, category, confidence):
        with open(
            self.history_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                category,
                f"{confidence:.1f}",
            ])

    def save_statistics(self):
        with open(
            self.statistics_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["Category", "Count"])

            for category in CATEGORIES:
                writer.writerow([
                    category,
                    self.counts[category],
                ])

            writer.writerow([
                "TOTAL",
                sum(self.counts.values()),
            ])

    def update_statistics(self):
        for category in CATEGORIES:
            self.count_labels[category].setText(
                str(self.counts[category])
            )

        self.total_label.setText(
            str(sum(self.counts.values()))
        )

        self.save_statistics()

    def reset_statistics(self):
        answer = QMessageBox.question(
            self,
            "Reset Statistics",
            "Reset all category counts to zero?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        self.counts = {name: 0 for name in CATEGORIES}
        self.update_statistics()

        # Keep the current item lock if an item is still present.
        # This prevents reset from making one item count repeatedly.
        self.status_label.setText(
            "Status: Statistics reset"
        )

    def stop_camera(self):
        self.timer.stop()

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_label.setText("Camera stopped")
        self.camera_label.setPixmap(QPixmap())

        self.status_label.setText("Status: Camera stopped")

        self.item_locked = False
        self.last_category = None
        self.stable_frames = 0
        self.removal_frames = 0
        self.counted_roi_gray = None
        self.empty_roi_gray = None

    def closeEvent(self, event):
        self.timer.stop()

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        event.accept()


def main():
    app = QApplication(sys.argv)
    window = SmartWasteUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
