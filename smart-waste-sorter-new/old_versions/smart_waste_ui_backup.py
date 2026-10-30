
import sys
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
)

MODEL_PATH = (
    "runs/classify/runs/classify/"
    "smart_waste_six/weights/best.pt"
)

CONFIDENCE_THRESHOLD = 60.0

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
        self.resize(1150, 760)

        self.counts = {name: 0 for name in CATEGORIES}
        self.camera = None
        self.model = None

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

        # Main panels
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

        result_panel = QFrame()
        result_panel.setObjectName("panel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setSpacing(14)

        result_heading = QLabel("Detection Result")
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

        # Statistics
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

            grid.addWidget(name_label, index // 3 * 2, index % 3)
            grid.addWidget(value_label, index // 3 * 2 + 1, index % 3)

            self.count_labels[name] = value_label

        stats_layout.addLayout(grid)

        self.total_label = QLabel("Total items: 0")
        self.total_label.setObjectName("value")
        stats_layout.addWidget(self.total_label)

        main_layout.addWidget(stats_panel, 1)

        # Load model and prepare camera timer
        try:
            self.model = YOLO(MODEL_PATH)
            self.status_label.setText("Status: Model ready")
        except Exception as error:
            self.status_label.setText("Status: Model loading failed")
            self.instruction_label.setText(str(error))
            self.start_button.setEnabled(False)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

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

        self.timer.start(30)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Camera running")

    def update_frame(self):
        if self.camera is None:
            return

        success, frame = self.camera.read()

        if not success:
            self.status_label.setText("Status: Camera read error")
            self.stop_camera()
            return

        try:
            result = self.model.predict(
                frame, imgsz=224, verbose=False
            )[0]

            class_id = int(result.probs.top1)
            category = result.names[class_id]
            confidence = float(result.probs.top1conf) * 100

            if confidence >= CONFIDENCE_THRESHOLD:
                self.category_label.setText(category.replace("_", "/"))
                self.instruction_label.setText(
                    INSTRUCTIONS.get(
                        category,
                        "Check local disposal instructions."
                    )
                )
                self.confidence_label.setText(
                    f"{confidence:.1f}%"
                )
            else:
                self.category_label.setText("UNKNOWN")
                self.instruction_label.setText(
                    "Please check this item manually."
                )
                self.confidence_label.setText(
                    f"{confidence:.1f}%"
                )

        except Exception as error:
            self.category_label.setText("Prediction error")
            self.instruction_label.setText(str(error))

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

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartWasteUI()
    window.show()
    sys.exit(app.exec())
