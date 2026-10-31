
import sys
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
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
    QMessageBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QListWidget,
    QHeaderView,
    QSlider,
    QSizePolicy,
)

# Project paths
MODEL_PATH = "model/best.pt"
HISTORY_FILE = "waste_history.csv"
STATISTICS_FILE = "waste_statistics.csv"

# Detection settings
CONFIDENCE_THRESHOLD = 60.0
REMOVAL_FRAMES_REQUIRED = 8
PRESENCE_FRAMES_REQUIRED = 5
MIN_ITEM_MEAN_DIFFERENCE = 8.0
MIN_ITEM_CHANGED_RATIO = 0.08

BINS = {
    "Plastic": ("BIN 1 (BLUE)", "#2448c3"),
    "Paper": ("BIN 2 (YELLOW)", "#a05b08"),
    "Can_Metal": ("BIN 3 (RED)", "#a91f27"),
    "Other": ("BIN 4 (GREY)", "#344358"),
    "Organic": ("BIN 5 (GREEN)", "#087343"),
    "Glass": ("BIN 6 (CYAN)", "#08738a"),
}
STABLE_FRAMES_REQUIRED = 8
REMOVAL_FRAMES_REQUIRED = 8

# Six waste categories
CATEGORIES = [
    "Organic",
    "Glass",
    "Can_Metal",
    "Other",
    "Paper",
    "Plastic",
]

# Sorting instructions
INSTRUCTIONS = {
    "Organic": "Use the organic waste bin.",
    "Glass": "Use the glass recycling bin.",
    "Can_Metal": "Use the metal recycling bin.",
    "Other": "Check local disposal instructions.",
    "Paper": "Use the paper recycling bin.",
    "Plastic": "Use the plastic recycling bin.",
}


class PredictionWorker(QObject):
    requested = Signal(object)
    completed = Signal(int, str, float, object, bool, str)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.requested.connect(self.predict)

    @Slot(object)
    def predict(self, request):
        frame, roi, item_present, session_id = request
        try:
            result = self.model.predict(
                frame,
                imgsz=224,
                verbose=False,
            )[0]
            class_id = int(result.probs.top1)
            category = result.names[class_id]
            confidence = float(result.probs.top1conf) * 100
            self.completed.emit(
                session_id,
                category,
                confidence,
                roi,
                item_present,
                "",
            )
        except Exception as error:
            self.completed.emit(
                session_id, "", 0.0, roi, item_present, str(error)
            )


class SpeechWorker(QObject):
    requested = Signal(str)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.engine = None
        self.requested.connect(self.speak)

    @Slot(str)
    def speak(self, message):
        try:
            if self.engine is None:
                import pyttsx3

                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 170)

            self.engine.say(message)
            self.engine.runAndWait()
        except Exception as error:
            self.engine = None
            self.failed.emit(str(error))


class SmartWasteUI(QMainWindow):
    PAGES = [
        ("Home", "⌂"),
        ("Camera", "◉"),
        ("Statistics", "▤"),
        ("History", "◷"),
        ("Settings", "⚙"),
        ("Help", "?"),
    ]

class SmartWasteUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Smart Waste Sorter")
        available = QApplication.primaryScreen().availableGeometry()
        min_width = min(900, available.width())
        min_height = min(620, available.height())
        self.setMinimumSize(min_width, min_height)
        self.resize(
            min(1180, max(min_width, int(available.width() * 0.94))),
            min(760, max(min_height, int(available.height() * 0.90))),
        )
        self.resize(1100, 750)

        self.project_dir = Path(__file__).resolve().parent
        self.history_path = self.project_dir / HISTORY_FILE
        self.statistics_path = self.project_dir / STATISTICS_FILE

        self.model = None
        self.camera = None
        self.prediction_pending = False
        self.current_item_present = False
        self.unknown_alert_active = False
        self.camera_session_id = 0
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.inference_thread = None
        self.prediction_worker = None
        self.speech_thread = None
        self.speech_worker = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.fps_window_started = 0.0
        self.fps_frame_count = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.counts = {name: 0 for name in CATEGORIES}

        # Prevent counting the same item repeatedly
        self.item_locked = False
        self.removal_frames = 0
        self.item_presence_frames = 0
        self.last_category = None
        self.stable_frames = 0
        self.item_locked = False
        self.removal_frames = 0

        # Reference images for item presence
        self.empty_roi_gray = None
        self.counted_roi_gray = None
        self.latest_detection_roi_gray = None
        self.last_confidence = 0.0

        self.create_ui()
        self.prepare_csv_files()
        self.update_statistics()

        # Load trained AI model
        try:
            model_file = self.project_dir / MODEL_PATH

            if not model_file.exists():
                raise FileNotFoundError(
                    f"Model file not found:\n{model_file}"
                )

            self.model = YOLO(str(model_file))
            self.validate_model_categories()
            self.start_prediction_worker()
            self.start_speech_worker()
            self.status_label.setText("Status: Model loaded")
            self.home_status_label.setText(
                "AI model is ready. Start the camera and place one item "
                "inside the guide."
            )
            self.status_label.setText("Status: Model loaded")

        except Exception as error:
            self.status_label.setText("Status: Model loading failed")
            self.instruction_label.setText(str(error))
            self.start_button.setEnabled(False)

    def start_prediction_worker(self):
        self.inference_thread = QThread(self)
        self.prediction_worker = PredictionWorker(self.model)
        self.prediction_worker.moveToThread(self.inference_thread)
        self.prediction_worker.completed.connect(self.handle_prediction)
        self.inference_thread.start()

    def start_speech_worker(self):
        self.speech_thread = QThread(self)
        self.speech_worker = SpeechWorker()
        self.speech_worker.moveToThread(self.speech_thread)
        self.speech_worker.failed.connect(self.handle_speech_error)
        self.speech_thread.start()

    @Slot(str)
    def handle_speech_error(self, error):
        self.status_label.setText(f"Voice output error: {error}")

    def show_unknown_alert(self, reason, confidence=None):
        self.category_label.setText("UNKNOWN")
        self.confidence_label.setText(
            "--" if confidence is None else f"Confidence: {confidence:.1f}%"
        )
        self.instruction_label.setText(reason)
        self.target_bin_label.setText("TARGET BIN: CHECK MANUALLY")
        self.target_bin_label.setStyleSheet(
            "background: #a05b08; color: white; font-weight: bold; "
            "padding: 6px; border-radius: 4px;"
        )
        self.result_status_label.setText("UNKNOWN")
        self.result_status_label.setStyleSheet(
            "background: #a05b08; color: white; font-weight: bold; "
            "padding: 4px 12px; border-radius: 4px;"
        )
        self.status_label.setText(f"Status: UNKNOWN — {reason}")

        if not self.unknown_alert_active and self.speech_worker is not None:
            self.speech_worker.requested.emit(f"Unknown. {reason}")
        self.unknown_alert_active = True

    def validate_model_categories(self):
        names = self.model.names
        model_categories = set(
            names.values() if isinstance(names, dict) else names
        )
        expected_categories = set(CATEGORIES)
        if model_categories != expected_categories:
            missing = sorted(expected_categories - model_categories)
            unexpected = sorted(model_categories - expected_categories)
            raise ValueError(
                "Model categories do not match the supported waste types. "
                f"Missing: {missing}; unexpected: {unexpected}"
            )

    def create_ui(self):
        central = QWidget()
        central.setObjectName("appRoot")
        central.setStyleSheet(
            "QWidget#appRoot { background-color: #0c1424; }"
            "QWidget { color: #edf2fa; font-family: Segoe UI; "
            "font-size: 14px; }"
            "QWidget#contentPage { background-color: #0c1424; }"
            "QWidget#card { background-color: #111c30; "
            "border: 1px solid #26334b; border-radius: 8px; }"
            "QTableWidget { background: #111c30; color: #edf2fa; "
            "gridline-color: #26334b; alternate-background-color: #172238; "
            "selection-background-color: #087343; }"
            "QHeaderView::section { background: #1a2639; color: #edf2fa; "
            "padding: 8px; border: 1px solid #26334b; }"
            "QSlider::groove:horizontal { background: #344358; height: 5px; }"
            "QSlider::handle:horizontal { background: #22b96f; width: 14px; "
            "margin: -5px 0; border-radius: 7px; }"
        )
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.build_header())

        self.pages = QStackedWidget()
        self.pages.setStyleSheet(
            "QStackedWidget { background-color: #0c1424; }"
        )
        root.addWidget(self.pages, 1)

        self.count_labels = {}
        self.home_total_label = QLabel("0")
        self.home_category_label = QLabel("0")
        self.home_status_label = QLabel("Model is loading...")

        self.build_home_page()
        self.build_camera_page()
        self.build_statistics_page()
        self.build_history_page()
        self.build_settings_page()
        self.build_help_page()
        self.show_page("Home")

    def build_header(self):
        header = QWidget()
        header.setObjectName("appHeader")
        header.setFixedHeight(118)
        header.setStyleSheet(
            "QWidget#appHeader { background-color: #0c1424; "
            "border-bottom: 1px solid #202b40; }"
        )
        layout = QVBoxLayout(header)
        layout.setContentsMargins(22, 10, 22, 8)
        layout.setSpacing(6)

        brand_row = QHBoxLayout()
        mark = QLabel("♻")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(42, 42)
        mark.setStyleSheet(
            "font-size: 22px; color: #7de4ad; background-color: #164b36; "
            "border-radius: 21px;"
        )
        brand_text = QVBoxLayout()
        brand = QLabel("Smart Waste Sorter")
        brand.setStyleSheet(
            "font-size: 19px; font-weight: bold; color: #f4f7fb;"
        )
        subtitle = QLabel("AI-powered waste sorting dashboard")
        subtitle.setStyleSheet("font-size: 12px; color: #91a0b7;")
        brand_text.addWidget(brand)
        brand_text.addWidget(subtitle)
        brand_row.addWidget(mark)
        brand_row.addSpacing(6)
        brand_row.addLayout(brand_text)
        brand_row.addStretch()
        layout.addLayout(brand_row)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        self.current = "Home"
        self.nav_buttons = {}
        for name, icon in self.PAGES:
            button = QPushButton(f"{icon}  {name}")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(
                "QPushButton { color: #aab6c9; background: transparent; "
                "border: none; border-radius: 5px; padding: 8px 14px; }"
                "QPushButton:hover { background: #18243a; color: white; }"
                "QPushButton:checked { background: #087343; color: white; "
                "font-weight: bold; }"
            )
            button.clicked.connect(
                lambda _checked=False, page=name: self.show_page(page)
            )
            nav_row.addWidget(button)
            self.nav_buttons[name] = button
        nav_row.addStretch()
        layout.addLayout(nav_row)
        self.nav_buttons[self.current].setChecked(True)
        return header

    def build_home_page(self):
        page = self.make_page()
        layout = page.layout()
        self.add_page_heading(layout, "Welcome to EcoSort AI",
                             "Your AI-powered recycling assistant")

        cards = QHBoxLayout()
        cards.setSpacing(14)
        cards.addWidget(self.make_summary_card(
            "Items sorted", self.home_total_label
        ))
        cards.addWidget(self.make_summary_card(
            "Waste categories", self.home_category_label
        ))
        layout.addLayout(cards)

        status_card = self.make_card()
        status_layout = QVBoxLayout(status_card)
        status_title = QLabel("Getting started")
        status_title.setStyleSheet("font-size: 18px; font-weight: bold;")
        status_layout.addWidget(status_title)
        self.home_status_label.setWordWrap(True)
        status_layout.addWidget(self.home_status_label)
        actions = QHBoxLayout()
        camera_action = self.make_button("Open camera", primary=True)
        camera_action.clicked.connect(lambda: self.show_page("Camera"))
        stats_action = self.make_button("View statistics")
        stats_action.clicked.connect(lambda: self.show_page("Statistics"))
        actions.addWidget(camera_action)
        actions.addWidget(stats_action)
        actions.addStretch()
        status_layout.addLayout(actions)
        layout.addWidget(status_card)
        layout.addStretch()
        self.pages.addWidget(page)

    def build_camera_page(self):
        page = QWidget()
        page.setObjectName("cameraPage")
        page.setStyleSheet(
            "QWidget { color: #edf2fa; font-family: Segoe UI; "
            "font-size: 13px; }"
            "QWidget#cameraPage { background-color: #0c1424; }"
            "QWidget#dashboardCard { background-color: #111c30; "
            "border: 1px solid #26334b; border-radius: 8px; }"
        )
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 12, 18, 14)
        layout.setSpacing(10)

        content = QHBoxLayout()
        content.setSpacing(12)
        layout.addLayout(content, 1)

        camera_card = QWidget()
        camera_card.setObjectName("dashboardCard")
        camera_layout = QVBoxLayout(camera_card)
        camera_layout.setContentsMargins(10, 10, 10, 8)
        self.camera_label = QLabel("Click START SCANNING to begin")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setMinimumSize(360, 250)
        self.camera_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.camera_label.setStyleSheet(
            "background-color: #070c16; color: #93a2b9; "
            "border: 1px solid #27334a; border-radius: 6px;"
        )
        camera_layout.addWidget(self.camera_label, 1)
        self.fps_label = QLabel("● Camera Inactive")
        self.fps_label.setStyleSheet("color: #91a0b7; font-size: 12px;")
        camera_layout.addWidget(self.fps_label)
        self.status_label = QLabel("Status: Waiting")
        self.status_label.setStyleSheet("color: #91a0b7; font-size: 12px;")
        camera_layout.addWidget(self.status_label)
        content.addWidget(camera_card, 3)

        right_column = QVBoxLayout()
        right_column.setSpacing(8)
        content.addLayout(right_column, 2)

        result_card = QWidget()
        result_card.setObjectName("dashboardCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(12, 9, 12, 9)
        result_layout.setSpacing(6)
        result_title = QLabel("LIVE CLASSIFICATION")
        result_title.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #bac6d8;"
        )
        result_layout.addWidget(result_title)

        category_row = QHBoxLayout()
        category_caption = QLabel("Category")
        self.category_label = QLabel("--")
        self.category_label.setStyleSheet(
            "color: #ffd84d; font-weight: bold; font-size: 14px; "
            "background: #1a2639; border-radius: 10px; padding: 3px 8px;"
        )
        category_row.addWidget(category_caption)
        category_row.addStretch()
        category_row.addWidget(self.category_label)
        result_layout.addLayout(category_row)

        confidence_row = QHBoxLayout()
        confidence_caption = QLabel("Confidence")
        self.confidence_label = QLabel("--")
        self.confidence_label.setStyleSheet(
            "color: #62e49a; font-weight: bold; font-size: 14px;"
        )
        confidence_row.addWidget(confidence_caption)
        confidence_row.addStretch()
        confidence_row.addWidget(self.confidence_label)
        result_layout.addLayout(confidence_row)

        self.instruction_label = QLabel("Point the camera at an item.")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet(
            "padding: 7px 9px; background-color: #14283a; color: #cbd6e6; "
            "border-radius: 5px;"
        )
        self.instruction_label.setMaximumHeight(48)
        result_layout.addWidget(self.instruction_label)

        target_bin_layout = QHBoxLayout()
        self.target_bin_label = QLabel("TARGET BIN: POINT CAMERA AT ITEM")
        self.target_bin_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.target_bin_label.setStyleSheet(
            "background: #344358; color: white; font-weight: bold; "
            "padding: 6px; border-radius: 4px;"
        )
        target_bin_layout.addWidget(self.target_bin_label)
        result_layout.addLayout(target_bin_layout)

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status"))
        status_row.addStretch()
        self.result_status_label = QLabel("WAITING")
        self.result_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_status_label.setStyleSheet(
            "background: #344358; color: #d7dfeb; font-weight: bold; "
            "padding: 4px 12px; border-radius: 4px;"
        )
        status_row.addWidget(self.result_status_label)
        result_layout.addLayout(status_row)
        right_column.addWidget(result_card, 2)

        controls_card = QWidget()
        controls_card.setObjectName("dashboardCard")
        controls_layout = QVBoxLayout(controls_card)
        controls_layout.setContentsMargins(12, 8, 12, 8)
        controls_layout.setSpacing(6)
        controls_title = QLabel("Scanner controls")
        controls_title.setStyleSheet("font-weight: bold;")
        controls_layout.addWidget(controls_title)

        self.start_button = QPushButton("SCAN")
        self.stop_button = QPushButton("PAUSE")
        self.reset_statistics_button = QPushButton("RESET")
        self.start_button.setToolTip("Start or resume scanning")
        self.stop_button.setToolTip("Pause the live camera preview")
        self.reset_statistics_button.setToolTip("Reset all saved category counts")
        self.stop_button.setEnabled(False)
        self.start_button.setStyleSheet(
            "QPushButton { background: #087a43; color: white; "
            "border: none; border-radius: 4px; padding: 8px 4px; "
            "font-weight: bold; }"
            "QPushButton:hover { background: #099451; }"
        )
        self.stop_button.setStyleSheet(
            "QPushButton { background: #b95708; color: white; "
            "border: none; border-radius: 4px; padding: 8px 4px; "
            "font-weight: bold; }"
            "QPushButton:hover { background: #d16a11; }"
            "QPushButton:disabled { background: #344358; color: #aeb9ca; }"
        )
        self.reset_statistics_button.setStyleSheet(
            "QPushButton { background: #344358; color: #e4e9f1; "
            "border: none; border-radius: 4px; padding: 8px 4px; }"
            "QPushButton:hover { background: #41536c; }"
        )
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.pause_camera)
        self.reset_statistics_button.clicked.connect(self.reset_statistics)
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(6)
        actions_layout.addWidget(self.start_button, 1)
        actions_layout.addWidget(self.stop_button, 1)
        actions_layout.addWidget(self.reset_statistics_button, 1)
        controls_layout.addLayout(actions_layout)

        self.scanned_items_list = QListWidget()
        self.scanned_items_list.setAlternatingRowColors(True)
        self.scanned_items_list.setMinimumHeight(34)
        self.scanned_items_list.setMaximumHeight(40)
        self.scanned_items_list.setStyleSheet(
            "QListWidget { background: #0d1728; color: #dbe5f3; "
            "border: 1px solid #26334b; }"
        )
        controls_layout.addWidget(self.scanned_items_list)
        right_column.addWidget(controls_card, 1)

        bins_card = QWidget()
        bins_card.setObjectName("dashboardCard")
        bins_layout = QVBoxLayout(bins_card)
        bins_layout.setContentsMargins(8, 6, 8, 6)
        bins_layout.setSpacing(4)
        bins_title = QLabel("SORTING BINS")
        bins_title.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #bac6d8;"
        )
        bins_layout.addWidget(bins_title)
        bin_row = QHBoxLayout()
        bin_row.setSpacing(6)
        self.bin_cards = {}
        for category in (
            "Plastic", "Paper", "Can_Metal", "Other", "Organic", "Glass"
        ):
            bin_name, color = BINS[category]
            bin_tile = QWidget()
            bin_tile.setStyleSheet(
                "background: #172238; border: 1px solid #28364d; "
                "border-radius: 5px;"
            )
            tile_layout = QVBoxLayout(bin_tile)
            tile_layout.setContentsMargins(4, 4, 4, 4)
            tile_layout.setSpacing(3)
            bin_name_label = QLabel(bin_name)
            bin_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            bin_name_label.setStyleSheet(
                f"background: {color}; color: white; font-size: 9px; "
                "font-weight: bold; padding: 4px 2px; border-radius: 4px;"
            )
            category_label = QLabel(category.replace("_", " / "))
            category_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            category_label.setStyleSheet(
                "color: #aebbd0; font-size: 9px;"
            )
            tile_layout.addWidget(bin_name_label)
            tile_layout.addWidget(category_label)
            bin_row.addWidget(bin_tile, 1)
            self.bin_cards[category] = bin_name_label
        bins_layout.addLayout(bin_row)
        layout.addWidget(bins_card)
        self.pages.addWidget(page)

    def show_page(self, name):
        self.current = name
        for page_name, button in self.nav_buttons.items():
            button.setChecked(page_name == name)
        page_index = [page for page, _ in self.PAGES].index(name)
        self.pages.setCurrentIndex(page_index)
        if name == "History":
            self.refresh_history()

    def make_page(self):
        page = QWidget()
        page.setObjectName("contentPage")
        page.setStyleSheet(
            "QWidget#contentPage { background-color: #0c1424; "
            "color: #edf2fa; font-family: Segoe UI; }"
        )
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 22)
        layout.setSpacing(16)
        return page

    def make_card(self):
        card = QWidget()
        card.setObjectName("card")
        card.setStyleSheet(
            "QWidget#card { background-color: #111c30; "
            "border: 1px solid #26334b; border-radius: 8px; }"
        )
        return card

    def make_button(self, text, primary=False):
        button = QPushButton(text)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        if primary:
            button.setStyleSheet(
                "QPushButton { background: #087a43; color: white; "
                "border: none; border-radius: 5px; padding: 10px 16px; "
                "font-weight: bold; }"
                "QPushButton:hover { background: #099451; }"
            )
        else:
            button.setStyleSheet(
                "QPushButton { background: #24334b; color: #e4e9f1; "
                "border: 1px solid #34445e; border-radius: 5px; "
                "padding: 10px 16px; }"
                "QPushButton:hover { background: #344358; }"
            )
        return button

    def add_page_heading(self, layout, title, subtitle):
        heading = QLabel(title)
        heading.setStyleSheet(
            "font-size: 25px; font-weight: bold; color: #f4f7fb;"
        )
        description = QLabel(subtitle)
        description.setStyleSheet("font-size: 14px; color: #a4b2c7;")
        layout.addWidget(heading)
        layout.addWidget(description)

    def make_summary_card(self, title, value_label):
        card = self.make_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 14px; color: #a4b2c7;")
        layout.addWidget(title_label)
        value_label.setStyleSheet(
            "font-size: 30px; font-weight: bold; color: #72e1a1; "
            "background-color: #17263a; padding: 12px; border-radius: 6px;"
        )
        layout.addWidget(value_label)
        return card

    def build_statistics_page(self):
        page = self.make_page()
        layout = page.layout()
        self.add_page_heading(
            layout, "Waste statistics", "A summary of items sorted by category"
        )
        card = self.make_card()
        card_layout = QVBoxLayout(card)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(16)
        for index, category in enumerate(CATEGORIES):
            row, column = divmod(index, 3)
            tile = self.make_card()
            tile_layout = QVBoxLayout(tile)
            name_label = QLabel(category.replace("_", "/"))
            name_label.setStyleSheet("color: #a4b2c7; font-size: 14px;")
            value_label = QLabel("0")
            value_label.setStyleSheet(
                "font-size: 28px; font-weight: bold; color: #72e1a1;"
            )
            tile_layout.addWidget(name_label)
            tile_layout.addWidget(value_label)
            grid.addWidget(tile, row, column)
            self.count_labels[category] = value_label
        card_layout.addLayout(grid)
        total_row = QHBoxLayout()
        total_row.addWidget(QLabel("Total items counted"))
        self.total_label = QLabel("0")
        self.total_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #72e1a1;"
        )
        total_row.addStretch()
        total_row.addWidget(self.total_label)
        card_layout.addLayout(total_row)
        layout.addWidget(card)

        reset_button = self.make_button("Reset statistics")
        reset_button.clicked.connect(self.reset_statistics)
        layout.addWidget(reset_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()
        self.pages.addWidget(page)

    def build_history_page(self):
        page = self.make_page()
        layout = page.layout()
        self.add_page_heading(
            layout, "Detection history", "Previously classified waste items"
        )
        self.history_table = QTableWidget(0, 3)
        self.history_table.setHorizontalHeaderLabels(
            ["Timestamp", "Category", "Confidence"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.history_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setStyleSheet(
            "QTableWidget { background: #111c30; color: #edf2fa; "
            "gridline-color: #26334b; alternate-background-color: #172238; "
            "selection-background-color: #087343; }"
            "QHeaderView::section { background: #1a2639; color: #edf2fa; "
            "padding: 8px; border: 1px solid #26334b; }"
        )
        layout.addWidget(self.history_table, 1)
        self.pages.addWidget(page)

    def build_settings_page(self):
        page = self.make_page()
        layout = page.layout()
        self.add_page_heading(
            layout, "Settings", "Adjust classification sensitivity"
        )
        card = self.make_card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        threshold_heading = QLabel("Minimum confidence")
        threshold_heading.setStyleSheet(
            "font-size: 17px; font-weight: bold;"
        )
        card_layout.addWidget(threshold_heading)
        self.threshold_value_label = QLabel(
            f"{self.confidence_threshold:.0f}%"
        )
        self.threshold_value_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #72e1a1;"
        )
        card_layout.addWidget(self.threshold_value_label)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 95)
        self.threshold_slider.setValue(int(self.confidence_threshold))
        self.threshold_slider.valueChanged.connect(
            self.set_confidence_threshold
        )
        card_layout.addWidget(self.threshold_slider)
        hint = QLabel(
            "Higher values require more confident predictions before "
            "showing a category."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #a4b2c7;")
        card_layout.addWidget(hint)
        layout.addWidget(card)
        layout.addStretch()
        self.pages.addWidget(page)

    def build_help_page(self):
        page = self.make_page()
        layout = page.layout()
        self.add_page_heading(
            layout, "Help", "How to sort an item with EcoSort AI"
        )
        card = self.make_card()
        help_layout = QVBoxLayout(card)
        help_layout.setContentsMargins(20, 18, 20, 18)
        help_text = QLabel(
            "1. Open Camera and select Start Camera.\n\n"
            "2. Keep the camera still and place one item in the center "
            "of the green guide.\n\n"
            "3. Wait for the category and recycling instruction. The app "
            "counts the item when it gets a confident result.\n\n"
            "4. Remove the counted item before presenting the next one.\n\n"
            "Use Statistics to review totals and History to see past "
            "classifications."
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("font-size: 15px; line-height: 1.5;")
        help_layout.addWidget(help_text)
        layout.addWidget(card)
        layout.addStretch()
        self.pages.addWidget(page)

    def set_confidence_threshold(self, value):
        self.confidence_threshold = float(value)
        self.threshold_value_label.setText(f"{value}%")

    def scan_again(self):
        if self.camera is None:
            self.start_camera()
            return

        if self.item_locked:
            self.status_label.setText(
                "Status: Remove the counted item before scanning again"
            )
            return

        self.category_label.setText("No item")
        self.confidence_label.setText("Confidence: --")
        self.instruction_label.setText("Point the camera at an item.")
        self.status_label.setText(
            "Status: Ready — place one item in the center"
        )

    def reset_scanned_items(self):
        self.scanned_items_list.clear()
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

        # Classification results
        result_panel = QVBoxLayout()

        result_title = QLabel("Classification Result")
        result_title.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )
        result_panel.addWidget(result_title)

        self.category_label = QLabel("No item")
        self.category_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        self.category_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; "
            "padding: 10px; background-color: #e8f5e9;"
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
                "font-size: 20px; font-weight: bold; padding: 5px;"
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

        # Control buttons
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

        if self.statistics_path.exists():
            self.load_statistics()
        else:
            self.save_statistics()

    def load_statistics(self):
        counts = {name: 0 for name in CATEGORIES}

        with open(
            self.statistics_path,
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            rows = list(csv.reader(file))

        if not rows or rows[0] != ["Category", "Count"]:
            raise ValueError(
                f"Invalid statistics file header: {self.statistics_path}"
            )

        loaded_categories = set()
        for line_number, row in enumerate(rows[1:], start=2):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(
                    f"Invalid statistics row {line_number} in "
                    f"{self.statistics_path}"
                )

            category, raw_count = row
            if category == "TOTAL":
                continue
            if category not in counts or category in loaded_categories:
                raise ValueError(
                    f"Invalid or duplicate category on statistics row "
                    f"{line_number}: {category}"
                )

            try:
                count = int(raw_count)
            except ValueError as error:
                raise ValueError(
                    f"Invalid count on statistics row {line_number}: "
                    f"{raw_count}"
                ) from error

            if count < 0:
                raise ValueError(
                    f"Negative count on statistics row {line_number}: "
                    f"{count}"
                )

            counts[category] = count
            loaded_categories.add(category)

        self.counts = counts

    def get_detection_bounds(self, frame):
        height, width = frame.shape[:2]
        if not self.statistics_path.exists():
            self.save_statistics()

    def get_detection_roi(self, frame):
        height, width = frame.shape[:2]

        x1 = width // 4
        x2 = 3 * width // 4
        y1 = height // 5
        y2 = 4 * height // 5
        return x1, y1, x2, y2

    def get_detection_roi(self, frame):
        x1, y1, x2, y2 = self.get_detection_bounds(frame)

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        return cv2.resize(gray, (160, 120))

    def roi_difference(self, first, second):
        if first is None or second is None:
            return 0.0, 0.0

        difference = cv2.absdiff(first, second)
        changed_ratio = float((difference > 25).mean())
        mean_difference = float(difference.mean())

        return mean_difference, changed_ratio

    def item_is_present(self, frame):
        if self.empty_roi_gray is None:
            return False

        current_roi = self.get_detection_roi(frame)
        difference = (
            current_roi.astype(np.int16)
            - self.empty_roi_gray.astype(np.int16)
        )
        brightness_shift = np.median(difference)
        difference = np.abs(difference - brightness_shift)
        mean_difference = float(difference.mean())
        changed_ratio = float((difference > 25).mean())

        return (
            mean_difference >= MIN_ITEM_MEAN_DIFFERENCE
            and changed_ratio > MIN_ITEM_CHANGED_RATIO
        )

    def has_confirmed_item_presence(self, detected_change):
        if not detected_change:
            self.item_presence_frames = 0
            return False

        self.item_presence_frames += 1
        return self.item_presence_frames >= PRESENCE_FRAMES_REQUIRED

        mean_difference, changed_ratio = self.roi_difference(
            self.empty_roi_gray,
            current_roi,
        )

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
            if not self.timer.isActive():
                self.timer.start(33)
                self.start_button.setText("SCAN")
                self.start_button.setEnabled(False)
                self.stop_button.setText("PAUSE")
                self.stop_button.setEnabled(True)
                self.fps_window_started = time.perf_counter()
                self.fps_frame_count = 0
                self.fps_label.setText("● Camera Active")
                self.status_label.setText("Status: Camera resumed")
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

        self.status_label.setText(
            "Status: Preparing camera — keep the guide empty"
        )
        success = False
        frame = None
        for _ in range(30):
            success, frame = self.camera.read()
            if not success:
                break
        success, frame = self.camera.read()

        if not success:
            self.camera.release()
            self.camera = None
            self.status_label.setText(
                "Status: Cannot read camera"
            )
            return

        # Let camera exposure settle before capturing the empty-scene baseline.
        self.empty_roi_gray = self.get_detection_roi(frame)
        self.counted_roi_gray = None
        self.latest_detection_roi_gray = self.empty_roi_gray.copy()
        self.current_item_present = False

        self.item_locked = False
        self.removal_frames = 0
        self.item_presence_frames = 0

        self.timer.start(33)
        self.camera_session_id += 1
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_button.setText("PAUSE")
        self.fps_window_started = time.perf_counter()
        self.fps_frame_count = 0
        self.fps_label.setText("● Camera Active")
        self.prediction_pending = False
        self.show_page("Camera")
        # Capture an empty area before placing an item
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
        self.unknown_alert_active = False
        self.result_status_label.setText("WAITING")
        self.target_bin_label.setText("TARGET BIN: WAITING FOR ITEM")
        self.category_label.setText("--")
        self.confidence_label.setText("--")
        self.instruction_label.setText("Point the camera at an item.")

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

        now = time.perf_counter()
        self.fps_frame_count += 1
        elapsed = now - self.fps_window_started
        if elapsed >= 1.0:
            fps = self.fps_frame_count / elapsed
            self.fps_label.setText(f"● Camera Active (FPS: {fps:.0f})")
            self.fps_window_started = now
            self.fps_frame_count = 0

        current_roi = self.get_detection_roi(frame)
        self.latest_detection_roi_gray = current_roi.copy()
        detected_change = self.item_is_present(frame)
        item_present = self.has_confirmed_item_presence(
            detected_change
        )
        self.current_item_present = item_present
        self.update_item_removal_state(current_roi, item_present)

        x1, y1, x2, y2 = self.get_detection_bounds(frame)
        prediction_frame = frame[y1:y2, x1:x2].copy()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if self.category_label.text() not in {"--", "No item", "UNKNOWN"}:
            overlay = (
                f"{self.category_label.text()} "
                f"{self.confidence_label.text().replace('Confidence: ', '')}"
            )
            cv2.putText(
                frame,
                overlay.upper(),
                (x1 + 8, y1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
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
        try:
            current_roi = self.get_detection_roi(frame)
            item_present = self.item_is_present(frame)

            # Check whether the counted item was removed or changed
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
                    self.item_locked = False
                    self.counted_roi_gray = None
                    self.last_category = None
                    self.stable_frames = 0
                    self.removal_frames = 0

                    # The current view becomes the new empty baseline
                    self.empty_roi_gray = current_roi.copy()

                    self.status_label.setText(
                        "Status: Ready for next item"
                    )
                else:
                    self.status_label.setText(
                        "Status: Remove the counted item"
                    )

            # Classify the camera image
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

                # Count only when an item is present and unlocked
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

            # Draw the camera detection area
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

        # Display camera preview
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

        if self.item_locked:
            return

        if not item_present and detected_change:
            self.category_label.setText("SCANNING...")
            self.confidence_label.setText("--")
            self.instruction_label.setText(
                "Hold the material inside the guide."
            )
            self.target_bin_label.setText(
                "TARGET BIN: CHECKING MATERIAL"
            )
            self.result_status_label.setText("SCANNING")
            self.status_label.setText(
                "Status: Confirming material in the guide"
            )
            return

        if not item_present:
            self.show_unknown_alert(
                "No material detected. Place an item inside the guide."
            )
            return

        if not self.prediction_pending and self.prediction_worker is not None:
            self.prediction_pending = True
            self.prediction_worker.requested.emit((
                prediction_frame,
                current_roi,
                item_present,
                self.camera_session_id,
            ))

    def update_item_removal_state(self, current_roi, item_present):
        if not self.item_locked:
            return

        if self.counted_roi_gray is not None:
            mean_difference, changed_ratio = self.roi_difference(
                self.counted_roi_gray,
                current_roi,
            )
            item_changed = mean_difference >= 12.0 and changed_ratio > 0.05
        else:
            item_changed = not item_present

        if item_changed:
            self.removal_frames += 1
        else:
            self.removal_frames = 0

        if self.removal_frames >= REMOVAL_FRAMES_REQUIRED:
            self.item_locked = False
            self.counted_roi_gray = None
            self.removal_frames = 0
            self.empty_roi_gray = current_roi.copy()
            self.status_label.setText("Status: Ready for next item")
        else:
            self.status_label.setText("Status: Remove the counted item")

    @Slot(int, str, float, object, bool, str)
    def handle_prediction(
        self, session_id, category, confidence, current_roi, item_present,
        error
    ):
        if session_id != self.camera_session_id or self.camera is None:
            return
        self.prediction_pending = False
        if not self.current_item_present:
            self.show_unknown_alert(
                "No material detected. Place an item inside the guide."
            )
            return

        if error:
            self.category_label.setText("Prediction error")
            self.instruction_label.setText(error)
            self.status_label.setText("Status: Processing error")
            self.result_status_label.setText("ERROR")
            return

        if self.latest_detection_roi_gray is not None:
            mean_difference, changed_ratio = self.roi_difference(
                current_roi,
                self.latest_detection_roi_gray,
            )
            if mean_difference >= 12.0 and changed_ratio > 0.05:
                self.category_label.setText("SCANNING...")
                self.confidence_label.setText("--")
                self.instruction_label.setText(
                    "The item moved. Checking the current item."
                )
                self.target_bin_label.setText(
                    "TARGET BIN: CHECKING MATERIAL"
                )
                self.result_status_label.setText("SCANNING")
                self.status_label.setText(
                    "Status: Item changed; checking the current material"
                )
                return

        if category not in CATEGORIES:
            self.show_unknown_alert(
                "Unsupported material. Check disposal guidance manually.",
                confidence,
            )
            return

        self.last_confidence = confidence
        if confidence >= self.confidence_threshold:
            self.unknown_alert_active = False
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
            bin_name, bin_color = BINS.get(
                category,
                ("CHECK LOCAL RULES", "#344358"),
            )
            self.target_bin_label.setText(f"TARGET BIN: {bin_name}")
            self.target_bin_label.setStyleSheet(
                f"background: {bin_color}; color: white; "
                "font-weight: bold; padding: 9px; border-radius: 4px;"
            )
            if not self.item_locked and item_present:
                self.counts[category] += 1
                self.item_locked = True
                self.removal_frames = 0
                self.counted_roi_gray = current_roi.copy()
                self.save_history(category, confidence)
                self.update_statistics()
                self.scanned_items_list.addItem(display_category)
                self.result_status_label.setText("CONFIRMED")
                self.result_status_label.setStyleSheet(
                    "background: #087a43; color: white; "
                    "font-weight: bold; padding: 5px 16px; "
                    "border-radius: 4px;"
                )
                self.status_label.setText(
                    "Counted item: " + display_category
                )
                if self.speech_worker is not None:
                    instruction = INSTRUCTIONS.get(
                        category,
                        "Check local disposal instructions.",
                    )
                    self.speech_worker.requested.emit(
                        f"{category.replace('_', ' ')}. {instruction}"
                    )
        else:
            self.show_unknown_alert(
                "Material not recognized confidently. Check it manually.",
                confidence,
            )

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
        if hasattr(self, "history_table"):
            self.refresh_history()

    def refresh_history(self):
        try:
            with open(
                self.history_path,
                "r",
                newline="",
                encoding="utf-8",
            ) as file:
                rows = list(csv.reader(file))
        except OSError as error:
            self.status_label.setText(f"History could not be loaded: {error}")
            return

        records = rows[1:] if rows else []
        self.history_table.setRowCount(len(records))
        for row_index, record in enumerate(records):
            for column_index, value in enumerate(record[:3]):
                self.history_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(value),
                )
        self.history_table.scrollToBottom()

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

        total = sum(self.counts.values())
        self.total_label.setText(str(total))
        self.home_total_label.setText(str(total))
        self.home_category_label.setText(str(len(CATEGORIES)))
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

        # Keep the item lock while the counted item remains present
        self.status_label.setText(
            "Status: Statistics reset"
        )

    def stop_camera(self):
        self.timer.stop()
        self.camera_session_id += 1
        self.prediction_pending = False

        if self.camera is not None:
            self.camera.release()
            self.camera = None

        self.start_button.setEnabled(True)
        self.start_button.setText("SCAN")
        self.stop_button.setEnabled(False)
        self.stop_button.setText("PAUSE")
        self.camera_label.setText("Camera paused")
        self.camera_label.setPixmap(QPixmap())

        self.status_label.setText("Status: Camera stopped")
        self.fps_label.setText("● Camera Inactive")

        self.item_locked = False
        self.removal_frames = 0
        self.item_presence_frames = 0
        self.counted_roi_gray = None
        self.empty_roi_gray = None
        self.latest_detection_roi_gray = None
        self.current_item_present = False
        self.unknown_alert_active = False

    def pause_camera(self):
        if self.camera is None or not self.timer.isActive():
            return

        self.timer.stop()
        self.camera_session_id += 1
        self.prediction_pending = False
        self.item_presence_frames = 0
        self.start_button.setEnabled(True)
        self.start_button.setText("RESUME")
        self.stop_button.setEnabled(False)
        self.fps_label.setText("● Camera Paused")
        self.status_label.setText("Status: Camera paused")
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

        if self.inference_thread is not None:
            self.inference_thread.quit()
            self.inference_thread.wait()

        if self.speech_thread is not None:
            self.speech_thread.quit()
            self.speech_thread.wait()

        event.accept()


def main():
    app = QApplication(sys.argv)
    window = SmartWasteUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
