# ♻️ Smart Waste Sorter

**AI-Powered Waste Detection and Classification System**

Smart Waste Sorter is a computer vision project that uses Artificial Intelligence (AI) to detect and classify different types of waste through a webcam or camera. The system helps users identify waste categories and provides appropriate sorting instructions to encourage recycling and protect the environment.

## 📌 Project Overview

The main goal of this project is to make waste sorting easier, faster, and smarter by using AI and computer vision technology.

The application captures images from a camera, detects waste objects, identifies their categories, and displays the results to users.

## ✨ Features

- **Real-Time Waste Detection:** Detect waste objects using a webcam.
- **AI Classification:** Identify different types of waste using a trained YOLO model.
- **Confidence Score:** Display the prediction confidence for detected objects.
- **Waste Sorting Instructions:** Provide guidance on how to sort different waste categories.
- **Detection History:** Save and review previous detection results.
- **Waste Statistics:** Track waste detection data and categories.
- **Unknown Detection:** Handle objects with low-confidence predictions.
- **User-Friendly Interface:** Provide a simple and easy-to-use desktop application.

## 🗂️ Waste Categories

The system is designed to classify waste into the following categories:

| Category | Description |
|---|---|
| Organic | Food waste and biodegradable materials |
| Plastic | Plastic bottles, bags, and containers |
| Paper | Paper, cardboard, and newspapers |
| Glass | Glass bottles and containers |
| Can/Metal | Aluminum cans and metal objects |
| Other | Waste that does not belong to the main categories |

## 🛠️ Technologies Used

- **Python** – Main programming language
- **OpenCV** – Image processing and camera integration
- **Ultralytics YOLO** – AI object detection
- **PyQt** – Desktop graphical user interface
- **CSV** – Store detection history and statistics
- **Git & GitHub** – Version control and team collaboration

## 📁 Project Structure

```text
smart-waste-sorter/
│
├── dataset/
├── dataset-six-class/
├── evaluation/
├── model/
├── old_versions/
├── runs/
├── venv/
│
├── main.py
├── README.md
├── requirements.txt
├── waste_history.csv
├── waste_statistics.csv
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd smart-waste-sorter
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python main.py
```

**Note:** Make sure the trained YOLO model is available in the correct model directory before running the application.

## 🚀 How It Works

1. Start the Smart Waste Sorter application.
2. Connect and activate the webcam.
3. Place a waste object in front of the camera.
4. The AI model detects and classifies the object.
5. The application displays the waste category and confidence score.
6. View the sorting instructions and detection history.

## 👥 Team Members and Responsibilities

| Member | Role | Responsibilities |
|---|---|---|
| Student 1 | AI / Computer Vision | Dataset preparation, AI model training, and waste detection |
| Student 2 | Backend Developer | Application logic, backend development, and AI integration |
| Student 3 | UI/UX & Frontend Developer | User interface, application design, and frontend development |

## 🎯 Project Objectives

- Improve waste sorting and recycling practices.
- Reduce incorrect waste classification.
- Apply AI and computer vision to real-world problems.
- Develop a practical and user-friendly waste management system.
- Promote environmental awareness and sustainability.

## 🔮 Future Improvements

- Integrate Arduino or ESP32 with servo motors.
- Develop an automatic smart waste-sorting bin.
- Improve model accuracy and detection speed.
- Add more waste categories.
- Develop a web or mobile application.

## 📸 Project Demo

*Screenshots and demonstration videos will be added here.*

## 📄 License

This project is developed for educational purposes as part of a student team project.

---

**Smart Waste Sorter – Making Waste Sorting Smarter for a Cleaner Environment.** ♻️
