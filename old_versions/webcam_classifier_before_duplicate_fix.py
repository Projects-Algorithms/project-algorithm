
import cv2
import csv
from ultralytics import YOLO

model_path = (
    "runs/classify/runs/classify/"
    "smart_waste_six/weights/best.pt"
)

model = YOLO(model_path)

class_instructions = {
    "Organic": "Use the organic waste bin.",
    "Glass": "Use the glass recycling bin.",
    "Can_Metal": "Use the metal recycling bin.",
    "Other": "Check local disposal instructions.",
    "Paper": "Use the paper recycling bin.",
    "Plastic": "Use the plastic recycling bin.",
}

CONFIDENCE_THRESHOLD = 60.0

item_counts = {
    "Organic": 0,
    "Glass": 0,
    "Can_Metal": 0,
    "Other": 0,
    "Paper": 0,
    "Plastic": 0,
}


def save_statistics():
    with open("waste_statistics.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Category", "Count"])
        for name, count in item_counts.items():
            writer.writerow([name, count])
        writer.writerow(["TOTAL", sum(item_counts.values())])


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

print("Smart Waste Sorter started!")
print("Show ONE waste item to the camera.")
print("Press C to count the visible classified item.")
print("Press S to show statistics.")
print("Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam image.")
            break

        results = model.predict(
            frame,
            imgsz=224,
            verbose=False
        )

        result = results[0]
        class_id = int(result.probs.top1)
        category = result.names[class_id]
        confidence = float(result.probs.top1conf) * 100

        if confidence >= CONFIDENCE_THRESHOLD:
            status = "Waste classified"
            instruction = class_instructions.get(
                category,
                "Check local disposal instructions."
            )
        else:
            category = "Uncertain"
            status = "Try again with better lighting"
            instruction = "Show one item clearly."

        cv2.putText(
            frame,
            f"Category: {category}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if category != "Uncertain"
            else (0, 165, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            status,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            instruction,
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Smart Waste Sorter", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if confidence >= CONFIDENCE_THRESHOLD and category in item_counts:
                item_counts[category] += 1
                save_statistics()
                print(f"Counted {category}. Total: {sum(item_counts.values())}")
            else:
                print("Item is uncertain. Not counted.")

        elif key == ord("s"):
            print("\\n--- Waste Statistics ---")
            for name, count in item_counts.items():
                print(f"{name}: {count}")
            print(f"Total items: {sum(item_counts.values())}")
            print("------------------------")

        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
