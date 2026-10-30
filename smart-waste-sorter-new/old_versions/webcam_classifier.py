
import cv2
import csv
from ultralytics import YOLO

# Load the trained AI model
model_path = (
    "runs/classify/runs/classify/"
    "smart_waste_six/weights/best.pt"
)
model = YOLO(model_path)

# Sorting instructions
class_instructions = {
    "Organic": "Use the organic waste bin.",
    "Glass": "Use the glass recycling bin.",
    "Can_Metal": "Use the metal recycling bin.",
    "Other": "Check local disposal instructions.",
    "Paper": "Use the paper recycling bin.",
    "Plastic": "Use the plastic recycling bin.",
}

CONFIDENCE_THRESHOLD = 60.0
REQUIRED_STABLE_FRAMES = 10
REQUIRED_EMPTY_FRAMES = 15

# Category statistics
item_counts = {
    "Organic": 0,
    "Glass": 0,
    "Can_Metal": 0,
    "Other": 0,
    "Paper": 0,
    "Plastic": 0,
}


def save_statistics():
    with open(
        "waste_statistics.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Category", "Count"])

        for name, count in item_counts.items():
            writer.writerow([name, count])

        writer.writerow(["TOTAL", sum(item_counts.values())])


# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

print("Smart Waste Sorter started!")
print("Show ONE waste item clearly to the camera.")
print("Items are counted automatically after stable predictions.")
print("Remove the item before showing the next one.")
print("Press S to show statistics.")
print("Press Q to quit.")

category = "Uncertain"
confidence = 0.0
instruction = "Show one item clearly."
status = "Waiting for prediction"

previous_category = ""
stable_frames = 0
empty_frames = 0
item_counted = False

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam image.")
            break

        # Predict category
        results = model.predict(
            frame,
            imgsz=224,
            verbose=False,
        )

        result = results[0]
        class_id = int(result.probs.top1)
        predicted_category = result.names[class_id]
        confidence = float(result.probs.top1conf) * 100

        # Apply confidence threshold
        if (
            confidence >= CONFIDENCE_THRESHOLD
            and predicted_category in item_counts
        ):
            category = predicted_category
            instruction = class_instructions[category]
        else:
            category = "Uncertain"
            instruction = "Please check manually."

        # Count only after stable predictions
        if category in item_counts:
            empty_frames = 0

            if category == previous_category:
                stable_frames += 1
            else:
                previous_category = category
                stable_frames = 1

            if (
                not item_counted
                and stable_frames >= REQUIRED_STABLE_FRAMES
            ):
                item_counts[category] += 1
                item_counted = True
                status = f"Counted {category}"
                save_statistics()

                print(
                    f"Counted {category}. "
                    f"Total: {sum(item_counts.values())}"
                )

        else:
            stable_frames = 0
            previous_category = ""

            if item_counted:
                empty_frames += 1

                if empty_frames >= REQUIRED_EMPTY_FRAMES:
                    item_counted = False
                    empty_frames = 0
                    status = "Ready for next item"
                    print("Ready for the next item.")
            else:
                status = "Low confidence - check manually"

        if item_counted:
            count_status = "Item counted - remove it"
        else:
            count_status = (
                f"Stabilizing: {stable_frames}/"
                f"{REQUIRED_STABLE_FRAMES}"
            )

        # Display category
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

        # Display confidence
        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Display status and instruction
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

        cv2.putText(
            frame,
            count_status,
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255) if item_counted
            else (0, 255, 0),
            2,
        )

        cv2.imshow("Smart Waste Sorter", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            print("\n--- Waste Statistics ---")

            for name, count in item_counts.items():
                print(f"{name}: {count}")

            print(f"Total items: {sum(item_counts.values())}")
            print("------------------------")

        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Smart Waste Sorter stopped.")
